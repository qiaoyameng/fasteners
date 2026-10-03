# -*- coding: utf-8 -*-

# Copyright 2011 OpenStack Foundation.
# All Rights Reserved.
#
#    Licensed under the Apache License, Version 2.0 (the "License"); you may
#    not use this file except in compliance with the License. You may obtain
#    a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
#    WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
#    License for the specific language governing permissions and limitations
#    under the License.

from contextlib import contextmanager
import errno
import functools
import logging
import os
from pathlib import Path
import threading
import time
import weakref
from typing import Callable
from typing import Optional
from typing import Union

from fasteners import _utils
from fasteners.process_mechanism import _interprocess_mechanism
from fasteners.process_mechanism import _interprocess_reader_writer_mechanism

LOG = logging.getLogger(__name__)

_semaphore_state_lock = threading.RLock()
_semaphore_held_slots = set()


class InterProcessSemaphoreError(Exception):
    """Base class for interprocess semaphore errors."""


class InterProcessSemaphoreCapacityError(
        InterProcessSemaphoreError, ValueError):
    """Raised when a semaphore path is used with a different capacity."""


class InterProcessSemaphoreReleaseError(
        InterProcessSemaphoreError, RuntimeError):
    """Raised when a semaphore permit cannot be released by the caller."""


def _ensure_tree(path):
    """Create a directory (and any ancestor directories required).

    :param path: Directory to create
    """
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno == errno.EEXIST:
            if not os.path.isdir(path):
                raise
            else:
                return False
        elif e.errno == errno.EISDIR:
            return False
        else:
            raise
    else:
        return True


class InterProcessSemaphorePermit:
    """A permit representing one acquired interprocess semaphore slot."""

    _instances = weakref.WeakSet()

    def __init__(self, slot: int, slot_path: bytes, slot_file,
                 semaphore: "InterProcessSemaphore"):
        self.slot = slot
        self.slot_path = slot_path
        self._slot_file = slot_file
        self._semaphore = semaphore
        self._owner_pid = os.getpid()
        self._released = False
        self._inherited_after_fork = False
        self._release_lock = threading.Lock()
        self._instances.add(self)

    def _mark_inherited_after_fork(self):
        self._inherited_after_fork = True
        slot_file = self._slot_file
        self._slot_file = None
        if slot_file is not None and not slot_file.closed:
            try:
                slot_file.close()
            except OSError:
                pass

    @classmethod
    def _after_fork_in_child(cls):
        for permit in list(cls._instances):
            permit._mark_inherited_after_fork()
        _semaphore_held_slots.clear()

    def release(self):
        """Release this permit.

        Release is allowed once and only from the process that acquired the
        permit. Permits inherited by a forked child are invalid in that child.
        """
        with self._release_lock:
            if self._inherited_after_fork:
                raise InterProcessSemaphoreReleaseError(
                    "Semaphore permit cannot be released by a forked child")
            if os.getpid() != self._owner_pid:
                raise InterProcessSemaphoreReleaseError(
                    "Semaphore permit can only be released by the process "
                    "that acquired it")
            if self._released:
                raise InterProcessSemaphoreReleaseError(
                    "Semaphore permit was already released")

            self._released = True
            try:
                _interprocess_mechanism.unlock(self._slot_file)
            except (IOError, OSError) as e:
                self._released = False
                raise InterProcessSemaphoreReleaseError(
                    "Could not release semaphore slot %s on `%s`"
                    % (self.slot, self.slot_path)) from e

            try:
                self._slot_file.close()
            except (IOError, OSError):
                self._semaphore.logger.exception(
                    "Could not close semaphore slot file `%s`",
                    self.slot_path)
            finally:
                self._slot_file = None
                _semaphore_held_slots.discard(self.slot_path)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(
        after_in_child=InterProcessSemaphorePermit._after_fork_in_child)


class InterProcessSemaphore:
    """A named counting semaphore backed by operating system locks."""

    def __init__(self,
                 path: Union[Path, str],
                 capacity: int,
                 sleep_func: Callable[[float], None] = time.sleep,
                 logger: Optional[logging.Logger] = None):
        if type(capacity) is not int or capacity <= 0:
            raise ValueError("Capacity must be a positive integer")

        self.path = os.path.realpath(_utils.canonicalize_path(path))
        self.capacity = capacity
        self.sleep_func = sleep_func
        self.logger = _utils.pick_first_not_none(logger, LOG)
        self._control_path = os.path.join(self.path, b"control")
        self._capacity_path = os.path.join(self.path, b"capacity")
        self._slot_paths = [
            os.path.join(self.path, ("slot-%d" % slot).encode())
            for slot in range(capacity)
        ]
        self._control_lock = InterProcessLock(
            self._control_path,
            sleep_func=sleep_func,
            logger=self.logger)
        self._initialize()

    def _read_capacity(self):
        try:
            with open(self._capacity_path, "r") as capacity_file:
                value = int(capacity_file.read().strip())
        except (IOError, OSError, ValueError) as e:
            raise InterProcessSemaphoreCapacityError(
                "Semaphore capacity metadata at `%s` is invalid"
                % self._capacity_path) from e
        return value

    def _write_capacity(self):
        temporary_path = self._capacity_path + b".tmp"
        with open(temporary_path, "w") as capacity_file:
            capacity_file.write(str(self.capacity))
            capacity_file.flush()
            os.fsync(capacity_file.fileno())
        os.replace(temporary_path, self._capacity_path)

    def _initialize(self):
        with _semaphore_state_lock:
            _ensure_tree(self.path)
            acquired = self._control_lock.acquire(blocking=True)
            if not acquired:
                raise InterProcessSemaphoreError(
                    "Unable to initialize semaphore at `%s`" % self.path)
            try:
                if os.path.exists(self._capacity_path):
                    existing_capacity = self._read_capacity()
                    if existing_capacity != self.capacity:
                        raise InterProcessSemaphoreCapacityError(
                            "Semaphore at `%s` has capacity %d, not %d"
                            % (self.path, existing_capacity, self.capacity))
                else:
                    for slot_path in self._slot_paths:
                        open(slot_path, "a").close()
                    self._write_capacity()

                for slot_path in self._slot_paths:
                    if not os.path.exists(slot_path):
                        open(slot_path, "a").close()
            finally:
                self._control_lock.release()

    def _try_acquire_slot(self, blocking: bool, watch: _utils.StopWatch):
        with _semaphore_state_lock:
            timeout = watch.leftover()
            acquired = self._control_lock.acquire(
                blocking=blocking, timeout=timeout)
            if not acquired:
                if not blocking or watch.expired():
                    return None
                raise _utils.RetryAgain()

            try:
                for slot, slot_path in enumerate(self._slot_paths):
                    if slot_path in _semaphore_held_slots:
                        continue

                    slot_file = None
                    try:
                        slot_file = open(slot_path, "a")
                        _interprocess_mechanism.trylock(slot_file)
                    except (IOError, OSError) as e:
                        if slot_file is not None:
                            slot_file.close()
                        if e.errno in (errno.EACCES, errno.EAGAIN):
                            continue
                        raise threading.ThreadError(
                            "Unable to acquire semaphore slot %s on `%s` due "
                            "to %s" % (slot, slot_path, e))

                    _semaphore_held_slots.add(slot_path)
                    return InterProcessSemaphorePermit(
                        slot, slot_path, slot_file, self)
            finally:
                self._control_lock.release()

            if not blocking or watch.expired():
                return None
            raise _utils.RetryAgain()

    def acquire(self,
                blocking: bool = True,
                delay: float = 0.01,
                max_delay: float = 0.1,
                timeout: Optional[float] = None
                ) -> Optional[InterProcessSemaphorePermit]:
        """Acquire one permit, waiting up to ``timeout`` seconds when blocking.

        Returns ``None`` when a blocking request times out or a non-blocking
        request cannot be satisfied immediately.
        """
        if delay < 0:
            raise ValueError("Delay must be greater than or equal to zero")
        if timeout is not None and timeout < 0:
            raise ValueError("Timeout must be greater than or equal to zero")
        if delay >= max_delay:
            max_delay = delay

        watch = _utils.StopWatch(duration=timeout)
        retry = _utils.Retry(
            delay, max_delay, sleep_func=self.sleep_func, watch=watch)
        with watch:
            return retry(self._try_acquire_slot, blocking, watch)


class InterProcessLock:
    """An interprocess lock."""

    MAX_DELAY = 0.1  # For backwards compatibility
    DELAY_INCREMENT = 0.01  # For backwards compatibility

    def __init__(self,
                 path: Union[Path, str],
                 sleep_func: Callable[[float], None] = time.sleep,
                 logger: Optional[logging.Logger] = None):
        """
        args:
            path:
                Path to the file that will be used for locking.
            sleep_func:
                Optional function to use for sleeping.
            logger:
                Optional logger to use for logging.
        """
        self.lockfile = None
        self.path = _utils.canonicalize_path(path)
        self.acquired = False
        self.sleep_func = sleep_func
        self.logger = _utils.pick_first_not_none(logger, LOG)

    def _try_acquire(self, blocking, watch):
        try:
            self.trylock()
        except (IOError, OSError) as e:
            if e.errno in (errno.EACCES, errno.EAGAIN):
                if not blocking or watch.expired():
                    return False
                else:
                    raise _utils.RetryAgain()
            else:
                raise threading.ThreadError("Unable to acquire lock on"
                                            " `%(path)s` due to"
                                            " %(exception)s" %
                                            {
                                                'path': self.path,
                                                'exception': e,
                                            })
        else:
            return True

    def _do_open(self):
        basedir = os.path.dirname(self.path)
        if basedir:
            made_basedir = _ensure_tree(basedir)
            if made_basedir:
                self.logger.log(_utils.BLATHER,
                                'Created lock base path `%s`', basedir)
        # Open in append mode so we don't overwrite any potential contents of
        # the target file. This eliminates the possibility of an attacker
        # creating a symlink to an important file in our lock path.
        if self.lockfile is None or self.lockfile.closed:
            self.lockfile = open(self.path, 'a')

    def acquire(self,
                blocking: bool = True,
                delay: float = 0.01,
                max_delay: float = 0.1,
                timeout: Optional[float] = None) -> bool:
        """Attempt to acquire the lock.

        Args:
            blocking:
                Whether to wait to try to acquire the lock.
            delay:
                When `blocking`, starting delay as well as the delay increment
                (in seconds).
            max_delay:
                When `blocking` the maximum delay in between attempts to
                acquire (in seconds).
            timeout:
                When `blocking`, maximal waiting time (in seconds).

        Returns:
            whether or not the acquisition succeeded
        """
        if delay < 0:
            raise ValueError("Delay must be greater than or equal to zero")
        if timeout is not None and timeout < 0:
            raise ValueError("Timeout must be greater than or equal to zero")
        if delay >= max_delay:
            max_delay = delay
        self._do_open()
        watch = _utils.StopWatch(duration=timeout)
        r = _utils.Retry(delay, max_delay,
                         sleep_func=self.sleep_func, watch=watch)
        with watch:
            gotten = r(self._try_acquire, blocking, watch)
        if not gotten:
            return False
        else:
            self.acquired = True
            self.logger.log(_utils.BLATHER,
                            "Acquired file lock `%s` after waiting %0.3fs [%s"
                            " attempts were required]", self.path,
                            watch.elapsed(), r.attempts)
            return True

    def _do_close(self):
        if self.lockfile is not None:
            self.lockfile.close()
            self.lockfile = None

    def __enter__(self):
        gotten = self.acquire()
        if not gotten:
            # This shouldn't happen, but just in case...
            raise threading.ThreadError("Unable to acquire a file lock"
                                        " on `%s` (when used as a"
                                        " context manager)" % self.path)
        return self

    def release(self):
        """Release the previously acquired lock."""
        if not self.acquired:
            raise threading.ThreadError("Unable to release an unacquired lock")
        try:
            self.unlock()
        except Exception as e:
            msg = "Could not unlock the acquired lock opened on `%s`", self.path
            self.logger.exception(msg)
            raise threading.ThreadError(msg) from e
        else:
            self.acquired = False
            try:
                self._do_close()
            except IOError:
                self.logger.exception("Could not close the file handle"
                                      " opened on `%s`", self.path)
            else:
                self.logger.log(_utils.BLATHER,
                                "Unlocked and closed file lock open on"
                                " `%s`", self.path)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()

    def exists(self):
        return os.path.exists(self.path)

    def trylock(self):
        _interprocess_mechanism.trylock(self.lockfile)

    def unlock(self):
        _interprocess_mechanism.unlock(self.lockfile)


class InterProcessReaderWriterLock:
    """An interprocess readers writer lock."""

    MAX_DELAY = 0.1  # for backwards compatibility
    DELAY_INCREMENT = 0.01  # for backwards compatibility

    def __init__(self,
                 path: Union[Path, str],
                 sleep_func: Callable[[float], None] = time.sleep,
                 logger: Optional[logging.Logger] = None):
        """
        Args:
            path:
                Path to the file that will be used for locking.
            sleep_func:
                Optional function to use for sleeping.
            logger:
                Optional logger to use for logging.
        """
        self.lockfile = None
        self.path = _utils.canonicalize_path(path)
        self.sleep_func = sleep_func
        self.logger = _utils.pick_first_not_none(logger, LOG)

    @contextmanager
    def read_lock(self, delay=0.01, max_delay=0.1):
        """Context manager that grans a read lock"""

        self.acquire_read_lock(blocking=True, delay=delay,
                               max_delay=max_delay, timeout=None)
        try:
            yield
        finally:
            self.release_read_lock()

    @contextmanager
    def write_lock(self, delay=0.01, max_delay=0.1):
        """Context manager that grans a write lock"""

        gotten = self.acquire_write_lock(blocking=True, delay=delay,
                                         max_delay=max_delay, timeout=None)

        if not gotten:
            # This shouldn't happen, but just in case...
            raise threading.ThreadError("Unable to acquire a file lock"
                                        " on `%s` (when used as a"
                                        " context manager)" % self.path)
        try:
            yield
        finally:
            self.release_write_lock()

    def _try_acquire(self, blocking, watch, exclusive):
        try:
            gotten = _interprocess_reader_writer_mechanism.trylock(self.lockfile, exclusive)
        except Exception as e:
            raise threading.ThreadError(
                "Unable to acquire lock on {} due to {}!".format(self.path, e))

        if gotten:
            return True

        if not blocking or watch.expired():
            return False

        raise _utils.RetryAgain()

    def _do_open(self):
        basedir = os.path.dirname(self.path)
        if basedir:
            made_basedir = _ensure_tree(basedir)
            if made_basedir:
                self.logger.log(_utils.BLATHER,
                                'Created lock base path `%s`', basedir)
        if self.lockfile is None:
            self.lockfile = _interprocess_reader_writer_mechanism.get_handle(self.path)

    def acquire_read_lock(self,
                          blocking: bool = True,
                          delay: float = 0.01,
                          max_delay: float = 0.1,
                          timeout: float = None) -> bool:
        """Attempt to acquire a reader's lock.

        Args:
            blocking:
                Whether to wait to try to acquire the lock.
            delay:
                When `blocking`, starting delay as well as the delay increment
                (in seconds).
            max_delay:
                When `blocking` the maximum delay in between attempts to
                acquire (in seconds).
            timeout:
                When `blocking`, maximal waiting time (in seconds).

        Returns:
            whether or not the acquisition succeeded
        """
        return self._acquire(blocking, delay, max_delay, timeout, exclusive=False)

    def acquire_write_lock(self,
                           blocking: bool = True,
                           delay: float = 0.01,
                           max_delay: float = 0.1,
                           timeout: float = None) -> bool:
        """Attempt to acquire a writer's lock.

        Args:
            blocking:
                Whether to wait to try to acquire the lock.
            delay:
                When `blocking`, starting delay as well as the delay increment
                (in seconds).
            max_delay:
                When `blocking` the maximum delay in between attempts to
                acquire (in seconds).
            timeout:
                When `blocking`, maximal waiting time (in seconds).

        Returns:
            whether or not the acquisition succeeded
        """
        return self._acquire(blocking, delay, max_delay, timeout, exclusive=True)

    def _acquire(self, blocking=True,
                 delay=0.01, max_delay=0.1,
                 timeout=None, exclusive=True):

        if delay < 0:
            raise ValueError("Delay must be greater than or equal to zero")
        if timeout is not None and timeout < 0:
            raise ValueError("Timeout must be greater than or equal to zero")
        if delay >= max_delay:
            max_delay = delay
        self._do_open()
        watch = _utils.StopWatch(duration=timeout)
        r = _utils.Retry(delay, max_delay,
                         sleep_func=self.sleep_func, watch=watch)
        with watch:
            gotten = r(self._try_acquire, blocking, watch, exclusive)
        if not gotten:
            return False
        else:
            self.logger.log(_utils.BLATHER,
                            "Acquired file lock `%s` after waiting %0.3fs [%s"
                            " attempts were required]", self.path,
                            watch.elapsed(), r.attempts)
            return True

    def _do_close(self):
        if self.lockfile is not None:
            _interprocess_reader_writer_mechanism.close_handle(self.lockfile)
            self.lockfile = None

    def release_write_lock(self):
        """Release the writer's lock."""
        try:
            _interprocess_reader_writer_mechanism.unlock(self.lockfile)
        except IOError:
            self.logger.exception("Could not unlock the acquired lock opened"
                                  " on `%s`", self.path)
        else:
            try:
                self._do_close()
            except IOError:
                self.logger.exception("Could not close the file handle"
                                      " opened on `%s`", self.path)
            else:
                self.logger.log(_utils.BLATHER,
                                "Unlocked and closed file lock open on"
                                " `%s`", self.path)

    def release_read_lock(self):
        """Release the reader's lock."""
        try:
            _interprocess_reader_writer_mechanism.unlock(self.lockfile)
        except IOError:
            self.logger.exception("Could not unlock the acquired lock opened"
                                  " on `%s`", self.path)
        else:
            try:
                self._do_close()
            except IOError:
                self.logger.exception("Could not close the file handle"
                                      " opened on `%s`", self.path)
            else:
                self.logger.log(_utils.BLATHER,
                                "Unlocked and closed file lock open on"
                                " `%s`", self.path)


def interprocess_write_locked(path: Union[Path, str]):
    """Acquires & releases an interprocess  **write** lock around the call into
    the decorated function

    Args:
        path: Path to the file used for locking.
    """
    lock = InterProcessReaderWriterLock(path)

    def decorator(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            with lock.write_lock():
                return f(*args, **kwargs)

        return wrapper

    return decorator


def interprocess_read_locked(path: Union[Path, str]):
    """Acquires & releases an interprocess **read** lock around the call into
    the decorated function

    Args:
        path: Path to the file used for locking.
    """
    lock = InterProcessReaderWriterLock(path)

    def decorator(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            with lock.read_lock():
                return f(*args, **kwargs)

        return wrapper

    return decorator


def interprocess_locked(path: Union[Path, str]):
    """Acquires & releases an interprocess lock around the call to the
    decorated function.

    Args:
        path: Path to the file used for locking.
   """
    lock = InterProcessLock(path)

    def decorator(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            with lock:
                return f(*args, **kwargs)

        return wrapper

    return decorator

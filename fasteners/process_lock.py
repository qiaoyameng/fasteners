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
from typing import Callable
from typing import Optional
from typing import Union

from fasteners import _utils
from fasteners.process_mechanism import _interprocess_mechanism
from fasteners.process_mechanism import _interprocess_reader_writer_mechanism

LOG = logging.getLogger(__name__)

PathLike = Union[str, bytes, os.PathLike]


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


class InterProcessLock:
    """An interprocess lock."""

    MAX_DELAY = 0.1  # For backwards compatibility
    DELAY_INCREMENT = 0.01  # For backwards compatibility

    def __init__(self,
                 path: PathLike,
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
                timeout: Optional[float] = None,
                watch: Optional[_utils.StopWatch] = None) -> bool:
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
            watch:
                Optional stopwatch supplying a shared timeout budget.

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
        owns_watch = watch is None
        if owns_watch:
            watch = _utils.StopWatch(duration=timeout)
            watch.start()
        r = _utils.Retry(delay, max_delay,
                         sleep_func=self.sleep_func, watch=watch)
        gotten = r(self._try_acquire, blocking, watch)
        if owns_watch:
            watch.stop()
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


class InterProcessLockReleaseError(threading.ThreadError):
    """Raised when releasing one or more interprocess locks fails."""

    def __init__(self, errors):
        self.errors = tuple(errors)
        if len(self.errors) == 1:
            lock, error = self.errors[0]
            message = "Unable to release lock `%s`: %s" % (lock.path, error)
        else:
            details = "; ".join(
                "`%s`: %s" % (lock.path, error)
                for lock, error in self.errors)
            message = "Unable to release %d interprocess locks: %s" % (
                len(self.errors), details)
        super().__init__(message)


class InterProcessMultiLock:
    """A set of interprocess locks acquired in a stable global order.

    Path aliases caused by relative paths, ``.``/``..`` components and
    symlinks are resolved when the object is constructed, and equivalent
    paths acquire one lock. User expansion is not performed, hard links are
    not coalesced, and a symlink changed after construction is not followed
    again.

    Locks which were already acquired when this object is constructed are
    reused but are not released by this object.
    """

    def __init__(self,
                 locks,
                 sleep_func: Callable[[float], None] = time.sleep,
                 logger: Optional[logging.Logger] = None):
        normalized_locks = []
        for item in locks:
            if _is_interprocess_lock(item):
                lock = item
            else:
                lock = InterProcessLock(
                    item, sleep_func=sleep_func, logger=logger)
            normalized_locks.append(lock)

        deduped = {}
        keys = {}
        for lock in normalized_locks:
            key = _lock_key(lock)
            existing = deduped.get(key)
            if existing is None or (
                    lock.acquired and not existing.acquired):
                if existing is not None:
                    keys.pop(id(existing), None)
                deduped[key] = lock
                keys[id(lock)] = key

        self.locks = tuple(sorted(
            deduped.values(), key=lambda item: keys[id(item)]))
        self.previously_acquired = tuple(
            lock for lock in self.locks if lock.acquired)
        self._previously_acquired_ids = {
            id(lock) for lock in self.previously_acquired
        }
        self._newly_acquired = []
        self.sleep_func = sleep_func
        self.logger = _utils.pick_first_not_none(logger, LOG)

    @property
    def acquired(self) -> bool:
        """Whether locks acquired by this object are currently held."""
        return bool(self._newly_acquired)

    def acquire(self,
                blocking: bool = True,
                delay: float = 0.01,
                max_delay: float = 0.1,
                timeout: Optional[float] = None) -> bool:
        """Acquire all locks using one shared timeout budget.

        Locks are always acquired by normalized path order regardless of the
        caller's input order. If acquisition fails or raises, locks acquired
        by this call are released in reverse order.
        """
        if delay < 0:
            raise ValueError("Delay must be greater than or equal to zero")
        if timeout is not None and timeout < 0:
            raise ValueError("Timeout must be greater than or equal to zero")
        if delay >= max_delay:
            max_delay = delay

        watch = _utils.StopWatch(duration=timeout)
        watch.start()
        newly_acquired = []
        try:
            for lock in self.locks:
                if (id(lock) in self._previously_acquired_ids or
                        lock.acquired):
                    continue
                remaining_timeout = (
                    watch.leftover() if timeout is not None else None)
                gotten = lock.acquire(
                    blocking=blocking,
                    delay=delay,
                    max_delay=max_delay,
                    timeout=remaining_timeout,
                    watch=watch)
                if not gotten:
                    self._release_locks(list(reversed(newly_acquired)))
                    return False
                newly_acquired.append(lock)
        except BaseException:
            self._release_locks(
                list(reversed(newly_acquired)))
            raise
        finally:
            watch.stop()
        self._newly_acquired = newly_acquired
        return True

    def release(self):
        """Release only locks acquired by this object, in reverse order."""
        self._release_acquired()

    def _release_acquired(self):
        locks = list(reversed(self._newly_acquired))
        self._newly_acquired = []
        self._release_locks(locks)

    def _release_locks(self, locks):
        errors = []
        for lock in locks:
            try:
                lock.release()
            except Exception as error:
                errors.append((lock, error))
        if errors:
            raise InterProcessLockReleaseError(errors)

    def __enter__(self):
        gotten = self.acquire()
        if not gotten:
            raise threading.ThreadError("Unable to acquire interprocess locks"
                                        " within the configured timeout")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            self._release_acquired()
        except Exception as release_error:
            if exc_val is not None:
                raise release_error from exc_val
            raise


def _lock_key(lock):
    return _utils.canonicalize_path_key(lock.path)


def _is_interprocess_lock(value):
    return (
        isinstance(value, InterProcessLock) or
        all(hasattr(value, name) for name in (
            'path', 'acquired', 'acquire', 'release'))
    )


@contextmanager
def interprocess_locks(
        *paths: Union[PathLike, InterProcessLock],
        delay: float = 0.01,
        max_delay: float = 0.1,
        timeout: Optional[float] = None,
        sleep_func: Callable[[float], None] = time.sleep,
        logger: Optional[logging.Logger] = None):
    """Acquire several interprocess locks as one ordered context."""
    multi_lock = InterProcessMultiLock(
        paths, sleep_func=sleep_func, logger=logger)
    if not multi_lock.acquire(
            delay=delay, max_delay=max_delay, timeout=timeout):
        raise threading.ThreadError("Unable to acquire interprocess locks"
                                    " within the configured timeout")
    with multi_lock:
        yield multi_lock


class InterProcessReaderWriterLock:
    """An interprocess readers writer lock."""

    MAX_DELAY = 0.1  # for backwards compatibility
    DELAY_INCREMENT = 0.01  # for backwards compatibility

    def __init__(self,
                 path: PathLike,
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

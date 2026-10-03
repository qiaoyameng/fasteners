# -*- coding: utf-8 -*-

# Copyright (C) 2014 Yahoo! Inc. All Rights Reserved.
# Copyright 2011 OpenStack Foundation.
#
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

import collections
import contextlib
import functools
import threading
import time
from typing import Optional

from fasteners import _utils


class ReaderWriterLock(object):
    """An inter-thread readers writer lock."""

    WRITER = 'w'  #: Writer owner type/string constant.
    READER = 'r'  #: Reader owner type/string constant.

    def __init__(self,
                 condition_cls=threading.Condition,
                 current_thread_functor=threading.current_thread):
        """
        Args:
            condition_cls:
                Optional custom `Condition` primitive used for synchronization.
            current_thread_functor:
                Optional function that returns the identity of the thread in case
                threads are not properly identified by threading.current_thread
        """
        self._writer = None
        self._writer_entries = 0
        self._pending_writers = collections.deque()
        self._readers = {}
        self._upgradeable_reader = None
        self._upgradeable_reader_entries = 0
        self._cond = condition_cls()
        self._current_thread = current_thread_functor

    @property
    def has_pending_writers(self) -> bool:
        """Check if there pending writers

        Returns:
            Whether there are pending writers.
        """
        return bool(self._pending_writers)

    def is_writer(self, check_pending: bool = True) -> bool:
        """Check if caller is a writer (optionally pending writer).

        Args:
            check_pending:
                Whether to check for pending writer status.

        Returns:
            Whether the caller is the active (or optionally pending) writer.
        """
        me = self._current_thread()
        if self._writer == me:
            return True
        if check_pending:
            return me in self._pending_writers
        else:
            return False

    def is_reader(self) -> bool:
        """Check if caller is a reader.

        Returns:
            Whether the caller is an active reader.
        """
        me = self._current_thread()
        return me in self._readers

    def is_upgradeable_reader(self) -> bool:
        """Check if caller owns an upgradeable read lock.

        Returns:
            Whether the caller is the upgradeable reader.
        """
        me = self._current_thread()
        return self._upgradeable_reader == me

    @property
    def owner(self) -> Optional[str]:
        """Caller ownership (if any) of the lock

        Returns:
            `'w'` if caller is a writer, `'r'` if caller is a reader, None otherwise.
        """
        """Returns whether the lock is locked by a writer or reader."""
        if self._writer is not None:
            return self.WRITER
        if self._readers:
            return self.READER
        return None

    def acquire_read_lock(self,
                          blocking: bool = True,
                          timeout: float = None) -> bool:
        """Acquire a read lock.

        Will wait until no active or pending writers.

        Args:
            blocking:
                Whether to wait for active or pending writers.
            timeout:
                When ``blocking``, maximum waiting time in seconds.

        Returns:
            Whether the read lock was acquired.

        Raises:
            RuntimeError: if a pending writer tries to acquire a read lock.
            ValueError: if ``timeout`` is negative.
        """
        me = self._current_thread()
        return self._acquire_read_lock(me, blocking, timeout)

    def release_read_lock(self):
        """Release a read lock.

        Raises:
            RuntimeError: if the current thread does not own a read lock.
        """
        me = self._current_thread()
        self._release_read_lock(me)

    def _acquire_read_lock(self, me, blocking=True, timeout=None):
        if me in self._pending_writers:
            raise RuntimeError("Writer %s can not acquire a read lock"
                               " while waiting for the write lock"
                               % me)
        self._validate_timeout(timeout)
        with self._cond:
            deadline = None
            if timeout is not None:
                deadline = time.monotonic() + timeout
            while True:
                # No active writer, or we are the writer;
                # Also no pending writers;
                # we are good to become a reader.
                if self._writer is None or self._writer == me:
                    if me in self._readers:
                        # ok to get a lock if current thread already has one
                        self._readers[me] = self._readers[me] + 1
                        break
                    elif ((self._writer == me)
                          or self._upgradeable_reader == me
                          or not self.has_pending_writers):
                        self._readers[me] = 1
                        break
                # An active or pending writer; guess we have to wait.
                if not blocking:
                    return False
                if deadline is None:
                    self._cond.wait()
                else:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        return False
                    if not self._cond.wait(remaining):
                        return False
            return True

    def _release_read_lock(self, me, raise_on_not_owned=True):
        # I am no longer a reader, remove *one* occurrence of myself.
        # If the current thread acquired two read locks, then it will
        # still have to remove that other read lock; this allows for
        # basic reentrancy to be possible.
        with self._cond:
            try:
                me_instances = self._readers[me]
                upgradeable_entries = 0
                if self._upgradeable_reader == me:
                    upgradeable_entries = self._upgradeable_reader_entries
                ordinary_entries = me_instances - upgradeable_entries
                if ordinary_entries <= 0:
                    raise KeyError(me)
                if me_instances > 1:
                    self._readers[me] = me_instances - 1
                else:
                    self._readers.pop(me)
            except KeyError:
                if raise_on_not_owned:
                    raise RuntimeError(f"Thread {me} does not own a read lock")
            self._cond.notify_all()

    @contextlib.contextmanager
    def read_lock(self, blocking: bool = True, timeout: float = None):
        """Context manager that grants a read lock.

        Will wait until no active or pending writers.

        Raises:
            RuntimeError: if a pending writer tries to acquire a read lock.
        """
        me = self._current_thread()
        acquired = self._acquire_read_lock(me, blocking, timeout)
        if not acquired:
            raise TimeoutError("Timed out acquiring read lock")
        try:
            yield self
        finally:
            self._release_read_lock(me, raise_on_not_owned=False)

    def acquire_upgradeable_read_lock(self,
                                      blocking: bool = True,
                                      timeout: float = None) -> bool:
        """Acquire the unique upgradeable read lock.

        Args:
            blocking:
                Whether to wait if another reader owns the upgradeable lock.
            timeout:
                When ``blocking``, maximum waiting time in seconds.

        Returns:
            Whether the upgradeable read lock was acquired.

        Raises:
            RuntimeError: if the caller already owns an ordinary read lock,
                is a pending writer, or owns the write lock.
            ValueError: if ``timeout`` is negative.
        """
        me = self._current_thread()
        return self._acquire_upgradeable_read_lock(me, blocking, timeout)

    def release_upgradeable_read_lock(self):
        """Release an upgradeable read lock.

        Raises:
            RuntimeError: if the current thread does not own the upgradeable
                read lock or is waiting to upgrade it.
        """
        me = self._current_thread()
        self._release_upgradeable_read_lock(me)

    @contextlib.contextmanager
    def upgradeable_read_lock(self,
                              blocking: bool = True,
                              timeout: float = None):
        """Context manager for the unique upgradeable read lock."""
        me = self._current_thread()
        acquired = self._acquire_upgradeable_read_lock(me, blocking, timeout)
        if not acquired:
            raise TimeoutError("Timed out acquiring upgradeable read lock")
        try:
            yield self
        finally:
            self._release_upgradeable_read_lock(me, raise_on_not_owned=False)

    def _validate_timeout(self, timeout):
        if timeout is not None and timeout < 0:
            raise ValueError("Timeout must be greater than or equal to zero")

    def _acquire_upgradeable_read_lock(self, me, blocking, timeout):
        self._validate_timeout(timeout)
        if me in self._pending_writers:
            raise RuntimeError("Writer %s can not acquire an upgradeable read"
                               " lock while waiting for the write lock" % me)
        with self._cond:
            deadline = None
            if timeout is not None:
                deadline = time.monotonic() + timeout
            while True:
                if self._upgradeable_reader == me:
                    self._upgradeable_reader_entries += 1
                    self._readers[me] = self._readers.get(me, 0) + 1
                    return True
                if self._writer == me:
                    raise RuntimeError("Writer %s can not acquire an"
                                       " upgradeable read lock while holding"
                                       " the write lock" % me)
                if me in self._readers:
                    raise RuntimeError("Reader %s must acquire the"
                                       " upgradeable read lock before acquiring"
                                       " a read lock" % me)
                if (self._writer is None
                        and not self.has_pending_writers
                        and self._upgradeable_reader is None):
                    self._upgradeable_reader = me
                    self._upgradeable_reader_entries = 1
                    self._readers[me] = 1
                    return True
                if not blocking:
                    return False
                if deadline is None:
                    self._cond.wait()
                else:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        return False
                    if not self._cond.wait(remaining):
                        return False

    def _release_upgradeable_read_lock(self, me, raise_on_not_owned=True):
        with self._cond:
            try:
                if self._upgradeable_reader != me:
                    raise KeyError(me)
                if self._writer == me:
                    raise RuntimeError("Writer %s can not release the"
                                       " upgradeable read lock until the write"
                                       " lock is downgraded or released" % me)
                if me in self._pending_writers:
                    raise RuntimeError("Upgradeable reader %s can not release"
                                       " the read lock while waiting to"
                                       " upgrade" % me)
                self._upgradeable_reader_entries -= 1
                if self._upgradeable_reader_entries == 0:
                    self._upgradeable_reader = None
                if self._readers[me] > 1:
                    self._readers[me] -= 1
                else:
                    self._readers.pop(me)
            except KeyError:
                if raise_on_not_owned:
                    raise RuntimeError(
                        f"Thread {me} does not own an upgradeable read lock")
            self._cond.notify_all()

    def _acquire_write_lock(self, me, blocking=True, timeout=None):
        if self.is_reader():
            raise RuntimeError("Reader %s to writer privilege"
                               " escalation not allowed" % me)
        self._validate_timeout(timeout)

        with self._cond:
            deadline = None
            if timeout is not None:
                deadline = time.monotonic() + timeout
            self._pending_writers.append(me)
            while True:
                # No readers, and no active writer, am I next??
                if not self._readers and self._writer is None:
                    if self._pending_writers[0] == me:
                        self._writer = self._pending_writers.popleft()
                        self._writer_entries = 1
                        return True
                if not blocking:
                    break
                if deadline is None:
                    self._cond.wait()
                else:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    if not self._cond.wait(remaining):
                        break
            self._pending_writers.remove(me)
            self._cond.notify_all()
            return False

    def _acquire_upgrade_write_lock(self, me, blocking, timeout):
        self._validate_timeout(timeout)
        if self._writer == me:
            self._writer_entries += 1
            return True
        if self._upgradeable_reader != me or me not in self._readers:
            raise RuntimeError("Thread %s does not own an upgradeable read"
                               " lock" % me)
        if me in self._pending_writers:
            raise RuntimeError("Thread %s is already waiting to upgrade" % me)

        with self._cond:
            deadline = None
            if timeout is not None:
                deadline = time.monotonic() + timeout
            self._pending_writers.appendleft(me)
            try:
                while True:
                    other_readers = [r for r in self._readers if r != me]
                    if (self._writer is None
                            and not other_readers
                            and self._pending_writers[0] == me):
                        self._writer = self._pending_writers.popleft()
                        self._writer_entries = 1
                        return True
                    if not blocking:
                        return False
                    if deadline is None:
                        self._cond.wait()
                    else:
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            return False
                        if not self._cond.wait(remaining):
                            return False
            except BaseException:
                self._pending_writers.remove(me)
                self._cond.notify_all()
                raise
            finally:
                if self._writer != me and me in self._pending_writers:
                    self._pending_writers.remove(me)
                    self._cond.notify_all()

    def _release_write_lock(self, me, raise_on_not_owned=True):
        with self._cond:
            if self._writer != me:
                if raise_on_not_owned:
                    raise RuntimeError(f"Thread {me} does not own a write lock")
                return
            self._writer = None
            self._writer_entries = 0
            self._cond.notify_all()

    def acquire_write_lock(self,
                           blocking: bool = True,
                           timeout: float = None) -> bool:
        """Acquire a write lock.

        Will wait until no active readers. Blocks readers after acquiring.

        Guaranteed for locks to be processed in fair order (FIFO).

        Args:
            blocking:
                Whether to wait for the write lock.
            timeout:
                When ``blocking``, maximum waiting time in seconds.

        Returns:
            Whether the write lock was acquired.

        Raises:
            RuntimeError: if an active reader attempts to acquire a lock.
        """
        me = self._current_thread()
        if self._writer == me:
            self._writer_entries += 1
            return True
        else:
            return self._acquire_write_lock(me, blocking, timeout)

    def acquire_upgrade_write_lock(self,
                                   blocking: bool = True,
                                   timeout: float = None) -> bool:
        """Upgrade the caller's upgradeable read lock to a write lock.

        The caller retains its read lock while waiting. Only one upgrade can
        be pending at a time, and pending writers continue to block new
        readers.

        Args:
            blocking:
                Whether to wait for existing readers and pending writers.
            timeout:
                When ``blocking``, maximum waiting time in seconds.

        Returns:
            Whether the upgrade succeeded. On failure the caller retains all
            of its previous upgradeable read lock layers.

        Raises:
            RuntimeError: if the caller does not own an upgradeable read lock.
            ValueError: if ``timeout`` is negative.
        """
        me = self._current_thread()
        return self._acquire_upgrade_write_lock(me, blocking, timeout)

    def downgrade_write_lock(self):
        """Atomically downgrade the caller's write lock to a read lock.

        An upgraded upgradeable reader becomes an upgradeable reader again. A
        plain writer becomes an ordinary reader.

        Raises:
            RuntimeError: if the caller is not the sole write-lock owner.
        """
        me = self._current_thread()
        with self._cond:
            if self._writer != me:
                raise RuntimeError(f"Thread {me} does not own a write lock")
            if self._writer_entries != 1:
                raise RuntimeError("Write lock can only be downgraded when"
                                   " its reentrancy count is one")
            if self._upgradeable_reader != me:
                self._readers[me] = self._readers.get(me, 0) + 1
            self._writer = None
            self._writer_entries = 0
            self._cond.notify_all()

    def release_write_lock(self):
        """Release a write lock.

        Raises:
            RuntimeError: if the current thread does not own a write lock.
        """
        me = self._current_thread()
        if self._writer == me:
            self._writer_entries -= 1
            if self._writer_entries == 0:
                self._release_write_lock(me)
        else:
            raise RuntimeError(f"Thread {me} does not own a write lock")

    @contextlib.contextmanager
    def write_lock(self, blocking: bool = True, timeout: float = None):
        """Context manager that grants a write lock.

        Will wait until no active readers. Blocks readers after acquiring.

        Guaranteed for locks to be processed in fair order (FIFO).

        Raises:
            TimeoutError: if ``blocking`` and the timeout expires.
            RuntimeError: if an active reader attempts to acquire a lock.
        """
        me = self._current_thread()
        if self.is_writer(check_pending=False):
            self._writer_entries += 1
            try:
                yield self
            finally:
                self._writer_entries -= 1
        else:
            acquired = self._acquire_write_lock(me, blocking, timeout)
            if not acquired:
                raise TimeoutError("Timed out acquiring write lock")
            try:
                yield self
            finally:
                self._release_write_lock(me)

    @contextlib.contextmanager
    def upgrade_write_lock(self,
                           blocking: bool = True,
                           timeout: float = None):
        """Context manager that upgrades an upgradeable read lock."""
        me = self._current_thread()
        was_writer = self._writer == me
        acquired = self._acquire_upgrade_write_lock(me, blocking, timeout)
        if not acquired:
            raise TimeoutError("Timed out upgrading read lock")
        try:
            yield self
        finally:
            if self._writer == me:
                if was_writer:
                    self.release_write_lock()
                else:
                    while self._writer_entries > 1:
                        self.release_write_lock()
                    self.downgrade_write_lock()

    upgrade = acquire_upgrade_write_lock
    downgrade = downgrade_write_lock


def locked(*args, **kwargs):
    """A locking **method** decorator.

    It will look for a provided attribute (typically a lock or a list
    of locks) on the first argument of the function decorated (typically this
    is the 'self' object) and before executing the decorated function it
    activates the given lock or list of locks as a context manager,
    automatically releasing that lock on exit.

    NOTE(harlowja): if no attribute name is provided then by default the
    attribute named '_lock' is looked for (this attribute is expected to be
    the lock/list of locks object/s) in the instance object this decorator
    is attached to.

    NOTE(harlowja): a custom logger (which will be used if lock release
    failures happen) can be provided by passing a logger instance for keyword
    argument ``logger``.

    NOTE(paulius): This function is DEPRECATED and will be kept until the end
    of time. It is potentially used by oslo, but too specific to be recommended
    for other projects
    """

    def decorator(f):
        attr_name = kwargs.get('lock', '_lock')
        logger = kwargs.get('logger')

        @functools.wraps(f)
        def wrapper(self, *args, **kwargs):
            attr_value = getattr(self, attr_name)
            if isinstance(attr_value, (tuple, list)):
                with _utils.LockStack(logger=logger) as stack:
                    for i, lock in enumerate(attr_value):
                        if not stack.acquire_lock(lock):
                            raise threading.ThreadError("Unable to acquire"
                                                        " lock %s" % (i + 1))
                    return f(self, *args, **kwargs)
            else:
                lock = attr_value
                with lock:
                    return f(self, *args, **kwargs)

        return wrapper

    # This is needed to handle when the decorator has args or the decorator
    # doesn't have args, python is rather weird here...
    if kwargs or not args:
        return decorator
    else:
        if len(args) == 1:
            return decorator(args[0])
        else:
            return decorator


def read_locked(*args, **kwargs):
    """Acquires & releases a read lock around call into decorated method.

    NOTE(harlowja): if no attribute name is provided then by default the
    attribute named '_lock' is looked for (this attribute is expected to be
    a :py:class:`.ReaderWriterLock`) in the instance object this decorator
    is attached to.

    NOTE(paulius): This function is DEPRECATED and will be kept until the end
    of time. It is potentially used by oslo, but too specific to be recommended
    for other projects
    """

    def decorator(f):
        attr_name = kwargs.get('lock', '_lock')

        @functools.wraps(f)
        def wrapper(self, *args, **kwargs):
            rw_lock = getattr(self, attr_name)
            with rw_lock.read_lock():
                return f(self, *args, **kwargs)

        return wrapper

    # This is needed to handle when the decorator has args or the decorator
    # doesn't have args, python is rather weird here...
    if kwargs or not args:
        return decorator
    else:
        if len(args) == 1:
            return decorator(args[0])
        else:
            return decorator


def write_locked(*args, **kwargs):
    """Acquires & releases a write lock around call into decorated method.

    NOTE(harlowja): if no attribute name is provided then by default the
    attribute named '_lock' is looked for (this attribute is expected to be
    a :py:class:`.ReaderWriterLock` object) in the instance object this
    decorator is attached to.

    NOTE(paulius): This function is DEPRECATED and will be kept until the end
    of time. It is potentially used by oslo, but too specific to be recommended
    for other projects
    """

    def decorator(f):
        attr_name = kwargs.get('lock', '_lock')

        @functools.wraps(f)
        def wrapper(self, *args, **kwargs):
            rw_lock = getattr(self, attr_name)
            with rw_lock.write_lock():
                return f(self, *args, **kwargs)

        return wrapper

    # This is needed to handle when the decorator has args or the decorator
    # doesn't have args, python is rather weird here...
    if kwargs or not args:
        return decorator
    else:
        if len(args) == 1:
            return decorator(args[0])
        else:
            return decorator


@contextlib.contextmanager
def try_lock(lock: threading.Lock) -> bool:
    """Context manager that attempts to acquire a lock without a timeout, and
    releases it on exit (if acquired).

    Args:
        lock:
            A lock to try to acquire.

    Returns:
        Whether the lock was acquired.

    # NOTE(harlowja): the keyword argument for 'blocking' does not work
    # in py2.x and only is fixed in py3.x (this adjustment is documented
    # and/or debated in http://bugs.python.org/issue10789); so we'll just
    # stick to the format that works in both (oddly the keyword argument
    # works in py2.x but only with reentrant locks).

    NOTE(paulius): This function is DEPRECATED and will be kept until the end
    of time. It is potentially used by oslo, but too specific to be recommended
    for other projects
    """
    was_locked = lock.acquire(False)
    try:
        yield was_locked
    finally:
        if was_locked:
            lock.release()

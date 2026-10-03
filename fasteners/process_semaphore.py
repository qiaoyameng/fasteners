# -*- coding: utf-8 -*-

# Copyright (C) 2026 Yahoo! Inc. All Rights Reserved.
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

"""A named inter-process semaphore built on OS file locks.

Each semaphore occupies a directory. The agreed capacity is recorded in a
``capacity`` file inside that directory the first time the semaphore is
created; later users must declare the same capacity or a :class:`ValueError`
is raised, so mismatched declarations can never silently create independent
limits.

Every slot is a small file locked with an OS level file lock (``flock`` on
POSIX, ``msvcrt.locking`` on Windows). Because the lock is held by the
operating system against the open file description, a holder that dies
(killed, crashed, ...) releases its slot automatically; recovery never
depends on deleting files or on heartbeat based liveness guesses.
"""

import errno
import os
import time

from fasteners import _utils
from fasteners.process_lock import _ensure_tree

if os.name == 'nt':
    import msvcrt

    def _try_lock_file(fileobj):
        try:
            msvcrt.locking(fileobj.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as e:
            if e.errno in (errno.EACCES, errno.EDEADLK):
                return False
            raise
        return True

    def _unlock_file(fileobj):
        fileobj.seek(0)
        msvcrt.locking(fileobj.fileno(), msvcrt.LK_UNLCK, 1)

else:
    import fcntl

    def _try_lock_file(fileobj):
        # flock locks are tied to the open file description, so a fork
        # child closing an inherited descriptor (or exiting) can not drop
        # the parent's lock, and two descriptors opened by the same
        # process lock independently of each other.
        try:
            fcntl.flock(fileobj.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as e:
            if e.errno in (errno.EACCES, errno.EAGAIN, errno.EWOULDBLOCK):
                return False
            raise
        return True

    def _unlock_file(fileobj):
        fcntl.flock(fileobj.fileno(), fcntl.LOCK_UN)


_CAPACITY_FILE = 'capacity'
_SLOT_FILE_TEMPLATE = 'slot_{:04d}'


def _claim_capacity(directory, capacity):
    """Record ``capacity`` in ``directory`` or verify the existing record.

    Returns the recorded capacity. Raises :class:`ValueError` when the
    directory already carries a different capacity.
    """
    capacity_path = os.path.join(directory, os.fsencode(_CAPACITY_FILE))
    tmp_path = os.path.join(
        directory, os.fsencode('.capacity.tmp.{}'.format(os.getpid())))
    with open(tmp_path, 'w') as tmp_file:
        tmp_file.write('{}\n'.format(capacity))
    try:
        # Atomic create-if-absent: fails with EEXIST when another process
        # already claimed the capacity, and never clobbers it.
        os.link(tmp_path, capacity_path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    finally:
        try:
            os.unlink(tmp_path)
        except OSError as e:
            if e.errno != errno.ENOENT:
                raise
    with open(capacity_path, 'r') as capacity_file:
        recorded = int(capacity_file.read().strip())
    if recorded != capacity:
        raise ValueError(
            "Semaphore at {!r} was created with capacity {}, "
            "which does not match the requested capacity {}".format(
                os.fsdecode(directory), recorded, capacity))
    return recorded


class SemaphorePermit:
    """Handle representing one acquired slot of an inter-process semaphore.

    The permit may only be released once, and only by the process that
    acquired it. A handle inherited across ``fork()`` must be discarded
    with :meth:`abandon` in the child instead of released.
    """

    def __init__(self, semaphore, slot, fileobj):
        self._semaphore = semaphore
        self._slot = slot
        self._fileobj = fileobj
        self._owner_pid = os.getpid()
        self._released = False

    @property
    def slot(self):
        """Index of the acquired slot (0 based)."""
        return self._slot

    @property
    def owner_pid(self):
        """PID of the process that acquired this permit."""
        return self._owner_pid

    @property
    def released(self):
        """Whether this permit has already been released or abandoned."""
        return self._released

    def release(self):
        """Release the slot back to the semaphore.

        :raises RuntimeError: if the permit was already released, or if it
            is released from a process other than the one that acquired it
            (for example a child that inherited the handle via ``fork()``).
        """
        if self._released:
            raise RuntimeError(
                "Permit for slot {} of {!r} was already released".format(
                    self._slot, os.fsdecode(self._semaphore.path)))
        if os.getpid() != self._owner_pid:
            raise RuntimeError(
                "Permit for slot {} of {!r} is owned by process {} and can "
                "not be released by process {}; use abandon() to discard "
                "an inherited handle".format(
                    self._slot, os.fsdecode(self._semaphore.path),
                    self._owner_pid, os.getpid()))
        self._released = True
        fileobj, self._fileobj = self._fileobj, None
        try:
            _unlock_file(fileobj)
        finally:
            fileobj.close()

    def abandon(self):
        """Discard this handle without unlocking the slot.

        Intended for a child process that inherited the handle through
        ``fork()``: closing the inherited descriptor does not drop the
        lock held by the acquiring process (the lock belongs to the open
        file description), so this is safe and simply detaches the child.
        """
        if self._released:
            return
        self._released = True
        fileobj, self._fileobj = self._fileobj, None
        fileobj.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if not self._released:
            self.release()

    def __del__(self):
        # Best effort cleanup; the OS releases the lock when the
        # descriptor is closed (or when the process dies) in any case.
        fileobj = getattr(self, '_fileobj', None)
        if fileobj is not None:
            self._fileobj = None
            self._released = True
            try:
                fileobj.close()
            except Exception:
                pass


class InterProcessSemaphore:
    """A named semaphore shared between processes on the same machine.

    :param path: directory that identifies the semaphore; created when
        missing. All processes using the same directory share the same
        semaphore.
    :param capacity: fixed number of slots. The first process to create
        the semaphore records the capacity; every later user must declare
        the same value or :class:`ValueError` is raised.
    :param sleep_func: optional function used to wait between attempts.
    """

    DELAY_INCREMENT = 0.01
    MAX_DELAY = 0.1

    def __init__(self, path, capacity, sleep_func=time.sleep):
        capacity = int(capacity)
        if capacity < 1:
            raise ValueError("capacity must be a positive integer")
        self.path = _utils.canonicalize_path(path)
        _ensure_tree(self.path)
        self.capacity = _claim_capacity(self.path, capacity)
        self.sleep_func = sleep_func

    def _slot_path(self, slot):
        return os.path.join(
            self.path, os.fsencode(_SLOT_FILE_TEMPLATE.format(slot)))

    def _try_acquire_once(self):
        for slot in range(self.capacity):
            fileobj = open(self._slot_path(slot), 'a+b')
            if _try_lock_file(fileobj):
                return SemaphorePermit(self, slot, fileobj)
            fileobj.close()
        return None

    def acquire(self, blocking=True, timeout=None):
        """Acquire one slot, returning a :class:`SemaphorePermit`.

        :param blocking: when false, make a single attempt and return
            ``None`` immediately when no slot is free.
        :param timeout: maximum seconds to wait for a slot; ``None``
            waits indefinitely. When the wait expires ``None`` is
            returned.
        """
        deadline = None if timeout is None else time.monotonic() + timeout
        delay = self.DELAY_INCREMENT
        while True:
            permit = self._try_acquire_once()
            if permit is not None:
                return permit
            if not blocking:
                return None
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self.sleep_func(min(delay, remaining))
            else:
                self.sleep_func(delay)
            delay = min(delay + self.DELAY_INCREMENT, self.MAX_DELAY)

    def __repr__(self):
        return "<{} path={!r} capacity={}>".format(
            type(self).__name__, os.fsdecode(self.path), self.capacity)

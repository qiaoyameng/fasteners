# -*- coding: utf-8 -*-

# Copyright 2011 OpenStack Foundation.
# Copyright 2011 Justin Santa Barbara
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

import contextlib
import errno
import multiprocessing
import os
import shutil
import sys
import tempfile
import threading
import time

import pytest

from fasteners import process_lock as pl
from fasteners.process_mechanism import _interprocess_mechanism

WIN32 = os.name == 'nt'


@contextlib.contextmanager
def scoped_child_processes(children, timeout=0.1, exitcode=0):
    for child in children:
        child.daemon = True
        child.start()
    yield
    start = time.time()
    timed_out = 0

    for child in children:
        child.join(max(timeout - (time.time() - start), 0))
        if child.is_alive():
            timed_out += 1
        child.terminate()

    if timed_out:
        msg = "{} child processes killed due to timeout\n".format(timed_out)
        sys.stderr.write(msg)

    if exitcode is not None:
        for child in children:
            c_code = child.exitcode
            msg = "Child exitcode {} != {}"
            assert c_code == exitcode, msg.format(c_code, exitcode)


def try_lock(lock_file):
    try:
        my_lock = pl.InterProcessLock(lock_file)
        my_lock.lockfile = open(lock_file, 'w')
        my_lock.trylock()
        my_lock.unlock()
        sys.exit(1)
    except IOError:
        sys.exit(0)


def inter_processlock_helper(lockname, lock_filename, pipe):
    lock2 = pl.InterProcessLock(lockname)
    lock2.lockfile = open(lock_filename, 'w')
    have_lock = False
    while not have_lock:
        try:
            lock2.trylock()
            have_lock = True
        except IOError:
            pass
    # Hold the lock and wait for the parent
    pipe.send(None)
    pipe.recv()


@pytest.fixture()
def lock_dir():
    tmp_dir = tempfile.mkdtemp()
    yield tmp_dir
    shutil.rmtree(tmp_dir, ignore_errors=True)


@pytest.fixture()
def handles_dir():
    tmp_dir = tempfile.mkdtemp()
    yield tmp_dir
    shutil.rmtree(tmp_dir, ignore_errors=True)


def test_lock_acquire_release_file_lock(lock_dir):
    lock_file = os.path.join(lock_dir, 'lock')
    lock = pl.InterProcessLock(lock_file)

    def attempt_acquire(count):
        children = [
            multiprocessing.Process(target=try_lock, args=(lock_file,))
            for i in range(count)]
        with scoped_child_processes(children, timeout=10, exitcode=None):
            pass
        return sum(c.exitcode for c in children)

    assert lock.acquire()
    try:
        acquired_children = attempt_acquire(10)
        assert acquired_children == 0
    finally:
        lock.release()

    acquired_children = attempt_acquire(5)
    assert acquired_children != 0


def test_nested_synchronized_external_works(lock_dir):
    sentinel = object()

    @pl.interprocess_locked(os.path.join(lock_dir, 'test-lock-1'))
    def outer_lock():
        @pl.interprocess_locked(os.path.join(lock_dir, 'test-lock-2'))
        def inner_lock():
            return sentinel

        return inner_lock()

    assert outer_lock() == sentinel


def _lock_files(lock_path, handles_dir, num_handles=50):
    with pl.InterProcessLock(lock_path):

        # Open some files we can use for locking
        handles = []
        for n in range(num_handles):
            path = os.path.join(handles_dir, ('file-%s' % n))
            handles.append(open(path, 'w'))

        # Loop over all the handles and try locking the file
        # without blocking, keep a count of how many files we
        # were able to lock and then unlock. If the lock fails
        # we get an IOError and bail out with bad exit code
        count = 0
        for handle in handles:
            try:
                _interprocess_mechanism.trylock(handle)
                count += 1
                _interprocess_mechanism.unlock(handle)
            except IOError:
                sys.exit(2)
            finally:
                handle.close()

        # Check if we were able to open all files
        if count != num_handles:
            raise AssertionError("Unable to open all handles")


def _do_test_lock_externally(lock_dir_, handles_dir_):
    lock_path = os.path.join(lock_dir_, "lock")

    num_handles = 50
    num_processes = 50
    args = (lock_path, handles_dir_, num_handles)
    children = [multiprocessing.Process(target=_lock_files, args=args)
                for _ in range(num_processes)]

    with scoped_child_processes(children, timeout=30, exitcode=0):
        pass


def test_lock_externally(lock_dir, handles_dir):
    _do_test_lock_externally(lock_dir, handles_dir)


def test_lock_externally_lock_dir_not_exist(lock_dir, handles_dir):
    os.rmdir(lock_dir)
    _do_test_lock_externally(lock_dir, handles_dir)


def test_lock_file_exists(lock_dir):
    lock_file = os.path.join(lock_dir, 'lock')

    @pl.interprocess_locked(lock_file)
    def foo():
        assert os.path.exists(lock_file)

    foo()


def test_bad_release(lock_dir):
    lock_file = os.path.join(lock_dir, 'lock')
    lock = pl.InterProcessLock(lock_file)
    with pytest.raises(threading.ThreadError):
        lock.release()


def test_interprocess_lock(lock_dir):
    lock_file = os.path.join(lock_dir, 'lock')
    lock_name = 'foo'

    child_pipe, them = multiprocessing.Pipe()
    child = multiprocessing.Process(
        target=inter_processlock_helper, args=(lock_name, lock_file, them))

    with scoped_child_processes((child,)):

        # Make sure the child grabs the lock first
        if not child_pipe.poll(5):
            pytest.fail('Timed out waiting for child to grab lock')

        start = time.time()
        lock1 = pl.InterProcessLock(lock_name)
        lock1.lockfile = open(lock_file, 'w')
        # NOTE(bnemec): There is a brief window between when the lock file
        # is created and when it actually becomes locked.  If we happen to
        # context switch in that window we may succeed in locking the
        # file. Keep retrying until we either get the expected exception
        # or timeout waiting.
        while time.time() - start < 5:
            try:
                lock1.trylock()
                lock1.unlock()
                time.sleep(0)
            except IOError:
                # This is what we expect to happen
                break
        else:
            pytest.fail('Never caught expected lock exception')

        child_pipe.send(None)


@pytest.mark.skipif(WIN32, reason='Windows cannot open file handles twice')
def test_non_destructive(lock_dir):
    lock_file = os.path.join(lock_dir, 'not-destroyed')
    with open(lock_file, 'w') as f:
        f.write('test')
    with pl.InterProcessLock(lock_file):
        with open(lock_file) as f:
            assert f.read() == 'test'


class BrokenLock(pl.InterProcessLock):
    def __init__(self, name, errno_code):
        super(BrokenLock, self).__init__(name)
        self.errno_code = errno_code

    def unlock(self):
        pass

    def trylock(self):
        err = IOError()
        err.errno = self.errno_code
        raise err


def test_bad_acquire(lock_dir):
    lock_file = os.path.join(lock_dir, 'lock')
    lock = BrokenLock(lock_file, errno.EBUSY)
    with pytest.raises(threading.ThreadError):
        lock.acquire()


def test_lock_twice(lock_dir):
    lock_file = os.path.join(lock_dir, 'lock')
    lock = pl.InterProcessLock(lock_file)

    ok = lock.acquire(blocking=False)
    assert ok

    # ok on Unix, not ok on Windows
    ok = lock.acquire(blocking=False)
    assert ok or not ok

    # should release without crashing
    lock.release()


def _semaphore_worker(path, capacity, conn):
    semaphore = pl.InterProcessSemaphore(path, capacity)
    permit = semaphore.acquire(timeout=10)
    if permit is None:
        conn.send(('error', 'timed out', os.getpid()))
        return

    conn.send(('entered', permit.slot, os.getpid()))
    command = conn.recv()
    if command == 'release':
        permit.release()
        conn.send(('released', permit.slot, os.getpid()))
        conn.recv()


def _release_inherited_permit(permit, conn):
    try:
        permit.release()
    except Exception as e:
        conn.send((type(e).__name__, str(e)))
    else:
        conn.send(None)


def test_interprocess_semaphore_limits_capacity(lock_dir):
    semaphore_path = os.path.join(lock_dir, 'semaphore')
    semaphore = pl.InterProcessSemaphore(semaphore_path, 3)

    permits = [semaphore.acquire() for _ in range(3)]
    try:
        assert len({permit.slot for permit in permits}) == 3
        assert semaphore.acquire(blocking=False) is None
    finally:
        for permit in permits:
            permit.release()

    permit = semaphore.acquire(blocking=False)
    assert permit is not None
    permit.release()


def test_interprocess_semaphore_acquire_timeout(lock_dir):
    semaphore_path = os.path.join(lock_dir, 'semaphore')
    semaphore = pl.InterProcessSemaphore(semaphore_path, 1)
    permit = semaphore.acquire()

    start = time.monotonic()
    try:
        assert semaphore.acquire(timeout=0.05) is None
    finally:
        permit.release()

    assert time.monotonic() - start >= 0.04


def test_interprocess_semaphore_capacity_mismatch(lock_dir):
    semaphore_path = os.path.join(lock_dir, 'semaphore')
    pl.InterProcessSemaphore(semaphore_path, 2)

    with pytest.raises(pl.InterProcessSemaphoreCapacityError):
        pl.InterProcessSemaphore(semaphore_path, 3)

    with open(os.path.join(semaphore_path, 'capacity')) as capacity_file:
        assert capacity_file.read() == '2'


@pytest.mark.parametrize('capacity', [0, -1, 1.0, True])
def test_interprocess_semaphore_invalid_capacity(lock_dir, capacity):
    semaphore_path = os.path.join(lock_dir, 'semaphore')
    with pytest.raises(ValueError):
        pl.InterProcessSemaphore(semaphore_path, capacity)


def test_interprocess_semaphore_rejects_double_release(lock_dir):
    semaphore_path = os.path.join(lock_dir, 'semaphore')
    semaphore = pl.InterProcessSemaphore(semaphore_path, 1)
    permit = semaphore.acquire()
    permit.release()

    with pytest.raises(pl.InterProcessSemaphoreReleaseError):
        permit.release()


def test_interprocess_semaphore_rejects_wrong_process_release(lock_dir):
    semaphore_path = os.path.join(lock_dir, 'semaphore')
    semaphore = pl.InterProcessSemaphore(semaphore_path, 1)
    permit = semaphore.acquire()
    permit._owner_pid += 1

    with pytest.raises(pl.InterProcessSemaphoreReleaseError):
        permit.release()

    assert not permit._released
    permit._owner_pid = os.getpid()
    permit.release()


@pytest.mark.skipif(WIN32, reason='fork is not available on Windows')
def test_interprocess_semaphore_rejects_forked_handle(lock_dir):
    ctx = multiprocessing.get_context('fork')
    semaphore_path = os.path.join(lock_dir, 'semaphore')
    semaphore = pl.InterProcessSemaphore(semaphore_path, 1)
    permit = semaphore.acquire()
    parent_conn, child_conn = ctx.Pipe(False)
    child = ctx.Process(
        target=_release_inherited_permit,
        args=(permit, child_conn))

    with scoped_child_processes((child,), timeout=10):
        assert parent_conn.poll(10)
        error_type, _ = parent_conn.recv()
        assert error_type == 'InterProcessSemaphoreReleaseError'

    permit.release()


@pytest.mark.skipif(WIN32, reason='fork is not available on Windows')
def test_interprocess_semaphore_recovers_after_holder_exit(lock_dir):
    ctx = multiprocessing.get_context('fork')
    semaphore_path = os.path.join(lock_dir, 'semaphore')
    workers = []
    try:
        for _ in range(4):
            parent_conn, child_conn = ctx.Pipe()
            child = ctx.Process(
                target=_semaphore_worker,
                args=(semaphore_path, 3, child_conn))
            child.daemon = True
            child.start()
            workers.append((child, parent_conn))

        entered = []
        deadline = time.time() + 10
        while len(entered) < 3 and time.time() < deadline:
            for child, conn in workers:
                if conn.poll(0.01):
                    event, slot, pid = conn.recv()
                    assert event == 'entered'
                    entered.append((child, conn, slot, pid))

        assert len(entered) == 3
        waiting = [
            (child, conn) for child, conn in workers
            if child not in [entered_child for entered_child, _, _, _ in entered]
        ]
        assert len(waiting) == 1
        waiting_child, waiting_conn = waiting[0]
        assert not waiting_conn.poll(0.5)

        terminated_child, terminated_conn, terminated_slot, _ = entered[0]
        terminated_child.terminate()
        terminated_child.join(10)
        assert not terminated_child.is_alive()

        assert waiting_conn.poll(10)
        event, slot, pid = waiting_conn.recv()
        assert event == 'entered'
        assert slot == terminated_slot
        assert pid == waiting_child.pid
        simultaneous = {
            (child.pid, held_slot)
            for child, _, held_slot, held_pid in entered[1:]
        }
        simultaneous.add((waiting_child.pid, slot))
        assert len(simultaneous) == 3

        for child, conn, _, _ in entered[1:]:
            conn.send('release')
            assert conn.recv()[0] == 'released'
            conn.send('exit')

        waiting_conn.send('release')
        assert waiting_conn.recv()[0] == 'released'
        waiting_conn.send('exit')
        waiting_child.join(10)
        assert not waiting_child.is_alive()
    finally:
        for child, conn in workers:
            if child.is_alive():
                try:
                    conn.send('exit')
                except (BrokenPipeError, EOFError, OSError):
                    pass
            child.join(10)
            if child.is_alive():
                child.terminate()
                child.join()

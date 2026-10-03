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


def multi_interprocesslock_helper(paths, pipe, reverse=False):
    locks = list(reversed(paths)) if reverse else list(paths)
    pipe.send('ready')
    pipe.recv()
    try:
        with pl.InterProcessMultiLock(locks):
            pipe.send(True)
            pipe.recv()
    except Exception as error:
        pipe.send((type(error).__name__, str(error)))
        raise


def multi_interprocesslock_try_helper(paths, pipe):
    multi_lock = pl.InterProcessMultiLock(paths)
    acquired = multi_lock.acquire(blocking=False)
    pipe.send(acquired)
    if acquired:
        multi_lock.release()


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


class RecordingLock(object):
    def __init__(self, path, acquired=False, release_error=None,
                 acquire_error=None):
        self.path = path
        self.acquired = acquired
        self.release_error = release_error
        self.acquire_error = acquire_error
        self.acquisitions = []
        self.releases = 0

    def acquire(self, blocking=True, delay=0.01, max_delay=0.1,
                timeout=None):
        self.acquisitions.append((blocking, delay, max_delay, timeout))
        if self.acquire_error is not None:
            raise self.acquire_error
        if delay:
            time.sleep(delay)
        self.acquired = True
        return True

    def release(self):
        self.releases += 1
        self.acquired = False
        if self.release_error is not None:
            raise self.release_error


class TimeAdvancingLock(RecordingLock):
    def __init__(self, path, clock, elapsed_after_acquire):
        super(TimeAdvancingLock, self).__init__(path)
        self._clock = clock
        self._elapsed_after_acquire = elapsed_after_acquire

    def acquire(self, blocking=True, delay=0.01, max_delay=0.1,
                timeout=None):
        self.acquisitions.append((blocking, delay, max_delay, timeout))
        self._clock[0] += self._elapsed_after_acquire
        self.acquired = True
        return True


def test_multi_lock_normalizes_and_deduplicates_paths(lock_dir):
    alias_dir = os.path.join(lock_dir, 'alias')
    os.symlink(lock_dir, alias_dir)
    first_path = os.path.join(lock_dir, 'first')
    second_path = os.path.join(alias_dir, 'second')
    held_lock = pl.InterProcessLock(os.path.join(alias_dir, 'first'))

    multi_lock = pl.InterProcessMultiLock([
        second_path,
        held_lock,
        os.path.join(lock_dir, 'first'),
        first_path,
    ])

    normalized_first = held_lock.path
    normalized_second = pl.InterProcessLock(second_path).path
    assert [lock.path for lock in multi_lock.locks] == [
        normalized_first, normalized_second]
    assert multi_lock.locks[0] is held_lock


def test_multi_lock_normalizes_relative_paths(lock_dir):
    relative_dir = os.path.relpath(lock_dir)
    direct_path = os.path.join(lock_dir, 'relative-lock')
    relative_path = os.path.join(relative_dir, 'relative-lock')

    multi_lock = pl.InterProcessMultiLock([relative_path])

    assert multi_lock.locks[0].path == pl.InterProcessLock(direct_path).path


def test_multi_lock_prefers_acquired_duplicate_regardless_of_order(lock_dir):
    path = os.path.join(lock_dir, 'lock')
    held_lock = pl.InterProcessLock(path)
    duplicate_lock = pl.InterProcessLock(path)

    assert held_lock.acquire(blocking=False)
    multi_lock = pl.InterProcessMultiLock([duplicate_lock, held_lock])

    assert multi_lock.locks[0] is held_lock
    multi_lock.acquire(blocking=False, delay=0)
    multi_lock.release()

    assert held_lock.acquired
    held_lock.release()


def test_multi_lock_rejects_invalid_lock_input():
    with pytest.raises(TypeError):
        pl.InterProcessMultiLock([object()])


def test_multi_lock_does_not_release_previously_acquired_lock():
    first = RecordingLock(b'first', acquired=True)
    second = RecordingLock(b'second')
    multi_lock = pl.InterProcessMultiLock([second, first])

    assert multi_lock.acquire(blocking=False, delay=0)
    multi_lock.release()

    assert first.acquired
    assert first.releases == 0
    assert not second.acquired
    assert second.releases == 1


def test_multi_lock_acquires_in_stable_order():
    first = RecordingLock(b'first')
    second = RecordingLock(b'second')
    third = RecordingLock(b'third')
    multi_lock = pl.InterProcessMultiLock([third, first, second])

    acquisition_order = []

    def record_acquire(lock):
        def acquire(blocking=True, delay=0.01, max_delay=0.1,
                    timeout=None):
            acquisition_order.append(lock.path)
            lock.acquired = True
            return True
        return acquire

    first.acquire = record_acquire(first)
    second.acquire = record_acquire(second)
    third.acquire = record_acquire(third)

    assert multi_lock.acquire(blocking=False, delay=0)
    assert acquisition_order == [b'first', b'second', b'third']


def test_multi_lock_uses_one_shared_timeout_budget(monkeypatch):
    clock = [100.0]
    monkeypatch.setattr(
        pl._utils.time, 'monotonic', lambda: clock[0])
    first = TimeAdvancingLock(b'first', clock, 0.1)
    second = TimeAdvancingLock(b'second', clock, 0.2)
    third = TimeAdvancingLock(b'third', clock, 0.3)
    multi_lock = pl.InterProcessMultiLock([third, second, first])

    assert multi_lock.acquire(timeout=1.0, delay=0)
    assert first.acquisitions[0][3] == 1.0
    assert second.acquisitions[0][3] == pytest.approx(0.9)
    assert third.acquisitions[0][3] == pytest.approx(0.7)


def test_multi_lock_rolls_back_in_reverse_order():
    first = RecordingLock(b'first')
    second = RecordingLock(b'second')
    events = []

    def release_first():
        events.append(('release', first.path))

    def release_second():
        events.append(('release', second.path))

    first.release = release_first
    second.release = release_second
    third = RecordingLock(b'third', acquire_error=RuntimeError('boom'))
    multi_lock = pl.InterProcessMultiLock([third, second, first])

    with pytest.raises(RuntimeError, match='boom'):
        multi_lock.acquire(blocking=False, delay=0)

    assert events == [('release', b'second'), ('release', b'first')]
    assert not multi_lock.acquired


def test_multi_lock_preserves_all_release_errors():
    first_error = RuntimeError('first failure')
    second_error = RuntimeError('second failure')
    first = RecordingLock(
        b'first', release_error=first_error)
    second = RecordingLock(
        b'second', release_error=second_error)
    third = RecordingLock(b'third')
    multi_lock = pl.InterProcessMultiLock([third, second, first])
    multi_lock.acquire(blocking=False, delay=0)

    with pytest.raises(pl.InterProcessMultiLockError) as exc_info:
        multi_lock.release()

    assert [error for lock, error in exc_info.value.errors] == [
        second_error, first_error]
    assert first.releases == 1
    assert 'first failure' in str(exc_info.value)
    assert 'second failure' in str(exc_info.value)
    assert third.releases == 1


def test_multi_lock_opposite_order_completes_and_releases(lock_dir):
    paths = [
        os.path.join(lock_dir, 'first'),
        os.path.join(lock_dir, 'second'),
    ]
    first_parent, first_child = multiprocessing.Pipe()
    second_parent, second_child = multiprocessing.Pipe()
    first = multiprocessing.Process(
        target=multi_interprocesslock_helper,
        args=(paths, first_child, True))
    second = multiprocessing.Process(
        target=multi_interprocesslock_helper,
        args=(paths, second_child, False))

    with scoped_child_processes((first, second), timeout=10):
        assert first_parent.poll(10)
        assert first_parent.recv() == 'ready'
        assert second_parent.poll(10)
        assert second_parent.recv() == 'ready'

        first_parent.send('start')
        second_parent.send('start')

        deadline = time.monotonic() + 10
        winner = None
        while deadline > time.monotonic():
            if first_parent.poll():
                winner = first_parent
                break
            if second_parent.poll():
                winner = second_parent
                break
            time.sleep(0.01)

        assert winner is not None
        assert winner.recv() is True
        winner.send('release')

        loser = second_parent if winner is first_parent else first_parent
        assert loser.poll(30)
        assert loser.recv() is True
        loser.send('release')

    third_parent, third_child = multiprocessing.Pipe()
    third = multiprocessing.Process(
        target=multi_interprocesslock_try_helper,
        args=(paths, third_child))
    with scoped_child_processes((third,), timeout=10):
        assert third_parent.poll(30)
        assert third_parent.recv() is True


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

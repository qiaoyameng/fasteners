# -*- coding: utf-8 -*-

#    Copyright (C) 2014 Yahoo! Inc. All Rights Reserved.
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
from concurrent import futures
import random
import threading
import time

import pytest

import fasteners
from fasteners import _utils

# NOTE(harlowja): Sleep a little so now() can not be the same (which will
# cause false positives when our overlap detection code runs). If there are
# real overlaps then they will still exist.
NAPPY_TIME = 0.05

# We will spend this amount of time doing some "fake" work.
WORK_TIMES = [(0.01 + x / 100.0) for x in range(0, 5)]

# If latches/events take longer than this to become empty/set, something is
# usually wrong and should be debugged instead of deadlocking...
WAIT_TIMEOUT = 300

THREAD_COUNT = 20


def _find_overlaps(times, start, end):
    overlaps = 0
    for (s, e) in times:
        if s >= start and e <= end:
            overlaps += 1
    return overlaps


def _spawn_variation(readers, writers, max_workers=None):
    start_stops = collections.deque()
    lock = fasteners.ReaderWriterLock()

    def read_func(ident):
        with lock.read_lock():
            enter_time = time.monotonic()
            time.sleep(WORK_TIMES[ident % len(WORK_TIMES)])
            exit_time = time.monotonic()
            start_stops.append((lock.READER, enter_time, exit_time))
            time.sleep(NAPPY_TIME)

    def write_func(ident):
        with lock.write_lock():
            enter_time = time.monotonic()
            time.sleep(WORK_TIMES[ident % len(WORK_TIMES)])
            exit_time = time.monotonic()
            start_stops.append((lock.WRITER, enter_time, exit_time))
            time.sleep(NAPPY_TIME)

    if max_workers is None:
        max_workers = max(0, readers) + max(0, writers)
    if max_workers > 0:
        with futures.ThreadPoolExecutor(max_workers=max_workers) as e:
            count = 0
            for _i in range(0, readers):
                e.submit(read_func, count)
                count += 1
            for _i in range(0, writers):
                e.submit(write_func, count)
                count += 1

    writer_times = []
    reader_times = []
    for (lock_type, start, stop) in list(start_stops):
        if lock_type == lock.WRITER:
            writer_times.append((start, stop))
        else:
            reader_times.append((start, stop))
    return writer_times, reader_times


def _daemon_thread(target):
    t = threading.Thread(target=target)
    t.daemon = True
    return t


@pytest.mark.parametrize("contextmanager", [True, False])
def test_no_double_writers(contextmanager):
    lock = fasteners.ReaderWriterLock()
    watch = _utils.StopWatch(duration=5)
    watch.start()
    dups = collections.deque()
    active = collections.deque()

    def acquire_check_ctx(me):
        with lock.write_lock():
            if len(active) >= 1:
                dups.append(me)
                dups.extend(active)
            active.append(me)
            time.sleep(random.random() / 100)
            active.remove(me)

    def acquire_check_plain(me):
        lock.acquire_write_lock()
        if len(active) >= 1:
            dups.append(me)
            dups.extend(active)
        active.append(me)
        time.sleep(random.random() / 100)
        active.remove(me)
        lock.release_write_lock()

    def run():
        me = threading.current_thread()
        while not watch.expired():
            if contextmanager:
                acquire_check_ctx(me)
            else:
                acquire_check_plain(me)

    threads = []
    for i in range(0, THREAD_COUNT):
        t = _daemon_thread(run)
        threads.append(t)
        t.start()
    while threads:
        t = threads.pop()
        t.join()

    assert not dups
    assert not active


@pytest.mark.parametrize("contextmanager", [True, False])
def test_no_concurrent_readers_writers(contextmanager):
    lock = fasteners.ReaderWriterLock()
    watch = _utils.StopWatch(duration=5)
    watch.start()
    dups = collections.deque()
    active = collections.deque()

    def acquire_check_ctx(me, reader):
        if reader:
            lock_func = lock.read_lock
        else:
            lock_func = lock.write_lock
        with lock_func():
            if not reader:
                # There should be no-one else currently active, if there
                # is ensure we capture them so that we can later blow-up
                # the test.
                if len(active) >= 1:
                    dups.append(me)
                    dups.extend(active)
            active.append(me)
            time.sleep(random.random() / 100)
            active.remove(me)

    def acquire_check_plain(me, reader):
        if reader:
            lock_func, unlock_func = lock.acquire_read_lock, lock.release_read_lock
        else:
            lock_func, unlock_func = lock.acquire_write_lock, lock.release_write_lock

        lock_func()
        if not reader:
            # There should be no-one else currently active, if there
            # is ensure we capture them so that we can later blow-up
            # the test.
            if len(active) >= 1:
                dups.append(me)
                dups.extend(active)
        active.append(me)
        time.sleep(random.random() / 100)
        active.remove(me)
        unlock_func()

    def run():
        me = threading.current_thread()
        while not watch.expired():
            if contextmanager:
                acquire_check_ctx(me, random.choice([True, False]))
            else:
                acquire_check_plain(me, random.choice([True, False]))

    threads = []
    for i in range(0, THREAD_COUNT):
        t = _daemon_thread(run)
        threads.append(t)
        t.start()
    while threads:
        t = threads.pop()
        t.join()

    assert not dups
    assert not active


def test_writer_abort():
    lock = fasteners.ReaderWriterLock()
    assert lock.owner is None

    with pytest.raises(RuntimeError):
        with lock.write_lock():
            assert lock.owner == lock.WRITER
            raise RuntimeError("Broken")

    assert lock.owner is None


def test_reader_abort():
    lock = fasteners.ReaderWriterLock()
    assert lock.owner is None

    with pytest.raises(RuntimeError):
        with lock.read_lock():
            assert lock.owner == lock.READER
            raise RuntimeError("Broken")

    assert lock.owner is None


def test_double_reader_abort():
    lock = fasteners.ReaderWriterLock()
    activated = collections.deque()

    def double_bad_reader():
        with lock.read_lock():
            with lock.read_lock():
                raise RuntimeError("Broken")

    def happy_writer():
        with lock.write_lock():
            activated.append(lock.owner)

    with futures.ThreadPoolExecutor(max_workers=20) as e:
        for i in range(0, 20):
            if i % 2 == 0:
                e.submit(double_bad_reader)
            else:
                e.submit(happy_writer)

    assert sum(a == 'w' for a in activated) == 10


def test_double_reader_writer():
    lock = fasteners.ReaderWriterLock()
    activated = collections.deque()
    active = threading.Event()

    def double_reader():
        with lock.read_lock():
            active.set()
            while not lock.has_pending_writers:
                time.sleep(0.001)
            with lock.read_lock():
                activated.append(lock.owner)

    def happy_writer():
        with lock.write_lock():
            activated.append(lock.owner)

    reader = _daemon_thread(double_reader)
    reader.start()
    active.wait(WAIT_TIMEOUT)
    assert active.is_set()

    writer = _daemon_thread(happy_writer)
    writer.start()

    reader.join()
    writer.join()
    assert list(activated) == ['r', 'w']


def test_reader_chaotic():
    lock = fasteners.ReaderWriterLock()
    activated = collections.deque()

    def chaotic_reader(blow_up):
        with lock.read_lock():
            if blow_up:
                raise RuntimeError("Broken")
            else:
                activated.append(lock.owner)

    def happy_writer():
        with lock.write_lock():
            activated.append(lock.owner)

    with futures.ThreadPoolExecutor(max_workers=20) as e:
        for i in range(0, 20):
            if i % 2 == 0:
                e.submit(chaotic_reader, blow_up=bool(i % 4 == 0))
            else:
                e.submit(happy_writer)

    assert sum(a == 'w' for a in activated) == 10
    assert sum(a == 'r' for a in activated) == 5


def test_writer_chaotic():
    lock = fasteners.ReaderWriterLock()
    activated = collections.deque()

    def chaotic_writer(blow_up):
        with lock.write_lock():
            if blow_up:
                raise RuntimeError("Broken")
            else:
                activated.append(lock.owner)

    def happy_reader():
        with lock.read_lock():
            activated.append(lock.owner)

    with futures.ThreadPoolExecutor(max_workers=20) as e:
        for i in range(0, 20):
            if i % 2 == 0:
                e.submit(chaotic_writer, blow_up=bool(i % 4 == 0))
            else:
                e.submit(happy_reader)

    assert sum(a == 'w' for a in activated) == 5
    assert sum(a == 'r' for a in activated) == 10


def test_writer_reader_writer_ctx():
    lock = fasteners.ReaderWriterLock()
    with lock.write_lock():
        assert lock.is_writer()
        with lock.read_lock():
            assert lock.is_reader()
            with lock.write_lock():
                assert lock.is_writer()


def test_writer_reader_writer_plain():
    lock = fasteners.ReaderWriterLock()
    lock.acquire_write_lock()
    assert lock.is_writer()
    lock.acquire_read_lock()
    assert lock.is_reader()
    lock.acquire_write_lock()
    assert lock.is_writer()


def test_single_reader_writer_ctx():
    lock = fasteners.ReaderWriterLock()
    with lock.read_lock():
        assert lock.is_reader()
    with lock.write_lock():
        assert lock.is_writer()
    with lock.read_lock():
        assert lock.is_reader()
    assert not lock.is_reader()
    assert not lock.is_writer()


def test_single_reader_writer_plain():
    lock = fasteners.ReaderWriterLock()

    lock.acquire_read_lock()
    assert lock.is_reader()
    lock.release_read_lock()

    lock.acquire_write_lock()
    assert lock.is_writer()
    lock.release_write_lock()

    lock.acquire_read_lock()
    assert lock.is_reader()
    lock.release_read_lock()

    assert not lock.is_reader()
    assert not lock.is_writer()


def test_reader_to_writer_ctx():
    lock = fasteners.ReaderWriterLock()

    with lock.read_lock():
        with pytest.raises(RuntimeError):
            with lock.write_lock():
                pass
        assert lock.is_reader()
        assert not lock.is_writer()

    assert not lock.is_reader()
    assert not lock.is_writer()


def test_reader_to_writer_plain():
    lock = fasteners.ReaderWriterLock()

    lock.acquire_read_lock()
    with pytest.raises(RuntimeError):
        lock.acquire_write_lock()
    assert lock.is_reader()
    assert not lock.is_writer()
    lock.release_read_lock()

    assert not lock.is_reader()
    assert not lock.is_writer()


def test_writer_to_reader_ctx():
    lock = fasteners.ReaderWriterLock()

    with lock.write_lock():
        with lock.read_lock():
            assert lock.is_writer()
            assert lock.is_reader()

        assert lock.is_writer()
        assert not lock.is_reader()

    assert not lock.is_writer()
    assert not lock.is_reader()


def test_writer_to_reader_plain():
    lock = fasteners.ReaderWriterLock()

    lock.acquire_write_lock()
    lock.acquire_read_lock()
    assert lock.is_writer()
    assert lock.is_reader()

    lock.release_read_lock()
    assert lock.is_writer()
    assert not lock.is_reader()

    lock.release_write_lock()
    assert not lock.is_writer()
    assert not lock.is_reader()


def test_double_writer_ctx():
    lock = fasteners.ReaderWriterLock()

    with lock.write_lock():
        assert not lock.is_reader()
        assert lock.is_writer()

        with lock.write_lock():
            assert lock.is_writer()

        assert lock.is_writer()

    assert not lock.is_reader()
    assert not lock.is_writer()


def test_double_writer_plain():
    lock = fasteners.ReaderWriterLock()

    lock.acquire_write_lock()
    assert not lock.is_reader()
    assert lock.is_writer()

    lock.acquire_write_lock()
    assert lock.is_writer()

    lock.release_write_lock()
    assert lock.is_writer()

    lock.release_write_lock()
    assert not lock.is_reader()
    assert not lock.is_writer()


def test_double_reader_ctx():
    lock = fasteners.ReaderWriterLock()

    with lock.read_lock():
        assert lock.is_reader()
        assert not lock.is_writer()

        with lock.read_lock():
            assert lock.is_reader()

        assert lock.is_reader()

    assert not lock.is_reader()
    assert not lock.is_writer()


def test_double_reader_plain():
    lock = fasteners.ReaderWriterLock()

    lock.acquire_read_lock()
    assert lock.is_reader()
    assert not lock.is_writer()

    lock.acquire_read_lock()
    assert lock.is_reader()

    lock.release_read_lock()
    assert lock.is_reader()

    lock.release_read_lock()
    assert not lock.is_reader()
    assert not lock.is_writer()


def test_multi_reader_multi_writer():
    writer_times, reader_times = _spawn_variation(10, 10)
    assert len(writer_times) == 10
    assert len(reader_times) == 10

    for (start, stop) in writer_times:
        assert _find_overlaps(reader_times, start, stop) == 0
        assert _find_overlaps(writer_times, start, stop) == 1
    for (start, stop) in reader_times:
        assert _find_overlaps(writer_times, start, stop) == 0


def test_multi_reader_single_writer():
    writer_times, reader_times = _spawn_variation(9, 1)
    assert len(writer_times) == 1
    assert len(reader_times) == 9

    start, stop = writer_times[0]
    assert _find_overlaps(reader_times, start, stop) == 0


def test_multi_writer():
    writer_times, reader_times = _spawn_variation(0, 10)
    assert len(writer_times) == 10
    assert len(reader_times) == 0

    for (start, stop) in writer_times:
        assert _find_overlaps(writer_times, start, stop) == 1


def test_deadlock_reproducer():
    lock = fasteners.ReaderWriterLock()

    def thread1():
        for _ in range(0, 1000):
            print("thread1 iteration: {}".format(_))
            print("thread1 waiting for write lock")
            with lock.write_lock():
                print("thread1 got write lock")
                print("thread1 waiting for read lock")
                with lock.read_lock():
                    print("thread1 got read lock")
                print("thread1 released read lock")
            print("thread1 released write lock")

    def thread2():
        for _ in range(0, 1000):
            print("thread2 iteration: {}".format(_))
            print("thread2 waiting for write lock")
            with lock.write_lock():
                print("thread2 got write lock")
            print("thread2 released write lock")

    funcs = (thread1, thread2)

    threads = []
    for func in funcs:
        t = _daemon_thread(func)
        threads.append(t)
        t.start()
    while threads:
        t = threads.pop()
        t.join()


def test_error_when_releasing_write_lock_without_write_lock():
    lock = fasteners.ReaderWriterLock()
    with pytest.raises(RuntimeError):
        lock.release_write_lock()


def test_error_when_releasing_read_lock_without_read_lock():
    lock = fasteners.ReaderWriterLock()
    with pytest.raises(RuntimeError):
        lock.release_read_lock()


def test_upgradable_read_lock_ctx():
    lock = fasteners.ReaderWriterLock()
    with lock.upgradable_read_lock():
        assert lock.is_reader()
        assert lock.is_upgrader()
        assert lock.has_pending_upgrader
        assert lock.upgrade_read_lock(timeout=WAIT_TIMEOUT)
        assert lock.is_writer(check_pending=False)
        assert not lock.is_reader()
        assert not lock.has_pending_upgrader
        lock.downgrade_write_lock()
        assert lock.is_reader()
        assert not lock.is_writer(check_pending=False)
    assert lock.owner is None


def test_plain_reader_can_not_upgrade():
    lock = fasteners.ReaderWriterLock()
    lock.acquire_read_lock()
    try:
        with pytest.raises(RuntimeError):
            lock.upgrade_read_lock(timeout=0.1)
        # A failed attempt must not have disturbed the read lock.
        assert lock.is_reader()
    finally:
        lock.release_read_lock()
    assert lock.owner is None


def test_only_one_upgradable_reader_at_a_time():
    lock = fasteners.ReaderWriterLock()
    first_holds = threading.Event()
    first_release = threading.Event()
    second_acquired = threading.Event()

    def first():
        lock.acquire_upgradable_read_lock()
        first_holds.set()
        first_release.wait(WAIT_TIMEOUT)
        lock.release_read_lock()

    def second():
        assert first_holds.wait(WAIT_TIMEOUT)
        lock.acquire_upgradable_read_lock()
        second_acquired.set()
        lock.release_read_lock()

    t1 = _daemon_thread(first)
    t2 = _daemon_thread(second)
    t1.start()
    t2.start()
    assert first_holds.wait(WAIT_TIMEOUT)
    time.sleep(NAPPY_TIME)
    # The second thread is blocked waiting for the upgrade intent even
    # though a plain read lock would be grantable right away.
    assert not second_acquired.is_set()
    first_release.set()
    assert second_acquired.wait(WAIT_TIMEOUT)
    t1.join()
    t2.join()
    assert not lock.has_pending_upgrader


def test_upgrade_read_lock_timeout_preserves_read_holds():
    lock = fasteners.ReaderWriterLock()
    lock.acquire_upgradable_read_lock()
    lock.acquire_upgradable_read_lock()
    other_has_read = threading.Event()
    other_release = threading.Event()

    def other_reader():
        lock.acquire_read_lock()
        other_has_read.set()
        other_release.wait(WAIT_TIMEOUT)
        lock.release_read_lock()

    t = _daemon_thread(other_reader)
    t.start()
    assert other_has_read.wait(WAIT_TIMEOUT)
    # The other reader never leaves in time, so the upgrade must fail and
    # our two read holds (and the upgrade intent) must be kept intact.
    assert lock.upgrade_read_lock(timeout=0.2) is False
    assert lock.is_reader()
    assert lock.is_upgrader()
    lock.release_read_lock()
    assert lock.is_reader()
    assert lock.is_upgrader()
    lock.release_read_lock()
    assert not lock.is_reader()
    assert not lock.has_pending_upgrader
    other_release.set()
    t.join()
    assert lock.owner is None


def test_downgrade_write_lock():
    lock = fasteners.ReaderWriterLock()
    lock.acquire_write_lock()
    lock.acquire_write_lock()
    lock.downgrade_write_lock()
    # All write entries were converted into a single read hold.
    assert lock.is_reader()
    assert not lock.is_writer(check_pending=False)
    assert lock.owner == lock.READER
    with pytest.raises(RuntimeError):
        lock.release_write_lock()
    lock.release_read_lock()
    assert lock.owner is None


def test_downgrade_requires_write_lock():
    lock = fasteners.ReaderWriterLock()
    with pytest.raises(RuntimeError):
        lock.downgrade_write_lock()
    lock.acquire_read_lock()
    try:
        with pytest.raises(RuntimeError):
            lock.downgrade_write_lock()
    finally:
        lock.release_read_lock()


def test_upgrade_downgrade_queued_writer_never_crosses():
    # Two readers (one of them upgradable) and one queued writer walk
    # through a barrier-scripted interleaving; the queued writer must
    # never enter its critical section while the upgrader holds either
    # its (upgraded) write lock or its (downgraded) read lock.
    lock = fasteners.ReaderWriterLock()
    events = collections.deque()
    r1_has_upgradable = threading.Event()
    r2_has_read = threading.Event()
    r2_release = threading.Event()
    r1_upgraded = threading.Event()
    r1_downgraded = threading.Event()
    r1_release = threading.Event()
    writer_done = threading.Event()

    def r1():
        lock.acquire_upgradable_read_lock()
        events.append(('r1', 'read-enter'))
        r1_has_upgradable.set()
        assert r2_has_read.wait(WAIT_TIMEOUT)
        assert lock.upgrade_read_lock(timeout=WAIT_TIMEOUT)
        events.append(('r1', 'write-enter'))
        r1_upgraded.set()
        time.sleep(NAPPY_TIME)
        lock.downgrade_write_lock()
        events.append(('r1', 'downgraded'))
        r1_downgraded.set()
        assert r1_release.wait(WAIT_TIMEOUT)
        lock.release_read_lock()
        events.append(('r1', 'read-exit'))

    def r2():
        assert r1_has_upgradable.wait(WAIT_TIMEOUT)
        lock.acquire_read_lock()
        events.append(('r2', 'read-enter'))
        r2_has_read.set()
        assert r2_release.wait(WAIT_TIMEOUT)
        lock.release_read_lock()
        events.append(('r2', 'read-exit'))

    def queued_writer():
        assert r2_has_read.wait(WAIT_TIMEOUT)
        lock.acquire_write_lock()
        events.append(('w', 'write-enter'))
        time.sleep(NAPPY_TIME)
        events.append(('w', 'write-exit'))
        lock.release_write_lock()
        writer_done.set()

    threads = [_daemon_thread(r1), _daemon_thread(r2),
               _daemon_thread(queued_writer)]
    for t in threads:
        t.start()

    # Both readers hold their locks; r1 is already blocked inside its
    # upgrade request because r2 is still an active reader.
    assert r2_has_read.wait(WAIT_TIMEOUT)
    time.sleep(NAPPY_TIME)
    assert not r1_upgraded.is_set()

    # Wait until the writer is actually queued behind the readers.
    deadline = time.monotonic() + WAIT_TIMEOUT
    while not lock.has_pending_writers:
        assert time.monotonic() < deadline
        time.sleep(0.01)

    # Let r2 go; the upgrader (not the queued writer) must win the lock.
    r2_release.set()
    assert r1_upgraded.wait(WAIT_TIMEOUT)
    assert not any(event == ('w', 'write-enter') for event in events)

    # After the downgrade the queued writer must still be locked out
    # because r1 atomically became a reader instead of fully releasing.
    assert r1_downgraded.wait(WAIT_TIMEOUT)
    time.sleep(NAPPY_TIME)
    assert not any(event == ('w', 'write-enter') for event in events)

    # Only once r1 releases its downgraded read lock may the writer in.
    r1_release.set()
    assert writer_done.wait(WAIT_TIMEOUT)
    for t in threads:
        t.join()

    kinds = {actor: [action for (a, action) in events if a == actor]
             for actor in ('r1', 'r2', 'w')}
    assert kinds['r1'] == ['read-enter', 'write-enter', 'downgraded',
                           'read-exit']
    assert kinds['r2'] == ['read-enter', 'read-exit']
    assert kinds['w'] == ['write-enter', 'write-exit']
    # No other writer crossed before or after the downgrade: the queued
    # writer's section comes strictly after r1 fully released.
    assert list(events).index(('w', 'write-enter')) > \
        list(events).index(('r1', 'read-exit'))
    assert lock.owner is None

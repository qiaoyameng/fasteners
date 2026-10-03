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


class FakeClock(object):
    def __init__(self):
        self._time = 0.0
        self._lock = threading.Lock()

    def time(self):
        with self._lock:
            return self._time

    def advance(self, seconds):
        with self._lock:
            self._time += seconds


class ControllableCondition(threading.Condition):
    def __init__(self, clock, poll_interval=100):
        super().__init__()
        self._clock = clock
        self._poll_interval = poll_interval
        self._stats_lock = threading.Lock()
        self.wait_calls = []
        self.active_waiters = 0

    def wait(self, timeout=None):
        ident = threading.get_ident()
        with self._stats_lock:
            self.active_waiters += 1
        try:
            if timeout is None:
                with self._stats_lock:
                    self.wait_calls.append((ident, timeout))
                return super().wait()

            deadline = self._clock.time() + timeout
            while True:
                remaining = deadline - self._clock.time()
                if remaining <= 0:
                    return False
                with self._stats_lock:
                    self.wait_calls.append((ident, remaining))
                super().wait(min(remaining, self._poll_interval))
        finally:
            with self._stats_lock:
                self.active_waiters -= 1

    def wake_all(self):
        with self:
            self.notify_all()

    def calls_for(self, thread):
        with self._stats_lock:
            return [timeout for ident, timeout in self.wait_calls
                    if ident == thread.ident]

    def wait_for_active_waiters(self, count):
        deadline = time.monotonic() + 5
        while True:
            with self._stats_lock:
                if self.active_waiters == count:
                    return
            assert time.monotonic() < deadline
            time.sleep(0.005)

    def wait_for_call_count(self, thread, count):
        deadline = time.monotonic() + 5
        while True:
            if len(self.calls_for(thread)) >= count:
                return
            assert time.monotonic() < deadline
            time.sleep(0.005)


def wait_for(event):
    assert event.wait(5)


def test_writer_timeout_removes_queued_writer_and_wakes_reader():
    clock = FakeClock()
    condition = ControllableCondition(clock)
    lock = fasteners.ReaderWriterLock(
        condition_cls=lambda: condition,
        time_func=clock.time)

    assert lock.acquire_write_lock()

    writer_result = []
    writer_done = threading.Event()

    def timed_writer():
        writer_result.append(lock.acquire_write_lock(timeout=1))
        writer_done.set()

    writer = _daemon_thread(timed_writer)
    writer.start()
    condition.wait_for_active_waiters(1)

    reader_got_lock = threading.Event()
    reader_release = threading.Event()
    reader_result = []

    def reader():
        reader_result.append(lock.acquire_read_lock())
        reader_got_lock.set()
        wait_for(reader_release)
        lock.release_read_lock()

    reader_thread = _daemon_thread(reader)
    reader_thread.start()
    condition.wait_for_active_waiters(2)

    condition.wake_all()
    condition.wake_all()
    condition.wait_for_call_count(writer, 3)

    waits = condition.calls_for(writer)
    assert waits[:3] == [1, 1, 1]

    clock.advance(1)
    condition.wake_all()
    wait_for(writer_done)
    assert writer_result == [False]
    assert not lock.has_pending_writers
    assert lock.is_writer(check_pending=False)
    assert lock._writer is not writer

    lock.release_write_lock()
    wait_for(reader_got_lock)
    assert reader_result == [True]
    assert lock.owner == lock.READER

    reader_release.set()
    reader_thread.join(5)
    writer.join(5)
    assert not reader_thread.is_alive()
    assert not writer.is_alive()


def test_timed_out_head_writer_allows_following_queue_to_continue():
    clock = FakeClock()
    condition = ControllableCondition(clock)
    lock = fasteners.ReaderWriterLock(
        condition_cls=lambda: condition,
        time_func=clock.time)

    assert lock.acquire_write_lock()

    first_writer_result = []
    first_writer_done = threading.Event()

    def first_writer():
        first_writer_result.append(lock.acquire_write_lock(timeout=1))
        first_writer_done.set()

    first_thread = _daemon_thread(first_writer)
    first_thread.start()
    condition.wait_for_active_waiters(1)

    second_got_lock = threading.Event()
    second_release = threading.Event()
    second_result = []

    def second_writer():
        second_result.append(lock.acquire_write_lock())
        second_got_lock.set()
        wait_for(second_release)
        lock.release_write_lock()

    second_thread = _daemon_thread(second_writer)
    second_thread.start()
    condition.wait_for_active_waiters(2)

    reader_got_lock = threading.Event()
    reader_release = threading.Event()
    reader_result = []

    def reader():
        reader_result.append(lock.acquire_read_lock())
        reader_got_lock.set()
        wait_for(reader_release)
        lock.release_read_lock()

    reader_thread = _daemon_thread(reader)
    reader_thread.start()
    condition.wait_for_active_waiters(3)

    condition.wake_all()
    condition.wake_all()
    condition.wait_for_call_count(first_thread, 3)
    clock.advance(1)
    condition.wake_all()
    wait_for(first_writer_done)

    assert first_writer_result == [False]
    assert list(lock._pending_writers) == [second_thread]
    assert lock._writer is not first_thread

    lock.release_write_lock()
    wait_for(second_got_lock)
    assert second_result == [True]
    assert not reader_got_lock.is_set()

    second_release.set()
    second_thread.join(5)
    wait_for(reader_got_lock)
    assert reader_result == [True]

    reader_release.set()
    reader_thread.join(5)
    first_thread.join(5)
    assert not reader_thread.is_alive()
    assert not second_thread.is_alive()
    assert not first_thread.is_alive()


def test_nonblocking_acquisition_returns_false_without_queueing():
    identity = ['owner']
    lock = fasteners.ReaderWriterLock(
        current_thread_functor=lambda: identity[0])

    assert lock.acquire_write_lock()
    identity[-1] = 'other'

    assert not lock.acquire_write_lock(blocking=False)
    assert not lock.has_pending_writers
    assert lock._writer == 'owner'

    assert not lock.acquire_read_lock(blocking=False)
    assert 'other' not in lock._readers

    identity[-1] = 'owner'
    assert lock.acquire_write_lock(blocking=False, timeout=1)
    assert lock._writer_entries == 2
    assert lock.acquire_read_lock(blocking=False)
    assert lock._readers['owner'] == 1

    lock.release_read_lock()
    assert lock._writer_entries == 2
    lock.release_write_lock()
    assert lock._writer_entries == 1
    lock.release_write_lock()
    assert lock.owner is None


def test_timed_attempts_return_false_without_blocking():
    identity = ['owner']
    lock = fasteners.ReaderWriterLock(
        current_thread_functor=lambda: identity[0])

    assert lock.acquire_write_lock()
    identity[-1] = 'reader'
    assert not lock.acquire_read_lock(timeout=0)
    assert 'reader' not in lock._readers

    identity[-1] = 'writer'
    assert not lock.acquire_write_lock(timeout=0)
    assert not lock.has_pending_writers
    assert lock._writer == 'owner'

    identity[-1] = 'owner'
    lock.release_write_lock()


def test_reentrant_locks_preserve_counts_with_timeout_options():
    lock = fasteners.ReaderWriterLock()

    assert lock.acquire_write_lock(timeout=1)
    assert lock.acquire_write_lock(timeout=1)
    assert lock._writer_entries == 2

    assert lock.acquire_read_lock(blocking=False)
    assert lock._readers[threading.current_thread()] == 1
    lock.release_read_lock()
    assert not lock.is_reader()
    assert lock._writer_entries == 2

    lock.release_write_lock()
    assert lock._writer_entries == 1
    lock.release_write_lock()

    assert lock.acquire_read_lock(timeout=1)
    assert lock.acquire_read_lock(blocking=False)
    assert lock._readers[threading.current_thread()] == 2
    lock.release_read_lock()
    assert lock._readers[threading.current_thread()] == 1
    lock.release_read_lock()
    assert lock.owner is None


@pytest.mark.parametrize("timeout", [-1, float("nan")])
@pytest.mark.parametrize("kind", ["read", "write"])
def test_invalid_timeout_is_rejected_before_state_change(timeout, kind):
    lock = fasteners.ReaderWriterLock()
    assert lock.acquire_write_lock()

    acquire = getattr(lock, "acquire_%s_lock" % kind)
    with pytest.raises(ValueError):
        acquire(timeout=timeout)

    assert lock._writer_entries == 1
    assert not lock.has_pending_writers
    lock.release_write_lock()


def test_context_managers_raise_when_lock_is_unavailable():
    identity = ['owner']
    lock = fasteners.ReaderWriterLock(
        current_thread_functor=lambda: identity[0])
    lock.acquire_write_lock()

    identity[-1] = 'other'
    with pytest.raises(RuntimeError):
        with lock.write_lock(blocking=False):
            pytest.fail("write lock should not be entered")

    with pytest.raises(RuntimeError):
        with lock.read_lock(blocking=False):
            pytest.fail("read lock should not be entered")

    assert lock._writer == 'owner'
    assert not lock.has_pending_writers

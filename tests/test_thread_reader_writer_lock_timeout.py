import threading
import time

import pytest

from fasteners import ReaderWriterLock


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def monotonic(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


class ControllableCondition:
    def __init__(self):
        self._lock = threading.RLock()
        self._waiters = []
        self.wait_count = 0
        self.notifications = 0
        self.timeouts = []

    def __enter__(self):
        self._lock.acquire()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._lock.release()

    def wait(self, timeout=None):
        if timeout is not None:
            self.timeouts.append(timeout)
        event = threading.Event()
        self._waiters.append(event)
        self.wait_count += 1
        self._lock.release()
        try:
            event.wait()
        finally:
            self._lock.acquire()
        return True

    def notify_all(self):
        self.notifications += 1
        waiters = self._waiters
        self._waiters = []
        for waiter in waiters:
            waiter.set()

    def wake_all(self):
        self.notify_all()


def make_lock():
    clock = FakeClock()
    condition = ControllableCondition()
    lock = ReaderWriterLock(
        condition_cls=lambda: condition,
        monotonic_functor=clock.monotonic,
    )
    return lock, clock, condition


def wait_for_count(condition, count):
    deadline = time.monotonic() + 5
    while condition.wait_count < count:
        assert time.monotonic() < deadline
        time.sleep(0.001)


def start_thread(target):
    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    return thread


def test_read_lock_timeout_uses_one_deadline_across_spurious_wakeups():
    lock, clock, condition = make_lock()
    holder = threading.current_thread()
    result = {}

    lock.acquire_write_lock()
    reader = start_thread(
        lambda: result.setdefault("reader", lock.acquire_read_lock(timeout=1))
    )

    wait_for_count(condition, 1)
    clock.advance(0.4)
    condition.wake_all()
    wait_for_count(condition, 2)

    clock.advance(0.4)
    condition.wake_all()
    wait_for_count(condition, 3)

    clock.advance(0.3)
    condition.wake_all()
    reader.join(5)
    assert not reader.is_alive()
    assert result["reader"] is False
    assert condition.timeouts == pytest.approx([1, 0.6, 0.2])
    assert lock._writer is holder
    assert not lock._readers

    lock.release_write_lock()


def test_timed_out_head_writer_leaves_queue_and_releases_waiting_reader():
    lock, clock, condition = make_lock()
    holder = threading.current_thread()
    results = {}
    reader_acquired = threading.Event()
    release_reader = threading.Event()

    def waiting_reader_target():
        results["reader"] = lock.acquire_read_lock()
        reader_acquired.set()
        release_reader.wait(5)
        lock.release_read_lock()

    lock.acquire_write_lock()

    waiting_writer = start_thread(
        lambda: results.setdefault("writer", lock.acquire_write_lock(timeout=1))
    )
    wait_for_count(condition, 1)

    waiting_reader = start_thread(waiting_reader_target)
    wait_for_count(condition, 2)

    clock.advance(0.4)
    condition.wake_all()
    wait_for_count(condition, 4)

    clock.advance(0.4)
    condition.wake_all()
    wait_for_count(condition, 6)

    clock.advance(0.3)
    condition.wake_all()
    waiting_writer.join(5)
    assert not waiting_writer.is_alive()
    assert results["writer"] is False
    assert not lock.has_pending_writers
    assert lock._writer is holder
    assert waiting_writer is not holder

    lock.release_write_lock()
    assert reader_acquired.wait(5)
    assert results["reader"] is True
    assert lock._readers == {waiting_reader: 1}
    assert lock._writer is None

    release_reader.set()
    waiting_reader.join(5)
    assert not waiting_reader.is_alive()
    assert lock.owner is None


def test_next_waiting_writer_continues_after_head_writer_times_out():
    lock, clock, condition = make_lock()
    results = {}
    second_acquired = threading.Event()
    release_second = threading.Event()
    reader_acquired = threading.Event()
    release_reader = threading.Event()

    def second_writer_target():
        results["second"] = lock.acquire_write_lock(timeout=10)
        second_acquired.set()
        release_second.wait(5)
        lock.release_write_lock()

    def waiting_reader_target():
        results["reader"] = lock.acquire_read_lock()
        reader_acquired.set()
        release_reader.wait(5)
        lock.release_read_lock()

    lock.acquire_write_lock()

    first_writer = start_thread(
        lambda: results.setdefault("first", lock.acquire_write_lock(timeout=1))
    )
    wait_for_count(condition, 1)

    second_writer = start_thread(second_writer_target)
    wait_for_count(condition, 2)

    waiting_reader = start_thread(waiting_reader_target)
    wait_for_count(condition, 3)

    clock.advance(1.1)
    condition.wake_all()
    first_writer.join(5)
    assert not first_writer.is_alive()
    assert results["first"] is False
    assert list(lock._pending_writers) == [second_writer]

    lock.release_write_lock()
    assert second_acquired.wait(5)
    assert results["second"] is True
    assert results.get("reader") is None

    release_second.set()
    second_writer.join(5)
    assert not second_writer.is_alive()

    assert reader_acquired.wait(5)
    assert results["reader"] is True
    release_reader.set()
    waiting_reader.join(5)
    assert not waiting_reader.is_alive()
    assert lock.owner is None
    assert not lock.has_pending_writers


@pytest.mark.parametrize("timeout", [-1, -0.01])
def test_invalid_timeout_is_rejected_before_state_changes(timeout):
    lock, _, _ = make_lock()

    with pytest.raises(ValueError):
        lock.acquire_read_lock(timeout=timeout)
    with pytest.raises(ValueError):
        lock.acquire_write_lock(timeout=timeout)

    with pytest.raises(ValueError):
        with lock.read_lock(timeout=timeout):
            pass
    with pytest.raises(ValueError):
        with lock.write_lock(timeout=timeout):
            pass

    assert lock.owner is None
    assert not lock.has_pending_writers


def test_blocking_timeout_returns_false_and_leaves_no_ownership():
    lock = ReaderWriterLock()
    holder_started = threading.Event()
    release_holder = threading.Event()

    def hold_write_lock():
        lock.acquire_write_lock()
        holder_started.set()
        release_holder.wait(5)
        lock.release_write_lock()

    holder = start_thread(hold_write_lock)
    assert holder_started.wait(5)

    assert lock.acquire_read_lock(timeout=0.01) is False
    assert lock.acquire_write_lock(timeout=0.01) is False
    assert not lock.is_reader()
    assert not lock.is_writer(check_pending=True)
    assert not lock.has_pending_writers

    with pytest.raises(threading.ThreadError):
        with lock.read_lock(timeout=0.01):
            pytest.fail("timed out read lock body should not run")
    with pytest.raises(threading.ThreadError):
        with lock.write_lock(timeout=0.01):
            pytest.fail("timed out write lock body should not run")

    assert not lock.is_reader()
    assert not lock.is_writer(check_pending=True)
    release_holder.set()
    holder.join(5)
    assert not holder.is_alive()


def test_nonblocking_acquisition_returns_false_without_waiting():
    lock, _, condition = make_lock()
    holder_started = threading.Event()
    release_holder = threading.Event()

    def hold_write_lock():
        lock.acquire_write_lock()
        holder_started.set()
        release_holder.wait(5)
        lock.release_write_lock()

    holder = start_thread(hold_write_lock)
    assert holder_started.wait(5)

    assert lock.acquire_read_lock(blocking=False) is False
    assert lock.acquire_write_lock(blocking=False) is False
    assert condition.wait_count == 0
    assert not lock.has_pending_writers

    with pytest.raises(threading.ThreadError):
        with lock.read_lock(blocking=False):
            pytest.fail("read lock body should not run")
    with pytest.raises(threading.ThreadError):
        with lock.write_lock(blocking=False):
            pytest.fail("write lock body should not run")

    assert condition.wait_count == 0
    assert not lock.has_pending_writers
    release_holder.set()
    holder.join(5)
    assert not holder.is_alive()


def test_nonblocking_reentrant_locks_preserve_existing_counts():
    lock, _, _ = make_lock()

    lock.acquire_read_lock()
    assert lock.acquire_read_lock(blocking=False, timeout=0) is True
    assert lock._readers == {threading.current_thread(): 2}
    lock.release_read_lock()
    assert lock._readers == {threading.current_thread(): 1}
    lock.release_read_lock()

    lock.acquire_write_lock()
    assert lock.acquire_write_lock(blocking=False, timeout=0) is True
    assert lock._writer_entries == 2
    lock.release_write_lock()
    assert lock.is_writer(check_pending=False)
    assert lock._writer_entries == 1
    lock.release_write_lock()
    assert lock.owner is None


def test_reader_reentry_remains_counted_while_writer_waits():
    lock, _, condition = make_lock()
    acquired = {}
    writer_acquired = threading.Event()
    release_writer = threading.Event()

    def waiting_writer_target():
        acquired["writer"] = lock.acquire_write_lock(timeout=5)
        writer_acquired.set()
        release_writer.wait(5)
        lock.release_write_lock()

    lock.acquire_read_lock()
    writer = start_thread(waiting_writer_target)
    wait_for_count(condition, 1)

    assert lock.acquire_read_lock(blocking=False) is True
    assert lock._readers == {threading.current_thread(): 2}
    lock.release_read_lock()
    assert lock._readers == {threading.current_thread(): 1}
    lock.release_read_lock()

    assert writer_acquired.wait(5)
    assert acquired["writer"] is True
    release_writer.set()
    writer.join(5)
    assert not writer.is_alive()
    assert lock.owner is None

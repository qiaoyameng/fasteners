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

import multiprocessing
import os
import select
import signal
import time

import pytest

import fasteners

WIN32 = os.name == 'nt'


def _read_bytes(fd, count, timeout):
    """Read exactly ``count`` bytes from ``fd`` within ``timeout`` seconds."""
    data = b''
    deadline = time.monotonic() + timeout
    while len(data) < count:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        ready, _, _ = select.select([fd], [], [], remaining)
        if not ready:
            break
        chunk = os.read(fd, count - len(data))
        if not chunk:
            break
        data += chunk
    return data


def _expect_no_bytes(fd, duration):
    ready, _, _ = select.select([fd], [], [], duration)
    assert not ready, "unexpected data on pipe: {!r}".format(os.read(fd, 64))


def _holder_child(path, capacity, wfd):
    # Acquire a slot, announce it, then hold until killed.
    sem = fasteners.InterProcessSemaphore(path, capacity)
    permit = sem.acquire(timeout=30)
    if permit is None:
        os.write(wfd, b'F')
        os._exit(1)
    os.write(wfd, b'E')
    while True:
        time.sleep(0.1)


def _waiter_child(path, capacity, wfd):
    # Wait for a slot, announce entry, then release and announce exit.
    sem = fasteners.InterProcessSemaphore(path, capacity)
    permit = sem.acquire(timeout=30)
    if permit is None:
        os.write(wfd, b'F')
        os._exit(1)
    os.write(wfd, b'E')
    permit.release()
    os.write(wfd, b'X')
    os._exit(0)


def _worker_child(path, capacity, wfd, hold):
    sem = fasteners.InterProcessSemaphore(path, capacity)
    permit = sem.acquire(timeout=30)
    if permit is None:
        os.write(wfd, b'F')
        os._exit(1)
    os.write(wfd, b'E')
    time.sleep(hold)
    permit.release()
    os.write(wfd, b'X')
    os._exit(0)


class TestInterProcessSemaphore:

    def test_capacity_recorded_and_mismatch_rejected(self, tmp_path):
        sem = fasteners.InterProcessSemaphore(str(tmp_path / 'sem'), 3)
        assert sem.capacity == 3
        again = fasteners.InterProcessSemaphore(str(tmp_path / 'sem'), 3)
        assert again.capacity == 3
        with pytest.raises(ValueError):
            fasteners.InterProcessSemaphore(str(tmp_path / 'sem'), 4)

    def test_invalid_capacity(self, tmp_path):
        with pytest.raises(ValueError):
            fasteners.InterProcessSemaphore(str(tmp_path / 'sem'), 0)

    def test_capacity_enforced_and_slots_reused(self, tmp_path):
        sem = fasteners.InterProcessSemaphore(str(tmp_path / 'sem'), 3)
        permits = [sem.acquire() for _ in range(3)]
        assert all(p is not None for p in permits)
        assert len({p.slot for p in permits}) == 3
        assert sem.acquire(blocking=False) is None
        permits[0].release()
        reused = sem.acquire(blocking=False)
        assert reused is not None
        assert reused.slot == permits[0].slot
        for permit in permits[1:] + [reused]:
            permit.release()

    def test_acquire_timeout(self, tmp_path):
        sem = fasteners.InterProcessSemaphore(str(tmp_path / 'sem'), 1)
        permit = sem.acquire()
        start = time.monotonic()
        assert sem.acquire(timeout=0.3) is None
        assert time.monotonic() - start >= 0.25
        permit.release()
        assert sem.acquire(timeout=0.3) is not None

    def test_double_release_raises(self, tmp_path):
        sem = fasteners.InterProcessSemaphore(str(tmp_path / 'sem'), 1)
        permit = sem.acquire()
        permit.release()
        assert permit.released
        with pytest.raises(RuntimeError):
            permit.release()

    def test_context_manager(self, tmp_path):
        sem = fasteners.InterProcessSemaphore(str(tmp_path / 'sem'), 1)
        with sem.acquire() as permit:
            assert sem.acquire(blocking=False) is None
        assert permit.released
        assert sem.acquire(blocking=False) is not None

    @pytest.mark.skipif(WIN32, reason="fork specific")
    def test_forked_child_cannot_release(self, tmp_path):
        sem = fasteners.InterProcessSemaphore(str(tmp_path / 'sem'), 1)
        permit = sem.acquire()
        rfd, wfd = os.pipe()
        pid = os.fork()
        if pid == 0:
            os.close(rfd)
            try:
                permit.release()
            except RuntimeError:
                os.write(wfd, b'R')
            else:
                os.write(wfd, b'B')
            # Discarding the inherited handle must not free the slot.
            permit.abandon()
            os._exit(0)
        os.close(wfd)
        try:
            assert _read_bytes(rfd, 1, 10) == b'R'
            os.waitpid(pid, 0)
            # The child is gone and abandoned its handle, yet the slot is
            # still held by this process.
            assert sem.acquire(blocking=False) is None
        finally:
            os.close(rfd)
        permit.release()
        assert sem.acquire(blocking=False) is not None

    def test_concurrency_never_exceeds_capacity(self, tmp_path):
        capacity = 3
        children = 6
        path = str(tmp_path / 'sem')
        rfd, wfd = os.pipe()
        processes = [
            multiprocessing.Process(
                target=_worker_child, args=(path, capacity, wfd, 0.3))
            for _ in range(children)
        ]
        for proc in processes:
            proc.start()
        os.close(wfd)
        try:
            data = _read_bytes(rfd, 2 * children, 30)
        finally:
            os.close(rfd)
            for proc in processes:
                proc.join(10)
                if proc.is_alive():
                    proc.terminate()
        assert b'F' not in data
        assert data.count(b'E') == children
        assert data.count(b'X') == children
        concurrent = 0
        peak = 0
        for byte in data:
            if byte == ord('E'):
                concurrent += 1
                peak = max(peak, concurrent)
            else:
                concurrent -= 1
        assert 1 <= peak <= capacity
        for proc in processes:
            assert proc.exitcode == 0

    def test_killed_holder_frees_slot_for_waiter(self, tmp_path):
        capacity = 3
        path = str(tmp_path / 'sem')
        rfd, wfd = os.pipe()
        holders = [
            multiprocessing.Process(
                target=_holder_child, args=(path, capacity, wfd))
            for _ in range(capacity)
        ]
        for proc in holders:
            proc.start()
        try:
            # All three slots are taken.
            assert _read_bytes(rfd, capacity, 30) == b'E' * capacity
            waiter = multiprocessing.Process(
                target=_waiter_child, args=(path, capacity, wfd))
            waiter.start()
            try:
                # The waiter can not enter while all holders are alive.
                _expect_no_bytes(rfd, 1.0)
                # Kill one holder without giving it a chance to clean up;
                # the OS releases its flock when the process dies.
                os.kill(holders[0].pid, signal.SIGKILL)
                holders[0].join(10)
                # The waiter takes over the freed slot and releases it.
                assert _read_bytes(rfd, 2, 30) == b'EX'
                waiter.join(10)
                assert waiter.exitcode == 0
            finally:
                if waiter.is_alive():
                    waiter.terminate()
                waiter.join()
        finally:
            os.close(rfd)
            os.close(wfd)
            for proc in holders:
                if proc.is_alive():
                    proc.terminate()
                proc.join()

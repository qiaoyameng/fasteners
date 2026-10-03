# Inter thread locks

Fasteners inter-thread locks were build specifically for the needs of
`oslo.concurrency` project and thus have a rather peculiar API. We do not
recommend using it fully, and it is hence not documented (but maintained until
the end of time).

Instead, we recommend limiting the use of fasteners inter-thread readers writer
lock to the basic API:

```python
import fasteners

# for threads
rw_lock = fasteners.ReaderWriterLock()

with rw_lock.write_lock():
    ...  # write access

with rw_lock.read_lock():
    ...  # read access
```

## Upgrading and downgrading

A read lock can not silently be turned into a write lock (a plain reader
attempting to acquire the write lock raises an exception, as that would
deadlock against another reader doing the same). Instead, a thread that
knows upfront that it may need to write can take the (single) upgrade
intent together with its read lock:

```python
with rw_lock.upgradable_read_lock():
    ...  # read access
    if needs_write:
        # Blocks new readers, waits for the remaining readers to leave
        # and then atomically becomes the writer. Pass a timeout (in
        # seconds) to give up instead; on timeout the original read
        # lock(s) are kept.
        if rw_lock.upgrade_read_lock():
            ...  # exclusive write access
            # Atomically become a reader again (no other writer can slip
            # in-between), or simply release the write lock instead.
            rw_lock.downgrade_write_lock()
    ...  # (still) read access
```

At most one thread at a time may hold the upgrade intent; other threads
requesting an upgradable read lock wait until the intent is released (or
consumed by a successful upgrade).

## (Lack of) Features

Fasteners inter-thread readers writer lock is

* explicitly [upgradeable] via the upgrade intent described above. An attempt
  to get a reader's lock while holding a writer's lock (or vice versa,
  without holding the upgrade intent) will raise an exception.

* [reentrant] (!). You can acquire (and correspondingly release) the lock
  multiple times.

* has writer preference. Readers will queue after writers and pending writers.

[upgradeable]: https://en.wikipedia.org/wiki/Readers%E2%80%93writer_lock#Upgradable_RW_lock>

[reentrant]: https://en.wikipedia.org/wiki/Reentrant_mutex

## Different thread creation mechanisms

If your threads are created by some other means than the standard library `threading`
module, you may need to provide corresponding thread identification and synchronisation
functions to the `ReaderWriterLock`.

### Eventlet

In particular, in case of `eventlet` threads, you should monkey_patch the stdlib threads with `eventlet.monkey_patch(tread=True)`
and initialise the `ReaderWriterLock` as `ReaderWriterLock(current_thread_functor=eventlet.getcurrent)`.

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

with rw_lock.upgradeable_read_lock():
    ...  # read access
    with rw_lock.upgrade_write_lock():
        ...  # write access
```

## (Lack of) Features

Fasteners inter-thread readers writer lock is

* supports an explicit [upgradeable] read lock. Only one thread may hold the
  upgradeable read lock, and ordinary read locks can not be upgraded. Use
  `upgrade_write_lock()` to upgrade, and `downgrade_write_lock()` to perform an
  atomic write-to-read transition.

* [reentrant] (!). You can acquire (and correspondingly release) the lock
  multiple times.

* has writer preference. Readers will queue after writers and pending writers.
  An upgrade request blocks new readers and waits for readers that already
  entered; failed or timed out upgrades retain the caller's read lock layers.

[upgradeable]: https://en.wikipedia.org/wiki/Readers%E2%80%93writer_lock#Upgradable_RW_lock>

[reentrant]: https://en.wikipedia.org/wiki/Reentrant_mutex

## Different thread creation mechanisms

If your threads are created by some other means than the standard library `threading`
module, you may need to provide corresponding thread identification and synchronisation
functions to the `ReaderWriterLock`.

### Eventlet

In particular, in case of `eventlet` threads, you should monkey_patch the stdlib threads with `eventlet.monkey_patch(tread=True)`
and initialise the `ReaderWriterLock` as `ReaderWriterLock(current_thread_functor=eventlet.getcurrent)`.

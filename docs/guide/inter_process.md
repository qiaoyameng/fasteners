# Inter process locks

Fasteners inter-process locks are cross-platform and are released automatically
if the process crashes. They are based on the platform specific locking
mechanisms:

* fcntl for posix (Linux and OSX)
* LockFileEx (via pywin32) and \_locking (via msvcrt) for Windows

## Difference from `multiprocessing.Lock`

Python standard library [multiprocessing.Lock] functions when the processes are
launched by a single main process, who is responsible for managing the 
synchronization. `fasteners` locks use the operating system mechanisms for 
synchronization management, and hence work between processes that were launched
independently.

## Timeouts

`fasteners` locks support timeouts, that can be used as follows:

```python
import fasteners

lock = fasteners.InterProcessLock('path/to/lock.file')

lock.acquire(timeout=10)
... # exclusive access
lock.release()
```

Equivalently for readers writer lock:


```python
import fasteners

lock = fasteners.InterProcessReaderWriterLock('path/to/lock.file')

lock.acquire_read_lock(timeout=10)
... # exclusive access
lock.release_read_lock()

lock.acquire_write_lock(timeout=10)
... # exclusive access
lock.release_write_lock()
```

## Multiple locks

When one operation needs several resources, acquire all of their locks with
`interprocess_locks` (or `InterProcessMultiLock`):

```python
import fasteners

with fasteners.interprocess_locks('a.lock', 'b.lock', timeout=10):
    ... # protected access to both resources
```

Paths are normalized and duplicates are coalesced using one stable global
order, so callers may pass paths in different orders without creating a lock
ordering deadlock. The timeout is one budget shared by every lock acquisition
in the group; it is not restarted for each lock.

Relative components, symlinks in existing parent directories, and a dangling
final symlink whose target is the requested lock file are recognized as
aliases. User shortcuts such as `~` are not expanded, hard links cannot be
recognized, and a symlink changed after constructing the multi-lock object is
not resolved again.

If one lock cannot be acquired, locks already acquired by that attempt are
released in reverse order. Locks which were already acquired before entering
the context remain held. Lock files are never deleted to resolve contention.
If multiple releases fail, the resulting exception preserves all failures.

## Decorators

For extra sugar, a function that always needs exclusive / read / write access
can be decorated using one of the provided decorators. Note that they do not 
expose the timeout parameter, and always block until the lock is acquired.

```python
import fasteners


@fasteners.interprocess_read_locked
def read_file():
  ...

@fasteners.interprocess_write_locked
def write_file():
  ...

@fasteners.interprocess_locked
def do_something_exclusive():
  ...
```

## (Lack of) Features

The intersection of fcntl and LockFileEx features is quite small, hence you
should always assume that:

* Locks are advisory. They do not prevent the modification of the locked file
  by other processes.

* Locks can be unintentionally released by simply opening and closing the file
  descriptor, so lock files must be accessed only using provided abstractions.

* Locks are not [reentrant]. An attempt to acquire a lock multiple times can
  result in a deadlock or a crash upon a release of the lock.

* Reader writer locks are not [upgradeable]. An attempt to get a reader's lock
  while holding a writer's lock (or vice versa) can result in a deadlock or a
  crash upon a release of the lock.

* There are no guarantees regarding usage by multiple threads in a
  single process. The locks work only between processes.

## Resources

To learn more about the complications of locking on different platforms we
recommend the following resources:

* [File locking in Linux (blog post)](https://gavv.github.io/articles/file-locks/)

* [On the Brokenness of File Locking (blog post)](http://0pointer.de/blog/projects/locking.html)

* [Everything you never wanted to know about file locking (blog post)](https://chris.improbable.org/2010/12/16/everything-you-never-wanted-to-know-about-file-locking/)

* [Record Locking (course notes)](http://poincare.matf.bg.ac.rs/~ivana/courses/ps/sistemi_knjige/pomocno/apue/APUE/0201433079/ch14lev1sec3.html)

* [Windows NT Files -- Locking (pywin32 docs)](http://timgolden.me.uk/pywin32-docs/Windows_NT_Files_.2d.2d_Locking.html)

* [_locking (Windows Dev Center)](https://docs.microsoft.com/en-us/cpp/c-runtime-library/reference/locking?view=vs-2019)

* [LockFileEx function (Windows Dev Center)](https://docs.microsoft.com/en-us/windows/win32/api/fileapi/nf-fileapi-lockfileex)

[upgradeable]: https://en.wikipedia.org/wiki/Readers%E2%80%93writer_lock#Upgradable_RW_lock>
[reentrant]: https://en.wikipedia.org/wiki/Reentrant_mutex
[multiprocessing.Lock]: https://docs.python.org/3/library/multiprocessing.html#multiprocessing.Lock

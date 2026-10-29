"""Process-owned file leases; stale files are harmless, live OS locks are not."""
from contextlib import contextmanager
import os
from pathlib import Path

@contextmanager
def file_lock(path,label='GPU lease'):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    stream=path.open('a+b')
    if stream.tell()==0:
        stream.write(b'0');stream.flush()
    stream.seek(0)
    try:
        try:
            if os.name=='nt':
                import msvcrt
                msvcrt.locking(stream.fileno(),msvcrt.LK_NBLCK,1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except OSError as error:
            raise RuntimeError(f'{label} is already held; no second job started') from error
        yield
    finally:
        stream.close()

@contextmanager
def gpu_lease(path=None):
    if path is None:
        yield
    else:
        with file_lock(path):
            yield

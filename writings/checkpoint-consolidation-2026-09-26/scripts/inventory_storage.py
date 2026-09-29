"""Read-only file metadata inventory. Paths and allocation only; no model loading."""
import json
import os
import sys
from pathlib import Path

root = Path(sys.argv[1])
host = sys.argv[2]
for directory, dirs, names in os.walk(root, followlinks=False):
    dirs[:] = sorted(d for d in dirs if d != 'GS-LSS-paper-checkpoints')
    for name in sorted(names):
        p = Path(directory) / name
        try:
            s = p.lstat()
            print(json.dumps(dict(host=host, path=str(p), bytes=s.st_size,
                allocated=s.st_blocks * 512, mtime_ns=s.st_mtime_ns,
                inode=s.st_ino, device=s.st_dev, nlink=s.st_nlink,
                symlink=os.readlink(p) if p.is_symlink() else '')))
        except OSError as error:
            print(json.dumps(dict(host=host, path=str(p), error=str(error))))

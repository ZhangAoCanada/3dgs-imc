"""Read-only inventory for the explicitly requested output cleanup."""
import collections
import datetime
import json
import os
from pathlib import Path
import shutil
import sqlite3
import stat
import time

ROOT = Path('/data1/zhangao/outputs')
WORK = Path(__file__).resolve().parent


def category(path):
    parts = path.split('/')
    name = parts[-1]
    suffix = Path(name).suffix.lower()
    if name == 'input.ply':
        return 'input_point_cloud'
    if suffix in {'.ply', '.pth', '.pt', '.ckpt', '.safetensors'}:
        return 'checkpoint_or_neural_state'
    if 'gt' in parts:
        return 'evaluation_ground_truth'
    if any(part.startswith('dataset') for part in parts):
        return 'frozen_dataset'
    if any(part in {'renders', 'render', 'depths', 'depth', 'normal', 'normals'} for part in parts):
        return 'rendered_outputs'
    if suffix in {'.png', '.jpg', '.jpeg', '.exr', '.tiff', '.tif'}:
        return 'other_images'
    if suffix in {'.mp4', '.avi', '.gif', '.mov'}:
        return 'videos'
    if '__pycache__' in parts or suffix == '.pyc':
        return 'python_cache'
    if suffix in {'.so', '.o', '.a'}:
        return 'compiled_code'
    return 'metadata_logs_or_other'


def main():
    start = time.time()
    dbpath = WORK / 'inventory.sqlite'
    assert not dbpath.exists(), 'Do not overwrite an earlier inventory'
    db = sqlite3.connect(dbpath)
    db.execute('CREATE TABLE files(path TEXT PRIMARY KEY, category TEXT, top TEXT, size INTEGER, allocated INTEGER, dev INTEGER, ino INTEGER, nlink INTEGER, mtime_ns INTEGER, mode INTEGER, uid INTEGER, gid INTEGER, kind TEXT, target TEXT)')
    db.execute('CREATE TABLE errors(path TEXT, error TEXT)')
    batch = []
    count = 0
    last_print = time.monotonic()
    usage = shutil.disk_usage(ROOT)
    for base, dirs, names in os.walk(ROOT, followlinks=False):
        entries = names + [d for d in dirs if os.path.islink(os.path.join(base, d))]
        for name in entries:
            p = Path(base) / name
            rel = str(p.relative_to(ROOT))
            try:
                s = p.lstat()
                kind = 'symlink' if stat.S_ISLNK(s.st_mode) else ('file' if stat.S_ISREG(s.st_mode) else 'special')
                batch.append((rel, category(rel), rel.split('/')[0], s.st_size, s.st_blocks * 512,
                              s.st_dev, s.st_ino, s.st_nlink, s.st_mtime_ns, stat.S_IMODE(s.st_mode),
                              s.st_uid, s.st_gid, kind, os.readlink(p) if kind == 'symlink' else None))
                count += 1
            except OSError as exc:
                db.execute('INSERT INTO errors VALUES (?,?)', (rel, repr(exc)))
            if len(batch) >= 5000:
                db.executemany('INSERT INTO files VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)', batch)
                db.commit()
                batch.clear()
            if time.monotonic() - last_print > 20:
                print(json.dumps({'scanned': count, 'elapsed_s': round(time.time() - start)}), flush=True)
                last_print = time.monotonic()
    if batch:
        db.executemany('INSERT INTO files VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)', batch)
    db.execute('CREATE INDEX size_idx ON files(size)')
    db.execute('CREATE INDEX inode_idx ON files(dev,ino)')
    db.commit()
    summaries = {}
    for key in ['category', 'top']:
        summaries[key] = [dict(zip(['name','files','logical_bytes','allocated_unique_bytes'], r)) for r in db.execute(
            f'SELECT {key},COUNT(*),SUM(size),SUM(allocated) FROM (SELECT * FROM files WHERE kind="file" GROUP BY dev,ino) GROUP BY {key} ORDER BY SUM(allocated) DESC')]
    summaries.update(root=str(ROOT), host=os.uname().nodename, started=datetime.datetime.fromtimestamp(start).astimezone().isoformat(),
                     elapsed_s=time.time()-start, disk_before=usage._asdict(), entries=count,
                     errors=list(db.execute('SELECT * FROM errors')))
    (WORK/'inventory_summary.json').write_text(json.dumps(summaries, indent=2)+'\n')
    print(json.dumps(summaries,indent=2),flush=True)


if __name__ == '__main__':
    main()

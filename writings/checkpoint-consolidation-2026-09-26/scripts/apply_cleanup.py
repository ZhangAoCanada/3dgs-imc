"""Apply only the reviewed warmup-snapshot cleanup plan, logging each removal."""
import csv
import datetime
import hashlib
import json
import os
from pathlib import Path
import re

WORK = Path(__file__).resolve().parents[1]
OUT = Path('/data1/zhangao/outputs')
rows = list(csv.DictReader((WORK/'cleanup_plan.csv').open()))
protected = {Path(r['path']).resolve() for r in csv.DictReader((WORK/'protected_checkpoints.csv').open()) if r['host']=='hust10'}
log_path = WORK/'cleanup_actions.jsonl'
assert not log_path.exists(), 'Cleanup has already been attempted; inspect its log instead of repeating blindly.'
assert rows and all(re.fullmatch(r'chkpnt3000_mv[123]\.pth',Path(r['path']).name) for r in rows)
types={'char':1,'uchar':1,'int8':1,'uint8':1,'short':2,'ushort':2,'int16':2,'uint16':2,'int':4,'uint':4,'int32':4,'uint32':4,'float':4,'float32':4,'double':8,'float64':8}

def check(row):
    p=Path(row['path']);keep=Path(row['retained_checkpoint'])
    assert not p.is_symlink() and p.resolve() not in protected
    assert any(p.resolve().is_relative_to(OUT/x) for x in ('baselines_multi_angle','baselines_discontinuous'))
    s=p.stat()
    assert (s.st_size,s.st_mtime_ns,s.st_ino,s.st_dev)==tuple(int(row[k]) for k in ('bytes','mtime_ns','inode','device'))
    with p.open('rb') as f:assert hashlib.sha256(f.read(4096)).hexdigest()==row['prefix_sha256']
    with keep.open('rb') as f:b=f.read(32768)
    end=b.index(b'end_header\n')+len(b'end_header\n');h=b[:end].decode('ascii')
    assert h.count('element ')==1 and 'binary_' in h
    n=int(re.search(r'element vertex (\d+)',h)[1]);fields=re.findall(r'^property (\w+) \w+',h,re.M)
    assert keep.stat().st_size==int(row['retained_bytes'])==end+n*sum(types[x] for x in fields)
    assert (p.parent/'cfg_args').is_file()
    return p

# Finish the complete preflight before the first deletion.
for row in rows:check(row)
removed=0;allocated=0
with log_path.open('x') as log:
    for row in rows:
        p=check(row)
        record=dict(row,phase='intent',time_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        log.write(json.dumps(record)+'\n');log.flush();os.fsync(log.fileno())
        p.unlink()
        log.write(json.dumps(dict(path=str(p),phase='removed',time_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))+'\n');log.flush();os.fsync(log.fileno())
        removed+=1
        if int(row['nlink'])==1:allocated+=int(row['allocated'])
print(json.dumps(dict(removed_files=removed,estimated_freed_GiB=allocated/2**30,log=str(log_path)),indent=2))

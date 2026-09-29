"""Read-only checkpoint and small provenance-file inventory. No tensor deserialization."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import sys

root = Path(sys.argv[1] if len(sys.argv) > 1 else '/data1/zhangao/outputs')
host = sys.argv[2] if len(sys.argv) > 2 else os.uname().nodename
skip = {'renders', 'gt', 'images', 'images_2', 'images_4', 'images_8', 'depth', 'depths', 'normal', 'normals', 'wandb', '__pycache__', '.git', 'node_modules', 'dav2', 'dav2_4', 'tensorboard', 'train_cameras', 'test_cameras'}
files, errors, top = [], [], []
small_exact = {'cfg_args', 'results.json', 'metrics.csv', 'training_time.txt', 'training_time.json', 'timing.json', 'timing_breakdown.json', 'keep_manifest.json', 'manifest.json', 'INDEX.csv', 'command.txt', 'config.json', 'run_meta.json', 'provenance.json'}
ply_types = {'char':1, 'uchar':1, 'int8':1, 'uint8':1, 'short':2, 'ushort':2, 'int16':2, 'uint16':2, 'int':4, 'uint':4, 'int32':4, 'uint32':4, 'float':4, 'float32':4, 'double':8, 'float64':8}
def ply_header(p, size):
    try:
        with p.open('rb') as f:
            b = f.read(32768)
        end = b.find(b'end_header\n')
        if end < 0: return {'check':'missing_header_terminator'}
        header_bytes = end + len(b'end_header\n')
        h = b[:header_bytes].decode('ascii')
        vm = re.search(r'element vertex (\d+)',h)
        n = int(vm[1]) if vm else None
        fields = re.findall(r'^property (\w+) (\w+)',h,re.M)
        result = {'format':re.search(r'format (\S+)',h)[1], 'vertices':n, 'properties':[x[1] for x in fields], 'header_sha256':hashlib.sha256(b[:header_bytes]).hexdigest()}
        if n is not None and 'binary_' in h and all(t in ply_types for t,_ in fields) and h.count('element ') == 1:
            expected = header_bytes + n * sum(ply_types[t] for t,_ in fields)
            result.update(expected_bytes=expected, check='size_matches_header' if size == expected else 'size_mismatch')
        else:
            result['check']='header_only'
        return result
    except Exception as e: return {'check':'error','error':str(e)}
for current, dirs, names in os.walk(root, followlinks=False, onerror=lambda e: errors.append(str(e))):
    dirs[:] = sorted(d for d in dirs if d not in skip)
    if Path(current) == root: top = dirs.copy()
    for name in sorted(names):
        p = Path(current) / name
        ext = p.suffix.lower()
        checkpoint = ext in {'.ply','.pth','.pt','.ckpt'}
        provenance = name in small_exact or ext in {'.csv','.json','.yaml','.yml','.md'} or any(k in name.lower() for k in ('timing','training_time','manifest','command','keep_dirs','aerial_runs'))
        if not (checkpoint or provenance): continue
        try:
            s = p.stat()
            r = dict(host=host,path=str(p),bytes=s.st_size,mtime_utc=datetime.datetime.fromtimestamp(s.st_mtime,datetime.timezone.utc).isoformat(), kind='checkpoint' if checkpoint else 'provenance')
            if p.is_symlink(): r['symlink_target']=str(p.resolve())
            if ext == '.ply': r['ply']=ply_header(p,s.st_size)
            read = not checkpoint and s.st_size <= 1048576 and name not in ('per_view.json','cameras.json') and not name.endswith('heartbeat.json')
            if read:
                r['text'] = p.read_text(errors='replace')
                r['sha256'] = hashlib.sha256(r['text'].encode()).hexdigest()
            files.append(r)
        except OSError as e: errors.append({'path':str(p),'error':str(e)})
print(json.dumps(dict(host=host,root=str(root),scanned_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),top_level_directories=top,files=files,errors=errors),ensure_ascii=False))

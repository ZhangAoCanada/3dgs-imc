"""Plan narrow cleanup of obsolete baseline intermediates; never delete in planning."""
import csv
import gzip
import hashlib
import json
from pathlib import Path
import re

WORK = Path(__file__).resolve().parents[1]
OUT = Path('/data1/zhangao/outputs')
protected = {r['path'] for r in csv.DictReader((WORK/'protected_checkpoints.csv').open()) if r['host']=='hust10'}
types = {'char':1,'uchar':1,'int8':1,'uint8':1,'short':2,'ushort':2,'int16':2,'uint16':2,'int':4,'uint':4,'int32':4,'uint32':4,'float':4,'float32':4,'double':8,'float64':8}

def valid_ply(p):
    try:
        with p.open('rb') as f: b=f.read(32768)
        end=b.index(b'end_header\n')+len(b'end_header\n');h=b[:end].decode('ascii')
        if 'binary_' not in h or h.count('element ')!=1:return False
        count=int(re.search(r'element vertex (\d+)',h)[1])
        fields=re.findall(r'^property (\w+) \w+',h,re.M)
        return count>0 and p.stat().st_size==end+count*sum(types[x] for x in fields)
    except (OSError,ValueError,KeyError,TypeError):return False

cache={}
def metrics(run):
    if run not in cache:
        rows={}
        for p in [run/'metrics.csv',run/'results.json']:
            if not p.is_file():continue
            try:
                d=json.loads(p.read_text()) if p.suffix=='.json' else {r['method']:r for r in csv.DictReader(p.open())}
                for k,v in d.items():
                    m=re.search(r'(\d+)',k)
                    if m and all(x in v for x in ['SSIM','PSNR','LPIPS']):rows[int(m[1])]={x:float(v[x]) for x in ['SSIM','PSNR','LPIPS']}
            except (ValueError,KeyError,TypeError):continue
        cache[run]=rows
    return cache[run]

plans=[]
with gzip.open(WORK/'evidence/local_files_before.jsonl.gz','rt') as f:
    for line in f:
        r=json.loads(line);p=Path(r['path'])
        if 'bytes' not in r or r.get('symlink') or str(p) in protected:continue
        if not any(p.is_relative_to(OUT/x) for x in ['baselines_multi_angle','baselines_discontinuous']):continue
        reason='';run=None
        if re.fullmatch(r'chkpnt3000_mv[123]\.pth',p.name):
            run=p.parent;reason='obsolete MVGS 3000-step optimizer snapshot; complete evaluated 30000-step rendering model retained'
        elif p.name=='point_cloud.ply' and p.parent.name=='iteration_7000':
            run=p.parent.parent.parent
            m=metrics(run)
            if 7000 not in m or 30000 not in m:continue
            early,final=m[7000],m[30000]
            if not(final['PSNR']>=early['PSNR'] and final['SSIM']>=early['SSIM'] and final['LPIPS']<=early['LPIPS']):continue
            reason='unselected 7000-step baseline PLY; retained evaluated 30000-step model is no worse on all three stored image metrics'
        else:continue
        retained=run/'point_cloud/iteration_30000/point_cloud.ply'
        if 30000 not in metrics(run) or not valid_ply(retained):continue
        s=p.stat()
        if s.st_size!=r['bytes'] or s.st_mtime_ns!=r['mtime_ns']:continue
        with p.open('rb') as b:prefix=b.read(4096)
        plans.append(dict(path=str(p),bytes=s.st_size,allocated=r['allocated'],mtime_ns=s.st_mtime_ns,
            device=s.st_dev,inode=s.st_ino,nlink=s.st_nlink,reason=reason,
            retained_checkpoint=str(retained),retained_bytes=retained.stat().st_size,
            retained_metrics=json.dumps(metrics(run)[30000],sort_keys=True),
            prefix_sha256=hashlib.sha256(prefix).hexdigest(),hash_scope='first 4096 bytes, not full content'))
with (WORK/'cleanup_plan.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=list(plans[0]) if plans else ['path']);w.writeheader();w.writerows(plans)
summary={}
for r in plans:
    k='MVGS warmup optimizer' if r['path'].endswith('.pth') else 'dominated early baseline PLY'
    d=summary.setdefault(k,dict(files=0,GiB=0));d['files']+=1;d['GiB']+=r['allocated']/2**30
print(json.dumps(dict(protected_local_checkpoints=len(protected),cleanup=summary,total_GiB=sum(r['allocated'] for r in plans)/2**30),indent=2))

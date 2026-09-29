"""Hash source or copied files, checking stability before/after each read."""
from pathlib import Path
import csv
import json
import shlex
import subprocess
import sys

WORK=Path(__file__).resolve().parents[1]
host,side=sys.argv[1:3]
extra='--extra' in sys.argv[3:]
assert host in ('hust10','hust2','wsl') and side in ('source','destination')
rows=[r for r in csv.DictReader((WORK/('copy_extra_file_manifest.csv' if extra else 'copy_file_manifest.csv')).open()) if r['host']==host and r['action']=='copy']
items={r[side]:dict(path=r[side],bytes=int(r['bytes'])) for r in rows}
out=WORK/'evidence'/f'sha256_{host}_{side}{"_extra" if extra else ""}.jsonl'
assert not out.exists(),out
code=r'''import sys,json,hashlib,os
from pathlib import Path
items=json.load(sys.stdin)
for i,r in enumerate(items):
 p=Path(r['path']);before=p.stat();assert before.st_size==r['bytes'],str(p)
 h=hashlib.sha256()
 with p.open('rb') as f:
  while True:
   b=f.read(4*1024*1024)
   if not b:break
   h.update(b)
 after=p.stat();assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns),str(p)
 print(json.dumps(dict(path=str(p),bytes=before.st_size,sha256=h.hexdigest())),flush=True)
'''
if side=='destination' or host=='hust10':command=['python3','-c',code]
else:
    command=['ssh','-o','BatchMode=yes','-o','ClearAllForwardings=yes','-o','ServerAliveInterval=30']
    if host=='hust2':command+=['-o','HostName=192.168.1.129','-o','Port=22','-o','HostKeyAlias=[222.20.99.102]:2088','-o','StrictHostKeyChecking=yes']
    command += [host,'python3 -c '+shlex.quote(code)]
with out.open('xb') as f:
    result=subprocess.run(command,input=json.dumps(list(items.values())).encode(),stdout=f,check=True)
print(json.dumps(dict(host=host,side=side,files=len(items),GiB=sum(r['bytes'] for r in items.values())/2**30,output=str(out))),flush=True)

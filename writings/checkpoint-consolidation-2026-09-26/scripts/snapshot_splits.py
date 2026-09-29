"""Preserve exact available split files referenced by selected run configurations."""
import base64
import csv
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

WORK=Path(__file__).resolve().parents[1]
LIB=Path('/data1/zhangao/outputs/GS-LSS-paper-checkpoints')
runs=json.loads((WORK/'selected_runs.json').read_text())
code=r'''import ast,base64,json,sys
from pathlib import Path
result=[]
for run in json.load(sys.stdin):
 p=Path(run['run_dir'])/'cfg_args';cfg={};error=''
 try:
  node=ast.parse(p.read_text(),mode='eval').body
  cfg={k.arg:ast.literal_eval(k.value) for k in node.keywords}
 except Exception as e:error=str(e)
 split=cfg.get('test_selection_path','') or ''
 r=dict(host=run['host'],run_dir=run['run_dir'],source_path=cfg.get('source_path',''),test_path=cfg.get('test_path',''),images=cfg.get('images',''),resolution=cfg.get('resolution',''),split_path=split,config_error=error)
 if split:
  q=Path(split)
  if not q.is_absolute():r['split_status']='relative path requires original working directory'
  elif q.is_file():r.update(split_status='copied exact source',content=base64.b64encode(q.read_bytes()).decode())
  else:r['split_status']='original split path missing'
 else:r['split_status']='config unavailable; split unknown' if error else 'no explicit split file recorded'
 result.append(r)
print(json.dumps(result))'''
records=[]
for host in ('hust10','hust2','wsl'):
    command=['python3','-c',code]
    if host!='hust10':
        command=['ssh','-o','BatchMode=yes','-o','ClearAllForwardings=yes']
        if host=='hust2':command+=['-o','HostName=192.168.1.129','-o','Port=22','-o','HostKeyAlias=[222.20.99.102]:2088','-o','StrictHostKeyChecking=yes']
        command += [host,'python3 -c '+shlex.quote(code)]
    data=json.loads(subprocess.check_output(command,input=json.dumps([r for r in runs if r['host']==host]).encode()))
    for r in data:
        content=r.pop('content',None)
        if content is not None:
            b=base64.b64decode(content);h=hashlib.sha256(b).hexdigest();p=LIB/'splits/by_sha256'/f'{h}.txt';p.parent.mkdir(parents=True,exist_ok=True)
            if p.exists():assert p.read_bytes()==b
            else:p.write_bytes(b)
            r.update(split_sha256=h,local_split=str(p),nonempty_lines=len([x for x in b.splitlines() if x.strip()]))
        records.append(r)
keys=list(dict.fromkeys(k for r in records for k in r))
for path in [WORK/'dataset_and_split_sources.csv',LIB/'indexes/dataset_and_split_sources.csv']:
    with path.open('w') as f:w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(records)
print(json.dumps(dict(runs=len(records),runs_with_exact_split=sum(bool(r.get('local_split')) for r in records),unique_split_files=len(list((LIB/'splits/by_sha256').glob('*.txt'))),missing_original_splits=sum(r['split_status']=='original split path missing' for r in records)),indent=2))

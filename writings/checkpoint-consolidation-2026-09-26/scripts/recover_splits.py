"""Recover moved split files only with explicit source or saved-camera evidence."""
import base64
import csv
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

WORK=Path(__file__).resolve().parents[1]
LIB=Path('/data1/zhangao/outputs/GS-LSS-paper-checkpoints')
rows=list(csv.DictReader((WORK/'dataset_and_split_sources.csv').open()))
missing=[r for r in rows if r['split_status']=='original split path missing']
code=r'''import base64,json,sys
from pathlib import Path
requests=json.load(sys.stdin);results=[]
for r in requests:
 original=Path(r['split_path']);run=Path(r['run_dir']);answer=dict(host=r['host'],run_dir=r['run_dir'],original_split=str(original))
 if str(original).startswith('/mnt/c/'):
  if original.is_file():answer.update(status='configured WSL split recovered from its owning host',content=base64.b64encode(original.read_bytes()).decode(),recovered_path=str(original),split_source_host='wsl')
  else:answer['status']='configured WSL split still missing'
  results.append(answer);continue
 camera_path=run/'cameras.json';per_view_path=run/'per_view.json';n=None
 try:
  cameras=json.loads(camera_path.read_text())
  if per_view_path.is_file():
   per=json.loads(per_view_path.read_text());counts={len(v['PSNR']) for v in per.values() if isinstance(v,dict) and isinstance(v.get('PSNR'),dict)}
   if len(counts)==1:n=counts.pop()
  if n is None:
   counts={len(list(p.glob('*.png')))+len(list(p.glob('*.jpg'))) for p in (run/'test').glob('*/renders')}
   counts.discard(0)
   if len(counts)==1:n=counts.pop()
  if not n:raise ValueError('no independent test-view count')
  expected={Path(str(v['img_name'])).stem for v in cameras[:n]}
  if len(expected)!=n:raise ValueError('non-unique saved test cameras')
  scene=original.parent.name
  roots=['/data1/zhangao/mipnerf360','/data1/zhangao/DATA/mipnerf360','/data1/zhangao/DATA/tandt','/data1/zhangao/tandt_db/tandt','/data2/zhangao/tandt_db/tandt','/data1/zhangao/DATA/DATA-small']
  candidates=[Path(root)/scene/original.name for root in roots]
  found=None
  for p in candidates:
   if p.is_file():
    names=[x.strip() for x in p.read_text().splitlines() if x.strip()]
    if len(names)==n and {Path(x).stem for x in names}==expected:found=p;break
  if found:answer.update(status='relocated split membership verified against saved test-camera prefix and independent view count',content=base64.b64encode(found.read_bytes()).decode(),recovered_path=str(found),split_source_host='hust2',test_views=n,camera_evidence=str(camera_path),view_count_evidence=str(per_view_path) if per_view_path.is_file() else str(run/'test'))
  else:
   # The inspected Scene implementations serialize test cameras first. Recover
   # membership from that prefix and an independent saved evaluation view count.
   # Require unique matching image names in an existing dataset; this is a
   # reconstructed split, never described as a byte-identical original file.
   image_dirs=[Path(r.get('source_path',''))/r.get('images','images')]+[Path(root)/scene/r.get('images','images') for root in roots]
   reconstructed=None
   for d in image_dirs:
    if not d.is_dir():continue
    names={p.stem:p.name for p in d.iterdir() if p.is_file()}
    if expected.issubset(names):reconstructed=('\n'.join(names[Path(str(v['img_name'])).stem] for v in cameras[:n])+'\n').encode();break
   if reconstructed:answer.update(status='reconstructed held-out membership from saved test-camera prefix and independent view count',content=base64.b64encode(reconstructed).decode(),recovered_path=str(camera_path),split_source_host='hust2',test_views=n,camera_evidence=str(camera_path),image_directory=str(d),view_count_evidence=str(per_view_path) if per_view_path.is_file() else str(run/'test'))
   else:answer.update(status='no matching relocated split',test_views=n)
 except Exception as e:answer.update(status='split recovery unresolved',error=str(e))
 results.append(answer)
print(json.dumps(results))'''
answers=[]
for host in ['wsl','hust2']:
    request=[r for r in missing if (r['split_path'].startswith('/mnt/c/'))==(host=='wsl')]
    command=['ssh','-o','BatchMode=yes','-o','ClearAllForwardings=yes']
    if host=='hust2':command+=['-o','HostName=192.168.1.129','-o','Port=22','-o','HostKeyAlias=[222.20.99.102]:2088','-o','StrictHostKeyChecking=yes']
    command += [host,'python3 -c '+shlex.quote(code)]
    answers+=json.loads(subprocess.check_output(command,input=json.dumps(request).encode()))
byrun={(r['host'],r['run_dir']):r for r in rows};recovered=0
for a in answers:
    content=a.pop('content',None)
    if content is None:continue
    data=base64.b64decode(content);sha=hashlib.sha256(data).hexdigest();p=LIB/'splits/by_sha256'/f'{sha}.txt'
    if p.exists():assert p.read_bytes()==data
    else:p.write_bytes(data)
    r=byrun[(a['host'],a['run_dir'])]
    r.update(split_status=a['status'],split_sha256=sha,local_split=str(p),nonempty_lines=len([x for x in data.splitlines() if x.strip()]),recovered_split_path=a['recovered_path'],split_source_host=a['split_source_host'])
    recovered+=1
for path in [WORK/'dataset_and_split_sources.csv',LIB/'indexes/dataset_and_split_sources.csv']:
    with path.open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)
evidence=WORK/'evidence/split_recovery.json'
round_number=2
while evidence.exists():
    evidence=WORK/f'evidence/split_recovery_round{round_number}.json'
    round_number+=1
evidence.write_text(json.dumps(answers,indent=2)+'\n')
print(json.dumps(dict(missing_before=len(missing),recovered=recovered,remaining=sum(r['split_status']=='original split path missing' for r in rows)),indent=2))

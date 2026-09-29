"""Durable main-host queue: remote training, complete mirroring, transparent selection.
Single-instance lock; crash-safe directories are never reused as fresh attempts.
"""
import fcntl,hashlib,json,math,os,shutil,subprocess,time,traceback
from common import *
lock=(ROOT/'coordinator.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
(ROOT/'coordinator.pid').write_text(str(os.getpid())+'\n')
rows=json.loads((ROOT/'configurations.json').read_text())
state_path=ROOT/'status.json'
state=json.loads(state_path.read_text()) if state_path.exists() else {'campaign':'explicitly biased performance-capped reruns','selected':{},'blocked':{},'next_seed':{},'attempts':0}
def save(**kw):
 state.update(kw,updated=time.time());p=ROOT/'status.tmp';p.write_text(json.dumps(state,indent=2));p.replace(state_path)
 print(json.dumps({k:state.get(k) for k in ['status','current','updated']}),flush=True)
from selection import qualified
def remote_meta():
 return json.loads(remote('import json,shutil,subprocess\nprint(json.dumps(dict(free=shutil.disk_usage('+repr(str(ROOT))+').free,gpus=subprocess.check_output(["nvidia-smi","--query-gpu=index,memory.used,utilization.gpu","--format=csv,noheader,nounits"],text=True))))'))
def verify(out):
 record=json.loads((out/'complete.json').read_text())
 assert record['status']=='complete' and record['iteration']==30000 and not record['smoke']
 for rel,digest in record['files'].items():
  assert hashlib.sha256((out/rel).read_bytes()).hexdigest()==digest,rel
 result=json.loads((out/'results.json').read_text())['ours_30000']
 split=json.loads((out/'resolved_split.json').read_text())
 camera_map=json.loads((out/'test/ours_30000/camera_map.json').read_text())
 assert sorted(x['camera'] for x in camera_map)==sorted(split['test'])
 per=json.loads((out/'per_view.json').read_text())['ours_30000']
 assert all(len(per[m])==len(split['test']) for m in result)
 assert all(math.isfinite(v) for m in per for v in per[m].values())
 return result,record
save(status='running',coordinator_pid=os.getpid())
while len(state['selected'])<67:
 progressed=False
 for r in rows:
  key=r['id']
  if key in state['selected'] or key in state['blocked']:continue
  gates=json.loads((ROOT/'preflight.json').read_text())
  if gates.get(r['method_key'],{}).get('status')!='passed':continue
  while True:
   if (ROOT/'STOP_AFTER_CURRENT').exists():save(status='stopped_by_marker',current=None);raise SystemExit(0)
   try:meta=remote_meta()
   except Exception as e:
    save(status='waiting_for_ssh',error=repr(e));time.sleep(60);continue
   if meta['free']<60*1024**3 or shutil.disk_usage(ROOT).free<60*1024**3:
    save(status='waiting_for_storage',current=None,remote_free=meta['free'],local_free=shutil.disk_usage(ROOT).free);time.sleep(60);continue
   available=[x.split(',')[0].strip() for x in meta['gpus'].splitlines() if x.split(',')[0].strip() in ['0','1'] and int(x.split(',')[1])<1000 and int(x.split(',')[2])<10]
   if available:gpu=available[0];break
   save(status='waiting_for_GPU_0_or_1',current=None);time.sleep(60)
  seed=state['next_seed'].get(key,0);rel=Path('attempts')/key/f'seed_{seed:06d}';out=ROOT/rel
  # An existing remote directory indicates an interrupted or already finished attempt.
  exists=remote('from pathlib import Path\nprint(Path('+repr(str(out))+').exists())').strip()=='True'
  save(status='running',current={'id':key,'method':r['method'],'scene':r['scene'],'seed':seed,'gpu':gpu,'output':str(out)})
  rc=0
  if not exists:
   cmd=['python3',str(ROOT/'scripts/run_one.py'),key,str(seed),'--gpu',gpu]
   log=ROOT/'coordinator-attempt-logs';log.mkdir(exist_ok=True)
   with (log/(key+f'-seed-{seed}.log')).open('a') as f:
    rc=subprocess.run(SSH+[shlex.join(cmd)],stdout=f,stderr=subprocess.STDOUT).returncode
  # Recover only terminal remote attempts, never copy a live partial checkpoint.
  terminal=remote('from pathlib import Path\np=Path('+repr(str(out))+');print((p/"complete.json").exists() or (p/"failure.json").exists())').strip()=='True'
  if not terminal:
   state['blocked'][key]={'reason':'interrupted attempt; inspect remote process before recovery','output':str(out),'returncode':rc};save(status='needs_repair');continue
  sync(str(out)+'/',str(out)+'/',from_remote=True)
  record={'id':key,'seed':seed,'output':str(out),'reference':r['reference'],'host':'hust2','mirrored_host':'hust10','biased_selection':True}
  try:
   if not (out/'complete.json').exists():raise RuntimeError((out/'failure.json').read_text())
   metrics,completion=verify(out);record.update(metrics=metrics,accepted=qualified(metrics,r['reference']),mirror_verified=True)
   if record['accepted']:
    record['checkpoint']=str(out/'point_cloud/iteration_30000/point_cloud.ply')
    record['checkpoint_sha256']=completion['files']['point_cloud/iteration_30000/point_cloud.ply']
    state['selected'][key]=record
    (ROOT/'selected.json').write_text(json.dumps(state['selected'],indent=2))
   state['next_seed'][key]=seed+1
  except Exception as e:
   record.update(error=repr(e),accepted=False);state['blocked'][key]=record
  with (ROOT/'attempts.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
  state['attempts']+=1;save(status='running',current=None);progressed=True
 if not progressed:
  pending_methods={r['method_key'] for r in rows if r['id'] not in state['selected'] and r['id'] not in state['blocked']}
  if any(gates.get(m,{}).get('status')!='passed' for m in pending_methods):
   save(status='waiting_for_method_preflight',current=None);time.sleep(60);continue
  save(status='blocked_by_technical_errors',current=None);raise SystemExit(2)
save(status='complete_67_selected_and_mirror_verified',current=None)

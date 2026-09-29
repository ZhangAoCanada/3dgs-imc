"""Local-only concurrent campaign, one independent attempt per physical GPU 0/1."""
import fcntl,hashlib,json,math,os,shutil,subprocess,sys,time
from pathlib import Path
from selection import qualified
ROOT=Path('/data1/zhangao/outputs/GS-LSS-baseline-capped-reruns-20260927')
lock=(ROOT/'coordinator.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
(ROOT/'coordinator.pid').write_text(str(os.getpid())+'\n')
rows=json.loads((ROOT/'configurations_local.json').read_text())
state_path=ROOT/'status.json';state=json.loads(state_path.read_text())
state.update(execution_host='local',allowed_physical_gpus=[0,1],remote_execution_disabled=True,coordinator_pid=os.getpid(),current={})
active={};stop=False

def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
 return h.hexdigest()

def save(status):
 state.update(status=status,current={gpu:{k:v for k,v in job.items() if k not in ['process','log']} for gpu,job in active.items()},updated=time.time())
 p=ROOT/'status.local.tmp';p.write_text(json.dumps(state,indent=2));p.replace(state_path)

def validate(out):
 record=json.loads((out/'complete.json').read_text())
 assert record['status']=='complete' and record['iteration']==30000 and not record['smoke']
 for rel,h in record['files'].items():assert sha(out/rel)==h,rel
 metrics=json.loads((out/'results.json').read_text())['ours_30000'];split=json.loads((out/'resolved_split.json').read_text())
 camera_map=json.loads((out/'test/ours_30000/camera_map.json').read_text());names={x['file'] for x in camera_map}
 assert len(names)==len(camera_map)>0
 assert sorted(x['camera'] for x in camera_map)==sorted(split['test'])
 per=json.loads((out/'per_view.json').read_text())['ours_30000']
 assert all(set(per[m])==names and all(math.isfinite(v) for v in per[m].values()) for m in ['SSIM','PSNR','LPIPS'])
 return metrics,record

def finish(job):
 out=Path(job['output']);r=next(x for x in rows if x['id']==job['id'])
 record=dict(id=r['id'],seed=job['seed'],output=str(out),reference=r['reference'],host='local (hust10 audit alias)',hostname=os.uname().nodename,gpu=job['gpu'],biased_selection=True)
 try:
  if not (out/'complete.json').exists():raise RuntimeError((out/'failure.json').read_text() if (out/'failure.json').exists() else 'Missing terminal completion file; inspect preserved attempt')
  metrics,completion=validate(out);record.update(metrics=metrics,accepted=qualified(metrics,r['reference']),artifacts_verified=True)
  if record['accepted']:
   record.update(checkpoint=str(out/'point_cloud/iteration_30000/point_cloud.ply'),checkpoint_sha256=completion['files']['point_cloud/iteration_30000/point_cloud.ply'])
   state['selected'][r['id']]=record
   p=ROOT/'selected.tmp';p.write_text(json.dumps(state['selected'],indent=2));p.replace(ROOT/'selected.json')
  state['next_seed'][r['id']]=job['seed']+1
 except Exception as e:
  record.update(accepted=False,error=repr(e));state['blocked'][r['id']]=record
 with (ROOT/'attempts.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
 state['attempts']+=1;print(json.dumps(record),flush=True)

def idle_gpus():
 rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).splitlines()
 compute=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader,nounits'],text=True).splitlines()
 occupied={x.strip() for x in compute}
 return [x[0] for line in rows if (x:=[p.strip() for p in line.split(',')])[0] in ['0','1'] and x[1] not in occupied and int(x[2])<1000 and int(x[3])<10]

save('running_local_two_GPUs')
while True:
 for gpu,job in list(active.items()):
  if job['process'].poll() is not None:
   job['log'].close();finish(job);del active[gpu];save('running_local_two_GPUs')
 if len(state['selected'])==67 and not active:save('complete_67_selected_and_verified');break
 if (ROOT/'STOP_LOCAL_AFTER_CURRENT').exists():
  if not active:save('stopped_by_local_marker');break
  save('draining_local_jobs');time.sleep(5);continue
 gates=json.loads((ROOT/'preflight_local.json').read_text())
 pending=[r for r in rows if r['id'] not in state['selected'] and r['id'] not in state['blocked'] and r['id'] not in {j['id'] for j in active.values()}]
 pending.sort(key=lambda r:(state['next_seed'].get(r['id'],0),r['id']))
 ready=[r for r in pending if gates.get(r['method_key'],{}).get('status')=='passed']
 free=shutil.disk_usage(ROOT).free
 if free>=60*1024**3:
  for gpu in idle_gpus():
   if gpu in active or not ready or (ROOT/f'RESERVE_GPU_{gpu}').exists():continue
   r=ready.pop(0);seed=state['next_seed'].get(r['id'],0);out=ROOT/'attempts_local'/r['id']/f'seed_{seed:06d}'
   if out.exists():
    # A restart never overwrites an old attempt or launches a duplicate process.
    if (out/'complete.json').exists() or (out/'failure.json').exists():
     finish(dict(id=r['id'],seed=seed,gpu=gpu,output=str(out)));continue
    state['blocked'][r['id']]={'reason':'Existing nonterminal attempt: check surviving local processes before recovery','output':str(out)};continue
   logs=ROOT/'local-coordinator-attempt-logs';logs.mkdir(exist_ok=True);f=(logs/(r['id']+f'-seed-{seed}.log')).open('a')
   cmd=[sys.executable,str(ROOT/'scripts_local/run_one.py'),r['id'],str(seed),'--gpu',gpu]
   process=subprocess.Popen(cmd,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
   active[gpu]=dict(id=r['id'],method=r['method'],scene=r['scene'],seed=seed,gpu=gpu,output=str(out),runner_pid=process.pid,process=process,log=f)
   print('LAUNCHED',gpu,r['id'],seed,process.pid,flush=True);save('running_local_two_GPUs')
 if not active and not pending:save('blocked_by_technical_errors');break
 status='running_local_two_GPUs' if active else ('waiting_for_storage' if free<60*1024**3 else ('waiting_for_local_preflight' if not ready else 'waiting_for_local_GPUs'))
 save(status);time.sleep(5)

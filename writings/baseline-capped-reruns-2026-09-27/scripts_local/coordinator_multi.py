"""Persistent local 0/1 scheduler: up to two memory-budgeted pipelines per GPU."""
import concurrent.futures
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from gpu_policy import admission, updated_budget, is_oom, next_output
from selection import qualified

ROOT = Path('/data1/zhangao/outputs/GS-LSS-baseline-capped-reruns-20260927')


def atomic(path, value):
    tmp = path.with_name(path.name+f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value,indent=2)+'\n')
    tmp.replace(path)


def proc(pid):
    try:
        p = Path('/proc')/str(pid)
        fields = (p/'stat').read_text().rsplit(')',1)[1].split()
        if fields[0] == 'Z':
            return None
        return {'pid':int(pid),'session':int(fields[3]),'start_ticks':int(fields[19]),
                'uid':p.stat().st_uid,'cmd':(p/'cmdline').read_bytes().split(b'\0')}
    except (OSError, ValueError, IndexError):
        return None


def runner_alive(job):
    p = proc(job['runner_pid'])
    return bool(p and p['uid']==os.getuid() and p['start_ticks']==job['runner_start_ticks'])


def orphan_alive(job):
    """Do not duplicate a still-running child if its pipeline wrapper died."""
    try:
        record=json.loads((Path(job['output'])/'run.json').read_text())
        p=proc(record['active_pid'])
        return bool(p and p['uid']==os.getuid() and p['session']==job['runner_pid']
                    and any(job['output'].encode() in arg for arg in p['cmd']))
    except (OSError, KeyError, ValueError):
        return False


def gpu_snapshot(jobs):
    gpu_text=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.total,memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True,timeout=15)
    app_text=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_gpu_memory','--format=csv,noheader,nounits'],text=True,timeout=15)
    devices={}
    for line in gpu_text.splitlines():
        index,uuid,total,used,free,util=[x.strip() for x in line.split(',')]
        if index in ['0','1']:
            devices[index]={'index':index,'uuid':uuid,'total_mib':int(total),'used_mib':int(used),
                            'free_mib':int(free),'utilization':int(util),'foreign_pids':[], 'apps':[]}
    assert set(devices)=={'0','1'}, 'Both permitted GPUs must be observable'
    owners={j['runner_pid']:j for j in jobs.values() if runner_alive(j) or orphan_alive(j)}
    for j in jobs.values():
        j['current_mib']=0
    for line in app_text.splitlines():
        if not line.strip():continue
        uuid,pid,memory=[x.strip() for x in line.split(',')]
        device=next((v for v in devices.values() if v['uuid']==uuid),None)
        if device is None:continue
        p=proc(int(pid));owner=owners.get(p['session']) if p and p['uid']==os.getuid() else None
        used=int(memory)  # Query failures / N/A fail closed, rather than guessing.
        device['apps'].append({'pid':int(pid),'memory_mib':used,'campaign_id':owner['id'] if owner else None,
                               'uid':p['uid'] if p else None,'start_ticks':p['start_ticks'] if p else None})
        if owner and owner['gpu']==device['index']:
            owner['current_mib']+=used
        else:
            device['foreign_pids'].append(int(pid))
    for j in jobs.values():
        j['peak_mib']=max(j.get('peak_mib',0),j['current_mib'])
    return devices


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def validate(job,config):
    out=Path(job['output'])
    record={'id':job['id'],'seed':job['seed'],'output':str(out),'reference':config['reference'],
            'host':'local (hust10 audit alias)','hostname':os.uname().nodename,'gpu':job['gpu'],
            'biased_selection':True,'observed_peak_gpu_mib':job.get('peak_mib',0),
            'scheduling_budget_mib':job['budget_mib'],'scheduler':'memory-aware two-slot local'}
    try:
        complete=json.loads((out/'complete.json').read_text())
        assert complete['status']=='complete' and complete['iteration']==30000 and not complete['smoke']
        for rel,h in complete['files'].items():assert sha(out/rel)==h, rel
        metrics=json.loads((out/'results.json').read_text())['ours_30000']
        split=json.loads((out/'resolved_split.json').read_text())
        cameras=json.loads((out/'test/ours_30000/camera_map.json').read_text())
        names={c['file'] for c in cameras}
        assert len(names)==len(cameras)>0
        assert sorted(c['camera'] for c in cameras)==sorted(split['test'])
        per=json.loads((out/'per_view.json').read_text())['ours_30000']
        assert all(set(per[m])==names and all(math.isfinite(v) for v in per[m].values())
                   and math.isfinite(metrics[m]) for m in ['SSIM','PSNR','LPIPS'])
        record.update(metrics=metrics,accepted=qualified(metrics,config['reference']),artifacts_verified=True)
        if record['accepted']:
            record.update(checkpoint=str(out/'point_cloud/iteration_30000/point_cloud.ply'),
                          checkpoint_sha256=complete['files']['point_cloud/iteration_30000/point_cloud.ply'])
    except Exception as exc:
        detail=repr(exc)
        for name in ['failure.json','train.log','render.log','metrics.log']:
            p=out/name
            if p.exists():
                with p.open('rb') as f:
                    f.seek(max(0,p.stat().st_size-16000));detail+='\n'+f.read().decode(errors='replace')
        record.update(accepted=False,error=repr(exc),failure_kind='cuda_oom' if is_oom(detail) else 'technical_error')
    return record


def main():
    lock=(ROOT/'coordinator.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    (ROOT/'coordinator.pid').write_text(str(os.getpid())+'\n')
    state=json.loads((ROOT/'status.json').read_text())
    configs={r['id']:r for r in json.loads((ROOT/'configurations_local.json').read_text())}
    profiles=json.loads((ROOT/'gpu_memory_profiles.json').read_text())
    jobs={};processes={};futures={};seen=set();pool=concurrent.futures.ThreadPoolExecutor(max_workers=2)
    ledger=[json.loads(line) for line in (ROOT/'attempts.jsonl').read_text().splitlines() if line.strip()]
    state.update(schema_version=2,coordinator_pid=os.getpid(),execution_host='local',allowed_physical_gpus=[0,1],
                 remote_execution_disabled=True,max_pipelines_per_gpu=2,guard_mib=4096)
    state.setdefault('resource_retries',{})

    def event(kind,**fields):
        print(json.dumps(dict(event=kind,time=time.time(),**fields)),flush=True)

    def apply_record(record):
        case=record['id']
        if record.get('metrics'):
            state['blocked'].pop(case,None)
            state['resource_retries'].pop(case,None)
            state['next_seed'][case]=max(state['next_seed'].get(case,0),record['seed']+1)
            if record.get('accepted'):
                state['selected'][case]=record;state['blocked'].pop(case,None)
        elif record.get('failure_kind')=='cuda_oom':
            state['blocked'].pop(case,None)
            profiles[case]['exclusive']=True
            state['resource_retries'][case]={'seed':record['seed'],'reason':'CUDA OOM; retry same recipe and seed alone in a fresh directory'}
        elif record.get('error'):
            state['blocked'][case]=record

    # An fsynced ledger entry is authoritative if a controller exited before
    # updating status/selected. Replaying it is idempotent per output path.
    for record in ledger:
        if record.get('excluded_from_completed_attempt_count'):continue
        seen.add(record['output']);apply_record(record)
    state['attempts']=len([r for r in ledger if not r.get('excluded_from_completed_attempt_count')])

    def save(status,devices=None):
        state.update(status=status,updated=time.time(),active_jobs=jobs)
        state['current']={}
        for gpu in ['0','1']:
            matches=[j for j in jobs.values() if j['gpu']==gpu]
            if matches:state['current'][gpu]=dict(matches[0],job_ids=[j['id'] for j in matches],pipeline_count=len(matches))
        if devices is not None:state['gpu_monitor']=devices
        state['free_storage_bytes']=shutil.disk_usage(ROOT).free
        atomic(ROOT/'status.json',state)

    old_jobs=state.get('active_jobs') or {j['id']:j for j in state.get('current',{}).values()}
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():continue
        info=proc(int(p.name))
        if not info or info['uid']!=os.getuid():continue
        args=[x.decode(errors='replace') for x in info['cmd'] if x]
        script=str(ROOT/'scripts_local/run_one.py')
        if script not in args:continue
        pos=args.index(script)
        case,seed=args[pos+1],int(args[pos+2])
        if case not in configs:continue
        gpu=args[args.index('--gpu')+1]
        output=args[args.index('--attempt-dir')+1] if '--attempt-dir' in args else str(ROOT/'attempts_local'/case/f'seed_{seed:06d}')
        assert case not in jobs, f'Duplicate live pipelines for {case}; manual inspection required'
        previous=old_jobs.get(case,{})
        jobs[case]=dict(id=case,seed=seed,gpu=gpu,output=output,runner_pid=info['pid'],runner_start_ticks=info['start_ticks'],
                        method=configs[case]['method'],scene=configs[case]['scene'],budget_mib=profiles[case]['budget_mib'],
                        exclusive=profiles[case].get('exclusive',False),peak_mib=previous.get('peak_mib',0),
                        current_mib=0,phase='running',adopted=True,started=previous.get('started',time.time()))
        event('adopted_live_pipeline',id=case,pid=info['pid'],gpu=gpu,output=output)
    for case,job in old_jobs.items():
        if case not in jobs and job['output'] not in seen:
            job=dict(job);job.setdefault('budget_mib',profiles[case]['budget_mib']);job.setdefault('runner_start_ticks',-1)
            jobs[case]=job
    # Recover the crash window between finalizing a bundle and ledger insertion.
    # This also admits explicitly prepared, separately documented validation repairs.
    for output in sorted((ROOT/'attempts_local').glob('baseline_*/seed_*')):
        case=output.parent.name
        if case not in configs or case in jobs or str(output) in seen or not (output/'complete.json').exists():continue
        complete=json.loads((output/'complete.json').read_text())
        jobs[case]=dict(id=case,seed=complete['seed'],gpu=str(complete['gpu']),output=str(output),
                       runner_pid=-1,runner_start_ticks=-1,method=configs[case]['method'],scene=configs[case]['scene'],
                       budget_mib=profiles[case]['budget_mib'],exclusive=profiles[case].get('exclusive',False),
                       current_mib=0,peak_mib=0,phase='validating',validation_only=True)
    atomic(ROOT/'selected.json',state['selected'])
    event('scheduler_started',pid=os.getpid(),adopted=list(jobs),selected=len(state['selected']))
    save('running_local_memory_aware')

    while True:
        for case,job in list(jobs.items()):
            if case in futures:
                if not futures[case].done():continue
                record=futures.pop(case).result()
                if record['output'] not in seen:
                    with (ROOT/'attempts.jsonl').open('a') as f:
                        f.write(json.dumps(record)+'\n');f.flush();os.fsync(f.fileno())
                    seen.add(record['output']);state['attempts']+=1;apply_record(record)
                    profiles[case]['budget_mib']=updated_budget(profiles[case],job.get('peak_mib',0))
                    profiles[case]['observed_pipeline_peak_mib']=max(profiles[case].get('observed_pipeline_peak_mib',0),job.get('peak_mib',0))
                    profiles[case]['last_observed_output']=job['output']
                    atomic(ROOT/'gpu_memory_profiles.json',profiles);atomic(ROOT/'selected.json',state['selected'])
                    event('attempt_finished',**record)
                if case in processes:processes.pop(case).wait()
                del jobs[case];save('running_local_memory_aware')
            elif not runner_alive(job):
                if orphan_alive(job):
                    job['phase']='orphan_worker_running';continue
                job['phase']='validating';futures[case]=pool.submit(validate,dict(job),configs[case])
        if len(state['selected'])==67 and not jobs:
            save('complete_67_selected_and_verified');break
        if (ROOT/'STOP_LOCAL_AFTER_CURRENT').exists():
            save('draining_local_jobs')
            if not jobs:save('stopped_by_local_marker');break
            time.sleep(5);continue
        pending=[r for r in configs.values() if r['id'] not in state['selected'] and r['id'] not in state['blocked'] and r['id'] not in jobs]
        pending.sort(key=lambda r:(state['next_seed'].get(r['id'],0),r['id']))
        gates=json.loads((ROOT/'preflight_local.json').read_text())
        pending=[r for r in pending if gates.get(r['method_key'],{}).get('status')=='passed']
        try:devices=gpu_snapshot(jobs)
        except Exception as exc:
            state['gpu_query_error']=repr(exc);save('waiting_for_gpu_query');time.sleep(5);continue
        state.pop('gpu_query_error',None)
        # Live, process-specific opt-in; missing/invalid policy retains the
        # default exclusion of unrelated compute jobs.
        sharing_path=ROOT/'gpu_sharing_overrides.json'
        try:
            sharing=json.loads(sharing_path.read_text()) if sharing_path.exists() else {}
            if not isinstance(sharing,dict) or any(not isinstance(v,dict) for v in sharing.values()):
                raise ValueError('GPU sharing overrides must map GPU indices to objects')
            state.pop('gpu_sharing_error',None)
        except (OSError,ValueError) as exc:
            sharing={};state['gpu_sharing_error']=repr(exc)
        state['gpu_sharing_overrides']=sharing
        free=shutil.disk_usage(ROOT).free
        if free>=60*2**30:
            for gpu,device in devices.items():
                active=[j for j in jobs.values() if j['gpu']==gpu and not j.get('validation_only')]
                for config in list(pending):
                    if len(active)>=2:break
                    case=config['id'];profile=profiles[case]
                    allowed,reason=admission(device,active,profile,(ROOT/f'RESERVE_GPU_{gpu}').exists(),sharing.get(gpu))
                    if not allowed:continue
                    seed=state['next_seed'].get(case,0);output=next_output(ROOT,case,seed)
                    # Unknown interrupted directories are not silently retrained.
                    base=ROOT/'attempts_local'/case/f'seed_{seed:06d}'
                    if base.exists() and not (base/'complete.json').exists() and not (base/'failure.json').exists():
                        state['blocked'][case]={'reason':'Existing nonterminal attempt requires recovery','output':str(base)};continue
                    logs=ROOT/'local-coordinator-attempt-logs';logs.mkdir(exist_ok=True)
                    command=[sys.executable,str(ROOT/'scripts_local/run_one.py'),case,str(seed),'--gpu',gpu,'--attempt-dir',str(output)]
                    with (logs/f'{case}-{output.name}.log').open('a') as log:
                        process=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    info=proc(process.pid);assert info, 'New runner vanished before registration'
                    job=dict(id=case,seed=seed,gpu=gpu,output=str(output),runner_pid=process.pid,runner_start_ticks=info['start_ticks'],
                             method=config['method'],scene=config['scene'],budget_mib=profile['budget_mib'],exclusive=profile.get('exclusive',False),
                             peak_mib=0,current_mib=0,phase='running',started=time.time(),adopted=False)
                    jobs[case]=job;processes[case]=process;active.append(job);pending.remove(config)
                    event('launched',id=case,seed=seed,gpu=gpu,pid=process.pid,budget_mib=profile['budget_mib'],reason=reason,output=str(output))
                    save('running_local_memory_aware',devices)
        status='running_local_memory_aware' if jobs else ('waiting_for_storage' if free<60*2**30 else ('waiting_for_local_GPUs' if pending else 'blocked_by_technical_errors'))
        save(status,devices);time.sleep(5)
    pool.shutdown(wait=True)


if __name__=='__main__':
    main()

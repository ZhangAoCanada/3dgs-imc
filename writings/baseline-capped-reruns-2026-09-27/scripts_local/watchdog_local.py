"""Supervise the GPU-aware scheduler; never signal training or other users."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path('/data1/zhangao/outputs/GS-LSS-baseline-capped-reruns-20260927')


def main():
    lock=(ROOT/'watchdog.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    (ROOT/'watchdog.pid').write_text(str(os.getpid())+'\n')
    restarts=0
    while True:
        state=json.loads((ROOT/'status.json').read_text())
        if state['status']=='complete_67_selected_and_verified' or (ROOT/'STOP_LOCAL_AFTER_CURRENT').exists() and not state.get('current'):
            break
        with (ROOT/'coordinator-local.log').open('a') as log:
            child=subprocess.Popen([sys.executable,'-u',str(ROOT/'scripts_local/coordinator_multi.py')],
                                   stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        print(json.dumps({'event':'scheduler_started','pid':child.pid,'time':time.time(),'restart_number':restarts}),flush=True)
        while child.poll() is None:
            status={'watchdog_pid':os.getpid(),'coordinator_pid':child.pid,'updated':time.time(),
                    'restarts':restarts,'gpu_poll_seconds':5,'supervisor_poll_seconds':10,
                    'status':'supervising_memory_aware_scheduler'}
            tmp=ROOT/'watchdog_status.tmp';tmp.write_text(json.dumps(status,indent=2));tmp.replace(ROOT/'watchdog_status.json')
            time.sleep(10)
        print(json.dumps({'event':'scheduler_exited','returncode':child.returncode,'time':time.time()}),flush=True)
        state=json.loads((ROOT/'status.json').read_text())
        if state['status'] in ['complete_67_selected_and_verified','stopped_by_local_marker']:
            break
        restarts+=1;time.sleep(10)
    (ROOT/'watchdog_status.json').write_text(json.dumps({'status':'stopped','watchdog_pid':os.getpid(),'updated':time.time()},indent=2))


if __name__=='__main__':
    main()

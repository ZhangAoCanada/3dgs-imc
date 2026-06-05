#!/usr/bin/env python3
import argparse
import os
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from queue import Queue, Empty


def now():
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def enqueue_stream(stream, queue):
    try:
        for line in iter(stream.readline, ''):
            queue.put(line)
    finally:
        try:
            stream.close()
        except Exception:
            pass


def run_text(cmd, timeout=5):
    proc = None
    try:
        proc = subprocess.Popen(
            cmd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid,
        )
        out, _ = proc.communicate(timeout=timeout)
        return (out or '').strip()
    except subprocess.TimeoutExpired:
        if proc is not None:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except Exception:
                pass
        cmd_text = ' '.join(cmd)
        return f'<timeout after {timeout}s: {cmd_text}>'
    except Exception as exc:
        return f'<failed: {exc}>'


def nvidia_smi_gpu_snapshot(gpu_index):
    query = [
        'nvidia-smi',
        f'--id={gpu_index}',
        '--query-gpu=index,memory.used,memory.total,utilization.gpu',
        '--format=csv,noheader,nounits',
    ]
    apps = [
        'nvidia-smi',
        '--query-compute-apps=gpu_uuid,pid,process_name,used_memory',
        '--format=csv,noheader,nounits',
    ]
    return run_text(query, timeout=3), run_text(apps, timeout=3)


def proc_gpu_snapshot(active_pids, gpu_index):
    if isinstance(gpu_index, str) and gpu_index.lower() in ('none', 'disabled', 'off', ''):
        return 'proc_gpu=disabled', 'proc_gpu_processes=[]', False
    rows = []
    for pid in sorted(active_pids):
        fd_dir = Path('/proc') / str(pid) / 'fd'
        seen = set()
        try:
            fds = list(fd_dir.iterdir())
        except Exception:
            continue
        for fd in fds:
            try:
                target = os.readlink(fd)
            except Exception:
                continue
            if target.startswith('/dev/nvidia'):
                seen.add(target)
        if seen:
            rows.append(f"{pid}:{','.join(sorted(seen))}")
    if rows:
        return 'proc_gpu_active=1 gpu_probe=procfs', 'proc_gpu_processes=[' + '; '.join(rows) + ']', True
    return 'proc_gpu_active=0 gpu_probe=procfs', 'proc_gpu_processes=[]', False


def process_group_pids(pgid):
    text = run_text(['ps', '-o', 'pid=', '-g', str(pgid)])
    pids = set()
    for line in text.splitlines():
        try:
            pids.add(int(line.strip().split()[0]))
        except Exception:
            pass
    return pids


def gpu_snapshot_timed_out(*snapshots):
    return any(str(snapshot).startswith('<timeout after') for snapshot in snapshots)


def parse_gpu_active(snapshot, apps_snapshot, active_pids):
    # Count only GPU processes from the monitored process group. Global GPU
    # utilization may come from another job and must not mask a stall.
    for line in apps_snapshot.splitlines():
        parts = [p.strip() for p in line.split(',')]
        if len(parts) < 2:
            continue
        try:
            if int(parts[1]) in active_pids:
                return True
        except ValueError:
            continue
    return False


def cpu_snapshot(pgid, active_threshold):
    cmd = [
        'ps',
        '-o', 'pid=,ppid=,pgid=,stat=,etimes=,pcpu=,pmem=,comm=',
        '-g', str(pgid),
    ]
    text = run_text(cmd)
    rows = []
    pids = set()
    total_cpu = 0.0
    total_non_d_cpu = 0.0
    total_mem = 0.0
    runnable = False
    for line in text.splitlines():
        parts = line.split(None, 7)
        if len(parts) < 8:
            continue
        pid, ppid, row_pgid, stat, etimes, pcpu, pmem, comm = parts
        try:
            pids.add(int(pid))
        except ValueError:
            pass
        try:
            cpu = float(pcpu)
            mem = float(pmem)
        except ValueError:
            cpu = 0.0
            mem = 0.0
        total_cpu += cpu
        if not stat.startswith('D'):
            total_non_d_cpu += cpu
        total_mem += mem
        if stat.startswith('R'):
            runnable = True
        rows.append(f'{pid}:{stat}:{cpu:.1f}%cpu:{mem:.1f}%mem:{comm}')
    active = runnable or total_non_d_cpu >= active_threshold
    summary = f'total_cpu={total_cpu:.1f}% non_d_cpu={total_non_d_cpu:.1f}% total_mem={total_mem:.1f}% active={int(active)} procs=[' + '; '.join(rows) + ']'
    if not rows:
        summary = 'total_cpu=0.0% total_mem=0.0% active=0 procs=[]'
        active = False
    return summary, active, pids


def current_uvm_d_state_processes():
    text = run_text(['ps', '-eo', 'pid=,ppid=,pgid=,stat=,comm=,args='], timeout=5)
    rows = []
    for line in text.splitlines():
        parts = line.split(None, 5)
        if len(parts) < 5:
            continue
        pid, ppid, pgid, stat, comm = parts[:5]
        args = parts[5] if len(parts) > 5 else ''
        if not stat.startswith('D'):
            continue
        try:
            wchan = Path('/proc') / pid / 'wchan'
            wait_channel = wchan.read_text().strip()
        except Exception:
            wait_channel = ''
        if wait_channel.startswith('uvm_'):
            rows.append({
                'pid': pid,
                'ppid': ppid,
                'pgid': pgid,
                'stat': stat,
                'comm': comm,
                'wchan': wait_channel,
                'args': args[:180],
            })
    return rows


def format_uvm_rows(rows):
    return '; '.join(
        f"pid={row['pid']} stat={row['stat']} wchan={row['wchan']} comm={row['comm']} args={row['args']}"
        for row in rows
    )


def record_simple_incident(path, label, reason, cmd):
    if path is None:
        return
    incident_path = Path(path)
    incident_path.parent.mkdir(parents=True, exist_ok=True)
    cmd_text = ' '.join(cmd)
    with incident_path.open('a', buffering=1) as incident:
        incident.write(f'[{now()}] INCIDENT {label}: {reason}; cmd={cmd_text}\n')


def terminate_group(proc, log):
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        log.write(f'[{now()}] sent SIGTERM to process group {proc.pid}\n')
        log.flush()
    except ProcessLookupError:
        return
    except Exception as exc:
        log.write(f'[{now()}] failed to SIGTERM process group: {exc}\n')
        log.flush()
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
            log.write(f'[{now()}] sent SIGKILL to process group {proc.pid}\n')
            log.flush()
        except Exception as exc:
            log.write(f'[{now()}] failed to SIGKILL process group: {exc}\n')
            log.flush()


def record_incident(path, label, reason, cmd, elapsed, idle_output, pid, gpu_line, apps_line, cpu_line):
    if path is None:
        return
    incident_path = Path(path)
    incident_path.parent.mkdir(parents=True, exist_ok=True)
    cmd_text = ' '.join(cmd)
    line = (
        f'[{now()}] INCIDENT {label}: {reason}; pid={pid}; elapsed={elapsed:.1f}s; '
        f'idle_output={idle_output:.1f}s; gpu={gpu_line}; gpu_apps={apps_line}; cpu={cpu_line}; cmd={cmd_text}\n'
    )
    with incident_path.open('a', buffering=1) as incident:
        incident.write(line)


def main():
    parser = argparse.ArgumentParser(description='Run an experiment with output, CPU, and GPU heartbeat monitoring.')
    parser.add_argument('--log', required=True, help='Combined stdout/stderr log file.')
    parser.add_argument('--gpu', default='1', help='Physical GPU index to monitor.')
    parser.add_argument('--gpu-probe-mode', choices=('proc', 'nvidia-smi', 'off'), default='proc', help='Use procfs GPU fd checks by default; nvidia-smi is optional because it can wedge when NVIDIA UVM is unhealthy.')
    parser.add_argument('--interval', type=float, default=30.0, help='Heartbeat interval in seconds.')
    parser.add_argument('--warmup', type=float, default=300.0, help='Seconds before stall checks are enforced.')
    parser.add_argument('--stall-timeout', type=float, default=900.0, help='Kill if no output, no CPU, and no GPU activity after this many seconds. Use 0 to disable.')
    parser.add_argument('--output-stall-timeout', type=float, default=0.0, help='Kill if no output after this many seconds, even when CPU/GPU appears active. Use 0 to disable.')
    parser.add_argument('--max-runtime', type=float, default=0.0, help='Kill after this many wall-clock seconds regardless of activity. Use 0 to disable.')
    parser.add_argument('--require-gpu-after', type=float, default=0.0, help='Kill if no monitored process-group GPU activity is observed by this many seconds. Use 0 to disable.')
    parser.add_argument('--cpu-active-pct', type=float, default=1.0, help='Process-group CPU percent considered active.')
    parser.add_argument('--incident-log', default='outputs/monitor/incidents.log', help='Append stall incidents here. Use \"none\" to disable.')
    parser.add_argument('--label', default='experiment_stall', help='Incident label used when this monitor terminates a command.')
    parser.add_argument('--skip-uvm-preflight', action='store_true', help='Skip the NVIDIA UVM D-state preflight check. Use only for non-CUDA commands/tests.')
    parser.add_argument('cmd', nargs=argparse.REMAINDER, help='Command to run after --')
    args = parser.parse_args()

    cmd = args.cmd
    if cmd and cmd[0] == '--':
        cmd = cmd[1:]
    if not cmd:
        parser.error('missing command after --')
    if isinstance(args.incident_log, str) and args.incident_log.lower() in ('none', 'disabled', 'off', ''):
        args.incident_log = None

    log_path = Path(args.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    heartbeat_path = log_path.with_suffix(log_path.suffix + '.heartbeat')

    with log_path.open('a', buffering=1) as log, heartbeat_path.open('a', buffering=1) as hb:

        if not args.skip_uvm_preflight:
            uvm_rows = current_uvm_d_state_processes()
            if uvm_rows:
                reason = 'NVIDIA UVM D-state processes already present; refusing to launch experiment: ' + format_uvm_rows(uvm_rows)
                msg = f'[{now()}] preflight failed: {reason}\n'
                log.write(msg)
                hb.write(msg)
                print(msg, end='', flush=True)
                record_simple_incident(args.incident_log, args.label + '_preflight', reason, cmd)
                return 126

        header = f'[{now()}] starting command: ' + ' '.join(cmd) + '\n'
        log.write(header)
        hb.write(header)
        print(header, end='', flush=True)

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            preexec_fn=os.setsid,
        )
        queue = Queue()
        thread = threading.Thread(target=enqueue_stream, args=(proc.stdout, queue), daemon=True)
        thread.start()

        start = time.time()
        last_output = time.time()
        last_heartbeat = 0.0
        returncode = None
        gpu_probe_enabled = args.gpu_probe_mode != 'off' and (args.gpu.lower() not in ('none', 'disabled', 'off') if isinstance(args.gpu, str) else True)
        gpu_seen = False

        while True:
            try:
                line = queue.get(timeout=1.0)
                last_output = time.time()
                log.write(line)
                print(line, end='', flush=True)
            except Empty:
                pass

            returncode = proc.poll()
            t = time.time()
            if t - last_heartbeat >= args.interval:
                cpu_line, cpu_active, group_pids = cpu_snapshot(proc.pid, args.cpu_active_pct)
                if gpu_probe_enabled and args.gpu_probe_mode == 'proc':
                    gpu_line, apps_line, gpu_active = proc_gpu_snapshot(group_pids, args.gpu)
                    gpu_seen = gpu_seen or gpu_active
                elif gpu_probe_enabled and args.gpu_probe_mode == 'nvidia-smi':
                    gpu_line, apps_line = nvidia_smi_gpu_snapshot(args.gpu)
                    if gpu_snapshot_timed_out(gpu_line, apps_line):
                        gpu_probe_enabled = False
                        apps_line = apps_line + ' | gpu probes disabled after timeout for this run'
                    gpu_active = parse_gpu_active(gpu_line, apps_line, group_pids)
                    gpu_seen = gpu_seen or gpu_active
                else:
                    gpu_line = 'gpu probes disabled'
                    apps_line = 'gpu probes disabled'
                    gpu_active = False
                idle_for = t - last_output
                msg = (
                    f'[{now()}] heartbeat pid={proc.pid} rc={returncode} '
                    f'idle_output={idle_for:.1f}s gpu_active={int(gpu_active)} gpu={gpu_line} '
                    f'gpu_apps={apps_line} cpu={cpu_line}\n'
                )
                hb.write(msg)
                log.write(msg)
                print(msg, end='', flush=True)
                last_heartbeat = t

                if returncode is None and args.max_runtime > 0 and t - start >= args.max_runtime:
                    msg = f'[{now()}] max runtime exceeded: elapsed={t - start:.1f}s limit={args.max_runtime:.1f}s; terminating.\n'
                    hb.write(msg)
                    log.write(msg)
                    print(msg, end='', flush=True)
                    record_incident(args.incident_log, args.label + '_max_runtime', msg.strip(), cmd, t - start, idle_for, proc.pid, gpu_line, apps_line, cpu_line)
                    terminate_group(proc, log)
                    returncode = 124
                    break

                if returncode is None and args.output_stall_timeout > 0 and t - start >= args.warmup:
                    if idle_for >= args.output_stall_timeout:
                        msg = f'[{now()}] output stall detected: no output for {idle_for:.1f}s with limit={args.output_stall_timeout:.1f}s; terminating even though CPU/GPU may appear active.\n'
                        hb.write(msg)
                        log.write(msg)
                        print(msg, end='', flush=True)
                        record_incident(args.incident_log, args.label + '_output_stall', msg.strip(), cmd, t - start, idle_for, proc.pid, gpu_line, apps_line, cpu_line)
                        terminate_group(proc, log)
                        returncode = 125
                        break

                if returncode is None and args.require_gpu_after > 0 and t - start >= args.require_gpu_after and not gpu_seen:
                    msg = f'[{now()}] GPU launch stall detected: no monitored process-group GPU activity observed after {t - start:.1f}s; terminating.\n'
                    hb.write(msg)
                    log.write(msg)
                    print(msg, end='', flush=True)
                    record_incident(args.incident_log, args.label + '_no_gpu', msg.strip(), cmd, t - start, idle_for, proc.pid, gpu_line, apps_line, cpu_line)
                    terminate_group(proc, log)
                    returncode = 126
                    break

                if returncode is None and args.stall_timeout > 0 and t - start >= args.warmup:
                    if idle_for >= args.stall_timeout and not gpu_active and not cpu_active:
                        msg = f'[{now()}] stall detected: no output for {idle_for:.1f}s, monitored process GPU inactive, and process group CPU inactive; terminating.\n'
                        hb.write(msg)
                        log.write(msg)
                        print(msg, end='', flush=True)
                        record_incident(args.incident_log, args.label, msg.strip(), cmd, t - start, idle_for, proc.pid, gpu_line, apps_line, cpu_line)
                        terminate_group(proc, log)
                        returncode = 124
                        break

            if returncode is not None:
                while True:
                    try:
                        line = queue.get_nowait()
                        log.write(line)
                        print(line, end='', flush=True)
                    except Empty:
                        break
                break

        footer = f'[{now()}] command exited rc={returncode}\n'
        hb.write(footer)
        log.write(footer)
        print(footer, end='', flush=True)
        return int(returncode or 0)


if __name__ == '__main__':
    sys.exit(main())

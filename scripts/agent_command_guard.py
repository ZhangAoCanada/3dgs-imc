#!/usr/bin/env python3
"""Bound short agent-side commands so local edits/status checks cannot hang silently.

Use this for short repo-maintenance commands. Use monitor_experiment.py for long
training/eval jobs. The guard uses a reader thread so wall-time checks are not
blocked by commands that produce no stdout.
"""
import argparse
import os
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from queue import Empty, Queue


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


def cpu_snapshot(pgid):
    text = run_text(['ps', '-o', 'pid=,ppid=,pgid=,stat=,etimes=,pcpu=,pmem=,comm=', '-g', str(pgid)])
    rows = []
    total_cpu = 0.0
    total_mem = 0.0
    for line in text.splitlines():
        parts = line.split(None, 7)
        if len(parts) < 8:
            continue
        pid, ppid, row_pgid, stat, etimes, pcpu, pmem, comm = parts
        try:
            total_cpu += float(pcpu)
            total_mem += float(pmem)
        except ValueError:
            pass
        rows.append(f'{pid}:{stat}:{pcpu}%cpu:{pmem}%mem:{comm}')
    return f'total_cpu={total_cpu:.1f}% total_mem={total_mem:.1f}% procs=[' + '; '.join(rows) + ']'


def process_group_pids(pgid):
    text = run_text(['ps', '-o', 'pid=', '-g', str(pgid)])
    pids = set()
    for line in text.splitlines():
        try:
            pids.add(int(line.strip().split()[0]))
        except Exception:
            pass
    return pids


def gpu_snapshot(gpu_index, pgid):
    if gpu_index is None:
        return 'proc_gpu=disabled'
    rows = []
    for pid in sorted(process_group_pids(pgid)):
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
        return 'proc_gpu_active=1 gpu_probe=procfs proc_gpu_processes=[' + '; '.join(rows) + ']'
    return 'proc_gpu_active=0 gpu_probe=procfs proc_gpu_processes=[]'


def terminate_group(proc, log, reason):
    msg = f'[{now()}] guard terminating command: {reason}\n'
    log.write(msg)
    print(msg, end='', flush=True)
    log.flush()
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    except Exception as exc:
        log.write(f'[{now()}] failed SIGTERM: {exc}\n')
        log.flush()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
            log.write(f'[{now()}] sent SIGKILL to process group {proc.pid}\n')
            log.flush()
        except Exception as exc:
            log.write(f'[{now()}] failed SIGKILL: {exc}\n')
            log.flush()


def record_incident(path, label, reason, cmd, elapsed, idle_output, pid, gpu_line, cpu_line):
    if path is None:
        return
    incident_path = Path(path)
    incident_path.parent.mkdir(parents=True, exist_ok=True)
    cmd_text = ' '.join(cmd)
    line = (
        f'[{now()}] INCIDENT {label}: {reason}; pid={pid}; '
        f'elapsed={elapsed:.1f}s; idle_output={idle_output:.1f}s; '
        f'gpu={gpu_line}; cpu={cpu_line}; cmd={cmd_text}\n'
    )
    with incident_path.open('a', buffering=1) as incident:
        incident.write(line)


def main():
    parser = argparse.ArgumentParser(description='Run a short command with heartbeat and hard wall timeout.')
    parser.add_argument('--log', default='outputs/monitor/agent_command_guard.log')
    parser.add_argument('--max-seconds', type=float, default=120.0)
    parser.add_argument('--interval', type=float, default=15.0)
    parser.add_argument('--idle-timeout', type=float, default=60.0, help='Kill if no output for this long. Use 0 to disable.')
    parser.add_argument('--gpu', default='none', help='Physical GPU id to include in heartbeat. Use \"none\" to disable.')
    parser.add_argument('--incident-log', default='outputs/monitor/incidents.log', help='Append timeout/stall incidents here. Use \"none\" to disable.')
    parser.add_argument('--label', default='agent_command_stall', help='Incident label used when this guard terminates a command.')
    parser.add_argument('cmd', nargs=argparse.REMAINDER)
    args = parser.parse_args()

    cmd = args.cmd
    if cmd and cmd[0] == '--':
        cmd = cmd[1:]
    if not cmd:
        parser.error('missing command after --')
    if isinstance(args.gpu, str) and args.gpu.lower() in ('none', 'disabled', 'off', ''):
        args.gpu = None
    if isinstance(args.incident_log, str) and args.incident_log.lower() in ('none', 'disabled', 'off', ''):
        args.incident_log = None

    log_path = Path(args.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open('a', buffering=1) as log:
        header = f'[{now()}] guard starting max={args.max_seconds}s idle={args.idle_timeout}s cmd=' + ' '.join(cmd) + '\n'
        log.write(header)
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
        last_output = start
        last_heartbeat = start
        rc = None
        while True:
            try:
                line = queue.get(timeout=0.25)
                last_output = time.time()
                log.write(line)
                print(line, end='', flush=True)
            except Empty:
                pass

            rc = proc.poll()
            t = time.time()
            if t - last_heartbeat >= args.interval:
                msg = (
                    f'[{now()}] guard heartbeat pid={proc.pid} rc={rc} elapsed={t-start:.1f}s '
                    f'idle_output={t-last_output:.1f}s {gpu_snapshot(args.gpu, proc.pid)} cpu={cpu_snapshot(proc.pid)}\n'
                )
                log.write(msg)
                print(msg, end='', flush=True)
                last_heartbeat = t
            if rc is not None:
                while True:
                    try:
                        line = queue.get_nowait()
                        log.write(line)
                        print(line, end='', flush=True)
                    except Empty:
                        break
                break
            if args.max_seconds > 0 and t - start >= args.max_seconds:
                reason = f'max wall time {args.max_seconds}s exceeded'
                gpu_line = gpu_snapshot(args.gpu, proc.pid)
                cpu_line = cpu_snapshot(proc.pid)
                terminate_group(proc, log, reason)
                record_incident(args.incident_log, args.label, reason, cmd, t - start, t - last_output, proc.pid, gpu_line, cpu_line)
                rc = 124
                break
            if args.idle_timeout > 0 and t - last_output >= args.idle_timeout:
                reason = f'no stdout for {args.idle_timeout}s'
                gpu_line = gpu_snapshot(args.gpu, proc.pid)
                cpu_line = cpu_snapshot(proc.pid)
                terminate_group(proc, log, reason)
                record_incident(args.incident_log, args.label, reason, cmd, t - start, t - last_output, proc.pid, gpu_line, cpu_line)
                rc = 125
                break
        footer = f'[{now()}] guard command exited rc={rc}\n'
        log.write(footer)
        print(footer, end='', flush=True)
        return int(rc or 0)


if __name__ == '__main__':
    sys.exit(main())

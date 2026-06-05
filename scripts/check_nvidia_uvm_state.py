#!/usr/bin/env python3
"""Fail fast when NVIDIA UVM is already wedged in kernel D-state.

This intentionally avoids nvidia-smi. It reads ps output and /proc/<pid>/wchan,
which keeps it safe when nvidia-smi itself is blocked.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path


def run_ps():
    out = subprocess.run(
        ['ps', '-eo', 'pid=,ppid=,pgid=,stat=,comm=,args='],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=5,
        check=False,
    )
    return out.stdout


def uvm_d_state_rows():
    rows = []
    for line in run_ps().splitlines():
        parts = line.split(None, 5)
        if len(parts) < 5:
            continue
        pid, ppid, pgid, stat, comm = parts[:5]
        args = parts[5] if len(parts) > 5 else ''
        if not stat.startswith('D'):
            continue
        try:
            wchan = (Path('/proc') / pid / 'wchan').read_text().strip()
        except Exception:
            wchan = ''
        if wchan.startswith('uvm_'):
            rows.append({
                'pid': int(pid),
                'ppid': int(ppid),
                'pgid': int(pgid),
                'stat': stat,
                'comm': comm,
                'wchan': wchan,
                'args': args,
            })
    return rows


def main():
    parser = argparse.ArgumentParser(description='Check for NVIDIA UVM D-state processes without calling nvidia-smi.')
    parser.add_argument('--json', action='store_true', help='Print machine-readable JSON.')
    parser.add_argument('--quiet', action='store_true', help='Only set the exit code.')
    args = parser.parse_args()

    rows = uvm_d_state_rows()
    if args.json:
        print(json.dumps(rows, indent=2, sort_keys=True))
    elif rows and not args.quiet:
        print('NVIDIA UVM D-state processes detected; refusing to launch CUDA work.', file=sys.stderr)
        for row in rows:
            cmd = row['args'][:220]
            print(
                f"  pid={row['pid']} stat={row['stat']} wchan={row['wchan']} comm={row['comm']} cmd={cmd}",
                file=sys.stderr,
            )
        print('Wait for these D-state processes to clear or recover/restart the NVIDIA driver before launching training.', file=sys.stderr)
    elif not rows and not args.quiet:
        print('No NVIDIA UVM D-state processes detected.')

    return 126 if rows else 0


if __name__ == '__main__':
    sys.exit(main())

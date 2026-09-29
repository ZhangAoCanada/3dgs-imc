"""Gather fresh read-only source snapshots; remote filesystem contents are unchanged."""
from pathlib import Path
import gzip
import json
import subprocess
import sys

repo = Path(__file__).resolve().parents[3]
work = Path(__file__).resolve().parents[1]
scan = (repo / 'writings/research-audit-2026-09-26/scripts/scan_outputs.py').read_bytes()
mode = sys.argv[1]
if mode == 'wsl':
    base = '/mnt/c/Users/Aooooo/Documents/wsl_repos'
    roots = [base + '/outputs', base + '/3dgs-imc-codex/output',
             base + '/3dgs-imc/output', base + '/MH-3DGS/output',
             '/home/ooooo/outputs']
elif mode == 'hust2':
    roots = ['/data1/zhangao/outputs']
else:
    raise SystemExit('host must be wsl or hust2')
for i, root in enumerate(roots):
    output = work / 'evidence' / f'fresh_{mode}_{i}.json.gz'
    assert not output.exists(), output
    command = ['ssh', '-o', 'BatchMode=yes', '-o', 'ClearAllForwardings=yes',
               '-o', 'ConnectTimeout=15', mode, 'python3', '-', root, mode]
    proc = subprocess.run(command, input=scan, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, check=True)
    snapshot = json.loads(proc.stdout)
    output.write_bytes(gzip.compress(proc.stdout))
    print(json.dumps(dict(host=mode, root=root, files=len(snapshot['files']),
                         errors=snapshot['errors'], output=str(output))), flush=True)

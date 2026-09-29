"""Select paper sources and existing revision candidates; make a non-destructive copy plan."""
import collections
import csv
import gzip
import json
from pathlib import Path
import re
import shlex
import subprocess

WORK = Path(__file__).resolve().parents[1]
REPO = WORK.parent.parent
AUDIT = REPO / 'writings/research-audit-2026-09-26'
LIB = Path('/data1/zhangao/outputs/GS-LSS-paper-checkpoints')
selections = {}
protected = collections.defaultdict(set)

def add(host, run, checkpoint, reason):
    if host not in ('hust10', 'hust2', 'wsl') or not run.startswith('/'):
        return
    key = (host, run)
    r = selections.setdefault(key, dict(host=host, run_dir=run, iterations=set(), reasons=set()))
    r['reasons'].add(reason)
    if checkpoint.startswith('/'):
        protected[(host, checkpoint)].add(reason)
        m = re.search(r'/iteration_(\d+)/', checkpoint)
        if m:
            r['iterations'].add(int(m[1]))

for name in ['main_table_checkpoint_manifest.csv', 'special_table_sources.csv',
             'timing_provenance.csv', 'reusable_experiments.csv',
             'unresolved_checkpoint_candidates.csv', 'ours_tables_I_II_preserve.csv']:
    for r in csv.DictReader((AUDIT/name).open()):
        cp = r.get('checkpoint', '')
        run = r.get('run_dir', '') or cp.split('/point_cloud/')[0]
        reason = name + ':' + ':'.join(r.get(k, '') for k in ('table','row','method','scene','angle','backs_metric'))
        add(r.get('host',''), run, cp, reason)

# WSL contains legacy models and incomplete ablation attempts. Preserve these
# as candidates, without asserting that they reproduce any submitted row.
wsl_files = []
for p in sorted((WORK/'evidence').glob('fresh_wsl_*.json.gz')):
    wsl_files += json.loads(gzip.decompress(p.read_bytes()))['files']
for f in wsl_files:
    path = f['path']
    if '/point_cloud/iteration_' in path and path.endswith('/point_cloud.ply'):
        add('wsl', path.split('/point_cloud/')[0], path, 'WSL recovered legacy/diagnostic model; paper linkage unverified')
    elif path.startswith('/home/ooooo/outputs/') and path.endswith('/cfg_args'):
        add('wsl', str(Path(path).parent), '', 'WSL incomplete historical experiment metadata')

def csv_out(name, rows):
    rows = list(rows)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with (WORK/name).open('w') as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)

runs = []
for (host,run), r in sorted(selections.items()):
    r = dict(r, iterations=sorted(r['iterations']), reasons=sorted(r['reasons']))
    r['local_run'] = run if host == 'hust10' and run.startswith('/data1/zhangao/outputs/') else str(LIB/'sources'/host/run.lstrip('/'))
    r['action'] = 'keep_in_place' if r['local_run'] == run else 'copy'
    runs.append(r)
(WORK/'selected_runs.json').write_text(json.dumps(runs, indent=2)+'\n')
csv_out('protected_checkpoints.csv', [dict(host=h,path=p,reasons='; '.join(sorted(why))) for (h,p),why in sorted(protected.items())])

# Enumerate actual root provenance files and selected iteration directories.
# Do not enumerate render trees or copy initial optimizer snapshots unrelated
# to the selected iteration. Each checkpoint directory's neural sidecars travel
# with the PLY. Remote sources are read-only.
list_code = r'''import json,sys,os,re
from pathlib import Path
answer=[]
for run in json.load(sys.stdin):
 root=Path(run['run_dir']);items=[];errors=[]
 if not root.is_dir():
  answer.append(dict(run=run,files=[],errors=['run directory missing']));continue
 for p in root.iterdir():
  if not p.is_file():continue
  m=re.match(r'chkpnt(\d+)',p.name)
  if m and int(m[1]) not in run['iterations']:continue
  s=p.stat();items.append(dict(path=str(p),bytes=s.st_size,mtime_ns=s.st_mtime_ns))
 for it in run['iterations']:
  d=root/'point_cloud'/('iteration_'+str(it))
  if not d.is_dir():errors.append('missing iteration '+str(it));continue
  for current,dirs,files in os.walk(d):
   for n in files:
    p=Path(current)/n;s=p.stat();items.append(dict(path=str(p),bytes=s.st_size,mtime_ns=s.st_mtime_ns))
 answer.append(dict(run=run,files=items,errors=errors))
print(json.dumps(answer))'''
all_records = []
for host in ('hust10','hust2','wsl'):
    host_runs = [r for r in runs if r['host'] == host]
    command = ['python3','-c',list_code] if host == 'hust10' else ['ssh','-o','BatchMode=yes','-o','ClearAllForwardings=yes',host,'python3 -c '+shlex.quote(list_code)]
    proc = subprocess.run(command,input=json.dumps(host_runs).encode(),stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=True)
    records = json.loads(proc.stdout); all_records += records
    (WORK/'evidence'/f'selected_files_{host}.json').write_text(json.dumps(records,indent=2)+'\n')
    fs = {f['path']:f for x in records if x['run']['action']=='copy' for f in x['files']}
    (WORK/f'copy_files_{host}.txt').write_text(''.join(p.lstrip('/')+'\n' for p in sorted(fs)))
    print(json.dumps(dict(host=host,runs=len(host_runs),copy_files=len(fs),copy_GiB=round(sum(f['bytes'] for f in fs.values())/2**30,3),errors=[dict(run=x['run']['run_dir'],errors=x['errors']) for x in records if x['errors']])),flush=True)
csv_out('selected_runs.csv', [dict(r,iterations=';'.join(map(str,r['iterations'])),reasons='; '.join(r['reasons'])) for r in runs])
csv_out('copy_file_manifest.csv', [dict(host=x['run']['host'],source=f['path'],destination=str(Path(x['run']['local_run'])/Path(f['path']).relative_to(x['run']['run_dir'])),bytes=f['bytes'],mtime_ns=f['mtime_ns'],action=x['run']['action']) for x in all_records for f in x['files']])

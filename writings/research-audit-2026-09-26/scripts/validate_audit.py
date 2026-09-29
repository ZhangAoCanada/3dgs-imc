"""Validate internal audit arithmetic, exact-iteration linkage, and local report links."""
import ast
import csv
import json
import hashlib
from pathlib import Path
import re
import statistics
from snapshot_io import load_json

B=Path(__file__).resolve().parent.parent
def read(name):return list(csv.DictReader((B/name).open()))
man=read('main_table_checkpoint_manifest.csv');agg=read('main_table_verification.csv')
ck={(r['host'],r['path']):r for r in read('checkpoint_inventory.csv')}
assert len(man)==1584 and len(agg)==360 and len(read('submitted_main_tables.csv'))==120
for a in agg:
    angle=lambda r:float(r['angle']) if r['angle'] else None
    rs=[r for r in man if all(r[k]==a[k] for k in ['table','method','dataset']) and angle(r)==angle(a) and r['backs_metric']==a['metric']]
    assert len(rs)==int(a['scenes_required']),(a, len(rs))
    values=[float(r[a['metric']]) for r in rs if r.get(a['metric'])]
    assert len(values)==int(a['scenes_with_metrics'])
    if a['recomputed']:
        v=statistics.mean(values)
        assert abs(v-float(a['recomputed']))<1e-10
        assert (round(v,3)==float(a['paper_value']))==(a['rounds_to_paper']=='True')
    assert sum(r['checkpoint_check']=='size_matches_header' for r in rs)==int(a['scenes_backed'])
for r in man:
    if r['checkpoint_check']!='size_matches_header':continue
    c=ck[(r['host'],r['checkpoint'])]
    assert c['ply_check']=='size_matches_header' and str(c['iteration'])==str(r['iteration'])
    assert r['checkpoint'].endswith('/iteration_'+r['iteration']+'/point_cloud.ply')
ours=[r for r in agg if r['method']=='GS-LSS' and r['table'] in ['I','II']]
assert len(ours)==39 and all(r['rounds_to_paper']=='True' for r in ours)
preserve=read('ours_tables_I_II_preserve.csv')
assert len(preserve)==131 and sum('/wsl_results/' in r['checkpoint'] for r in preserve)==14
for r in preserve:
    p=Path(r['checkpoint']);assert r['host']=='hust10' and p.is_file()
    assert p.stat().st_size==int(ck[(r['host'],str(p))]['bytes'])
assert len(read('ours_best_psnr_checkpoints.csv'))==64
assert len(read('history_index.csv'))==47
assert len(read('claude_memory_index.csv'))==20
assert len(read('damaged_or_unverified_ply.csv'))==5
sidecars=read('scaffold_sidecar_audit.csv')
assert len(sidecars)==64 and all(r['required_networks_present']=='True' for r in sidecars)
assert all(r['ratio_rounds_to_paper']=='True' for r in read('timing_provenance.csv'))
ffr=[r for r in read('special_table_verification.csv') if r['table']=='IV']
assert len(ffr)==48 and sum(r['rounds_to_paper']=='True' for r in ffr)==46
for p in (B/'scripts').glob('*.py'):ast.parse(p.read_text(),filename=str(p))
for r in read('source_fingerprints.csv'):
    if r['host']=='hust10':assert hashlib.sha256(Path(r['path']).read_bytes()).hexdigest()==r['sha256']
for p in B.glob('*.md'):
    for target in re.findall(r'\]\((/[^)]+)\)',p.read_text()):
        path=re.sub(r':\d+$','',target)
        assert Path(path).exists(),(p,target)
print('PASS: 1,584 expected main-table source positions; 360 aggregate checks; 39 matching GS-LSS Table I/II values; 131 selected PLYs re-statted; 14 WSL files; 64 Scaffold sidecar sets; 48 FFR entries; 8 timing ratios; session counts; script syntax; all absolute Markdown file links.')

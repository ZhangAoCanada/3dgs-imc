"""Build a local, table-oriented index without changing the frozen research audit."""
import collections
import csv
import hashlib
import json
from pathlib import Path
import re
import shutil

WORK=Path(__file__).resolve().parents[1]
REPO=WORK.parent.parent
AUDIT=REPO/'writings/research-audit-2026-09-26'
LIB=Path('/data1/zhangao/outputs/GS-LSS-paper-checkpoints')
runs=json.loads((WORK/'selected_runs.json').read_text())
runmap={(r['host'],r['run_dir']):r for r in runs}
splits={(r['host'],r['run_dir']):r for r in csv.DictReader((WORK/'dataset_and_split_sources.csv').open())}
hashes={}
for p in (WORK/'evidence').glob('sha256_*_destination*.jsonl'):
    for line in p.open():
        r=json.loads(line);hashes[r['path']]=r['sha256']

def write_csv(path,rows):
    rows=list(rows);keys=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w') as f:w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)

def slug(s):return re.sub(r'[^A-Za-z0-9_.+-]+','_',str(s)).strip('_') or 'unspecified'

def remap(host,path,run=''):
    if not path:return ''
    if not run:run=path.split('/point_cloud/')[0]
    record=runmap.get((host,run))
    if not record:return ''
    try:return str(Path(record['local_run'])/Path(path).relative_to(run))
    except ValueError:return ''

ply_cache={}
def ply_check(path):
    if not path:return 'not_located'
    if path in ply_cache:return ply_cache[path]
    p=Path(path);result='missing'
    try:
        with p.open('rb') as f:b=f.read(32768)
        end=b.index(b'end_header\n')+len(b'end_header\n');h=b[:end].decode('ascii')
        types={'char':1,'uchar':1,'int8':1,'uint8':1,'short':2,'ushort':2,'int16':2,'uint16':2,'int':4,'uint':4,'int32':4,'uint32':4,'float':4,'float32':4,'double':8,'float64':8}
        n=int(re.search(r'element vertex (\d+)',h)[1]);fields=re.findall(r'^property (\w+) \w+',h,re.M)
        if h.count('element ')==1 and 'binary_' in h:
            result='size_matches_header' if n>0 and p.stat().st_size==end+n*sum(types[t] for t in fields) else 'size_mismatch'
        else:result='header_only'
    except (OSError,ValueError,KeyError,TypeError,UnicodeError):pass
    ply_cache[path]=result;return result

def localize(row):
    r=dict(row);host=r.get('host','');cp=r.get('checkpoint','');run=r.get('run_dir','') or cp.split('/point_cloud/')[0]
    record=runmap.get((host,run),{})
    r['source_host']=host;r['source_checkpoint']=cp
    r['local_run_dir']=record.get('local_run','')
    r['local_checkpoint']=remap(host,cp,run)
    r['local_checkpoint_check']=ply_check(r['local_checkpoint'])
    r['import_sha256']=hashes.get(r['local_checkpoint'],'')
    r['local_config_path']=str(Path(r['local_run_dir'])/'cfg_args') if r['local_run_dir'] else ''
    metric=r.get('metric_path','') or r.get('metric_source','')
    r['local_metric_path']=remap(host,metric,run)
    # The frozen floater audit explicitly identifies these local metric files
    # even when the associated original checkpoint's host is unresolved.
    local_floater = r.get('table')=='IV' and Path(metric).is_relative_to('/data1/zhangao/outputs/viz_multiangle')
    if not r['local_metric_path'] and metric.startswith('/') and Path(metric).is_file() and (host=='hust10' or Path(metric).is_relative_to(REPO) or local_floater):
        target=LIB/'reference/source_metrics'/metric.lstrip('/')
        target.parent.mkdir(parents=True,exist_ok=True)
        if not target.exists():shutil.copy2(metric,target)
        r['local_metric_path']=str(target)
    r['local_timing_path']=remap(host,r.get('timing_path',''),run)
    net=remap(host,r.get('net_pth',''),run)
    r['local_net_pth']=net if net and Path(net).is_file() else ''
    sp=splits.get((host,run),{})
    for key in ('local_split','split_sha256','split_status'):r[key]=sp.get(key,'')
    r['freshly_rendered']='False'
    r['sidecar_check']='available run/iteration files retained; renderer not tested' if r['local_checkpoint_check']=='size_matches_header' else 'not audited; no located valid checkpoint'
    if r.get('method')=='Scaffold-GS' and r['local_checkpoint_check']=='size_matches_header':
        folder=Path(r['local_checkpoint']).parent
        need=['opacity_mlp.pt','cov_mlp.pt','color_mlp.pt','embedding_appearance.pt']
        missing=[n for n in need if not (folder/n).is_file()]
        r['sidecar_check']='complete Scaffold network set' if not missing else 'missing '+','.join(missing)
    return r

main=[localize(r) for r in csv.DictReader((AUDIT/'main_table_checkpoint_manifest.csv').open())]
for r in main:
    if r['local_checkpoint_check']!='size_matches_header':r['library_status']='checkpoint unresolved or invalid'
    elif r.get('aggregate_rounds_to_paper')=='True':r['library_status']='local model linked to stored metrics; aggregate matches paper'
    else:r['library_status']='local candidate; stored aggregate does not establish the submitted value'
    group='gap_'+str(float(r['angle'])).removesuffix('.0') if r.get('angle') else r.get('protocol','')
    d=LIB/'tables'/r['table']/slug(r['method'])/slug(r['dataset'])/slug(r['scene'])/slug(group)/r['backs_metric']
    d.mkdir(parents=True,exist_ok=True)
    r['library_entry']=str(d)
    (d/'provenance.json').write_text(json.dumps(r,indent=2)+'\n')
    if r['local_checkpoint_check']=='size_matches_header':
        for name,target in [('checkpoint.ply',r['local_checkpoint']),('run',r['local_run_dir'])]:
            link=d/name
            if link.is_symlink():assert str(link.resolve())==str(Path(target).resolve())
            elif not link.exists():link.symlink_to(target,target_is_directory=name=='run')
            else:raise RuntimeError('Unexpected existing library entry: '+str(link))
write_csv(LIB/'indexes/main_tables.csv',main)

for name in ['special_table_sources.csv','timing_provenance.csv','reusable_experiments.csv','unresolved_checkpoint_candidates.csv','ours_best_psnr_checkpoints.csv','ours_tables_I_II_preserve.csv']:
    rows=[localize(r) for r in csv.DictReader((AUDIT/name).open())]
    write_csv(LIB/'indexes'/name,rows)
    if name in ('special_table_sources.csv','timing_provenance.csv'):
        for i,r in enumerate(rows):
            table=r.get('table','VIII' if name=='timing_provenance.csv' else 'unknown')
            d=LIB/'tables'/table/'sources'/f'{i:03d}_{slug(r.get("method") or r.get("row"))}_{slug(r.get("scene"))}'
            d.mkdir(parents=True,exist_ok=True)
            (d/'provenance.json').write_text(json.dumps(r,indent=2)+'\n')
            if r['local_checkpoint_check']=='size_matches_header' and not (d/'checkpoint.ply').exists():
                (d/'checkpoint.ply').symlink_to(r['local_checkpoint'])
    if name=='ours_tables_I_II_preserve.csv':assert len(rows)==131 and all(r['local_checkpoint_check']=='size_matches_header' for r in rows)

models=[]
for r in runs:
    sp=splits.get((r['host'],r['run_dir']),{})
    for it in r['iterations']:
        cp=str(Path(r['local_run'])/'point_cloud'/f'iteration_{it}'/'point_cloud.ply')
        models.append(dict(source_host=r['host'],source_run=r['run_dir'],local_run=r['local_run'],iteration=it,checkpoint=cp,checkpoint_check=ply_check(cp),import_sha256=hashes.get(cp,''),reasons='; '.join(r['reasons']),local_split=sp.get('local_split',''),source_path=sp.get('source_path',''),test_path=sp.get('test_path',''),images=sp.get('images',''),resolution=sp.get('resolution','')))
write_csv(LIB/'indexes/all_selected_models.csv',models)
write_csv(LIB/'indexes/unresolved_selected_models.csv',[r for r in models if r['checkpoint_check']!='size_matches_header'])

missing=[r for r in main if r['local_checkpoint_check']!='size_matches_header']
write_csv(LIB/'indexes/unresolved_main_tables.csv',missing)
missing_groups={}
for r in missing:
    key=tuple(r.get(k,'') for k in ['table','method','dataset','scene','protocol','angle'])
    group=missing_groups.setdefault(key,{k:r.get(k,'') for k in ['table','method','dataset','scene','protocol','angle']})
    group.update(original_run_path=r.get('original_run_path',''),original_host=r.get('host',''),checkpoint_status=r.get('checkpoint_check',''))
write_csv(LIB/'indexes/unresolved_main_groups.csv',missing_groups.values())
write_csv(LIB/'indexes/main_table_candidates_or_mismatches.csv',[r for r in main if r['local_checkpoint_check']=='size_matches_header' and r.get('aggregate_rounds_to_paper')!='True'])
summary=[]
for (table,method),rows in __import__('itertools').groupby(sorted(main,key=lambda r:(r['table'],r['method'])),lambda r:(r['table'],r['method'])):
    rows=list(rows)
    summary.append(dict(table=table,method=method,scene_metric_rows=len(rows),local_checkpoint_rows=sum(r['local_checkpoint_check']=='size_matches_header' for r in rows),missing_or_invalid_rows=sum(r['local_checkpoint_check']!='size_matches_header' for r in rows),rows_in_matching_paper_aggregate=sum(r.get('aggregate_rounds_to_paper')=='True' for r in rows)))
write_csv(LIB/'indexes/coverage_summary.csv',summary)

ref=LIB/'reference';ref.mkdir(exist_ok=True)
for name in ['REPORT.md','RESEARCH_HISTORY.md','main_table_verification.csv','submitted_main_tables.csv','special_table_verification.csv','revision_issue_map.csv','source_fingerprints.csv']:
    shutil.copy2(AUDIT/name,ref/name)
for name in ['GS-LSS.tex','GS-LSS.pdf']:
    shutil.copy2(REPO/'writings/GS-LSS-submission'/name,ref/name)
for name in ['cleanup_plan.csv','cleanup_actions.jsonl','copy_file_manifest.csv','copy_extra_file_manifest.csv','protected_checkpoints.csv','selected_runs.csv']:
    shutil.copy2(WORK/name,LIB/'indexes'/name)
for name in ['lookup.py','prepare_eval.py']:
    shutil.copy2(WORK/'scripts'/name,LIB/name)
for table in ['IV','V','VI','VII','VIII']:
    d=LIB/'tables'/table;d.mkdir(parents=True,exist_ok=True)
    (d/'README.md').write_text('See ../../indexes/special_table_sources.csv and ../../reference/special_table_verification.csv for source associations and unresolved values. Timing uses ../../indexes/timing_provenance.csv. A linked model is not proof that a submitted scalar matches it. Table V has no recovered source for the submitted MatrixCity component rows.\n')
counts=dict(main_rows=len(main),main_rows_with_local_checkpoint=len(main)-len(missing),unresolved_main_rows=len(missing),unresolved_scene_method_protocols=len({tuple(r.get(k,'') for k in ['table','method','scene','angle']) for r in missing}),unique_main_checkpoints=len({r['local_checkpoint'] for r in main if r['local_checkpoint_check']=='size_matches_header'}),selected_model_records=len(models),valid_selected_models=sum(r['checkpoint_check']=='size_matches_header' for r in models),invalid_or_missing_selected_models=sum(r['checkpoint_check']!='size_matches_header' for r in models),ours_tables_I_II_unique_checkpoints=131)
(WORK/'library_summary.json').write_text(json.dumps(counts,indent=2)+'\n')
(LIB/'indexes/summary.json').write_text(json.dumps(counts,indent=2)+'\n')
print(json.dumps(counts,indent=2))

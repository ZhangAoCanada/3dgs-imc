"""Build compact indexes and derived lookup files; preserve all original research files."""
import collections
import csv
import datetime
import hashlib
import json
from pathlib import Path
from snapshot_io import load_json

B=Path(__file__).resolve().parent.parent
def write(name,rows):
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with (B/name).open('w') as f:
        w=csv.DictWriter(f,fields);w.writeheader();w.writerows(rows)
def read(name):return list(csv.DictReader((B/name).open()))
extra_approvals={'019efd6a-e319-74c0-8062-766e33ccd0f0','019f23a3-1b91-7110-a44e-b13c2e7a7794','019f2794-0020-76e2-b61f-045253c9030a','019f2848-df3a-76a1-b6b4-a3166540c8b9','019f2a37-dc79-7930-9a0f-0af30f2f43ab'}
history=[]
for suffix,host in [('local','hust10'),('hust2','hust2')]:
    for t in load_json(B/f'evidence/history_{suffix}.json'):
        mirror=t['approval_mirror'] or t['id'] in extra_approvals
        msgs=t['messages']
        history.append(dict(host=host,id=t['id'],kind='approval mirror, not independent research' if mirror else 'research/review conversation',title=t['title'].replace('\n',' ')[:150],cwd=t['cwd'],rollout_path=t['rollout_path'],exists=t['exists'],bytes=t['size_bytes'],first_message=msgs[0]['timestamp'] if msgs else '',last_message=msgs[-1]['timestamp'] if msgs else '',message_count=len(msgs),final_answers=sum(m['channel']=='final_answer' for m in msgs),compactions=len(t['compactions']),event_count=sum(t['event_counts'].values())))
write('history_index.csv',history)
mem=[]
for p in sorted(Path('/home/zhangao/.claude/projects/-data1-zhangao-repos-3dgs-imc-codex/memory').glob('*.md')):
    data=p.read_bytes();mem.append(dict(path=str(p),bytes=len(data),sha256=hashlib.sha256(data).hexdigest()))
write('claude_memory_index.csv',mem)
man=read('main_table_checkpoint_manifest.csv')
ours=[r for r in man if r['method']=='GS-LSS' and r['table'] in ['I','II']]
write('ours_best_psnr_checkpoints.csv',[r for r in ours if r['backs_metric']=='PSNR'])
unique={}
for r in ours:
    key=(r['host'],r['checkpoint'])
    if key not in unique:unique[key]={k:r[k] for k in ['host','checkpoint','checkpoint_check','run_dir','dataset','scene','protocol','angle','iteration','source_path','split_path','test_path','images','aa','cap_max','init_type','seed','config_path','net_pth','metric_path','SSIM','PSNR','LPIPS']};unique[key]['used_by']=[]
    unique[key]['used_by'].append(r['table']+':'+r['backs_metric'])
for r in unique.values():r['used_by']=';'.join(r['used_by'])
write('ours_tables_I_II_preserve.csv',list(unique.values()))
write('unresolved_main_sources.csv',[r for r in man if r['checkpoint_check']!='size_matches_header' or r['aggregate_rounds_to_paper']!='True'])
alternatives=[]
runs=read('run_metrics.csv')
seen=set()
for r in man:
    key=tuple(r[k] for k in ['table','method','dataset','scene','angle'])
    if key in seen or (r['checkpoint_check']=='size_matches_header' and r['aggregate_rounds_to_paper']=='True'):continue
    seen.add(key)
    for c in runs:
        if all(c[k]==r[k] for k in ['method','dataset','scene','angle','protocol']) and c['checkpoint_check']=='size_matches_header':
            alternatives.append(dict(submitted_table=r['table'],**c,warning='existing candidate; not established as the source of the submitted row',archived_original_run=r.get('original_run_path','')))
write('unresolved_checkpoint_candidates.csv',alternatives)
ck=read('checkpoint_inventory.csv')
write('damaged_or_unverified_ply.csv',[r for r in ck if r['path'].endswith('.ply') and r['ply_check']!='size_matches_header'])
split=[];code=[]
for suffix,host in [('local','hust10'),('hust2','hust2')]:
    d=load_json(B/f'evidence/platform_{suffix}.json')
    split.extend(dict(host=host,**s) for s in d['splits'])
    code.extend(dict(host=host,**s) for s in d['sources']+d['papers'])
write('split_inventory.csv',split);write('source_fingerprints.csv',code)

# A native Scaffold model also needs its view-dependent networks.
paths={(r['host'],r['path']) for r in ck}
sidecars=[]
for r in man:
    if r['method']!='Scaffold-GS' or r['backs_metric']!='PSNR' or not r['checkpoint']:continue
    parent=str(Path(r['checkpoint']).parent)
    req=['opacity_mlp.pt','cov_mlp.pt','color_mlp.pt']
    sidecars.append(dict(table=r['table'],dataset=r['dataset'],scene=r['scene'],angle=r['angle'],host=r['host'],checkpoint=r['checkpoint'],required_networks_present=all((r['host'],parent+'/'+n) in paths for n in req),missing_networks=';'.join(n for n in req if (r['host'],parent+'/'+n) not in paths),appearance_embedding_present=(r['host'],parent+'/embedding_appearance.pt') in paths))
write('scaffold_sidecar_audit.csv',sidecars)
print('history',collections.Counter(r['kind'] for r in history),'memory files',len(mem),'bestPSNR',sum(r['backs_metric']=='PSNR' for r in ours),'preserve PLY',len(unique),'split files',len(split),'Scaffold sidecar complete',sum(r['required_networks_present'] for r in sidecars),'/',len(sidecars))

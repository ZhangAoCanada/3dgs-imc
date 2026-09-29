"""Audit submitted Tables IV-VIII without rendering or changing experiment outputs."""
import collections
import csv
import json
from pathlib import Path
import re
import statistics
from snapshot_io import load_json

B=Path(__file__).resolve().parent.parent
R=B.parent.parent
K=['SSIM','PSNR','LPIPS']
def read(name):return list(csv.DictReader((B/name).open()))
def write(name,rows):
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with (B/name).open('w') as f:
        w=csv.DictWriter(f,keys);w.writeheader();w.writerows(rows)
files=[f for s in ['local','hust2','repo'] for f in load_json(B/f'evidence/outputs_{s}.json')['files']]
lookup={(f['host'],f['path']):f for f in files}
runs=read('run_metrics.csv');ckpts=read('checkpoint_inventory.csv');arch=read('archived_metric_sources.csv')
tex=(R/'writings/GS-LSS-submission/GS-LSS.tex').read_text()
tex='\n'.join(re.split(r'(?<!\\)%',s)[0] for s in tex.splitlines())
blocks={re.search(r'\\label\{([^}]+)\}',b)[1]:b for b in re.findall(r'\\begin\{table\*?\}(.*?)\\end\{table\*?\}',tex,re.S) if re.search(r'\\label\{([^}]+)\}',b)}
def decimals(line):return [float(n) for c in line.split('&')[1:] for n in re.findall(r'(?<![A-Za-z])\d+\.\d+',c)]
verification=[];sources=[]
vm=json.loads(lookup[('hust10','/data1/zhangao/outputs/viz_multiangle/manifest.json')]['text'])
labels={'3DGS':'3DGS','Scaffold-GS':'ScaffoldGS','Hierarchical-GS':'Hier','MVGS':'MVGS','PGSR':'PGSR','3DGS-MCMC':'MCMC','MH-3DGS':'MH-3DGS','GS-LSS':'Ours'}
for line in blocks['tab:ffr'].splitlines():
    name=line.split('&')[0].strip().split('~')[0].strip()
    if name.startswith('GS-LSS'):name='GS-LSS'
    vals=decimals(line)
    if name not in labels or len(vals)!=6:continue
    p=f'/data1/zhangao/outputs/viz_multiangle/global_all/bonsai__{labels[name]}.frac.json'
    d=json.loads(lookup[('hust10',p)]['text'])
    nums=[100*d[str(a)] for a in [20,40,60,80,100]]
    for a,paper,v in zip([20,40,60,80,100,'mean'],vals,nums+[statistics.mean(nums)]):
        verification.append(dict(table='IV',row=name,setting=a,metric='FFR_percent',paper_value=paper,stored_value=v,rounds_to_paper=round(v,3)==paper,evidence='stored Bonsai fraction JSON; not freshly recomputed',source=p))
        if a=='mean':continue
        entry=vm[f'bonsai@{a}'];e=entry.get('ours') if name=='GS-LSS' else entry.get('base',{}).get(labels[name])
        if e:
            it=int(re.search(r'ours_(\d+)',e['rdir'])[1]);run=e['dir'];candidates=[c for c in ckpts if c['path']==f'{run}/point_cloud/iteration_{it}/point_cloud.ply']
            association='visualization manifest; fraction file does not embed checkpoint hash'
        else:
            candidates=[c for c in ckpts if c['method']==name and c['scene']=='bonsai' and c['angle']==str(float(a)) and c['iteration']=='30000' and c['path'].endswith('/point_cloud.ply') and '/baselines_multi_angle/' in c['path']]
            association='candidate inferred from exporter layout; original exporter config not located'
        c=sorted(candidates,key=lambda c:c['host']!='hust10')[0] if candidates else {}
        sources.append(dict(table='IV',row=name,dataset='Mip-NeRF 360',scene='bonsai',protocol='coverage_gap',angle=a,metric_source=p,host=c.get('host',''),checkpoint=c.get('path',''),checkpoint_check=c.get('ply_check','not located'),iteration=c.get('iteration',''),association=association,weighting='count of neural Gaussian centers' if name=='Scaffold-GS' else 'opacity mass',reference='full-coverage SfM input.ply in legacy exporter; not a verified train-only gap ROI',metric_script='experiments/scaffold_bev_export.py' if name=='Scaffold-GS' else 'experiments/floaters_export.py'))

# Candidate paired Train/Truck means from original CSV exports, keeping the last
# saved iteration per scene, plus the retained checkpoint-backed tnt60_table5 runs.
groups=collections.defaultdict(dict)
for r in arch:
    if not r['table'].startswith('VI/VII') or r['scene'] not in ['train_resize','truck_resize']:continue
    p=Path(r['metric_path']);key=str(p.parent.relative_to(R/'experiments/metrics_ablation'))+'/'+p.name.replace(r['scene'],'SCENE')
    old=groups[key].get(r['scene'])
    if old is None or int(r['iteration'])>=int(old['iteration']):groups[key][r['scene']]=r
for r in runs:
    if '/tnt60_table5/' in r['run_dir']:
        key='checkpoint_tnt60_table5/'+Path(r['run_dir']).name.replace(r['scene'],'SCENE')
        groups[key][r['scene']]=r
summary=[]
for key,items in sorted(groups.items()):
    if len(items)!=2:continue
    summary.append(dict(group=key,**{k:statistics.mean(float(r[k]) for r in items.values()) for k in K},scenes='train_resize;truck_resize',protocol='coverage_gap_60',checkpoint_linked=sum(r['checkpoint_check']=='size_matches_header' for r in items.values())))
write('ablation_candidate_means.csv',summary)
for table,label in [('VI','tab:ablation_CPF'),('VII','tab:detailed_metrics')]:
    trialvals=[]
    for line in blocks[label].splitlines():
        vals=decimals(line)
        if len(vals)!=3 or (table=='VII' and not line.strip().startswith('Trial ')):continue
        row=line.split('&')[0].strip()
        if table=='VII':trialvals.append(vals)
        matches=[g for g in summary if all(round(g[k],4)==v for k,v in zip(K,vals))]
        for k,paper in zip(K,vals):
            verification.append(dict(table=table,row=row,metric=k,paper_value=paper,stored_value=matches[0][k] if matches else '',rounds_to_paper=bool(matches),evidence='paired scene mean matched' if matches else 'no exact paired-scene triple located',source=';'.join(m['group'] for m in matches)))
        for g in matches:
            for sc,r in groups[g['group']].items():
                sources.append(dict(table=table,row=row,candidate_group=g['group'],dataset='Tanks & Temples',scene=sc,protocol='coverage_gap',angle=60,**{k:r.get(k,'') for k in ['SSIM','PSNR','LPIPS','host','iteration','checkpoint','checkpoint_check','run_dir','metric_path','source_path','split_path','original_run_path']},selection='mean of Train and Truck; exact triple match is arithmetic evidence, not proof of setting label'))
    if table=='VII':
        for j,k in enumerate(K):
            vals=[r[j] for r in trialvals]
            for stat,v in [('mean_of_printed_rows',statistics.mean(vals)),('sample_variance_of_printed_rows',statistics.variance(vals)),('population_variance_of_printed_rows',statistics.pvariance(vals))]:
                verification.append(dict(table='VII',row=stat,metric=k,stored_value=v,evidence='arithmetic on rounded paper text only; not checkpoint-backed',source=str(R/'writings/GS-LSS-submission/GS-LSS.tex')))

for i,line in enumerate(l for l in blocks['tab:ablation_LSS'].splitlines() if len(decimals(l))==3):
    vals=decimals(line);matches=[r for r in runs if r['dataset']=='MatrixCity' and all(round(float(r[k]),3)==v for k,v in zip(K,vals))]
    for k,v in zip(K,vals):verification.append(dict(table='V',row=['fixed_noise_no_CPF','learned_noise_only','CPF_drift_fixed_noise','full_LSS'][i],metric=k,paper_value=v,stored_value='',evidence='paper-only; no matching stored-result triple located' if not matches else 'candidate found',source=';'.join(r['metric_path'] for r in matches)))

platform={h:load_json(B/f'evidence/platform_{h}.json') for h in ['local','hust2']}
timings=[]
tdirs={'3DGS':'3dgs','Scaffold-GS':'scaffold','Hierarchical-GS':'hier','3DGS-MCMC':'mcmc','MH-3DGS':'mh','PGSR':'pgsr','MVGS':'mvgs'}
tenvs={'3DGS':'gaussian_splatting','Hierarchical-GS':'gaussian_splatting','Scaffold-GS':'scaffold','3DGS-MCMC':'mcmc','MH-3DGS':'mh','PGSR':'pgsr','MVGS':'mvgs'}
for line in blocks['tab:training-time'].splitlines():
    name=line.split('&')[0].strip();vals=decimals(line)
    if name.startswith('GS-LSS'):name='GS-LSS'
    if name not in list(tdirs)+['GS-LSS'] or len(vals)!=4:continue
    if name=='GS-LSS':
        host='hust10';run='/data1/zhangao/outputs/ours_timing_diag/tandt/truck_resize_discontinuous_clean_shipped_i30000';path=run+'/training_time.txt';t=lookup[(host,path)]['text'];seconds=float(re.search(r'[\d.]+(?=seconds)',t)[0]);scope='in-training wall timer; excludes conda/import/load and post-training render/metrics; save operations inside loop may be included';gpu='RTX 5090 GPU1';split='discontinuous.txt';env='3dgs-imc';script='tools/ours_timing_diag.sh';selection='one FAST run; not per-metric table-quality checkpoints'
    else:
        host='hust2';run='/data1/zhangao/outputs/baselines_t7_clean/'+tdirs[name];path=run+'/wall_seconds.txt';seconds=float(next(a['text'] for a in platform['hust2']['extra_artifacts'] if a['path']==path));scope='external wall clock around conda run; includes startup, loading, training and configured evaluation/save; excludes later standalone rendering/metrics';gpu='RTX 4090 GPU2';split='standard --eval; test_selection_path empty';env=tenvs[name];script='tools/hust2_baseline_retime_clean.sh';selection='one Truck timing run; table quality from Train+Truck gap60'
    c=[r for r in ckpts if r['host']==host and r['path']==run+'/point_cloud/iteration_30000/point_cloud.ply']
    timings.append(dict(table='VIII',method=name,host=host,gpu=gpu,conda_env=env,dataset='Tanks & Temples',scene='truck_resize',protocol=split,iterations=30000,seconds=seconds,ratio_to_840s=seconds/840,paper_ratio=vals[3],ratio_rounds_to_paper=round(seconds/840,2)==vals[3],timing_path=path,checkpoint=c[0]['path'] if c else '',checkpoint_check=c[0]['ply_check'] if c else 'not located',timer_scope=scope,command_source=script,selection=selection,important_config='FAST shipped, RUN_EVAL=0' if name=='GS-LSS' else 'native defaults; MVGS mv3; MCMC cap2M; Hier -d dav2; WANDB_MODE=disabled'))
write('timing_provenance.csv',timings)
write('special_table_verification.csv',verification)
write('special_table_sources.csv',sources)

# Reusable later experiments are explicitly kept separate from submitted tables.
reuse=[]
roots=['component_ablation','table4_','split_center','hyperparam_sensitivity','cpf_reliability','tnt60_table5','tables567','hust10_best_checkpoints_20260703','t7_bonsai','ours_timing_diag']
for r in runs:
    tail=r['run_dir'].split('/outputs/',1)[-1];family=tail.split('/')[0]
    if not any(family.startswith(k) for k in roots):continue
    ffr=[f for f in files if f['host']==r['host'] and f['path'].startswith(r['run_dir']+'/') and f['path'].endswith('ffr.json')]
    reuse.append(dict(experiment_family=family,**r,ffr_artifacts=';'.join(f['path'] for f in ffr),use='later diagnostic or candidate; not automatically a submitted-table source'))
write('reusable_experiments.csv',reuse)
print('special scalar comparisons',len(verification),'source rows',len(sources),'paired candidate groups',len(summary),'timing rows',len(timings),'reusable result rows',len(reuse))
for r in verification:
    if r['table'] in ['VI','VII'] and r.get('metric')=='PSNR':print(r)
for r in verification:
    if r['table']=='IV' and not r['rounds_to_paper']:print(r)

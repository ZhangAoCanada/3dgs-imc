"""Derive a host-qualified catalog and main-table provenance from read-only snapshots."""
import ast
import collections
import csv
import io
import json
from pathlib import Path
import re
from snapshot_io import load_json

BASE = Path(__file__).resolve().parent.parent
REPO = BASE.parent.parent
MIP = ['bicycle','bonsai','counter','garden','kitchen','room','stump','treehill']
TNT = ['truck_resize','train_resize']
MILL = ['building-small','residence-small','rubble-small','sci-art-small']
SCENES = MIP + TNT + MILL + ['flowers','Zeche1','Zeche2','blockA_fusion_small_aerial+somestreet']
METRICS = ['SSIM','PSNR','LPIPS']
METHODS = ['3DGS','Scaffold-GS','Hierarchical-GS','MVGS','PGSR','3DGS-MCMC','MH-3DGS','GS-LSS']
files=[]
for suffix in ['local','hust2','repo']:
    files += load_json(BASE/f'evidence/outputs_{suffix}.json')['files']
lookup={(f['host'],f['path']):f for f in files}

def dump_csv(name, rows):
    cols=list(dict.fromkeys(k for r in rows for k in r))
    with (BASE/name).open('w') as f:
        w=csv.DictWriter(f,fieldnames=cols);w.writeheader();w.writerows(rows)

def cfg_at(host, run):
    t=lookup.get((host,run+'/cfg_args'),{}).get('text','')
    if not t:return {}
    try:
        n=ast.parse(t,mode='eval').body
        return {k.arg:ast.literal_eval(k.value) for k in n.keywords}
    except Exception:return {}

def identify(path,cfg):
    scene=''
    src=cfg.get('source_path','')
    for s in sorted(SCENES,key=len,reverse=True):
        if re.search(r'(?:^|[/_])'+re.escape(s)+r'(?:[/_.]|$)',src+' '+path):scene=s;break
    split=cfg.get('test_selection_path','') or ''
    tp=cfg.get('test_path','') or ''
    search=split+' '+path
    a=re.search(r'(?:angle_|angle|gap)(\d+(?:\.\d+)?)',search)
    angle=float(a[1]) if a else None
    if 'discontinuous' in search:protocol='discontinuous';angle=None
    elif angle is not None:protocol='coverage_gap'
    elif tp or any(s in search.lower() for s in ['aerial','traintest','zeche','blocka']):protocol='aerial_ground'
    else:protocol='unclassified'
    dataset='Mip-NeRF 360' if scene in MIP+['flowers'] else 'Tanks & Temples' if scene in TNT else 'Mill 19' if scene in MILL else 'MatrixCity' if scene.startswith('blockA') else 'Zeche' if scene.startswith('Zeche') else ''
    lower=path.lower()
    method=''
    mapping=[('mh-3dgs','MH-3DGS'),('mh_3dgs','MH-3DGS'),('3dgs-mcmc','3DGS-MCMC'),('3dgs_mcmc','3DGS-MCMC'),('scaffold','Scaffold-GS'),('3dgs-depth','Hierarchical-GS'),('/hier/','Hierarchical-GS'),('mvgs','MVGS'),('pgsr','PGSR'),('cdgs','CDGS (not paper Hierarchical-GS)'),('3dgs-original','3DGS'),('/3dgs/','3DGS'),('/mcmc/','3DGS-MCMC'),('/mh/','MH-3DGS')]
    for needle,label in mapping:
        if needle in lower:method=label;break
    if '/split_center_baselines_' in lower and '/3dgs_i' in lower:method='3DGS'
    if not method and any(t in lower for t in ['develop3','/diag/','/best_','/tables567/','tnt60','mip60','/mip_ablation/','gslss','gs-lss','/ours','/fast_lss','/new_dev','/val_','/repro/','/proto/','/component_ablation','/table4_','hyperparam_sensitivity','/sweep_models/','/split_center_robustness']):method='GS-LSS'
    return dataset,scene,protocol,angle,method

ckpts=[]
for f in files:
    if f['kind']!='checkpoint':continue
    p=f['path'];run=p.split('/point_cloud/')[0] if '/point_cloud/' in p else str(Path(p).parent)
    cfg=cfg_at(f['host'],run)
    ds,sc,protocol,angle,method=identify(run,cfg)
    it=re.search(r'(?:iteration_|chkpnt)(\d+)',p)
    ckpts.append(dict(host=f['host'],path=p,method=method,dataset=ds,scene=sc,protocol=protocol,angle=angle,iteration=int(it[1]) if it else '',bytes=f['bytes'],mtime_utc=f['mtime_utc'],ply_check=f.get('ply',{}).get('check','not_deserialized'),vertices=f.get('ply',{}).get('vertices',''),run_dir=run,symlink_target=f.get('symlink_target','')))
dump_csv('checkpoint_inventory.csv',ckpts)

candidates={}
conflicts=[]
for f in sorted(files,key=lambda f:Path(f['path']).name=='results.json'):
    p=f['path'];name=Path(p).name
    if name not in ('results.json','metrics.csv') or 'text' not in f:continue
    run=str(Path(p).parent);cfg=cfg_at(f['host'],run)
    ds,sc,protocol,angle,method=identify(run,cfg)
    try:
        raw=json.loads(f['text']) if name=='results.json' else {r.get('method',''):r for r in csv.DictReader(io.StringIO(f['text']))}
        if not isinstance(raw,dict):continue
        for tag,v in raw.items():
            if not isinstance(v,dict) or not all(k in v for k in METRICS):continue
            im=re.search(r'(\d+)',tag)
            if not im:continue
            it=int(im[1]);ply=f'{run}/point_cloud/iteration_{it}/point_cloud.ply';pf=lookup.get((f['host'],ply))
            vals={k:float(v[k]) for k in METRICS}
            key=(f['host'],run,tag)
            if key in candidates:
                old=candidates[key]
                if max(abs(old[k]-vals[k]) for k in METRICS)>1e-6:conflicts.append(dict(host=f['host'],run=run,tag=tag,first=old['metric_path'],second=p,first_values={k:old[k] for k in METRICS},second_values=vals))
            aa='not recorded'
            low=run.lower()
            if 'woantialiasing' in low or 'woaa' in low or 'aa0' in low:aa='off (path inference)'
            elif '_aa' in low or 'antialiasing' in low or '/develop3_multi_angle2/' in low or '/develop3_discontinuous2/' in low:aa='on (path inference)'
            r=dict(host=f['host'],method=method,dataset=ds,scene=sc,protocol=protocol,angle=angle,iteration=it,result_key=tag,**vals,run_dir=run,metric_path=p,checkpoint=ply,checkpoint_exists=bool(pf),checkpoint_check=pf.get('ply',{}).get('check','missing') if pf else 'missing',source_path=cfg.get('source_path',''),split_path=cfg.get('test_selection_path',''),test_path=cfg.get('test_path',''),images=cfg.get('images',''),resolution=cfg.get('resolution',''),aa=aa,config_path=run+'/cfg_args' if cfg else '',net_pth=run+'/net.pth' if (f['host'],run+'/net.pth') in lookup else '',metric_script='metrics.py (historical results; no fresh rendering)',selection='single stored run/iteration')
            r.update(cap_max=cfg.get('cap_max',''),init_type=cfg.get('init_type',''),seed=cfg.get('seed','not recorded in cfg_args'))
            for k in ('BRISQUE','NIQE','PIQE'):
                if k in v:r[k]=float(v[k])
            candidates[key]=r
    except Exception:continue
rows=list(candidates.values())
dump_csv('run_metrics.csv',rows)
(BASE/'evidence/metric_source_conflicts.json').write_text(json.dumps(conflicts,indent=2))

# Preserve original metric exports, including their original absolute run paths.
# A matching filename is not sufficient: link only identical SSIM/PSNR/LPIPS triples.
archived=[]
for directory in ['metrics_multiangle','metrics_discontinuous','metrics_ablation']:
    for path in sorted((REPO/'experiments'/directory).rglob('*.csv')):
        for v in csv.DictReader(path.open()):
            if not all(v.get(k) for k in METRICS):continue
            original=v.get('scene','');tag=v.get('method','');im=re.search(r'(\d+)',tag)
            if not im:continue
            try:vals={k:float(v[k]) for k in METRICS}
            except ValueError:continue
            ds,sc,protocol,angle,method=identify(original+' '+str(path),{})
            if directory=='metrics_discontinuous':protocol='discontinuous';angle=None
            if directory=='metrics_multiangle':protocol='coverage_gap'
            family=str(path.relative_to(REPO/'experiments'/directory)).split('/')[0]
            matches=[r for r in rows if r['scene']==sc and r['iteration']==int(im[1]) and (not method or r['method']==method) and (directory=='metrics_ablation' or (r['protocol']==protocol and r['angle']==angle)) and all(abs(r[k]-vals[k])<1e-6 for k in METRICS)]
            matches.sort(key=lambda r:(r['checkpoint_check']!='size_matches_header',r['host']!='hust10',r['run_dir']))
            chosen=matches[0] if matches else None
            record=dict(table='I' if directory=='metrics_multiangle' else 'II' if directory=='metrics_discontinuous' else 'VI/VII legacy candidates',family=family,method=method,dataset=ds,scene=sc,protocol=protocol,angle=angle,iteration=int(im[1]),**vals,metric_path=str(path),original_run_path=original,host=chosen['host'] if chosen else 'unresolved original host',run_dir=chosen['run_dir'] if chosen else original,checkpoint=chosen['checkpoint'] if chosen else '',checkpoint_check=chosen['checkpoint_check'] if chosen else 'no identical stored-result checkpoint located',source_matches=len(matches),selection='archived metric export; exact triple linkage',source_path=chosen.get('source_path','') if chosen else '',split_path=chosen.get('split_path','') if chosen else '',test_path=chosen.get('test_path','') if chosen else '',images=chosen.get('images','') if chosen else '',aa=chosen.get('aa','not recorded') if chosen else 'not recorded',metric_script='historical metrics.py export; no fresh rendering')
            archived.append(record)
            if chosen:
                record.update({k:chosen.get(k,'') for k in ['cap_max','init_type','seed','config_path','net_pth','resolution']})
dump_csv('archived_metric_sources.csv',archived)

# Read numeric rows from the submitted source itself, never from old hardcoded targets.
tex=(REPO/'writings/GS-LSS-submission/GS-LSS.tex').read_text()
tex='\n'.join(re.split(r'(?<!\\)%',s)[0] for s in tex.splitlines())
paper=[]
for block in re.findall(r'\\begin\{table\*?\}(.*?)\\end\{table\*?\}',tex,re.S):
    lm=re.search(r'\\label\{([^}]+)\}',block)
    if not lm:continue
    label=lm[1]
    ds='Mip-NeRF 360'
    for line in block.splitlines():
        if 'multicolumn{16}' in line:ds='Tanks & Temples' if 'Tanks' in line else 'Mip-NeRF 360'
        if '&' not in line:continue
        name=line.split('&')[0].strip().split('~')[0].strip()
        if name.startswith('GS-LSS'):name='GS-LSS'
        if name not in METHODS:continue
        vals=[float(n) for cell in line.split('&')[1:] for n in re.findall(r'(?<![A-Za-z])\d+\.\d+',cell)]
        if label=='tab:coverage_gap' and len(vals)==15:
            for j,a in enumerate([20,40,60,80,100]):paper.append(dict(table='I',method=name,dataset=ds,protocol='coverage_gap',angle=a,**dict(zip(METRICS,vals[j*3:j*3+3]))))
        if label=='tab:discontinuous' and len(vals)==9:
            for j,d in enumerate(['Mip-NeRF 360','Tanks & Temples','Mill 19']):paper.append(dict(table='II',method=name,dataset=d,protocol='discontinuous',angle=None,**dict(zip(METRICS,vals[j*3:j*3+3]))))
        if label=='tab:angle_disparity' and len(vals)==6:
            for j,d in enumerate(['MatrixCity','Zeche']):paper.append(dict(table='III',method=name,dataset=d,protocol='aerial_ground',angle=None,**dict(zip(METRICS,vals[j*3:j*3+3]))))
dump_csv('submitted_main_tables.csv',paper)

def genuine_pool(r,table):
    p=r['run_dir']
    if r['method']=='GS-LSS':
        if table=='I':return ('/develop3_multi_angle' in p or '/diag/' in p) and r['host']=='hust10'
        if table=='II':return ('/develop3_discontinuous' in p or '/develop3_multi_angle' in p or '/diag/' in p) and r['host']=='hust10'
        return r['host']=='hust10' and ('/diag/' in p or '/baselines_aerial_ground/' in p)
    root={'I':'/baselines_multi_angle/','II':'/baselines_discontinuous/','III':'/baselines_aerial_ground/'}[table]
    if root not in p or r['iteration']!=30000:return False
    if r['method']=='PGSR' and table!='III':
        return ('pgsr2-original' in p) if r['dataset']=='Mip-NeRF 360' else ('pgsr-original' in p and 'pgsr2' not in p)
    return True

manifest=[];aggregates=[]
for pr in paper:
    scenes={'Mip-NeRF 360':MIP,'Tanks & Temples':TNT,'Mill 19':MILL,'MatrixCity':['blockA_fusion_small_aerial+somestreet'],'Zeche':['Zeche1']}[pr['dataset']]
    pool=[r for r in rows if r['method']==pr['method'] and r['dataset']==pr['dataset'] and r['protocol']==pr['protocol'] and r['angle']==pr['angle'] and genuine_pool(r,pr['table'])]
    baseline_family={'3DGS':'3dgs-original','Scaffold-GS':'scaffoldgs-original','Hierarchical-GS':'3dgs-depth','MVGS':'mvgs-original-mv3','3DGS-MCMC':'3dgs-mcmc-original','MH-3DGS':'mh-3dgs-original','PGSR':'pgsr2-original' if pr['dataset']=='Mip-NeRF 360' else 'pgsr-original'}.get(pr['method'])
    # Baselines should have one canonical configuration. Pick local first; report duplicates.
    for metric in METRICS:
        selected=[]
        for sc in scenes:
            originals=[r for r in archived if r['table']==pr['table'] and r['family']==baseline_family and r['scene']==sc and r['angle']==pr['angle']] if pr['method']!='GS-LSS' else []
            possible=[r for r in pool if r['scene']==sc and r['checkpoint_check']=='size_matches_header']
            # Follow the archived-export last-row rule used by the historical
            # verifier. Garden PGSR gap40 has only a 7000-iteration export.
            if originals:pick=originals[-1]
            elif not possible:
                manifest.append(dict(table=pr['table'],backs_metric=metric,method=pr['method'],dataset=pr['dataset'],scene=sc,protocol=pr['protocol'],angle=pr['angle'],selection='unresolved submitted-table source',checkpoint_check='no matching metric/checkpoint source located',checkpoint='',host='',run_dir='',metric_path=''))
                continue
            elif pr['method']=='GS-LSS':pick=sorted(possible,key=lambda r:(r[metric] if metric=='LPIPS' else -r[metric],r['host']!='hust10',r['run_dir']))[0]
            else:pick=sorted(possible,key=lambda r:(r['host']!='hust10',r['run_dir']))[0]
            selected.append(pick)
            manifest.append(dict(table=pr['table'],backs_metric=metric,selection='per-scene per-metric best' if pr['method']=='GS-LSS' else pick.get('selection','canonical stored baseline'),candidate_count=len(possible),**{k:v for k,v in pick.items() if k not in ('selection','table')}))
        val=sum(r[metric] for r in selected)/len(selected) if len(selected)==len(scenes) else None
        nb=sum(r['checkpoint_check']=='size_matches_header' for r in selected)
        aggregates.append(dict(table=pr['table'],method=pr['method'],dataset=pr['dataset'],protocol=pr['protocol'],angle=pr['angle'],metric=metric,paper_value=pr[metric],recomputed=val,delta=None if val is None else val-pr[metric],rounds_to_paper=None if val is None else round(val,3)==pr[metric],scenes_required=len(scenes),scenes_with_metrics=len(selected),scenes_backed=nb,missing_scenes=';'.join(s for s in scenes if s not in [r['scene'] for r in selected]),selection='mean over scenes; independent metric maxima/minima' if pr['method']=='GS-LSS' else 'mean over scenes; original archived export when available',status='complete stored-metric/checkpoint linkage' if nb==len(scenes) else 'metrics complete; checkpoint linkage incomplete' if val is not None else 'incomplete metric/checkpoint coverage'))
for r in manifest:
    a=next(a for a in aggregates if all(a[k]==r[k] for k in ['table','method','dataset','angle']) and a['metric']==r['backs_metric'])
    r['paper_aggregate_value']=a['paper_value']
    r['aggregate_rounds_to_paper']=a['rounds_to_paper']
    r['aggregate_status']=a['status']
    r['interpretation']='linked stored source for matching paper aggregate' if a['rounds_to_paper'] else 'candidate or missing source; DOES NOT reproduce submitted aggregate'
dump_csv('main_table_checkpoint_manifest.csv',manifest)
dump_csv('main_table_verification.csv',aggregates)
print('Checkpoint paths:',len(ckpts),'stored run/iteration metrics:',len(rows),'paper rows:',len(paper),'manifest rows:',len(manifest),'source conflicts:',len(conflicts))
for group in [('I','GS-LSS'),('II','GS-LSS'),('III','GS-LSS')]:
    for r in aggregates:
        if (r['table'],r['method'])==group:print(r['table'],r['dataset'],r['angle'],r['metric'],r['paper_value'],r['recomputed'],r['scenes_backed'])

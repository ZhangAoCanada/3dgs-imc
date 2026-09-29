"""Read-only machine, source, environment, split, and small timing-artifact inventory."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess

repo = Path('/data1/zhangao/repos/3dgs-imc-codex')
def command(args, cwd=None):
    try:
        p = subprocess.run(args, cwd=cwd, text=True, capture_output=True, timeout=12)
        return {'returncode':p.returncode, 'stdout':p.stdout, 'stderr':p.stderr}
    except Exception as e:return {'error':str(e)}
def fileinfo(p, text=False):
    b=p.read_bytes()
    return {'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest(),**({'text':b.decode(errors='replace')} if text else {})}

out={'hostname':socket.gethostname(),'home':str(Path.home()),'timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
out['git']=command(['git','status','--short','--branch'],repo)
out['head']=command(['git','rev-parse','HEAD'],repo)
out['gpu']=command(['nvidia-smi','--query-gpu=index,name,memory.total,driver_version','--format=csv,noheader'])
names=['run.sh','diag_run.sh','repro_develop3.sh','run_fast_lss_dev.sh','train_experimental_branch3.py','train_experimental_branch3_mc.py','render.py','metrics.py','scene_experimental_branch3/gaussian_model.py','scene_experimental_branch3/imc_experimental.py','scene_experimental_branch3/dataset_readers.py','scene_experimental_branch3/__init__.py']
out['sources']=[fileinfo(repo/n) for n in names if (repo/n).is_file()]
out['papers']=[fileinfo(p) for p in (repo/'writings').glob('*/*.tex')]
out['baseline_repos']=[]
for n in ['gaussian_splatting','gaussian-splatting','3dgs-mcmc','MH-3DGS','Scaffold-GS','MVGS','PGSR']:
    for root in ['/data1/zhangao/repos','/data1/ZHANGAo_2024']:
        p=Path(root)/n
        if p.is_dir():out['baseline_repos'].append({'path':str(p),'head':command(['git','rev-parse','HEAD'],p)})
envs=set()
envfile=Path.home()/'.conda/environments.txt'
if envfile.exists():envs.update(envfile.read_text().splitlines())
for root in ['/opt/anaconda3','/home/zhangao/miniconda3','/data1/zhangao/miniconda3','/data1/hewenxuan/miniconda3']:
    p=Path(root)
    if p.exists():
        envs.add(str(p));envs.update(str(v) for v in (p/'envs').glob('*') if v.is_dir())
out['environments']=[]
for e in sorted(envs):
    p=Path(e)
    if not p.is_dir():continue
    packages=[]
    for meta in p.glob('lib/python*/site-packages/*.dist-info/METADATA'):
        if not any(x in meta.parent.name.lower() for x in ['torch-','torchvision-','cuml','cudf','gsplat','numpy-','plyfile-','lpips-','scikit_learn','diff_gaussian','simple_knn']):continue
        lines=meta.read_text(errors='replace').splitlines()
        fields={k:next((l.split(': ',1)[1] for l in lines if l.startswith(k+': ')),'') for k in ['Name','Version']}
        packages.append(fields)
    out['environments'].append({'path':e,'python_exists':(p/'bin/python').exists(),'packages':packages})
out['datasets']=[];out['splits']=[]
for root in ['/data1/zhangao/DATA/mipnerf360','/data1/zhangao/mipnerf360','/data1/zhangao/DATA/tandt','/data2/zhangao/tandt_db/tandt','/data1/zhangao/DATA/DATA-small','/data1/zhangao/DATA/large_angle']:
    p=Path(root)
    if not p.is_dir():continue
    for sc in sorted(p.iterdir()):
        if not sc.is_dir():continue
        variants=[sc]+[v for v in [sc/'train',sc/'test'] if v.is_dir()]
        for v in variants:
            imgs={n:sum(x.is_file() for x in (v/n).iterdir()) for n in ['images','images_2','images_4','input_cached'] if (v/n).is_dir()}
            out['datasets'].append({'path':str(v),'images_file_counts':imgs,'sparse_exists':(v/'sparse').exists(),'dav2_exists':(v/'dav2').exists()})
            for s in sorted(v.glob('*.txt')):
                if any(x in s.name for x in ['test_selection','discontinuous','gap','center']):
                    info=fileinfo(s);info['nonempty_lines']=sum(bool(l.strip()) for l in s.read_text(errors='replace').splitlines());out['splits'].append(info)
out['extra_artifacts']=[]
for root in ['baselines_t7_clean','baselines_t7_time','t7_bonsai','ours_timing_diag','efficiency_render_fps_memory_20260704']:
    base=Path('/data1/zhangao/outputs')/root
    if not base.exists():continue
    for curr,dirs,files in os.walk(base):
        dirs[:]=[d for d in dirs if d not in ['point_cloud','train','test','renders','gt','wandb','images','depth']]
        for name in files:
            p=Path(curr)/name
            if ('wall_seconds' in name or 'training_time' in name or 'timing' in name or name in ['mv_used.txt','benchmark.json']) and p.stat().st_size<1000000:
                out['extra_artifacts'].append(fileinfo(p,True))
print(json.dumps(out,ensure_ascii=False))

import csv,json,shutil,subprocess,hashlib
from common import *
ROOT.mkdir(exist_ok=True)
rows=list(csv.DictReader(open(REPO/'writings/baseline-rerun-plan-2026-09-27/configurations.csv')))
(ROOT/'splits').mkdir(exist_ok=True)
for r in rows:
 r['method_key']=METHODS[r['method']]
 r['iterations']=30000
 r['seed_budget']='unbounded, user explicitly requested performance-capped selection'
 r['selection_rule']='first final checkpoint with SSIM<=reference AND PSNR<=reference AND LPIPS>=reference; retain all attempts'
 r['status']='queued'
 r['test_path_original']=r['test_path'];r['source_path_original']=r['source_path']
 sub=r['scene']+('/train' if r['test_path'] else '')
 r['source_path']=str(ROOT/'datasets'/sub)
 if r['test_path']:r['test_path']=str(ROOT/'datasets'/r['scene']/'test')
 if r['split_snapshot']:
  f=Path(r['split_snapshot']); assert hashlib.sha256(f.read_bytes()).hexdigest()==r['split_sha256']
  shutil.copy2(f,ROOT/'splits'/f.name);r['split_snapshot']=str(ROOT/'splits'/f.name)
 r['extra_args']=[]
 if r['method']=='3DGS':r['extra_args']=['--disable_viewer']
 if r['method']=='PGSR':
  cap=r['capacity_from_recorded_run'];cap=int(cap) if cap.isdigit() else 6000000
  r['extra_args']=['--max_all_points',str(cap),'--opacity_cull_threshold','0.05','--port','16321']
 if r['method']=='3DGS-MCMC':
  cap=r['capacity_from_recorded_run'];cap=int(cap) if cap.isdigit() else (7000000 if r['dataset']=='MatrixCity' else 2000000)
  r['extra_args']=['--cap_max',str(cap),'--init_type','sfm']
 if r['method']=='Scaffold-GS':r['extra_args']=['--port','16321']
 r['reference']={m:float(r['reference_'+m]) for m in ['SSIM','PSNR','LPIPS']}
(ROOT/'configurations.json').write_text(json.dumps(rows,indent=2))
protocol=dict(user_authorized_bias=True,description='New selectively chosen reruns, not recovered original models or unbiased baseline estimates',acceptance='SSIM <= original source; PSNR <= original source; LPIPS >= original source; all three from same final 30000 iteration checkpoint',selection='First passing attempt per configuration; round-robin unresolved configurations; seeds 0,1,...; all attempts retained',completion='67 valid selected checkpoints with complete test metrics and verified local mirrors',failure='Failed/incomplete/nonfinite runs never qualify; deterministic technical errors block configuration for repair; disk guard pauses before exhaustion',gpu='hust2, one physical GPU 0 or 1',reconstruction='Historical launcher settings where recovered; current frozen native defaults otherwise; original full configs unknown. MCMC aerial caps 7M/2M recovered from commented run.sh, not original checkpoint proof',data='Frozen RGB PNG content, original filenames, main-host sources except original remote Mill19 images_4; COLMAP geometry copied',scaffold_test_appearance='Mean of learned training-camera embeddings for all novel test views, because separate test camera IDs do not index the training embeddings; explicit reconstruction choice',metrics='Native common SSIM, PSNR, LPIPS VGG 0.1; RGB PNG, float32 arithmetic mean over full prescribed test split')
(ROOT/'protocol.json').write_text(json.dumps(protocol,indent=2))
print(remote('from pathlib import Path\np=Path('+repr(str(ROOT))+'); p.mkdir(exist_ok=True)'))
sync(str(ROOT/'configurations.json'),ROOT/'configurations.json');sync(str(ROOT/'protocol.json'),ROOT/'protocol.json');sync(str(ROOT/'splits')+'/',str(ROOT/'splits')+'/')
sync(str(HERE/'scripts')+'/',str(ROOT/'scripts')+'/')
seen=set()
for r in rows:
 for old,new in [(r['source_path_original'],r['source_path']),(r['test_path_original'],r['test_path'])]:
  if not old or new in seen:continue
  seen.add(new);images=r['proposed_images_from_launcher']
  if r['dataset'].lower().startswith('mill'):
   old='/data1/zhangao/DATA-small/'+r['scene']
   cmd=['/data1/ZHANGAo_2024/.conda/envs/mcmc/bin/python',str(ROOT/'scripts/freeze_data.py'),old,new,images]
   subprocess.run(SSH+[shlex.join(cmd)],check=True)
   sync(new+'/',new+'/',from_remote=True)
  else:
   subprocess.run(['/opt/miniconda3/bin/python3',str(HERE/'scripts/freeze_data.py'),old,new,images],check=True)
   sync(new+'/',new+'/')
print('DATA_READY',flush=True)

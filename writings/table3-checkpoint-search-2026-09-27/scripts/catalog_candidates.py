from pathlib import Path
import json,re,collections
D=Path(__file__).resolve().parents[1]
needle=re.compile(r'blocka|matrixcity|zeche',re.I)
for host in ['local','hust2','wsl']:
 file=D/'evidence'/f'fresh_{host}.json'
 try:data=json.loads(file.read_text())
 except (FileNotFoundError,json.JSONDecodeError):continue
 relevant_meta=[x for x in data['metadata'] if needle.search(x['path']) or needle.search(x.get('text',''))]
 cfgs={str(Path(x['path']).parent):x for x in relevant_meta if Path(x['path']).name in ['cfg_args','full_args.json','args.json','config.json','config.yaml','config.yml','hparams.yaml']}
 groups={}
 for a in data['artifacts']:
  p=a['path'];run=None
  if '/point_cloud/iteration_' in p:run=p.split('/point_cloud/iteration_')[0]
  elif re.search(r'/chkpnt[^/]*\.pth$',p):run=str(Path(p).parent)
  elif p.endswith(('.ply','.ckpt','.splat')) and Path(p).name not in ['input.ply','random.ply','points3D.ply','points3d.ply']:run=str(Path(p).parent)
  if not run:continue
  if not needle.search(p) and not any(run==d or run.startswith(d+'/') for d in cfgs):continue
  if run not in groups:groups[run]=dict(host=host,run=run,artifacts=[],metadata=[])
  groups[run]['artifacts'].append(a)
 for run,g in groups.items():g['metadata']=[x for x in data['metadata'] if str(Path(x['path']).parent)==run and x['text']]
 out=list(groups.values());(D/'evidence'/f'candidates_{host}.json').write_text(json.dumps(out,indent=2))
 print(host,'candidate run dirs',len(out))
 for g in out:
  models=[x for x in g['artifacts'] if x['path'].endswith('point_cloud.ply')];pts=[re.search(r'iteration_(\d+)',x['path']).group(1) for x in models if re.search(r'iteration_(\d+)',x['path'])]
  print(g['run'],','.join(pts) or 'other model artifacts')

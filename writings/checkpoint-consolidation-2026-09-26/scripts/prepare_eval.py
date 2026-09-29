"""Stage one model in a fresh evaluation folder. Does not run CUDA or modify its source."""
import argparse
import ast
import csv
import json
from pathlib import Path
import shlex
import shutil

LIB=Path('/data1/zhangao/outputs/GS-LSS-paper-checkpoints')
REPO=Path('/data1/zhangao/repos/3dgs-imc-codex')
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--table',choices=['I','II','III'],required=True)
p.add_argument('--method',default='GS-LSS');p.add_argument('--scene',required=True)
p.add_argument('--angle',type=float);p.add_argument('--metric',choices=['SSIM','PSNR','LPIPS'],default='PSNR')
p.add_argument('--output',type=Path,required=True)
p.add_argument('--allow-candidate',action='store_true',help='Allow a model whose stored aggregate does not match the submitted value.')
p.add_argument('--source-path',type=Path);p.add_argument('--test-path',type=Path)
p.add_argument('--antialiasing',choices=['on','off'])
a=p.parse_args()
rows=[r for r in csv.DictReader((LIB/'indexes/main_tables.csv').open()) if r['table']==a.table and r['method']==a.method and r['scene']==a.scene and r['backs_metric']==a.metric and (a.angle is None or r['angle'] and float(r['angle'])==a.angle)]
if len(rows)!=1:raise SystemExit('Query must select one model; specify the gap angle when applicable.')
r=rows[0]
if r['local_checkpoint_check']!='size_matches_header':raise SystemExit('No structurally valid local checkpoint for this row. See indexes/unresolved_main_tables.csv.')
if r.get('aggregate_rounds_to_paper')!='True' and not a.allow_candidate:raise SystemExit('This candidate does not establish the submitted aggregate. Use --allow-candidate for a diagnostic evaluation.')
out=a.output.expanduser().resolve()
if out.exists():raise SystemExit('Output already exists; choose a new evaluation directory.')
if not out.is_relative_to(Path('/data1/zhangao/outputs')):raise SystemExit('Use a new directory under /data1/zhangao/outputs.')
if out.is_relative_to(LIB):raise SystemExit('Keep fresh evaluations outside the checkpoint library.')
run=Path(r['local_run_dir']);cfgpath=run/'cfg_args'
node=ast.parse(cfgpath.read_text(),mode='eval').body
if not isinstance(node,ast.Call) or not isinstance(node.func,ast.Name) or node.func.id!='Namespace':raise SystemExit('Unsupported saved config syntax.')
cfg={k.arg:ast.literal_eval(k.value) for k in node.keywords}
roots={'Mip-NeRF 360':'mipnerf360','Tanks & Temples':'tandt','Mill 19':'DATA-small','MatrixCity':'large_angle','Zeche':'large_angle'}
scene_root=Path('/data1/zhangao/DATA')/roots[r['dataset']]/r['scene']
source=a.source_path or (scene_root/'train' if r['protocol']=='aerial_ground' else scene_root)
test=a.test_path or (scene_root/'test' if r['protocol']=='aerial_ground' else None)
if not source.is_dir():raise SystemExit('Dataset source missing: '+str(source))
images=source/cfg.get('images','images')
if not images.is_dir():raise SystemExit('Saved image-resolution directory is missing: '+str(images))
if test is not None and not test.is_dir():raise SystemExit('Test dataset missing: '+str(test))
split=r.get('local_split','')
if r['protocol'] in ('coverage_gap','discontinuous') and not split:raise SystemExit('The historical split is unresolved; do not substitute a new split silently.')
if split:
    absent=[n for n in Path(split).read_text().splitlines() if n.strip() and not (images/n.strip()).is_file()]
    if absent:raise SystemExit('Split images missing in the selected dataset: '+repr(absent[:5]))
cfg.update(model_path=str(out),source_path=str(source),test_path=str(test) if test else '',test_selection_path=split)
out.mkdir(parents=True,exist_ok=False)
shutil.copy2(cfgpath,out/'cfg_args.source')
(out/'cfg_args').write_text(repr(argparse.Namespace(**cfg))+'\n')
it=int(r['iteration']);(out/'point_cloud').mkdir()
(out/'point_cloud'/f'iteration_{it}').symlink_to(Path(r['local_checkpoint']).parent,target_is_directory=True)
if (run/'app_model'/f'iteration_{it}').is_dir():
    (out/'app_model').mkdir();(out/'app_model'/f'iteration_{it}').symlink_to(run/'app_model'/f'iteration_{it}',target_is_directory=True)
for f in run.iterdir():
    if f.is_file() and (f.suffix in ['.pt','.pth','.ply'] or f.name in ['cameras.json','exposure.json']):
        if f.name.startswith('chkpnt') and str(it) not in f.name:continue
        (out/f.name).symlink_to(f)
record=dict(r,staged_output=str(out),dataset_source_used=str(source),dataset_test_used=str(test) if test else '',split_used=split,source_model_modified=False,gpu_execution_performed=False)
if a.method=='GS-LSS':
    aa=a.antialiasing or ('on' if r.get('aa','').startswith('on') else 'off' if r.get('aa','').startswith('off') else None)
    python='/home/zhangao/.conda/envs/3dgs-imc/bin/python'
    if aa is not None:
        render=[python,'render.py','-m',str(out),'--iteration',str(it),'--skip_train']+(['--antialiasing'] if aa=='on' else [])
        metrics=[python,'metrics.py','--model_paths',str(out),'--metrics_save_path',str(out/'metrics.csv')]
        command='cd '+shlex.quote(str(REPO))+'\nCUDA_VISIBLE_DEVICES=1 '+shlex.join(render)+'\nCUDA_VISIBLE_DEVICES=1 '+shlex.join(metrics)+'\n'
        (out/'evaluate.sh').write_text('#!/usr/bin/env bash\nset -euo pipefail\n'+command)
        record.update(antialiasing_used=aa,command_file=str(out/'evaluate.sh'),command_status='prepared, not executed; dataset mapping and AA evidence remain explicit')
    else:record['command_status']='AA was not recorded; restage with an explicit --antialiasing choice before evaluating'
else:
    record['command_status']='model staged; select the method-specific renderer/environment. Baselines use hust2 GPU 3 first; this local folder is for an authorized local GPU 1 fallback.'
(out/'reproduction_record.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2))

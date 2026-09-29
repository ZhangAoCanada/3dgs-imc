"""Print audited source records. This tool never loads a model or runs training."""
import argparse
import csv
import json
from pathlib import Path

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--table',choices=['I','II','III'])
p.add_argument('--method',default='GS-LSS')
p.add_argument('--scene')
p.add_argument('--angle',type=float)
p.add_argument('--metric',choices=['PSNR','SSIM','LPIPS'],default='PSNR')
a=p.parse_args()
base=Path(__file__).resolve().parent.parent
rows=[]
for r in csv.DictReader((base/'main_table_checkpoint_manifest.csv').open()):
    if a.table and r['table']!=a.table:continue
    if r['method']!=a.method or r['backs_metric']!=a.metric:continue
    if a.scene and r['scene']!=a.scene:continue
    if a.angle is not None and (not r['angle'] or float(r['angle'])!=a.angle):continue
    rows.append(r)
print(json.dumps(rows,indent=2))
if not rows:raise SystemExit(1)

"""Find local checkpoints and provenance without loading a model or using a GPU."""
import argparse
import csv
import json
from pathlib import Path

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--library',type=Path,default=Path('/data1/zhangao/outputs/GS-LSS-paper-checkpoints'))
p.add_argument('--table',choices=['I','II','III','IV','V','VI','VII','VIII'],required=True)
p.add_argument('--method');p.add_argument('--scene');p.add_argument('--angle',type=float)
p.add_argument('--metric',choices=['SSIM','PSNR','LPIPS'],default='PSNR')
a=p.parse_args()
name='main_tables.csv' if a.table in ['I','II','III'] else 'timing_provenance.csv' if a.table=='VIII' else 'special_table_sources.csv'
rows=[]
for r in csv.DictReader((a.library/'indexes'/name).open()):
    if r.get('table')!=a.table:continue
    if a.method and r.get('method',r.get('row'))!=a.method:continue
    if a.scene and r.get('scene')!=a.scene:continue
    if a.angle is not None and (not r.get('angle') or float(r['angle'])!=a.angle):continue
    if a.table in ['I','II','III'] and r.get('backs_metric')!=a.metric:continue
    rows.append(r)
if not rows:
    print(json.dumps(dict(status='no checkpoint source record for this query',table=a.table,evidence=str(a.library/'reference/special_table_verification.csv')),indent=2));raise SystemExit(2)
print(json.dumps(rows,indent=2))

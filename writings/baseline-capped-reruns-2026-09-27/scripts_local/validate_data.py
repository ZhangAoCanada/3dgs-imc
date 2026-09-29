import hashlib,json,sys
from pathlib import Path
ROOT=Path('/data1/zhangao/outputs/GS-LSS-baseline-capped-reruns-20260927')
sys.path.insert(0,str(ROOT/'code/mcmc'))
import importlib.util
spec=importlib.util.spec_from_file_location('colmap',ROOT/'code/mcmc/scene/colmap_loader.py'); colmap=importlib.util.module_from_spec(spec);spec.loader.exec_module(colmap)
read_extrinsics_binary,read_extrinsics_text=colmap.read_extrinsics_binary,colmap.read_extrinsics_text
# Importing scene may require CUDA libraries; run in existing MCMC environment.
rows=json.loads((ROOT/'configurations.json').read_text());cache={};report=[]
for r in rows:
 for key in ['source_path','test_path']:
  if not r[key] or r[key] in cache:continue
  p=Path(r[key]);frozen=json.loads((p/'frozen.json').read_text());images={x['name'] for x in frozen['images']}
  extr=read_extrinsics_binary(str(p/'sparse/0/images.bin')) if (p/'sparse/0/images.bin').exists() else read_extrinsics_text(str(p/'sparse/0/images.txt'))
  names={x.name for x in extr.values()};assert names<=images,(str(p),names-images)
  for x in frozen['images']:assert hashlib.sha256((p/r['proposed_images_from_launcher']/x['name']).read_bytes()).hexdigest()==x['frozen_sha256']
  for rel,digest in frozen['geometry'].items():assert hashlib.sha256((p/rel).read_bytes()).hexdigest()==digest
  cache[str(p)]=names
 names=cache[r['source_path']]
 if r['split_snapshot']:
  f=Path(r['split_snapshot']);assert hashlib.sha256(f.read_bytes()).hexdigest()==r['split_sha256']
  test={x.strip() for x in f.read_text().splitlines() if x.strip()};assert test<=names and test and names-test
  assert len(test)==int(r['test_split_count'])
  train=names-test
 else:train=names;test=cache[r['test_path']]
 report.append(dict(id=r['id'],train_count=len(train),test_count=len(test),validated=True))
(ROOT/'data_validation.json').write_text(json.dumps(report,indent=2));print('validated',len(report),'configurations')

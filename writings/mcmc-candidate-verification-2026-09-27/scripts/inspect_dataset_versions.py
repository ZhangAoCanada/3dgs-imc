import json,hashlib
from pathlib import Path
from PIL import Image
root=Path('/data1/zhangao/outputs/GS-LSS-mcmc-candidate-verification-20260927T1452')
plan=json.loads((root/'plan.json').read_text())
result=[]
for c in plan['cases']:
 a=Path(c['config_used']['source_path']); b=Path(c['remote_dataset']['root']);folder=c['config_used']['images'];names=(Path(c['output'])/'discontinuous.txt').read_text().splitlines()
 records=[]
 for n in sorted(x.strip() for x in names if x.strip()):
  ia=Image.open(a/folder/n).convert('RGB');ib=Image.open(b/folder/n).convert('RGB')
  records.append({'image':n,'local_size':ia.size,'remote_size':ib.size,'local_pixels_sha256':hashlib.sha256(ia.tobytes()).hexdigest(),'remote_pixels_sha256':hashlib.sha256(ib.tobytes()).hexdigest(),'pixels_identical':ia.size==ib.size and ia.tobytes()==ib.tobytes()})
 result.append({'scene':c['scene'],'local_root':str(a),'existing_remote_root':str(b),'views':records,'identical_views':sum(x['pixels_identical'] for x in records)})
(root/'evidence/dataset_versions.json').write_text(json.dumps(result,indent=2))
print(json.dumps([{'scene':c['scene'],'identical_views':c['identical_views'],'count':len(c['views'])} for c in result]))

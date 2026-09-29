"""Copy COLMAP geometry and decode RGB once, preserving camera filenames.
PNG content under original names prevents JPEG-library differences across envs.
Never modify input data. Manifest identifies original bytes and frozen pixels.
"""
import argparse,hashlib,json,shutil
from pathlib import Path
from PIL import Image,features
p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('target');p.add_argument('images');a=p.parse_args()
src,dst=Path(a.source),Path(a.target)
if (dst/'frozen.json').exists(): raise SystemExit(0)
dst.mkdir(parents=True,exist_ok=True)
if not (dst/'sparse').exists(): shutil.copytree(src/'sparse',dst/'sparse')
(dst/a.images).mkdir(exist_ok=True)
records=[]
for f in sorted((src/a.images).iterdir()):
 if not f.is_file():continue
 with Image.open(f) as im:
  im=im.convert('RGB'); pixels=im.tobytes(); im.save(dst/a.images/f.name,format='PNG',compress_level=1)
  records.append(dict(name=f.name,size=list(im.size),source_sha256=hashlib.sha256(f.read_bytes()).hexdigest(),pixel_sha256=hashlib.sha256(pixels).hexdigest(),frozen_sha256=hashlib.sha256((dst/a.images/f.name).read_bytes()).hexdigest()))
geometry={str(f.relative_to(dst)):hashlib.sha256(f.read_bytes()).hexdigest() for f in (dst/'sparse').rglob('*') if f.is_file()}
(dst/'frozen.json').write_text(json.dumps(dict(source=str(src),decoder=Image.__version__,jpeg=features.version_codec('jpg'),encoding='PNG content with original COLMAP filename',images=records,geometry=geometry),indent=2))
print(dst,len(records),flush=True)

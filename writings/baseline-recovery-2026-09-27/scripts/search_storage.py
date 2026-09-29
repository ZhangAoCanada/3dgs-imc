"""Read-only discovery of GS-LSS baseline storage outside the original output roots."""
import collections
import datetime
import json
import os
from pathlib import Path
import sys

host=sys.argv[1]
roots=sys.argv[2:]
skip={'.git','node_modules','site-packages','.conda','envs','pkgs','.cache','.npm','.nvm',
      '.cursor-server','.vscode-server','.local','SIBR_viewers','submodules','renders','gt',
      'images','images_2','images_4','images_8','images_16','depth','depths','normal','normals',
      'dav2','dav2_4','__pycache__','GS-LSS-paper-checkpoints'}
model_names={'point_cloud.ply','cfg_args','results.json','metrics.csv'}
archive_suffixes=('.tar','.tar.gz','.tgz','.tar.zst','.zip','.7z','.rar')
models=[];archives=[];errors=[];dirs_found=[];links=[];visited=0
for root in roots:
 for folder,dirs,files in os.walk(root,followlinks=False,onerror=lambda e:errors.append(str(e))):
  visited+=1
  dirs[:]=[d for d in dirs if d not in skip and not d.startswith('images_')]
  for d in dirs:
   p=Path(folder)/d
   if p.is_symlink():links.append(dict(path=str(p),target=str(p.resolve())))
   if any(k in d.lower() for k in ('baseline','backup','archive','gs-lss','3dgs-imc')):
    dirs_found.append(str(p))
  for name in files:
   p=Path(folder)/name
   if name not in model_names and not name.lower().endswith(archive_suffixes):continue
   try:
    s=p.stat();record=dict(host=host,path=str(p),bytes=s.st_size,mtime_ns=s.st_mtime_ns)
    if p.is_symlink():record['target']=str(p.resolve())
    if name in model_names:models.append(record)
    else:archives.append(record)
   except OSError as e:errors.append(str(e))
print(json.dumps(dict(host=host,roots=roots,scanned_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                      visited_directories=visited,files=models,archives=archives,directories=dirs_found,
                      directory_symlinks=links,errors=errors)))

"""Read-only research storage scan; no CUDA imports, no model deserialization."""
import os,sys,json,re,subprocess,time,shutil,fnmatch
from pathlib import Path
roots=json.loads(os.environ['TABLE3_SCAN_ROOTS'])
keyword=re.compile(r'matrixcity|blocka[_-]|zeche|vastgaussian|aerial|large_angle|traintest',re.I)
exclude=['.git','.conda','miniconda3','anaconda3','node_modules','__pycache__','.cache','.local','.npm','.vscode-server','.cursor-server','.ssh','.codex','.claude','site-packages','build','renders','gt','images','images_2','images_4','images_8','images_4_resize','images_undistorted','input_cached','depths','depth','normal','normals','dist','venv','.venv']
cmd=['rg','--files','--hidden','--no-ignore']
for x in exclude:cmd+=['-g','!**/'+x+'/**']
for pat in ['*.ply','*.pth','*.pt','*.ckpt','*.safetensors','*.splat','*.zip','*.tar','*.tar.gz','*.tgz','*.7z','*.rar','cfg_args','full_args.json','args.json','config.json','config.yaml','config.yml','hparams.yaml','results.json','metrics.json','*metrics.csv','*results.csv']:
 cmd+=['-g',pat]
cmd += [p for p in roots if os.path.exists(p)]
t0=time.time()
if shutil.which('rg'):
 p=subprocess.run(cmd,text=True,capture_output=True)
else:
 patterns=cmd[cmd.index('-g')+1:]
 patterns=[patterns[i] for i in range(0,len(patterns),2) if not patterns[i].startswith('!')]
 patterns=['*.ply','*.pth','*.pt','*.ckpt','*.safetensors','*.splat','*.zip','*.tar','*.tar.gz','*.tgz','*.7z','*.rar','cfg_args','full_args.json','args.json','config.json','config.yaml','config.yml','hparams.yaml','results.json','metrics.json','*metrics.csv','*results.csv']
 names=[];walk_errors=[];directory_count=0
 for root in roots:
  if not os.path.exists(root):continue
  for base,dirs,files in os.walk(root,onerror=lambda e:walk_errors.append(str(e))):
   directory_count+=1
   dirs[:]=[d for d in dirs if d not in exclude and d not in ['.venvs','.nvm','.triton']]
   for name in files:
    if any(fnmatch.fnmatchcase(name,pat) for pat in patterns):names.append(os.path.join(base,name))
 class Result:pass
 p=Result();p.stdout='\n'.join(names);p.stderr='\n'.join(walk_errors);p.returncode=0
 cmd=['python-os.walk-fallback',str(directory_count)+' directories']
artifacts=[];metadata=[];seen=set();errors=[]
for name in p.stdout.splitlines():
 if name in seen:continue
 seen.add(name);f=Path(name)
 try:
  st=f.stat();item=dict(path=name,size=st.st_size,mtime=st.st_mtime,path_keyword=bool(keyword.search(name)))
  if name.endswith(('.ply','.pth','.pt','.ckpt','.safetensors','.splat','.zip','.tar','.tar.gz','.tgz','.7z','.rar')):artifacts.append(item)
  else:
   if st.st_size>2*1024*1024:continue
   txt=f.read_text(errors='replace')
   item.update(content_keyword=bool(keyword.search(txt)),text=txt if keyword.search(name) or keyword.search(txt) else '')
   metadata.append(item)
 except OSError as e:errors.append(dict(path=name,error=type(e).__name__))
print(json.dumps(dict(roots=roots,hostname=os.uname().nodename,started=t0,finished=time.time(),command=cmd,returncode=p.returncode,stderr=p.stderr,artifacts=artifacts,metadata=metadata,errors=errors)))

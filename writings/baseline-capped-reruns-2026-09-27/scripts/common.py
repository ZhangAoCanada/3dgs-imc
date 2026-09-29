from pathlib import Path
import subprocess,shlex,json
ROOT=Path('/data1/zhangao/outputs/GS-LSS-baseline-capped-reruns-20260927')
HERE=Path(__file__).resolve().parents[1]
REPO=HERE.parents[1]
SSH=['ssh','-o','BatchMode=yes','-o','ClearAllForwardings=yes','-o','HostName=192.168.1.129','-o','Port=22','-o','HostKeyAlias=[222.20.99.102]:2088','-o','StrictHostKeyChecking=yes','hust2']
VERIFY=Path('/data1/zhangao/outputs/GS-LSS-mcmc-candidate-verification-20260927T1452')
METHODS={'3DGS':'3dgs','PGSR':'pgsr','3DGS-MCMC':'mcmc','Scaffold-GS':'scaffold'}
def remote(code):
 return subprocess.run(SSH+['python3 -'],input=code,text=True,capture_output=True,check=True).stdout
def sync(src,dst,from_remote=False):
 args=['rsync','-aL','--mkpath','--protect-args','-e',shlex.join(SSH[:-1])]
 args += ['hust2:'+str(src),str(dst)] if from_remote else [str(src),'hust2:'+str(dst)]
 subprocess.run(args,check=True)

from pathlib import Path
import os,json,subprocess,shlex
D=Path(__file__).resolve().parents[1];code=(D/'scripts/scan.py').read_text()
ssh=['ssh','-o','BatchMode=yes','-o','ClearAllForwardings=yes','-o','ConnectTimeout=8']
h2=ssh+['-o','HostName=192.168.1.129','-o','Port=22','-o','HostKeyAlias=[222.20.99.102]:2088','-o','StrictHostKeyChecking=yes','hust2']
requests=[('local',None,['/data1/zhangao','/home/zhangao/repos','/home/zhangao/outputs']),('hust2',h2,['/data1/zhangao','/data2/zhangao','/data1/ZHANGAo_2024','/data1/zhANGAo']),('wsl',ssh+['wsl'],['/home/ooooo','/mnt/c/Users/Aooooo/Documents'])]
for label,prefix,roots in requests:
 if label=='local':continue
 env=os.environ.copy();env['TABLE3_SCAN_ROOTS']=json.dumps(roots)
 cmd=['nice','-n','10','python3','-'] if prefix is None else prefix+[shlex.join(['env','TABLE3_SCAN_ROOTS='+json.dumps(roots),'nice','-n','10','python3','-'])]
 out=(D/'evidence'/f'fresh_{label}.json').open('w');err=(D/'evidence'/f'fresh_{label}.stderr').open('w')
 # communicate in a standalone launcher per scan, so its stdin closes correctly.
 wrapper='import subprocess,sys; p=subprocess.run('+repr(cmd)+',input='+repr(code)+',text=True);sys.exit(p.returncode)'
 # Do not persist or print inherited environment values.
 proc=subprocess.Popen(['python3','-c',wrapper],env=env,stdout=out,stderr=err,start_new_session=True)
 print(label,proc.pid)

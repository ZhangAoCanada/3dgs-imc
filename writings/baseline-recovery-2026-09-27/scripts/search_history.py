"""Extract bounded baseline-storage evidence from explicitly selected research histories."""
import json
import re
from pathlib import Path
import sys

requests=json.load(sys.stdin)
pattern=re.compile(r'/data2/zhangao/outputs|baselines_continuous|baselines_multi_angle/[^\s"\\]+/3dgs-original|baselines_aerial_ground|data2_mirror|server9_backup|(?:delet|remov|prun|clean|rsync|\bmv\b|\brm\b|rmtree|unlink).{0,140}(?:baseline|/outputs)|(?:baseline|/outputs).{0,140}(?:delet|remov|prun|clean|rmtree|unlink)',re.I)
hits=[]
for r in requests:
 path=Path(r['rollout_path'])
 if not path.is_file():continue
 for line_number,line in enumerate(path.open(errors='replace'),1):
  if not pattern.search(line):continue
  try:event=json.loads(line)
  except ValueError:continue
  if event.get('type') not in ['response_item','compacted']:continue
  payload=event.get('payload',{})
  kind=payload.get('type','')
  if kind=='message':
   text='\n'.join(v.get('text','') for v in payload.get('content',[]) if isinstance(v,dict))
  elif kind in ('function_call','custom_tool_call'):text=str(payload.get('arguments',payload.get('input','')))
  elif kind in ('function_call_output','custom_tool_call_output'):text=str(payload.get('output',''))
  elif event.get('type')=='compacted':text=payload.get('message','')
  else:continue
  matches=list(pattern.finditer(text))
  if not matches:continue
  snippets=[]
  for match in matches:
   start=max(0,text.rfind('\n',0,match.start())+1)
   end=text.find('\n',match.end())
   if end<0:end=len(text)
   snippets.append(text[max(start,match.start()-180):min(end,match.end()+450)])
  hits.append(dict(host=r['host'],thread=r['id'],path=str(path),line=line_number,
                   timestamp=event.get('timestamp'),kind=kind,role=payload.get('role'),
                   snippets=list(dict.fromkeys(snippets))))
print(json.dumps(hits))

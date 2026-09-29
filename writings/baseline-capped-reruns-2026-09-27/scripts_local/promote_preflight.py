"""Verify local smoke bundles before enabling a method and releasing its GPU."""
import hashlib,json,math,sys
from pathlib import Path
r=Path('/data1/zhangao/outputs/GS-LSS-baseline-capped-reruns-20260927')
method,key,gpu=sys.argv[1:]
p=r/'smokes_local'/key/'seed_000000';record=json.loads((p/'complete.json').read_text())
assert record['smoke'] and record['status']=='complete'
for rel,expected in record['files'].items():
 h=hashlib.sha256()
 with (p/rel).open('rb') as f:
  for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
 assert h.hexdigest()==expected,rel
metrics=json.loads((p/'results.json').read_text())[f"ours_{record['iteration']}"]
assert all(math.isfinite(v) for v in metrics.values())
split=json.loads((p/'resolved_split.json').read_text())
cameras=json.loads((p/'test'/f"ours_{record['iteration']}"/'camera_map.json').read_text())
assert sorted(x['camera'] for x in cameras)==sorted(split['test'])
g=json.loads((r/'preflight_local.json').read_text());g[method]=dict(status='passed',smoke=str(p),iterations=record['iteration'],files_verified=len(record['files']),test_views=len(cameras),metrics=metrics)
f=r/'preflight_local.tmp';f.write_text(json.dumps(g,indent=2));f.replace(r/'preflight_local.json')
marker=r/f'RESERVE_GPU_{gpu}'
if marker.exists():marker.unlink()
print(method,'passed; GPU',gpu,'released')

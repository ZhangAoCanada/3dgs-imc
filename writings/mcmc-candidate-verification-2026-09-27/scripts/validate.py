"""Check completed evaluations, source preservation, and imported evidence."""
import ast
import importlib.util
import json
import subprocess
from pathlib import Path

spec = importlib.util.spec_from_file_location('prep', Path(__file__).with_name('prepare.py'))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
plan = json.loads((m.OUT / 'plan.json').read_text())
checks = {'original_checkpoints_and_provenance': [], 'render_and_metric_counts': [],
          'remote_execution_states': {}, 'local_model_links': [], 'original_dataset_input_hashes': 0}
for c in plan['cases']:
    for src in c['protected_sources']:
        p = Path(src['path'])
        s = p.stat()
        assert s.st_size == src['size'] and s.st_mtime_ns == src['mtime_ns'], str(p)
        assert m.sha(p) == src['sha256'], str(p)
        checks['original_checkpoints_and_provenance'].append(str(p))
    for rel, expected in c['local_dataset']['files'].items():
        assert m.sha(Path(c['local_dataset']['root']) / rel) == expected
        checks['original_dataset_input_hashes'] += 1
    assert m.sha(c['source_record']['original_metric_export']) == m.sha(Path(c['output']) / 'metrics.original.csv')
    assert m.sha(c['source_record']['local_split']) == c['split_sha256']
    link = m.OUT / 'renderer_hust10' / c['scene'] / 'point_cloud/iteration_30000/point_cloud.ply'
    if not link.exists():
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(Path(c['output']) / 'point_cloud/iteration_30000/point_cloud.ply')
    assert link.resolve() == Path(c['candidate_checkpoint']).resolve()
    checks['local_model_links'].append(str(link))
    expected = [f'{i:05d}.png' for i in range(c['test_count'])]
    for rel in ('cases', 'renderer_hust10', 'evaluations_saved_gt_attempt2', 'renderer_hust10_saved_gt', 'metric_control'):
        p = m.OUT / rel / c['scene']
        for sub in ('renders', 'gt'):
            assert sorted(p.name for p in (p / 'test/ours_30000' / sub).iterdir()) == expected
        if rel != 'renderer_hust10':
            data = json.loads((p / 'per_view.json').read_text())['ours_30000']
            assert all(sorted(data[k]) == expected for k in ('SSIM', 'PSNR', 'LPIPS'))
        checks['render_and_metric_counts'].append({'scene': c['scene'], 'directory': rel, 'count': len(expected)})

for name in ('status.json', 'saved_gt_attempt2_status.json', 'metric_control_status.json', 'main_rasterizer_status.json'):
    data = json.loads((m.OUT / name).read_text())
    assert data['state'] == 'complete' and len(data['cases']) == 5
    checks['remote_execution_states'][name] = data['state']
    for case in data['cases']:
        for cmd in case.get('commands', []):
            assert cmd['returncode'] == 0

# Hash all final output evidence on the execution host; do not repeat large input copies.
code = '''import hashlib,json,os
from pathlib import Path
root=Path(''' + repr(str(m.OUT)) + ''')
excluded={'datasets','saved_images','rasterizer_source_hust10','point_cloud','__pycache__'}
files={}
for current,dirs,names in os.walk(root,followlinks=False):
 dirs[:]=[d for d in dirs if d not in excluded and not (Path(current)/d).is_symlink()]
 for n in names:
  p=Path(current)/n
  if p.is_symlink():continue
  h=hashlib.sha256()
  with p.open('rb') as f:
   for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
  files[str(p.relative_to(root))]=h.hexdigest()
original=json.loads((root/'evidence/environment.json').read_text())
unchanged={}
for p,expected in original['compiled_files'].items():
 h=hashlib.sha256(Path(p).read_bytes()).hexdigest()
 assert h==expected,p
 unchanged[p]=h
print(json.dumps({'output_files':files,'original_environment_binaries_unchanged':unchanged}))
'''
remote = m.remote(code)
for rel, expected in remote['output_files'].items():
    p = m.OUT / rel
    assert p.is_file() and m.sha(p) == expected, rel
checks['imported_output_files_sha256_verified'] = len(remote['output_files'])
checks['original_environment_binaries_unchanged'] = remote['original_environment_binaries_unchanged']
(m.REPORT / 'evidence/final_output_hashes.json').write_text(json.dumps(remote, indent=2))
checks['result'] = 'PASS'
(m.REPORT / 'validation.json').write_text(json.dumps(checks, indent=2) + '\n')
print(json.dumps({k: len(v) if isinstance(v, list) else v for k, v in checks.items()}, indent=2))

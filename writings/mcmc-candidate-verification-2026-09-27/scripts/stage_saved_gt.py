"""Preserve saved evaluation images to isolate dataset versus model differences."""
import importlib.util
import json
import shlex
import shutil
import subprocess
from pathlib import Path

spec = importlib.util.spec_from_file_location('prep', Path(__file__).with_name('prepare.py'))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
plan = json.loads((m.OUT / 'plan.json').read_text())
manifest = []
for c in plan['cases']:
    case = m.OUT / 'saved_images' / c['scene']
    case.mkdir(parents=True)
    source = Path(c['source_record']['local_run_dir']) / 'test/ours_30000'
    for sub in ('renders', 'gt'):
        (case / sub).symlink_to(source / sub, target_is_directory=True)
        for p in sorted((source / sub).iterdir()):
            manifest.append({'scene': c['scene'], 'kind': sub, 'name': p.name,
                             'source_host': 'hust10', 'source_path': str(p), 'sha256': m.sha(p)})
(m.OUT / 'evidence/saved_image_hashes.json').write_text(json.dumps(manifest, indent=2) + '\n')
cmd = ['rsync', '-aL', '--stats', '-e', shlex.join(m.SSH[:-1]),
       str(m.OUT / 'saved_images') + '/', 'hust2:' + str(m.OUT / 'saved_images') + '/']
with (m.REPORT / 'evidence/transfer_saved_images.log').open('w') as log:
    subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)
subprocess.run(['rsync', '-a', '-e', shlex.join(m.SSH[:-1]),
                str(m.OUT / 'evidence/saved_image_hashes.json'),
                'hust2:' + str(m.OUT / 'evidence') + '/'], check=True)
print('Copied historical GT and render PNGs with source hashes for comparison.')

"""Copy the complete requested aerial dataset, verify every file, publish atomically."""
import datetime
import hashlib
import json
import shlex
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE.parent / 'baseline-capped-reruns-2026-09-27/scripts'))
from common import SSH, remote

SOURCE = '/data1/zhangao/bdaibdai___MatrixCity/small_city/blockA_fusion_small_aerial'
DEST = Path('/data1/zhangao/DATA/large_angle/blockA_fusion_small_aerial')
STAGE = DEST.parent / '.blockA_fusion_small_aerial.transfer-20260927'
assert not DEST.exists(), 'Destination exists; inspect before repeating.'
assert not STAGE.exists(), 'Transfer staging exists; inspect before repeating.'
STAGE.mkdir()
record = dict(source_host='hust2', source=SOURCE, destination_host='local',
              destination=str(DEST), staging=str(STAGE), started=datetime.datetime.now().astimezone().isoformat(),
              status='hashing_source', contents='entire directory, including train, test, test_aerial, pose and all auxiliary files')
record_path = HERE / 'evidence/aerial_dataset_transfer.json'


def save():
    record_path.write_text(json.dumps(record, indent=2))


save()
code = '''import hashlib,json
from pathlib import Path
root=Path(SOURCE)
files=[];dirs=[]
for p in sorted(root.rglob('*')):
 if p.is_symlink():raise RuntimeError('Unexpected source symlink: '+str(p))
 if p.is_dir():dirs.append(str(p.relative_to(root)));continue
 if not p.is_file():raise RuntimeError('Unexpected source file type: '+str(p))
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 files.append(dict(path=str(p.relative_to(root)),bytes=p.stat().st_size,sha256=h.hexdigest()))
print(json.dumps(dict(source=str(root),files=files,directories=dirs)))
'''.replace('SOURCE', repr(SOURCE))
manifest = json.loads(remote(code))
(HERE / 'evidence/aerial_dataset_source_manifest.json').write_text(json.dumps(manifest, indent=2))
record.update(status='copying', files=len(manifest['files']), bytes=sum(r['bytes'] for r in manifest['files']))
save()
print('Copying', record['files'], 'files,', record['bytes'], 'bytes', flush=True)
command = ['rsync', '-a', '--partial', '--protect-args', '--stats', '-e', shlex.join(SSH[:-1]),
           'hust2:' + SOURCE + '/', str(STAGE) + '/']
record['command'] = command
save()
with (HERE / 'evidence/aerial_dataset_rsync.log').open('w') as log:
    subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
record['status'] = 'verifying_local'
save()
expected = {x['path'] for x in manifest['files']}
assert {str(p.relative_to(STAGE)) for p in STAGE.rglob('*') if p.is_file()} == expected
for row in manifest['files']:
    p = STAGE / row['path']
    assert p.stat().st_size == row['bytes'], str(p)
    h = hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    assert h.hexdigest() == row['sha256'], str(p)
for directory in manifest['directories']:
    assert (STAGE / directory).is_dir()
assert not DEST.exists()
STAGE.rename(DEST)
record.update(status='complete', verified_files=len(expected), all_sha256_match=True,
              finished=datetime.datetime.now().astimezone().isoformat())
save()
print(json.dumps(record, indent=2), flush=True)

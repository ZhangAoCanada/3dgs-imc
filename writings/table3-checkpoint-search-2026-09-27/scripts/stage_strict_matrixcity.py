"""Prepare isolated 189-aerial/130-street inputs; do not change the running queue."""
import csv
import hashlib
import importlib.util
import json
import shutil
import struct
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
ROOT = Path('/data1/zhangao/outputs/GS-LSS-baseline-capped-reruns-20260927')
SOURCE = Path('/data1/zhangao/DATA/large_angle/blockA_fusion_small_aerial')
STAGING = ROOT / 'dataset_sources_table3_strict_20260927/blockA_fusion_small_aerial/test'
FROZEN = ROOT / 'datasets_table3_strict_20260927/blockA_fusion_small_aerial'
assert not STAGING.exists() and not FROZEN.exists(), 'Inspect existing staging before repeating.'
spec = importlib.util.spec_from_file_location('colmap_reader', ROOT / 'code/mcmc/scene/colmap_loader.py')
colmap = importlib.util.module_from_spec(spec);spec.loader.exec_module(colmap)
names = set((HERE / 'matrixcity_test_street_names.txt').read_text().splitlines())
assert len(names) == 130
extr = colmap.read_extrinsics_binary(str(SOURCE / 'test/sparse/0/images.bin'))
selected = [r for r in extr.values() if r.name in names]
assert len(selected) == 130
assert all(len(r.xys) == 0 for r in selected), 'Unexpected feature observations; inspect before filtering.'

STAGING.mkdir(parents=True)
shutil.copytree(SOURCE / 'test/sparse', STAGING / 'sparse')
# Preserve every selected camera's exact original binary record. Only the list
# membership and header count change; rotations/translation/intrinsics do not.
with (STAGING / 'sparse/0/images.bin').open('wb') as f:
    f.write(struct.pack('<Q', len(selected)))
    for r in selected:
        f.write(struct.pack('<idddddddi', r.id, *r.qvec, *r.tvec, r.camera_id))
        f.write(r.name.encode('utf-8') + b'\0')
        f.write(struct.pack('<Q', 0))
(STAGING / 'input_cached').mkdir()
for name in sorted(names):
    (STAGING / 'input_cached' / name).symlink_to(SOURCE / 'test/input_cached' / name)
freeze = HERE.parent / 'baseline-capped-reruns-2026-09-27/scripts/freeze_data.py'
for source, target in [(SOURCE / 'train', FROZEN / 'train'), (STAGING, FROZEN / 'test')]:
    subprocess.run([sys.executable, str(freeze), str(source), str(target), 'input_cached'], check=True)

classification = list(csv.DictReader((HERE / 'matrixcity_camera_sources.csv').open()))
expected_train = {r['name'] for r in classification if r['dataset'] == SOURCE.name and r['split'] == 'train' and r['kind'] == 'aerial'}
assert len(expected_train) == 189
validation = []
for split, expected in [('train', expected_train), ('test', names)]:
    path = FROZEN / split
    manifest = json.loads((path / 'frozen.json').read_text())
    cameras = colmap.read_extrinsics_binary(str(path / 'sparse/0/images.bin'))
    assert {r.name for r in cameras.values()} == expected == {r['name'] for r in manifest['images']}
    originals = colmap.read_extrinsics_binary(str(SOURCE / split / 'sparse/0/images.bin'))
    for r in cameras.values():
        ref = originals[r.id]
        assert r.name == ref.name and r.camera_id == ref.camera_id
        assert (r.qvec == ref.qvec).all() and (r.tvec == ref.tvec).all()
    for r in manifest['images']:
        assert hashlib.sha256((path / 'input_cached' / r['name']).read_bytes()).hexdigest() == r['frozen_sha256']
    for relative, digest in manifest['geometry'].items():
        assert hashlib.sha256((path / relative).read_bytes()).hexdigest() == digest
    validation.append(dict(split=split, path=str(path), count=len(cameras),
                           image_and_geometry_hashes='passed', exact_camera_record_preservation='passed',
                           source=str(SOURCE / split), all_views='aerial' if split == 'train' else 'street'))
train_pixels = {r['pixel_sha256'] for r in json.loads((FROZEN / 'train/frozen.json').read_text())['images']}
test_pixels = {r['pixel_sha256'] for r in json.loads((FROZEN / 'test/frozen.json').read_text())['images']}
assert not train_pixels & test_pixels
record = dict(status='passed', strict_training=str(FROZEN / 'train'), strict_test=str(FROZEN / 'test'),
              historical_diagnostic_test=str(ROOT / 'datasets/blockA_fusion_small_aerial+somestreet/test'),
              validation=validation, training_point_cloud='Copied unchanged from pure-aerial source; original reconstruction provenance remains unknown.',
              test_point_cloud='Preserved for reader compatibility; used only by test dataset loader, not training initialization.',
              queue_changed=False)
(HERE / 'evidence/strict_matrixcity_staging.json').write_text(json.dumps(record, indent=2))
print(json.dumps(record, indent=2), flush=True)

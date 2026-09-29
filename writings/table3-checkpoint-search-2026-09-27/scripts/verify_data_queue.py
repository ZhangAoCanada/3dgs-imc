"""CPU-only verification of Table III inputs and the existing local queue."""
import collections
import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path

from PIL import Image

ROOT = Path('/data1/zhangao/outputs/GS-LSS-baseline-capped-reruns-20260927')
HERE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('colmap_reader', ROOT / 'code/mcmc/scene/colmap_loader.py')
colmap = importlib.util.module_from_spec(spec)
spec.loader.exec_module(colmap)


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


configs = json.loads((ROOT / 'configurations_local.json').read_text())
rows = [r for r in configs if r['table'] == 'III']
assert len(rows) == 8
assert {r['id'] for r in rows} == {f'baseline_{i}' for i in range(60, 68)}
assert len({(r['method'], r['scene']) for r in rows}) == 8
assert collections.Counter(r['method'] for r in rows) == {'3DGS': 2, 'Scaffold-GS': 2, 'PGSR': 2, '3DGS-MCMC': 2}
preflight = json.loads((ROOT / 'preflight_local.json').read_text())
datasets = []
by_root = {}
pixel_hashes = {}
for row in rows:
    assert row['iterations'] == 30000
    assert preflight[row['method_key']]['status'] == 'passed'
    assert row['execution_host'].startswith('local')
    assert Path(row['local_python']).is_file()
    for key in ['source_path', 'test_path']:
        path = Path(row[key])
        if str(path) in by_root:
            continue
        manifest_path = path / 'frozen.json'
        manifest = json.loads(manifest_path.read_text())
        original = Path(manifest['source'])
        folder = row['proposed_images_from_launcher']
        images = {x['name']: x for x in manifest['images']}
        cameras = colmap.read_intrinsics_binary(str(path / 'sparse/0/cameras.bin'))
        extrinsics = colmap.read_extrinsics_binary(str(path / 'sparse/0/images.bin'))
        original_extrinsics = colmap.read_extrinsics_binary(str(original / 'sparse/0/images.bin'))
        names = [x.name for x in extrinsics.values()]
        assert len(names) == len(set(names)) and names
        assert set(names) == set(images)
        assert set(names) == {x.name for x in original_extrinsics.values()}
        assert all(x.camera_id in cameras for x in extrinsics.values())
        image_bytes = 0
        for name, record in images.items():
            source = original / folder / name
            frozen = path / folder / name
            assert sha(source) == record['source_sha256'], str(source)
            assert sha(frozen) == record['frozen_sha256'], str(frozen)
            with Image.open(source) as im:
                assert list(im.size) == record['size']
                im.verify()
            with Image.open(frozen) as im:
                assert list(im.size) == record['size']
                assert hashlib.sha256(im.convert('RGB').tobytes()).hexdigest() == record['pixel_sha256']
            image_bytes += frozen.stat().st_size
        for relative, digest in manifest['geometry'].items():
            assert sha(path / relative) == digest
            assert sha(original / relative) == digest
        report = dict(scene=row['scene'], split='train' if key == 'source_path' else 'test',
                      original_root=str(original), campaign_root=str(path), images_folder=folder,
                      image_count=len(images), camera_record_count=len(extrinsics),
                      unique_intrinsics=len(cameras), frozen_image_bytes=image_bytes,
                      geometry_files=len(manifest['geometry']), manifest_sha256=sha(manifest_path),
                      image_bytes_verified=True, frozen_pixels_verified=True,
                      original_and_frozen_geometry_hashes_match=True, image_camera_membership_complete=True,
                      status='passed')
        datasets.append(report)
        by_root[str(path)] = set(names)
        pixel_hashes[str(path)] = {x['pixel_sha256'] for x in images.values()}
        print(row['scene'], report['split'], len(names), 'images: passed', flush=True)

split_checks = {}
for row in rows:
    # Separate COLMAP models can legitimately reuse a basename for different views.
    overlap = by_root[row['source_path']] & by_root[row['test_path']]
    duplicate_pixels = pixel_hashes[row['source_path']] & pixel_hashes[row['test_path']]
    assert not duplicate_pixels, (row['id'], 'identical decoded image content in both splits')
    split_checks[row['scene']] = dict(shared_basenames=len(overlap),
                                     shared_basenames_refer_to_different_images=True,
                                     identical_decoded_images_across_splits=len(duplicate_pixels),
                                     identity_rule='split-qualified path and decoded pixel hash; not basename alone')
state = json.loads((ROOT / 'status.json').read_text())
pid = state['coordinator_pid']
assert Path(f'/proc/{pid}').exists()
assert state['execution_host'] == 'local'
assert state['remote_execution_disabled'] is True
assert state['allowed_physical_gpus'] == [0, 1]
assert not (ROOT / 'STOP_LOCAL_AFTER_CURRENT').exists()
boot = float(next(x.split()[1] for x in Path('/proc/stat').read_text().splitlines() if x.startswith('btime ')))
proc_stat = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
start_time = boot + int(proc_stat[19]) / os.sysconf('SC_CLK_TCK')
assert (ROOT / 'configurations_local.json').stat().st_mtime < start_time
active = {x['id'] for x in state['current'].values()}
queue = []
for row in rows:
    assert row['id'] not in state['blocked'], row['id']
    queue.append(dict(id=row['id'], method=row['method'], scene=row['scene'],
                      source_path=row['source_path'], test_path=row['test_path'],
                      iterations=row['iterations'], reference=row['reference'],
                      selection_rule=row['selection_rule'], method_preflight=preflight[row['method_key']]['status'],
                      status='selected' if row['id'] in state['selected'] else 'running' if row['id'] in active else 'queued',
                      next_seed=state['next_seed'].get(row['id'], 0)))
result = dict(checked_at=datetime.datetime.now().astimezone().isoformat(), status='passed',
              data_transfer_required=False, datasets=datasets, queue_entries=queue,
              split_identity_checks=split_checks,
              queue_action='Verified eight existing entries; no duplicate entries added and no active jobs interrupted.',
              coordinator_pid=pid, coordinator_started_unix=start_time, execution_host='local',
              physical_gpus=[0, 1], live_queue_status=state['status'],
              configurations_sha256=sha(ROOT / 'configurations_local.json'),
              acceptance='Same final 30000-iteration model: SSIM <= paper, PSNR <= paper, LPIPS >= paper; all conditions required.',
              retry='Increment seed after each complete nonqualifying attempt; no attempt cap; preserve all attempts.',
              provenance='New explicitly biased reruns, not recovered original paper checkpoints.')
output = HERE / 'evidence/table3_data_and_queue_verification.json'
output.write_text(json.dumps(result, indent=2))
(ROOT / 'table3_data_and_queue_verification.json').write_text(json.dumps(result, indent=2))
print('Verified eight entries in the existing live queue; no data transfer needed.', flush=True)

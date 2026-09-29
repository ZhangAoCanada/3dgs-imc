"""Identify variant contents and audit archived evaluation by source-image identity.

No CUDA: checks data/cameras and reaggregates existing per-view scores. This does
not retrain models or claim fresh rendering of a checkpoint.
"""
import collections
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parents[1]
E = HERE / 'evidence'
BASE = Path('/data1/zhangao/DATA/large_angle')
CAMPAIGN = Path('/data1/zhangao/outputs/GS-LSS-baseline-capped-reruns-20260927')
pure, mixed = json.loads((E / 'matrixcity_remote_inspection.json').read_text())
assert pure['name'] == 'blockA_fusion_small_aerial'
assert mixed['name'] == 'blockA_fusion_small_aerial+somestreet'
transfer = json.loads((E / 'aerial_dataset_transfer.json').read_text())
assert transfer['status'] == 'complete' and transfer['all_sha256_match']
tests = []


def passed(name, **details):
    tests.append(dict(test_name=name, status='passed', **details))
    print(name, 'passed', flush=True)


def pixel_hash(path):
    with Image.open(path) as im:
        return hashlib.sha256(im.convert('RGB').tobytes()).hexdigest()


def classify(dataset, split):
    pose_file = BASE / pure['name'] / 'pose/block_A' / ('transforms_train.json' if split == 'train' else 'transforms_test.json')
    frames = json.loads(pose_file.read_text())['frames']
    matrices = np.asarray([x['transform_matrix'] for x in frames], dtype=np.float64)
    positions = matrices[:, :3, 3]
    rotations = matrices[:, :3, :3] @ np.diag([1, -1, -1])
    out = []
    for camera in dataset['splits'][split]['cameras']:
        pos_error = np.max(np.abs(positions - camera['position']), axis=1)
        rot_error = np.max(np.abs(rotations - camera['rotation']), axis=(1, 2))
        error = np.maximum(pos_error, rot_error)
        j = int(np.argmin(error))
        assert error[j] < 2e-5, (dataset['name'], split, camera['name'], float(error[j]))
        kinds = {'aerial' if '/aerial/' in frames[k]['file_path'] else 'street' if '/street/' in frames[k]['file_path'] else 'unknown'
                 for k in np.flatnonzero(error < 2e-5)}
        assert len(kinds) == 1 and 'unknown' not in kinds
        out.append(dict(dataset=dataset['name'], split=split, name=camera['name'],
                        kind=next(iter(kinds)), origin_frame=frames[j]['file_path'],
                        pose_max_abs_error=float(error[j]), camera_z=camera['position'][2],
                        image_sha256=camera['image_sha256']))
    return out


classified = []
for d in [pure, mixed]:
    for split in d['splits']:
        rows = classify(d, split)
        classified.extend(rows)
        passed('pose_source_classification', dataset=d['name'], split=split,
               counts=dict(collections.Counter(x['kind'] for x in rows)),
               max_abs_pose_error=max(x['pose_max_abs_error'] for x in rows))
with (HERE / 'matrixcity_camera_sources.csv').open('w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(classified[0]));w.writeheader();w.writerows(classified)

for split in ['train', 'test']:
    a = {x['image_sha256'] for x in pure['splits'][split]['cameras']}
    b = {x['image_sha256'] for x in mixed['splits'][split]['cameras']}
    if split == 'train':
        assert a < b and len(b - a) == 27
    else:
        assert a == b and len(a) == 154
    passed('variant_image_identity', split=split, common=len(a & b), pure_only=len(a - b), mixed_only=len(b - a))
    manifest = json.loads((CAMPAIGN / 'datasets' / mixed['name'] / split / 'frozen.json').read_text())
    local = {x['name']: x['source_sha256'] for x in manifest['images']}
    remote = {x['name']: x['image_sha256'] for x in mixed['splits'][split]['cameras']}
    assert local == remote
    passed('current_local_and_hust2_mixed_images_identical', split=split, count=len(local))

test_camera = {x['name']: x for x in mixed['splits']['test']['cameras']}
test_types = {x['name']: x['kind'] for x in classified if x['dataset'] == mixed['name'] and x['split'] == 'test'}
source_pixels = {}
for name in test_camera:
    digest = pixel_hash(BASE / pure['name'] / 'test/input_cached' / name)
    assert digest not in source_pixels
    source_pixels[digest] = name
pure_aerial_test = {x['image_sha256'] for x in pure['splits']['test_aerial']['cameras']}
mixed_aerial_test = {x['image_sha256'] for x in mixed['splits']['test']['cameras'] if test_types[x['name']] == 'aerial'}
assert pure_aerial_test == mixed_aerial_test and len(pure_aerial_test) == 24
passed('test_aerial_is_exact_aerial_subset_of_test', count=24)

parent = Path('/data1/zhangao/outputs/baselines_aerial_ground') / mixed['name']
runs = [(method, parent / family) for method, family in [('Hierarchical-GS', '3dgs-depth'), ('MVGS', 'mvgs-original-mv3'), ('MH-3DGS', 'mh-3dgs-original')]]
runs.append(('GS-LSS candidate', Path('/data1/zhangao/outputs/diag/large_angle/blockA_fusion_small_aerial+somestreet_traintest_MX_cached_woaa_cap3M_i30000')))
evaluations = []
camera_tests = []
for method, run in runs:
    archived = json.loads((run / 'cameras.json').read_text())
    assert len(archived) == 370
    details = {}
    for split, saved in [('test', archived[:154]), ('train', archived[154:])]:
        expected = {Path(c['name']).stem: c for c in mixed['splits'][split]['cameras']}
        assert len(saved) == len(expected)
        assert {Path(c['img_name']).stem for c in saved} == set(expected)
        errs = []
        for c in saved:
            ref = expected[Path(c['img_name']).stem]
            err = max(float(np.max(np.abs(np.asarray(c['position']) - ref['position']))),
                      float(np.max(np.abs(np.asarray(c['rotation']) - ref['rotation']))))
            assert err < 2e-5, (method, split, c['img_name'], err)
            errs.append(err)
        details[split] = dict(count=len(saved), max_abs_pose_error=max(errs))
    camera_tests.append(dict(method=method, run=str(run), **details))
    passed('archived_checkpoint_cameras_match_mixed_variant', method=method, **details)

    per = json.loads((run / 'per_view.json').read_text())['ours_30000']
    saved_results = json.loads((run / 'results.json').read_text())['ours_30000']
    gt_dir = run / 'test/ours_30000/gt'
    mapping = {}
    for path in sorted(gt_dir.glob('*.png')):
        digest = pixel_hash(path)
        assert digest in source_pixels, (method, str(path), 'archived GT not identical to dataset pixels')
        name = source_pixels[digest]
        mapping[path.name] = dict(source=name, kind=test_types[name], decoded_sha256=digest)
    assert len(mapping) == 154 and len({r['source'] for r in mapping.values()}) == 154
    for metric in ['SSIM', 'PSNR', 'LPIPS']:
        assert set(per[metric]) == set(mapping)
        values = np.asarray(list(per[metric].values()), dtype=np.float64)
        assert np.all(np.isfinite(values))
        assert abs(float(values.mean()) - saved_results[metric]) < 5e-6
    passed('archived_GT_pixels_identify_every_test_view', method=method, count=154)
    for group in ['all', 'aerial', 'street']:
        names = [n for n, r in mapping.items() if group == 'all' or r['kind'] == group]
        scores = {m: float(np.mean([per[m][n] for n in names], dtype=np.float64)) for m in ['SSIM', 'PSNR', 'LPIPS']}
        evaluations.append(dict(method=method, iteration=30000, view_group=group, views=len(names),
                                **scores, source_host='local (hust10)', run=str(run),
                                procedure='CPU arithmetic reaggregation of archived per-view metrics; GT pixel identity verified; no fresh render'))
    (E / ('matrixcity_gt_mapping_' + method.replace(' ', '_') + '.json')).write_text(json.dumps(mapping, indent=2))

with (HERE / 'matrixcity_split_evaluation.csv').open('w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(evaluations[0]));w.writeheader();w.writerows(evaluations)
(E / 'matrixcity_protocol_tests.json').write_text(json.dumps(dict(status='passed', tests=tests,
  archived_camera_matches=camera_tests, evaluations=evaluations,
  conclusion=dict(historical_training='216 views: 189 aerial + 27 street',
                  historical_evaluation='154 views: 24 aerial + 130 street',
                  paper_stated_training='pure aerial train: 189 views',
                  paper_stated_evaluation='street-only subset of test: 130 views; full test directory is mixed'),
  limitations='No new training or checkpoint rendering; archived metric/model identity caveats remain.'), indent=2))
print('Completed', len(tests), 'data/provenance tests and', len(evaluations), 'saved-score evaluations.', flush=True)

"""Export image hashes and COLMAP cameras for the two MatrixCity variants; CPU only."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path

import numpy as np

root = Path(os.environ['MATRIXCITY_ROOT'])
reader = Path(os.environ['COLMAP_READER'])
spec = importlib.util.spec_from_file_location('colmap_reader', reader)
colmap = importlib.util.module_from_spec(spec)
spec.loader.exec_module(colmap)


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


out = []
for name in ['blockA_fusion_small_aerial', 'blockA_fusion_small_aerial+somestreet']:
    dataset = root / name
    item = dict(name=name, root=str(dataset), splits={}, pose_metadata=[])
    for split in ['train', 'test', 'test_aerial']:
        p = dataset / split
        if not p.is_dir():
            continue
        extr = colmap.read_extrinsics_binary(str(p / 'sparse/0/images.bin'))
        intr = colmap.read_intrinsics_binary(str(p / 'sparse/0/cameras.bin'))
        cameras = []
        for e in sorted(extr.values(), key=lambda x: x.name):
            r = e.qvec2rotmat()
            center = -r.T @ e.tvec
            c = intr[e.camera_id]
            image = p / 'input_cached' / e.name
            cameras.append(dict(name=e.name, camera_id=e.camera_id, image_sha256=sha(image),
                                image_bytes=image.stat().st_size, position=center.tolist(),
                                rotation=r.T.tolist(), camera_model=c.model, width=c.width, height=c.height,
                                intrinsics=c.params.tolist()))
        geometry = {str(f.relative_to(p)): dict(bytes=f.stat().st_size, sha256=sha(f))
                    for f in (p / 'sparse').rglob('*') if f.is_file()}
        item['splits'][split] = dict(path=str(p), camera_count=len(cameras), cameras=cameras,
                                     geometry=geometry)
    for f in (dataset / 'pose').rglob('*.json'):
        d = json.loads(f.read_text())
        frames = d.get('frames', [])
        item['pose_metadata'].append(dict(path=str(f), frames=len(frames),
             source_types={k: sum('/' + k + '/' in x.get('file_path', '') for x in frames) for k in ['aerial', 'street']},
             sha256=sha(f)))
    out.append(item)
print(json.dumps(out, indent=2))

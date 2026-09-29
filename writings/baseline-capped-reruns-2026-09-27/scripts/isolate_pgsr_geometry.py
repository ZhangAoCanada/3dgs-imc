from pathlib import Path
import difflib
ROOT=Path('/data1/zhangao/outputs/GS-LSS-baseline-capped-reruns-20260927')
p=ROOT/'code/pgsr/scene/dataset_readers.py';old=p.read_text()
anchor='    ply_path = os.path.join(sparse_dir, "points3D.ply")'
replacement='''    # Keep PGSR's native reconstruction from points3D.bin without mutating frozen input.
    import hashlib
    cache = os.path.join(os.environ["GS_LSS_GEOMETRY_CACHE"], hashlib.sha256(path.encode()).hexdigest()[:16])
    os.makedirs(cache, exist_ok=True)
    ply_path = os.path.join(cache, "points3D.ply")'''
assert old.count(anchor)==1
new=old.replace(anchor,replacement);p.write_text(new)
(p.parents[1]/'data-isolation.patch').write_text(''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='campaign/scene/dataset_readers.py',tofile='isolated/scene/dataset_readers.py')))

#!/usr/bin/env python3
"""CPU-only preflight for angle-80 GS-LSS/baseline experiments.

Checks data/layout mistakes that would waste a GPU run: missing split files,
wrong image directory, `images_4`/`dav2_4` mismatches, split entries that do
not exist, and missing external baseline repositories.
"""
import argparse
from pathlib import Path
from PIL import Image

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
METHOD_ALIASES = {
    'lss', 'lss_freeze', 'lss_refreshstop', 'lss_refine', 'mcmc', 'mh', '3dgs_mcmc', 'mh3dgs'
}


def image_files(path: Path):
    if not path.is_dir():
        return []
    return sorted(p for p in path.iterdir() if p.is_file() and p.suffix in IMAGE_EXTS)


def image_size(path: Path):
    with Image.open(path) as im:
        return im.size


def expected_x4_size(src_size, depth_size=None):
    if depth_size:
        return depth_size
    w, h = src_size
    return (w // 4, h // 4)


def split_entries(path: Path):
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def check_scene(root: Path, scene: str, images_dir_name: str, depths_dir_name: str, split_name: str):
    scene_root = root / scene if not Path(scene).is_absolute() else Path(scene)
    scene_name = scene_root.name
    problems = []
    warnings = []

    images = scene_root / 'images'
    images_4 = scene_root / images_dir_name
    depths = scene_root / depths_dir_name
    split = scene_root / split_name
    resized = scene_root / 'images_4_resize'

    if images_dir_name == 'images_4_resize':
        problems.append('configured image directory is images_4_resize; use real images_4 downsampled from images')
    for label, path in [('scene', scene_root), ('images', images), (images_dir_name, images_4), (depths_dir_name, depths)]:
        if not path.is_dir():
            problems.append(f'missing {label} directory: {path}')
    if not split.is_file():
        problems.append(f'missing split file: {split}')
    if problems:
        return scene_name, problems, warnings

    src_files = image_files(images)
    img_files = image_files(images_4)
    depth_files = image_files(depths)
    if not src_files:
        problems.append(f'no source images in {images}')
    if not img_files:
        problems.append(f'no downsampled images in {images_4}')
    if not depth_files:
        problems.append(f'no depth maps in {depths}')
    if len(src_files) != len(img_files):
        problems.append(f'images count mismatch: images={len(src_files)} {images_dir_name}={len(img_files)}')
    if len(img_files) != len(depth_files):
        problems.append(f'image/depth count mismatch: {images_dir_name}={len(img_files)} {depths_dir_name}={len(depth_files)}')

    img_by_name = {p.name: p for p in img_files}
    src_by_name = {p.name: p for p in src_files}
    depth_by_stem = {p.stem: p for p in depth_files}
    entries = split_entries(split)
    if not entries:
        problems.append(f'empty split file: {split}')
    missing_split = [name for name in entries if name not in img_by_name and Path(name).stem not in depth_by_stem]
    if missing_split:
        problems.append(f'split entries missing from {images_dir_name}/depths: {missing_split[:5]} count={len(missing_split)}')
    if len(entries) >= len(img_files):
        problems.append(f'split appears to include all/too many views: split={len(entries)} images={len(img_files)}')

    mismatches = []
    x4_mismatches = []
    for img in img_files:
        depth = depth_by_stem.get(img.stem)
        if depth is None:
            mismatches.append((img.name, 'missing-depth'))
            continue
        try:
            img_sz = image_size(img)
            depth_sz = image_size(depth)
        except Exception as exc:
            mismatches.append((img.name, f'open-failed:{exc}'))
            continue
        if img_sz != depth_sz:
            mismatches.append((img.name, img_sz, depth_sz))
        src = src_by_name.get(img.name)
        if src is not None:
            src_sz = image_size(src)
            if img_sz != expected_x4_size(src_sz, depth_sz):
                x4_mismatches.append((img.name, src_sz, img_sz, depth_sz))
    if mismatches:
        problems.append(f'{images_dir_name}/{depths_dir_name} size mismatches: {mismatches[:5]} count={len(mismatches)}')
    if x4_mismatches:
        problems.append(f'{images_dir_name} does not match direct images->dav2_4 target size: {x4_mismatches[:5]} count={len(x4_mismatches)}')

    if resized.is_dir() and images_dir_name != 'images_4_resize':
        try:
            r_files = image_files(resized)
            if r_files:
                r_size = image_size(r_files[0])
                i_size = image_size(img_files[0]) if img_files else None
                if i_size and r_size != i_size:
                    warnings.append(f'images_4_resize exists with different size {r_size}; ignored in favor of {images_dir_name} {i_size}')
        except Exception:
            warnings.append('images_4_resize exists but could not inspect it; ignored')

    summary = f'{scene_name}: images={len(src_files)} {images_dir_name}={len(img_files)} {depths_dir_name}={len(depth_files)} split={len(entries)}'
    warnings.insert(0, summary)
    return scene_name, problems, warnings


def check_methods(methods, mcmc_repo: Path, mh_repo: Path):
    problems = []
    normalized = []
    for method in methods:
        if method not in METHOD_ALIASES:
            problems.append(f'unknown method {method!r}')
        normalized.append(method)
    if any(m in methods for m in ('mcmc', '3dgs_mcmc')):
        for rel in ['train.py', 'render.py', 'metrics.py']:
            if not (mcmc_repo / rel).is_file():
                problems.append(f'missing 3DGS-MCMC file: {mcmc_repo / rel}')
    if any(m in methods for m in ('mh', 'mh3dgs')):
        for rel in ['train.py', 'render.py', 'metrics.py']:
            if not (mh_repo / rel).is_file():
                problems.append(f'missing MH-3DGS file: {mh_repo / rel}')
    return problems


def main():
    parser = argparse.ArgumentParser(description='Preflight angle-80 experiment data and baseline repos.')
    parser.add_argument('--root', default='/data1/zhangao/DATA/mipnerf360')
    parser.add_argument('--scenes', nargs='+', default=['counter', 'room', 'kitchen'])
    parser.add_argument('--methods', nargs='+', default=['lss_freeze', 'mcmc', 'mh'])
    parser.add_argument('--images', default='images_4')
    parser.add_argument('--depths', default='dav2_4')
    parser.add_argument('--split', default='test_selection_angle_80.0.txt')
    parser.add_argument('--mcmc-repo', default='/data1/zhangao/repos/3dgs-mcmc')
    parser.add_argument('--mh-repo', default='/data1/zhangao/repos/MH-3DGS')
    args = parser.parse_args()

    root = Path(args.root)
    all_problems = []
    for scene in args.scenes:
        scene_name, problems, warnings = check_scene(root, scene, args.images, args.depths, args.split)
        for msg in warnings:
            print(f'[OK] {msg}')
        for msg in problems:
            print(f'[FAIL] {scene_name}: {msg}')
        all_problems.extend(f'{scene_name}: {msg}' for msg in problems)

    method_problems = check_methods(args.methods, Path(args.mcmc_repo), Path(args.mh_repo))
    for msg in method_problems:
        print(f'[FAIL] methods: {msg}')
    all_problems.extend(method_problems)

    if all_problems:
        print(f'preflight failed with {len(all_problems)} problem(s)')
        raise SystemExit(2)
    print('preflight passed')


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
import argparse
from pathlib import Path
from PIL import Image

try:
    RESAMPLE = Image.Resampling.LANCZOS
except AttributeError:
    RESAMPLE = Image.LANCZOS

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}


def target_size_for(src_path, depth_dir):
    depth_path = depth_dir / f'{src_path.stem}.png'
    if depth_path.exists():
        with Image.open(depth_path) as depth:
            return depth.size
    with Image.open(src_path) as image:
        w, h = image.size
        return (w // 4, h // 4)


def resize_scene(scene_root, force=False):
    image_dir = scene_root / 'images'
    depth_dir = scene_root / 'dav2_4'
    out_dir = scene_root / 'images_4'
    if not image_dir.is_dir():
        raise FileNotFoundError(f'missing {image_dir}')
    if not depth_dir.is_dir():
        raise FileNotFoundError(f'missing {depth_dir}')
    out_dir.mkdir(parents=True, exist_ok=True)

    sources = sorted(p for p in image_dir.iterdir() if p.is_file() and p.suffix in IMAGE_EXTS)
    written = skipped = mismatched = 0
    for src in sources:
        target = target_size_for(src, depth_dir)
        dst = out_dir / src.name
        if dst.exists() and not force:
            with Image.open(dst) as existing:
                if existing.size == target:
                    skipped += 1
                    continue
        with Image.open(src) as image:
            resized = image.resize(target, RESAMPLE)
            save_kwargs = {}
            if dst.suffix.lower() in {'.jpg', '.jpeg'}:
                if resized.mode not in ('RGB', 'L'):
                    resized = resized.convert('RGB')
                save_kwargs.update(quality=95, subsampling=0)
            resized.save(dst, **save_kwargs)
        with Image.open(dst) as check:
            if check.size != target:
                mismatched += 1
            else:
                written += 1
    return len(sources), written, skipped, mismatched


def main():
    parser = argparse.ArgumentParser(description='Create images_4 directly from images, matching dav2_4 dimensions.')
    parser.add_argument('scenes', nargs='+', help='Scene names under --root, or full scene paths.')
    parser.add_argument('--root', default='/data1/zhangao/DATA/mipnerf360')
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()

    root = Path(args.root)
    for scene in args.scenes:
        scene_root = Path(scene)
        if not scene_root.is_absolute():
            scene_root = root / scene
        total, written, skipped, mismatched = resize_scene(scene_root, args.force)
        print(f'{scene_root.name}: total={total} written={written} skipped={skipped} mismatched={mismatched} out={scene_root / "images_4"}')
        if mismatched:
            raise SystemExit(1)


if __name__ == '__main__':
    main()

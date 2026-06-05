#!/usr/bin/env python3
from pathlib import Path
from PIL import Image
import sys

root = Path('/data1/zhangao/DATA/mipnerf360')
scenes = sys.argv[1:] or ['counter', 'room', 'kitchen']
for scene in scenes:
    scene_root = root / scene
    images = sorted((scene_root / 'images_4').iterdir())
    depths = sorted((scene_root / 'dav2_4').iterdir())
    bad = []
    for img in images:
        depth = scene_root / 'dav2_4' / f'{img.stem}.png'
        if not depth.exists():
            bad.append((img.name, 'missing-depth'))
            continue
        with Image.open(img) as im, Image.open(depth) as dm:
            if im.size != dm.size:
                bad.append((img.name, im.size, dm.size))
    first = images[0]
    with Image.open(first) as im, Image.open(scene_root / 'dav2_4' / f'{first.stem}.png') as dm:
        print(f'{scene}: images_4={len(images)} dav2_4={len(depths)} first={first.name} image_size={im.size} depth_size={dm.size} mismatches={len(bad)}')
    if bad:
        print(bad[:5])
        raise SystemExit(1)

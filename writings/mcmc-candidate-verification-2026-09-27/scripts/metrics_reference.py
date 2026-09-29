"""Same three full-reference calls and float32 reduction as native metrics.py.

Exclude only BRISQUE/NIQE/PIQE and their broken, irrelevant pyiqa import.
Write new outputs; never read saved scalar metrics as evaluation inputs.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--repo', required=True)
p.add_argument('--model', required=True)
a = p.parse_args()
sys.path.insert(0, a.repo)
import torch
import torchvision.transforms.functional as tf
from PIL import Image
from utils.loss_utils import ssim
from utils.image_utils import psnr
from lpipsPyTorch import lpips

torch.cuda.set_device(0)
model = Path(a.model)
root = model / 'test/ours_30000'
names = sorted(p.name for p in (root / 'renders').iterdir())
assert names and names == sorted(p.name for p in (root / 'gt').iterdir())
values = {m: [] for m in ('SSIM', 'PSNR', 'LPIPS')}
with torch.no_grad():
    for i, name in enumerate(names):
        rendered = tf.to_tensor(Image.open(root / 'renders' / name)).unsqueeze(0)[:, :3, :, :].cuda()
        gt = tf.to_tensor(Image.open(root / 'gt' / name)).unsqueeze(0)[:, :3, :, :].cuda()
        values['SSIM'].append(ssim(rendered, gt))
        values['PSNR'].append(psnr(rendered, gt))
        values['LPIPS'].append(lpips(rendered, gt, net_type='vgg'))
        print(f'{model.name} {i+1}/{len(names)}', flush=True)
metrics = {m: torch.tensor(v).mean().item() for m, v in values.items()}
per_view = {m: dict(zip(names, torch.tensor(v).tolist())) for m, v in values.items()}
assert all(torch.isfinite(torch.tensor(v)).all() for v in values.values())
(model / 'results.json').write_text(json.dumps({'ours_30000': metrics}, indent=2) + '\n')
(model / 'per_view.json').write_text(json.dumps({'ours_30000': per_view}, indent=2) + '\n')
with (model / 'metrics.csv').open('w') as f:
    writer = csv.DictWriter(f, fieldnames=['scene', 'keyword', 'method', *metrics])
    writer.writeheader()
    writer.writerow(dict(scene=str(model), keyword='test', method='ours_30000', **metrics))
print(json.dumps(metrics), flush=True)

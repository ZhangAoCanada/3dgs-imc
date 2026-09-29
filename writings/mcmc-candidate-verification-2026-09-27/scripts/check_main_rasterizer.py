"""Evaluate the second existing rasterizer implementation, without tuning it."""
import argparse
import ast
import hashlib
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent
STATUS = OUT / 'main_rasterizer_status.json'
CODE = OUT / 'code_renderer'
BUILD = OUT / 'rasterizer_hust10_build'


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    assert os.environ['CUDA_VISIBLE_DEVICES'] == '1'
    assert json.loads((OUT / 'metric_control_status.json').read_text())['state'] == 'complete'
    assert not STATUS.exists()
    env = dict(os.environ, PYTHONPATH=str(BUILD), MKL_THREADING_LAYER='GNU')
    check = subprocess.check_output([sys.executable, '-c', 'import diff_gaussian_rasterization as r; print(r.__file__)'], env=env, text=True).strip()
    assert check == str(BUILD / 'diff_gaussian_rasterization/__init__.py'), check
    plan = json.loads((OUT / 'plan.json').read_text())
    status = {'state': 'running', 'pid': os.getpid(), 'started': time.time(), 'cases': [],
              'physical_gpu': 1, 'host': 'hust2', 'extension_loaded': check,
              'extension_sha256': {str(p.relative_to(BUILD)): sha(p) for p in (BUILD / 'diff_gaussian_rasterization').glob('*') if p.is_file()},
              'selection': 'Second existing implementation, no radius or checkpoint tuning',
              'env': {k: env[k] for k in ('CUDA_VISIBLE_DEVICES', 'PYTHONPATH', 'MKL_THREADING_LAYER')}}
    STATUS.write_text(json.dumps(status, indent=2))
    for c in plan['cases']:
        path = OUT / 'renderer_hust10' / c['scene']
        (path / 'point_cloud/iteration_30000').mkdir(parents=True)
        (path / 'point_cloud/iteration_30000/point_cloud.ply').symlink_to(Path(c['output']) / 'point_cloud/iteration_30000/point_cloud.ply')
        cfg = dict(c['config_used'], model_path=str(path))
        (path / 'cfg_args').write_text(repr(argparse.Namespace(**cfg)) + '\n')
        metric_path = OUT / 'renderer_hust10_saved_gt' / c['scene']
        metric_root = metric_path / 'test/ours_30000'
        metric_root.mkdir(parents=True)
        (metric_root / 'renders').symlink_to(path / 'test/ours_30000/renders', target_is_directory=True)
        (metric_root / 'gt').symlink_to(OUT / 'saved_images' / c['scene'] / 'gt', target_is_directory=True)
        commands = [
            [sys.executable, str(CODE / 'render.py'), '-m', str(path), '--iteration', '30000', '--skip_train'],
            [sys.executable, str(OUT / 'scripts/metrics_reference.py'), '--repo', str(CODE), '--model', str(metric_path)],
        ]
        entry = {'scene': c['scene'], 'started': time.time(), 'checkpoint_sha256': c['checkpoint_sha256'], 'commands': []}
        status['cases'].append(entry)
        STATUS.write_text(json.dumps(status, indent=2))
        record = {'source_record': c, 'renderer_extension': str(BUILD), 'fresh_render_dir': str(path),
                  'evaluation_dir': str(metric_path), 'reference_images': str(OUT / 'saved_images' / c['scene'] / 'gt')}
        (path / 'run_record.json').write_text(json.dumps(record, indent=2))
        (path / 'command.txt').write_text('\n'.join('CUDA_VISIBLE_DEVICES=1 MKL_THREADING_LAYER=GNU PYTHONPATH=' + shlex.quote(str(BUILD)) + ' ' + shlex.join(cmd) for cmd in commands) + '\n')
        for name, cmd in zip(('render', 'metrics'), commands):
            t = time.time()
            print(c['scene'] + ': ' + name, flush=True)
            with (path / (name + '.log')).open('w') as log:
                r = subprocess.run(cmd, env=env, cwd=CODE, stdout=log, stderr=subprocess.STDOUT)
            entry['commands'].append({'kind': name, 'returncode': r.returncode, 'elapsed_seconds': time.time()-t, 'argv': cmd})
            STATUS.write_text(json.dumps(status, indent=2))
            assert r.returncode == 0, str(path / (name + '.log'))
        expected = [f'{i:05d}.png' for i in range(c['test_count'])]
        assert sorted(p.name for p in (path / 'test/ours_30000/renders').iterdir()) == expected
        metrics = json.loads((metric_path / 'results.json').read_text())['ours_30000']
        per_view = json.loads((metric_path / 'per_view.json').read_text())['ours_30000']
        assert all(sorted(per_view[k]) == expected for k in ('SSIM', 'PSNR', 'LPIPS'))
        entry.update(state='complete', metrics=metrics, finished=time.time())
        STATUS.write_text(json.dumps(status, indent=2))
        print(c['scene'] + ': ' + json.dumps(metrics), flush=True)
    status.update(state='complete', finished=time.time())
    STATUS.write_text(json.dumps(status, indent=2))


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        if STATUS.exists():
            status = json.loads(STATUS.read_text())
            status.update(state='failed', error=repr(e), finished=time.time())
            STATUS.write_text(json.dumps(status, indent=2))
        raise

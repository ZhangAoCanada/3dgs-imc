"""Run all five declared evaluations once on hust2 physical GPU 1."""
import hashlib
import importlib.metadata
import json
import os
import shlex
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent
NATIVE = Path('/data1/zhangao/repos/3dgs-mcmc')
CODE = OUT / 'code_renderer'
ENV = Path('/data1/ZHANGAo_2024/.conda/envs/mcmc')


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def dump(p, obj):
    p.write_text(json.dumps(obj, indent=2) + '\n')


def main():
    assert os.environ['CUDA_VISIBLE_DEVICES'] == '1'
    assert not (OUT / 'status.json').exists(), 'Run already attempted; inspect its artifacts first'
    plan = json.loads((OUT / 'plan.json').read_text())
    before = subprocess.check_output(['nvidia-smi', '--query-gpu=index,name,uuid,memory.used,utilization.gpu', '--format=csv,noheader'], text=True)
    (OUT / 'evidence/gpu_before.csv').write_text(before)
    status = {'pid': os.getpid(), 'host': socket.gethostname(), 'started': time.time(),
              'physical_gpu': 1, 'state': 'preflight', 'cases': []}
    dump(OUT / 'status.json', status)
    code_files = [NATIVE / 'render.py', NATIVE / 'metrics.py']
    for d in ('arguments', 'scene', 'utils', 'gaussian_renderer', 'lpipsPyTorch'):
        code_files.extend((NATIVE / d).rglob('*.py'))
    code_hashes = {}
    for p in code_files:
        dest = CODE / p.relative_to(NATIVE)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dest)
        code_hashes[str(p.relative_to(NATIVE))] = sha(dest)
    dump(OUT / 'evidence/code_hashes.json', code_hashes)
    commit = subprocess.check_output(['git', '-C', str(NATIVE), 'rev-parse', 'HEAD'], text=True).strip()
    (OUT / 'evidence/native_code.diff').write_text(subprocess.check_output(['git', '-C', str(NATIVE), 'diff', '--', '*.py'], text=True))
    sys.path.insert(0, str(CODE))
    import torch
    import torchvision
    import diff_gaussian_rasterization as raster
    import diff_gaussian_rasterization._C as binary
    import simple_knn._C as knn
    versions = {p: importlib.metadata.version(p) for p in ('numpy', 'Pillow', 'plyfile', 'scipy', 'pyiqa')}
    metadata = {'host': socket.gethostname(), 'python': sys.version, 'executable': sys.executable,
                'native_repo': str(NATIVE), 'native_commit': commit, 'snapshot_used': str(CODE),
                'torch': torch.__version__, 'torchvision': torchvision.__version__, 'cuda': torch.version.cuda,
                'gpu': torch.cuda.get_device_name(0), 'visible_devices': os.environ['CUDA_VISIBLE_DEVICES'],
                'settings_fields': raster.GaussianRasterizationSettings._fields, 'versions': versions,
                'compiled_files': {m.__file__: sha(m.__file__) for m in (raster, binary, knn)},
                'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32,
                'matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32}
    dump(OUT / 'evidence/environment.json', metadata)
    probe = subprocess.run([str(ENV / 'bin/python'), '-c', 'import pyiqa'], capture_output=True, text=True)
    (OUT / 'evidence/native_pyiqa_import.log').write_text(probe.stdout + probe.stderr)
    for case in plan['cases']:
        path = Path(case['output'])
        assert sha(path / 'point_cloud/iteration_30000/point_cloud.ply') == case['checkpoint_sha256']
        assert sha(path / 'discontinuous.txt') == case['split_sha256']
        dataset = Path(case['config_used']['source_path'])
        for rel, expected in case['local_dataset']['files'].items():
            assert sha(dataset / rel) == expected, str(dataset / rel)
        assert sorted(p.name for p in (dataset / case['config_used']['images']).iterdir() if p.is_file()) == case['local_dataset']['all_image_names']
    status['state'] = 'running'
    status['input_hashes_verified'] = True
    dump(OUT / 'status.json', status)
    for case in plan['cases']:
        path = Path(case['output'])
        commands = [
            [str(ENV / 'bin/python'), str(CODE / 'render.py'), '-m', str(path), '--iteration', '30000', '--skip_train'],
            [str(ENV / 'bin/python'), str(OUT / 'scripts/metrics_reference.py'), '--repo', str(CODE), '--model', str(path)],
        ]
        (path / 'command.txt').write_text('cd ' + shlex.quote(str(CODE)) + '\n' + '\n'.join('CUDA_VISIBLE_DEVICES=1 ' + shlex.join(c) for c in commands) + '\n')
        outcome = {'scene': case['scene'], 'started': time.time(), 'commands': []}
        status['cases'].append(outcome)
        dump(OUT / 'status.json', status)
        for name, command in zip(('render', 'metrics'), commands):
            print(f'{case["scene"]}: {name} started', flush=True)
            start = time.time()
            with (path / (name + '.log')).open('w') as log:
                p = subprocess.run(command, cwd=CODE, stdout=log, stderr=subprocess.STDOUT)
            outcome['commands'].append({'kind': name, 'argv': command, 'returncode': p.returncode, 'elapsed_seconds': time.time()-start})
            dump(OUT / 'status.json', status)
            if p.returncode:
                raise RuntimeError(case['scene'] + ' ' + name + ' failed; inspect preserved log')
        expected = [f'{i:05d}.png' for i in range(case['test_count'])]
        for sub in ('renders', 'gt'):
            assert sorted(p.name for p in (path / 'test/ours_30000' / sub).iterdir()) == expected
        metrics = json.loads((path / 'results.json').read_text())['ours_30000']
        values = json.loads((path / 'per_view.json').read_text())['ours_30000']
        assert set(metrics) == {'SSIM', 'PSNR', 'LPIPS'}
        assert all(sorted(values[k]) == expected for k in metrics)
        outcome.update(state='complete', metrics=metrics, finished=time.time())
        dump(OUT / 'status.json', status)
        print(case['scene'] + ': ' + json.dumps(metrics), flush=True)
    weights = {}
    for name in ('vgg.pth', 'vgg16-397923af.pth'):
        p = Path(torch.hub.get_dir()) / 'checkpoints' / name
        if p.exists():
            weights[str(p)] = sha(p)
    dump(OUT / 'evidence/lpips_weight_hashes.json', weights)
    status.update(state='complete', finished=time.time())
    dump(OUT / 'status.json', status)
    print('ALL_FIVE_COMPLETE', flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        path = OUT / 'status.json'
        status = json.loads(path.read_text()) if path.exists() else {}
        status.update(state='failed', error=repr(error), finished=time.time())
        dump(path, status)
        raise

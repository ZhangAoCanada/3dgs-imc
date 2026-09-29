"""Stage the five fixed MCMC candidates and fingerprint evaluation inputs."""
import argparse
import ast
import csv
import hashlib
import json
import shlex
import shutil
import subprocess
from pathlib import Path

REPO = Path('/data1/zhangao/repos/3dgs-imc-codex')
REPORT = REPO / 'writings/mcmc-candidate-verification-2026-09-27'
OUT = Path('/data1/zhangao/outputs/GS-LSS-mcmc-candidate-verification-20260927T1452')
SSH = ['ssh', '-o', 'BatchMode=yes', '-o', 'ClearAllForwardings=yes', '-o',
       'HostName=192.168.1.129', '-o', 'Port=22', '-o',
       'HostKeyAlias=[222.20.99.102]:2088', '-o', 'StrictHostKeyChecking=yes', 'hust2']


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def remote(code):
    p = subprocess.run(SSH + ['python3 -'], input=code, text=True, capture_output=True, check=True)
    return json.loads(p.stdout)


def dataset_record(root, images, names):
    root = Path(root)
    assert (root / images).is_dir(), str(root / images)
    files = {}
    for sub in ('images.bin', 'cameras.bin', 'images.txt', 'cameras.txt', 'points3D.ply'):
        p = root / 'sparse/0' / sub
        if p.is_file():
            files[str(p.relative_to(root))] = sha(p)
    assert 'sparse/0/points3D.ply' in files, 'Avoid implicit dataset modification'
    for name in names:
        p = root / images / name
        files[str(p.relative_to(root))] = sha(p)
    return {'root': str(root), 'files': files,
            'all_image_names': sorted(p.name for p in (root / images).iterdir() if p.is_file())}


def main():
    import inspect
    OUT.mkdir(exist_ok=False)
    (OUT / 'evidence').mkdir()
    (REPORT / 'evidence').mkdir(exist_ok=True)
    rows = list(csv.DictReader((REPO / 'writings/baseline-recovery-2026-09-27/mcmc_candidate_details.csv').open()))
    records = []
    for r in rows:
        case = OUT / 'cases' / r['scene']
        case.mkdir(parents=True)
        source = Path(r['local_run_dir'])
        cfg = {k.arg: ast.literal_eval(k.value) for k in ast.parse((source / 'cfg_args').read_text(), mode='eval').body.keywords}
        names = sorted(n.strip() for n in Path(r['local_split']).read_text().splitlines() if n.strip())
        local = dataset_record(cfg['source_path'], cfg['images'], names)
        remote_root = '/data1/zhangao/mipnerf360/' + r['scene'] if r['dataset'] == 'Mip-NeRF 360' else '/data2/zhangao/tandt_db/tandt/' + r['scene']
        code = 'import hashlib,json\nfrom pathlib import Path\n' + inspect.getsource(sha) + inspect.getsource(dataset_record)
        other = remote(code + '\nprint(json.dumps(dataset_record(' + repr(remote_root) + ',' + repr(cfg['images']) + ',' + repr(names) + ')))')
        # Preserve the candidate's local dataset version, never substitute a host copy.
        dataset_stage = OUT / 'datasets' / r['scene']
        (dataset_stage / 'sparse/0').mkdir(parents=True)
        (dataset_stage / cfg['images']).symlink_to(Path(cfg['source_path']) / cfg['images'], target_is_directory=True)
        for name in ('images.bin', 'cameras.bin', 'images.txt', 'cameras.txt', 'points3D.ply'):
            p = Path(cfg['source_path']) / 'sparse/0' / name
            if p.is_file():
                (dataset_stage / 'sparse/0' / name).symlink_to(p)
        remote_root = str(dataset_stage)
        cp = Path(r['local_checkpoint'])
        cp_hash = sha(cp)
        assert cp_hash == r['candidate_checkpoint_sha256'], 'Candidate changed since recovery audit'
        split = case / 'discontinuous.txt'
        shutil.copy2(r['local_split'], split)
        assert sha(split) == r['split_sha256']
        (case / 'point_cloud/iteration_30000').mkdir(parents=True)
        (case / 'point_cloud/iteration_30000/point_cloud.ply').symlink_to(cp)
        shutil.copy2(source / 'cfg_args', case / 'cfg_args.source')
        shutil.copy2(source / 'results.json', case / 'results.saved_candidate.json')
        shutil.copy2(source / 'per_view.json', case / 'per_view.saved_candidate.json')
        shutil.copy2(source / 'cameras.json', case / 'cameras.saved_candidate.json')
        shutil.copy2(r['original_metric_export'], case / 'metrics.original.csv')
        cfg.update(model_path=str(case), source_path=remote_root, test_selection_path=str(split))
        (case / 'cfg_args').write_text(repr(argparse.Namespace(**cfg)) + '\n')
        cameras = json.loads((source / 'cameras.json').read_text())
        test_names = [c['img_name'] for c in cameras[:len(names)]]
        assert test_names == sorted(Path(n).stem for n in names), 'Saved test camera prefix mismatch'
        stored_per_view = json.loads((source / 'per_view.json').read_text())['ours_30000']
        assert all(len(stored_per_view[m]) == len(names) for m in ('SSIM', 'PSNR', 'LPIPS'))
        protected = []
        for p in [cp, source / 'cfg_args', source / 'results.json', source / 'per_view.json', source / 'cameras.json']:
            s = p.stat()
            protected.append({'path': str(p), 'size': s.st_size, 'mtime_ns': s.st_mtime_ns, 'sha256': cp_hash if p == cp else sha(p)})
        record = {'scene': r['scene'], 'dataset': r['dataset'], 'iteration': 30000,
                  'source_host': 'hust10', 'execution_host': 'hust2', 'physical_gpu': 1,
                  'candidate_checkpoint': str(cp), 'checkpoint_sha256': cp_hash,
                  'source_record': r, 'output': str(case), 'config_used': cfg,
                  'local_dataset': local, 'remote_dataset': other,
                  'split_sha256': sha(split), 'test_count': len(names), 'test_names': test_names,
                  'seed': 'training seed unknown; native renderer safe_state seeds 0',
                  'pipeline': {'convert_SHs_python': False, 'compute_cov3D_python': False, 'debug': False},
                  'protected_sources': protected}
        (case / 'run_record.json').write_text(json.dumps(record, indent=2) + '\n')
        records.append(record)
        print(r['scene'] + ': local dataset staged, split/camera/checkpoint verified; existing remote dataset identical=' + str(local['files'] == other['files']), flush=True)
    plan = {'question': 'Do these five fixed 30k checkpoints reproduce the original Table II source triples?',
            'selection': 'All five preidentified candidates; no model/iteration/parameter search',
            'metric_procedure': 'Native 8-bit PNG render/GT, repository SSIM and PSNR, lpipsPyTorch VGG v0.1, arithmetic mean over held-out views; float32 reduction',
            'thresholds': {'all_three_absolute_error_below': 1e-6, 'also_report_rounding_to': 3},
            'scope': 'Fresh inference/evaluation only; no training or source overwrite',
            'output_root': str(OUT), 'cases': records}
    for p in [OUT / 'plan.json', REPORT / 'plan.json']:
        p.write_text(json.dumps(plan, indent=2) + '\n')
    remote('from pathlib import Path\nimport json\np=Path(' + repr(str(OUT)) + ')\np.mkdir(exist_ok=False)\nprint(json.dumps({"created":str(p)}))')
    command = ['rsync', '-aL', '--stats', '-e', shlex.join(SSH[:-1]), str(OUT) + '/', 'hust2:' + str(OUT) + '/']
    with (REPORT / 'evidence/transfer_to_hust2.log').open('w') as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    print('Staged all five on hust2: ' + str(OUT), flush=True)


if __name__ == '__main__':
    main()

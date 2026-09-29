"""Inventory the requested 67 reruns without launching or selecting results."""
import collections
import csv
import hashlib
import json
from pathlib import Path

REPO = Path('/data1/zhangao/repos/3dgs-imc-codex')
OUT = Path(__file__).resolve().parent
LIB = Path('/data1/zhangao/outputs/GS-LSS-paper-checkpoints/indexes')
groups = list(csv.DictReader((REPO / 'writings/baseline-recovery-2026-09-27/baseline_recovery_groups.csv').open()))
table = list(csv.DictReader((LIB / 'main_tables.csv').open()))
roots = {'Mip-NeRF 360': 'mipnerf360', 'Tanks & Temples': 'tandt', 'Mill 19': 'DATA-small', 'MatrixCity': 'large_angle', 'Zeche': 'large_angle'}
repos = {'3DGS': '/data1/zhangao/repos/gaussian_splatting', 'PGSR': '/data1/ZHANGAo_2024/PGSR', 'Scaffold-GS': '/data1/zhangao/repos/Scaffold-GS', '3DGS-MCMC': '/data1/zhangao/repos/3dgs-mcmc'}
envs = {'3DGS': 'gaussian_splatting', 'PGSR': 'pgsr', 'Scaffold-GS': 'scaffold', '3DGS-MCMC': 'mcmc'}
rows = []
for i, g in enumerate(groups, 1):
    matching = [r for r in table if all(r[k] == g[k] for k in ('table', 'method', 'dataset', 'scene', 'protocol', 'angle'))]
    assert len(matching) == 3
    aerial = g['protocol'] == 'aerial_ground'
    root = Path('/data1/zhangao/DATA') / roots[g['dataset']] / g['scene']
    source = root / 'train' if aerial else root
    test = root / 'test' if aerial else None
    images = 'input_cached' if g['dataset'] == 'MatrixCity' else 'images_4' if g['dataset'] in ('Mip-NeRF 360', 'Mill 19') else 'images'
    split = None if aerial else root / ('test_selection_angle_' + g['angle'] + '.txt' if g['protocol'] == 'coverage_gap' else 'discontinuous.txt')
    digest, snapshot, names, absent = '', '', [], []
    if split:
        data = split.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        target = OUT / 'splits' / (digest + '.txt')
        target.write_bytes(data)
        snapshot = str(target)
        names = [n.strip() for n in data.decode().splitlines() if n.strip()]
        absent = [n for n in names if not (source / images / n).is_file()]
    if g['archived_metric_path']:
        exported = list(csv.DictReader(Path(g['archived_metric_path']).open()))
        assert len(exported) == 1
        reference = {k: float(exported[0][k]) for k in ('SSIM', 'PSNR', 'LPIPS')}
    else:
        reference = {r['backs_metric']: float(r['paper_aggregate_value']) for r in matching}
    cap = 'unknown' if aerial and g['method'] in ('PGSR', '3DGS-MCMC') else 'not specified in recovered launcher'
    if g['method'] in ('PGSR', '3DGS-MCMC') and not aerial:
        cap = g['original_run_path'].rstrip('_').rsplit('-', 1)[1]
    evidence = 'Historical launcher; full original cfg, optimizer settings and seed unverified'
    if g['method'] == '3DGS-MCMC' and not aerial:
        evidence = 'Existing candidate cfg and verified evaluation; not identified as original paper source'
    if aerial:
        evidence = 'Protocol and paper triple known; original full config, seed and iteration unlocated'
    required = 'final PLY; full args; iteration; optimizer state; logs; split/camera map; fresh per-view metrics; source/environment hashes'
    if g['method'] == 'PGSR':
        required += '; iteration-matched app_model sidecar'
    if g['method'] == 'Scaffold-GS':
        required += '; all iteration-matched MLP/network sidecars'
    row = {
        'id': f'baseline_{i:02d}', **{k: g[k] for k in ('table', 'method', 'dataset', 'scene', 'protocol', 'angle')},
        'execution_host': 'hust2 first', 'allowed_physical_gpu': '0 or 1; one selected device per run',
        'proposed_repo': repos[g['method']], 'proposed_python': '/data1/ZHANGAo_2024/.conda/envs/' + envs[g['method']] + '/bin/python',
        'proposed_iterations': 30000, 'iteration_evidence': 'Proposed; original unknown' if aerial else 'Archived ours_30000 export',
        'proposed_images_from_launcher': images, 'proposed_resolution': 1, 'capacity_from_recorded_run': cap,
        'dataset_candidate_host': 'hust10', 'source_path': str(source), 'test_path': str(test) if test else '',
        'source_images_present': (source / images).is_dir(), 'test_images_present': (test / images).is_dir() if test else '',
        'split_source': str(split) if split else '', 'split_snapshot': snapshot, 'split_sha256': digest,
        'test_split_count': len(names) if split else '', 'split_images_missing_from_proposed_directory': len(absent),
        'data_issue': 'Historical images_4 directory absent locally; establish correct remote dataset/resolution before staging' if absent else '',
        'configuration_evidence': evidence, 'original_seed': 'unknown', 'proposed_first_seed': 0,
        'seed_budget': 'pending user protocol decision', 'selection_rule': 'Proposed: retain all fixed-budget runs, independent of scores',
        'status': 'not launched; protocol and method preflight pending', 'checkpoint_requirements': required,
        'original_run_path': g['original_run_path'], 'existing_candidate_checkpoint': g['candidate_checkpoint'],
        'reference_kind': 'paper-only scene triple' if aerial else 'archived per-scene triple',
        'reference_source': g['archived_metric_path'] or str(REPO / 'writings/GS-LSS-submission/GS-LSS.tex'),
        **{'reference_' + k: reference[k] for k in ('SSIM', 'PSNR', 'LPIPS')},
        **{'paper_dataset_aggregate_' + r['backs_metric']: r['paper_aggregate_value'] for r in matching},
    }
    rows.append(row)
assert len(rows) == len({tuple(r[k] for k in ('table', 'method', 'dataset', 'scene', 'protocol', 'angle')) for r in rows}) == 67
with (OUT / 'configurations.csv').open('w') as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
summary = {'status': 'prepared inventory; training not launched; protocol decision pending', 'configurations': 67,
           'by_table': dict(collections.Counter(r['table'] for r in rows)), 'by_method': dict(collections.Counter(r['method'] for r in rows)),
           'explicit_split_rows': sum(bool(r['split_sha256']) for r in rows), 'unique_split_files': len(list((OUT / 'splits').iterdir())),
           'aerial_original_full_configs_unlocated': 8,
           'local_dataset_image_directory_issues': [r['id'] for r in rows if r['data_issue']],
           'remote_free_space_observed_GiB': 240, 'cuda_preflight_performed': False, 'training_runs_launched': 0}
(OUT / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
(OUT / 'validation.json').write_text(json.dumps({'result': 'PASS for inventory only; not execution readiness',
                                              'unique_rows': 67, 'reference_triples': 67,
                                              'split_rows_snapshotted': summary['explicit_split_rows'],
                                              'issues_preserved': summary['local_dataset_image_directory_issues']}, indent=2) + '\n')
print(json.dumps(summary, indent=2))

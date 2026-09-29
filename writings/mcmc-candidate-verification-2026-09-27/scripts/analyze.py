"""Compare all five fixed candidates against original exports and saved results."""
import csv
import importlib.util
import json
from pathlib import Path

spec = importlib.util.spec_from_file_location('prep', Path(__file__).with_name('prepare.py'))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
plan = json.loads((m.OUT / 'plan.json').read_text())
primary = json.loads((m.OUT / 'status.json').read_text())
fixed = json.loads((m.OUT / 'saved_gt_attempt2_status.json').read_text())
control = json.loads((m.OUT / 'metric_control_status.json').read_text())
main_renderer = json.loads((m.OUT / 'main_rasterizer_status.json').read_text())
assert primary['state'] == fixed['state'] == control['state'] == main_renderer['state'] == 'complete'
assert len(primary['cases']) == len(fixed['cases']) == 5
fixed_by_scene = {r['scene']: r for r in fixed['cases']}
control_by_scene = {r['scene']: r for r in control['cases']}
main_by_scene = {r['scene']: r for r in main_renderer['cases']}
pixel_data = {r['scene']: r for r in json.loads((m.OUT / 'evidence/fresh_saved_pixel_comparison_attempt2.json').read_text())}
rows = []
for c in plan['cases']:
    p = Path(c['output'])
    fresh = json.loads((p / 'results.json').read_text())['ours_30000']
    historical_gt = fixed_by_scene[c['scene']]['metrics']
    control_metrics = control_by_scene[c['scene']]['metrics']
    main_metrics = main_by_scene[c['scene']]['metrics']
    saved = json.loads((p / 'results.saved_candidate.json').read_text())['ours_30000']
    original = next(csv.DictReader((p / 'metrics.original.csv').open()))
    r = {'scene': c['scene'], 'dataset': c['dataset'], 'test_views': c['test_count'],
         'iteration': 30000, 'checkpoint': c['candidate_checkpoint'], 'sha256': c['checkpoint_sha256'],
         'execution_host': 'hust2', 'physical_gpu': 1, 'split_sha256': c['split_sha256']}
    for k in ('SSIM', 'PSNR', 'LPIPS'):
        r['original_' + k] = float(original[k])
        r['saved_candidate_' + k] = saved[k]
        r['fresh_current_dataset_' + k] = fresh[k]
        r['fresh_saved_gt_' + k] = historical_gt[k]
        r['metric_control_' + k] = control_metrics[k]
        r['fresh_main_rasterizer_' + k] = main_metrics[k]
        r['saved_gt_minus_original_' + k] = historical_gt[k] - float(original[k])
        r['saved_gt_minus_saved_candidate_' + k] = historical_gt[k] - saved[k]
        r['main_rasterizer_minus_original_' + k] = main_metrics[k] - float(original[k])
        r['main_rasterizer_minus_saved_candidate_' + k] = main_metrics[k] - saved[k]
    for label, values in [('fresh_current_dataset', fresh), ('fresh_saved_gt', historical_gt),
                          ('metric_control', control_metrics), ('fresh_main_rasterizer', main_metrics)]:
        r[label + '_matches_original_1e6'] = all(abs(values[k] - float(original[k])) < 1e-6 for k in ('SSIM', 'PSNR', 'LPIPS'))
        r[label + '_matches_original_3dp'] = all(f'{values[k]:.3f}' == f'{float(original[k]):.3f}' for k in ('SSIM', 'PSNR', 'LPIPS'))
        r[label + '_matches_candidate_3dp'] = all(f'{values[k]:.3f}' == f'{saved[k]:.3f}' for k in ('SSIM', 'PSNR', 'LPIPS'))
    for sub in ('gt', 'renders'):
        v = [v[sub] for v in pixel_data[c['scene']]['views']]
        r['fresh_vs_saved_' + sub + '_identical_views'] = sum(p['pixels_identical'] for p in v)
        r['fresh_vs_saved_' + sub + '_mean_abs_255'] = sum(p['mean_abs_255'] for p in v) / len(v)
        r['fresh_vs_saved_' + sub + '_max_abs_255'] = max(p['max_abs_255'] for p in v)
    rows.append(r)
with (m.REPORT / 'comparison.csv').open('w') as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
summary = {'question': plan['question'], 'completed_candidates': 5, 'rendered_views': sum(c['test_count'] for c in plan['cases']),
           'execution_host': 'hust2', 'physical_gpu': 1,
           'matches_original_1e6_with_current_dataset': sum(r['fresh_current_dataset_matches_original_1e6'] for r in rows),
           'matches_original_1e6_with_saved_gt': sum(r['fresh_saved_gt_matches_original_1e6'] for r in rows),
           'matches_original_3dp_with_saved_gt': sum(r['fresh_saved_gt_matches_original_3dp'] for r in rows),
           'matches_stored_candidate_3dp_with_saved_gt': sum(r['fresh_saved_gt_matches_candidate_3dp'] for r in rows),
           'matches_original_1e6_main_rasterizer': sum(r['fresh_main_rasterizer_matches_original_1e6'] for r in rows),
           'matches_original_3dp_main_rasterizer': sum(r['fresh_main_rasterizer_matches_original_3dp'] for r in rows),
           'matches_stored_candidate_3dp_main_rasterizer': sum(r['fresh_main_rasterizer_matches_candidate_3dp'] for r in rows),
           'metric_control_matches_stored_candidate_3dp': sum(r['metric_control_matches_candidate_3dp'] for r in rows),
           'source_checkpoint_identity': 'Numerical agreement alone does not establish historical identity',
           'results': rows}
(m.REPORT / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps(summary, indent=2))

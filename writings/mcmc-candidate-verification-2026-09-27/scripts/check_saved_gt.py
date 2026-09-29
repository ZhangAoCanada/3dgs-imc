"""Evaluate fresh renders with historical GT; compare saved/fresh pixels."""
import hashlib
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
from PIL import Image

OUT = Path(__file__).resolve().parent.parent
CODE = OUT / 'code_renderer'
ATTEMPT = os.environ.get('GS_LSS_EVAL_ATTEMPT', '1')
SUFFIX = '' if ATTEMPT == '1' else '_attempt' + ATTEMPT
STATUS_PATH = OUT / ('saved_gt' + SUFFIX + '_status.json')


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    assert os.environ['CUDA_VISIBLE_DEVICES'] == '1'
    plan = json.loads((OUT / 'plan.json').read_text())
    assert json.loads((OUT / 'status.json').read_text())['state'] == 'complete'
    status_path = STATUS_PATH
    assert not status_path.exists()
    status = {'pid': os.getpid(), 'started': time.time(), 'state': 'running', 'cases': [],
              'attempt': ATTEMPT, 'metrics_env': {'CUDA_VISIBLE_DEVICES': '1', 'MKL_THREADING_LAYER': 'GNU'}}
    status_path.write_text(json.dumps(status, indent=2))
    for item in json.loads((OUT / 'evidence/saved_image_hashes.json').read_text()):
        assert sha(OUT / 'saved_images' / item['scene'] / item['kind'] / item['name']) == item['sha256']
    comparisons = []
    for c in plan['cases']:
        fresh = Path(c['output']) / 'test/ours_30000'
        saved = OUT / 'saved_images' / c['scene']
        names = [f'{i:05d}.png' for i in range(c['test_count'])]
        compare = {'scene': c['scene'], 'views': []}
        for n in names:
            view = {'name': n, 'image_name': c['test_names'][int(n[:5])]}
            for sub in ('gt', 'renders'):
                a = np.asarray(Image.open(fresh / sub / n).convert('RGB'))
                b = np.asarray(Image.open(saved / sub / n).convert('RGB'))
                assert a.shape == b.shape
                d = np.abs(a.astype(np.int16) - b.astype(np.int16))
                view[sub] = {'pixels_identical': bool(np.array_equal(a, b)),
                             'mean_abs_255': float(d.mean()), 'max_abs_255': int(d.max()),
                             'differing_components': int((d != 0).sum()), 'components': int(d.size)}
            compare['views'].append(view)
        comparisons.append(compare)
        (OUT / ('evidence/fresh_saved_pixel_comparison' + SUFFIX + '.json')).write_text(json.dumps(comparisons, indent=2))
        entry = {'scene': c['scene']}
        if all(v['gt']['pixels_identical'] for v in compare['views']):
            entry.update(state='identical_gt', metrics=json.loads((Path(c['output']) / 'results.json').read_text())['ours_30000'])
        else:
            path = OUT / ('evaluations_saved_gt' + SUFFIX) / c['scene']
            root = path / 'test/ours_30000'
            root.mkdir(parents=True)
            (root / 'renders').symlink_to(fresh / 'renders', target_is_directory=True)
            (root / 'gt').symlink_to(saved / 'gt', target_is_directory=True)
            command = [sys.executable, str(OUT / 'scripts/metrics_reference.py'), '--repo', str(CODE), '--model', str(path)]
            (path / 'command.txt').write_text('CUDA_VISIBLE_DEVICES=1 MKL_THREADING_LAYER=GNU ' + shlex.join(command) + '\n')
            record = {'purpose': 'Isolate GT-version effect using the same freshly rendered images',
                      'fresh_render_source': str(fresh / 'renders'), 'historical_gt_source': str(saved / 'gt'),
                      'camera_mapping': c['test_names'], 'checkpoint_sha256': c['checkpoint_sha256'],
                      'all_views_included': c['test_count']}
            (path / 'run_record.json').write_text(json.dumps(record, indent=2))
            with (path / 'metrics.log').open('w') as log:
                result = subprocess.run(command, cwd=CODE, stdout=log, stderr=subprocess.STDOUT,
                                        env=dict(os.environ, MKL_THREADING_LAYER='GNU'))
            assert result.returncode == 0, c['scene'] + ': inspect metrics.log'
            entry.update(state='complete', metrics=json.loads((path / 'results.json').read_text())['ours_30000'])
        status['cases'].append(entry)
        status_path.write_text(json.dumps(status, indent=2))
        print(c['scene'] + ': ' + json.dumps(entry), flush=True)
    status.update(state='complete', finished=time.time())
    status_path.write_text(json.dumps(status, indent=2))


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        if STATUS_PATH.exists():
            status = json.loads(STATUS_PATH.read_text())
            status.update(state='failed', error=repr(error), finished=time.time())
            STATUS_PATH.write_text(json.dumps(status, indent=2))
        raise

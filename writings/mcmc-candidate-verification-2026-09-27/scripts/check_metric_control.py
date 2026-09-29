"""Check the metric implementation against each candidate's saved image pair."""
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent
STATUS = OUT / 'metric_control_status.json'


def main():
    assert os.environ['CUDA_VISIBLE_DEVICES'] == '1'
    assert json.loads((OUT / 'saved_gt_attempt2_status.json').read_text())['state'] == 'complete'
    assert not STATUS.exists()
    plan = json.loads((OUT / 'plan.json').read_text())
    status = {'state': 'running', 'pid': os.getpid(), 'started': time.time(), 'cases': []}
    STATUS.write_text(json.dumps(status, indent=2))
    for c in plan['cases']:
        p = OUT / 'metric_control' / c['scene']
        root = p / 'test/ours_30000'
        root.mkdir(parents=True)
        for sub in ('renders', 'gt'):
            (root / sub).symlink_to(OUT / 'saved_images' / c['scene'] / sub, target_is_directory=True)
        command = [sys.executable, str(OUT / 'scripts/metrics_reference.py'), '--repo', str(OUT / 'code_renderer'), '--model', str(p)]
        (p / 'command.txt').write_text('CUDA_VISIBLE_DEVICES=1 MKL_THREADING_LAYER=GNU ' + shlex.join(command) + '\n')
        with (p / 'metrics.log').open('w') as log:
            r = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                               env=dict(os.environ, MKL_THREADING_LAYER='GNU'))
        assert r.returncode == 0, c['scene'] + ' metric control failed'
        result = json.loads((p / 'results.json').read_text())['ours_30000']
        status['cases'].append({'scene': c['scene'], 'metrics': result, 'returncode': r.returncode})
        STATUS.write_text(json.dumps(status, indent=2))
        print(c['scene'] + ': ' + json.dumps(result), flush=True)
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

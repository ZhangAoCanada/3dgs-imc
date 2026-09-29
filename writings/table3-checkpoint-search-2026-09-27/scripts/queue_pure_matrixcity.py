"""Apply the user's pure-aerial-first choice to four unstarted local cases."""
import copy
import hashlib
import json
import os
import shutil
import time
from pathlib import Path

ROOT = Path('/data1/zhangao/outputs/GS-LSS-baseline-capped-reruns-20260927')
REPORT = Path(__file__).resolve().parents[1]
CHANGE = ROOT / 'protocol_changes/20260927_matrixcity_pure_first'
IDS = {'baseline_60', 'baseline_62', 'baseline_64', 'baseline_66'}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


staged = json.loads((REPORT / 'evidence/strict_matrixcity_staging.json').read_text())
assert staged['status'] == 'passed'
assert json.loads((CHANGE / 'validation.json').read_text())['status'] == 'passed'
config_path = ROOT / 'configurations_local.json'
before = json.loads(config_path.read_text())
assert before == json.loads((CHANGE / 'configurations_local.before.json').read_text())
state = json.loads((ROOT / 'status.json').read_text())
os.kill(state['coordinator_pid'], 0)
assert time.time() - state['updated'] < 30
active = {r['id'] for r in state['current'].values()}
for ident in IDS:
    assert ident not in active and ident not in state['selected'] and ident not in state['blocked']
    assert not state['next_seed'].get(ident, 0)
    assert not (ROOT / 'attempts_local' / ident).exists()
after = copy.deepcopy(before)
for r in after:
    if r['id'] not in IDS:
        continue
    r.update(source_path=staged['strict_training'], test_path=staged['strict_test'],
             source_path_original='/data1/zhangao/DATA/large_angle/blockA_fusion_small_aerial/train',
             test_path_original=str(ROOT / 'dataset_sources_table3_strict_20260927/blockA_fusion_small_aerial/test'),
             test_path_unfiltered='/data1/zhangao/DATA/large_angle/blockA_fusion_small_aerial/test',
             protocol='aerial_only_train_street_only_test', dataset_variant='blockA_fusion_small_aerial',
             scene_field_role='Historical table-group key; active data is specified by source_path and test_path.',
             train_split_count=189, test_split_count=130,
             diagnostic_test_path=staged['historical_diagnostic_test'], diagnostic_test_count=154,
             diagnostic_selection_role='Comparison only; acceptance uses primary street-only metrics.',
             reference_kind='Submitted Table III triple retained as biased acceptance target across a changed protocol; not evidence of same-protocol reproduction.',
             configuration_evidence='User chose pure-aerial 189 train / 130 street test; original missing model configuration remains unknown. Also evaluate identical model on historical 154 test views.',
             protocol_change_record=str(CHANGE / 'queue_change.json'),
             mixed_training_fallback='No automatic fallback. Review pure-aerial results before a separately recorded mixed-training comparison.')
assert len(after) == 67
changed = {a['id'] for a, b in zip(after, before) if a != b}
assert changed == IDS
for a, b in zip(after, before):
    for field in ['id', 'method_key', 'reference', 'iterations', 'extra_args', 'local_python', 'selection_rule']:
        assert a[field] == b[field]
    if a['id'] in IDS:
        for split, count in [('source_path', 189), ('test_path', 130), ('diagnostic_test_path', 154)]:
            assert len(json.loads((Path(a[split]) / 'frozen.json').read_text())['images']) == count
record = dict(status='prepared', host=os.uname().nodename, time=time.time(), changed_ids=sorted(IDS),
              coordinator_pid=state['coordinator_pid'], active_jobs_preserved=state['current'],
              unchanged_configurations=63, total_configurations=67, acceptance='Primary street-only 130 metrics against unchanged biased reference thresholds.',
              scheduler_behavior='Running scheduler retains IDs and reference values; each new run_one process reloads configurations_local.json. Legacy scene labels retained for scheduler compatibility.',
              before_sha256=sha(config_path), validation=str(CHANGE / 'validation.json'))
(CHANGE / 'queue_change.json').write_text(json.dumps(record, indent=2))
tmp = config_path.with_suffix('.pure-first.tmp')
tmp.write_text(json.dumps(after, indent=2))
tmp.replace(config_path)
assert json.loads(config_path.read_text()) == after
record.update(status='applied', after_sha256=sha(config_path), finished=time.time())
(CHANGE / 'queue_change.json').write_text(json.dumps(record, indent=2))
shutil.copy2(config_path, CHANGE / 'configurations_local.after.json')
shutil.copy2(CHANGE / 'queue_change.json', REPORT / 'evidence/strict_matrixcity_queue_change.json')
shutil.copy2(CHANGE / 'validation.json', REPORT / 'evidence/secondary_evaluation_validation.json')
for name in ['run_one.py', 'secondary_evaluation.py']:
    shutil.copy2(ROOT / 'scripts_local' / name, CHANGE / (name + '.after'))
print(json.dumps(record, indent=2))

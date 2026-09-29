"""Verify local originals for four imported Mill 19 backup runs."""
import csv
import hashlib
import json
from pathlib import Path

WORK = Path(__file__).resolve().parents[1]
LIB = Path('/data1/zhangao/outputs/GS-LSS-paper-checkpoints')
rows = list(csv.DictReader((WORK/'dataset_and_split_sources.csv').open()))
runs = {(r['host'], r['run_dir']): r for r in json.loads((WORK/'selected_runs.json').read_text())}
evidence = []
for row in rows:
    if row['split_status'] != 'original split path missing':
        continue
    original = Path(row['split_path'])
    if not original.is_relative_to('/data1/zhangao/DATA/DATA-small') or not original.is_file():
        continue
    run = Path(runs[row['host'], row['run_dir']]['local_run'])
    cameras = json.loads((run/'cameras.json').read_text())
    per_view = json.loads((run/'per_view.json').read_text())
    counts = {len(v['PSNR']) for v in per_view.values()}
    assert len(counts) == 1
    n = counts.pop()
    expected = {Path(v['img_name']).stem for v in cameras[:n]}
    data = original.read_bytes()
    lines = [s.strip() for s in data.decode().splitlines() if s.strip()]
    assert n > 0 and len(expected) == n == len(lines)
    assert expected == {Path(s).stem for s in lines}, original
    sha = hashlib.sha256(data).hexdigest()
    target = LIB/'splits/by_sha256'/f'{sha}.txt'
    if target.exists():
        assert target.read_bytes() == data
    else:
        target.write_bytes(data)
    row.update(split_status='local original-host split membership verified against imported saved cameras and independent view count',
               split_sha256=sha, local_split=str(target), nonempty_lines=n,
               recovered_split_path=str(original), split_source_host='hust10')
    evidence.append(dict(source_host=row['host'], source_run=row['run_dir'],
                         split_source_host='hust10', split_source=str(original),
                         camera_evidence=str(run/'cameras.json'),
                         view_count_evidence=str(run/'per_view.json'),
                         test_views=n, sha256=sha, local_split=str(target),
                         status=row['split_status']))
for path in [WORK/'dataset_and_split_sources.csv', LIB/'indexes/dataset_and_split_sources.csv']:
    with path.open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader()
        writer.writerows(rows)
output = WORK/'evidence/split_recovery_local.json'
assert not output.exists()
output.write_text(json.dumps(evidence, indent=2)+'\n')
print(json.dumps(dict(recovered=len(evidence), remaining=sum(r['split_status']=='original split path missing' for r in rows))))

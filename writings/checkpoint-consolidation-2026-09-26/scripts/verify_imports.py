"""Compare every imported file's complete source and destination SHA-256."""
import collections
import csv
import json
from pathlib import Path
import shutil

WORK = Path(__file__).resolve().parents[1]
LIB = Path('/data1/zhangao/outputs/GS-LSS-paper-checkpoints')
hashes = {}
for host in ['hust10', 'hust2', 'wsl']:
    for side in ['source', 'destination']:
        records = {}
        for path in (WORK/'evidence').glob(f'sha256_{host}_{side}*.jsonl'):
            for line in path.open():
                record = json.loads(line)
                previous = records.get(record['path'])
                assert previous is None or previous == record, record['path']
                records[record['path']] = record
        hashes[host, side] = records

verified = []
for name in ['copy_file_manifest.csv', 'copy_extra_file_manifest.csv']:
    for row in csv.DictReader((WORK/name).open()):
        if row['action'] != 'copy':
            continue
        source = hashes[row['host'], 'source'][row['source']]
        destination = hashes[row['host'], 'destination'][row['destination']]
        assert source['bytes'] == destination['bytes'] == int(row['bytes']), row
        assert source['sha256'] == destination['sha256'], row
        assert Path(row['destination']).stat().st_size == int(row['bytes']), row
        verified.append(dict(host=row['host'], source=row['source'],
                             destination=row['destination'], bytes=row['bytes'],
                             sha256=source['sha256'], status='full SHA-256 matches source'))
assert len({r['destination'] for r in verified}) == len(verified)
path = WORK/'import_verification.csv'
with path.open('w') as f:
    writer = csv.DictWriter(f, fieldnames=list(verified[0]))
    writer.writeheader()
    writer.writerows(verified)
summary = dict(files=len(verified), bytes=sum(int(r['bytes']) for r in verified),
               files_by_host=dict(collections.Counter(r['host'] for r in verified)),
               bytes_by_host={h: sum(int(r['bytes']) for r in verified if r['host'] == h)
                              for h in ['hust10', 'hust2', 'wsl']},
               mismatch_count=0, missing_count=0)
(WORK/'import_verification.json').write_text(json.dumps(summary, indent=2)+'\n')
for name in ['import_verification.csv', 'import_verification.json']:
    shutil.copy2(WORK/name, LIB/'indexes'/name)
print(json.dumps(summary, indent=2))

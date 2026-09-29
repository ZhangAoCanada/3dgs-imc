"""Find byte-identical immutable output files; never changes the output tree."""
import collections
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import stat
import time

ROOT = Path('/data1/zhangao/outputs')
WORK = Path(__file__).resolve().parent
MIN_BYTES = 4096
SAMPLE_BYTES = 4096
EXCLUDED_TOPS = {'GS-LSS-R1.2-compare-20260928-142236', 'sentinel_state'}
EXCLUDED_NAMES = {'status.json', 'selected.json', 'run.json', 'attempts.jsonl'}


def unchanged(row):
    p = ROOT / row['path']
    s = p.lstat()
    assert stat.S_ISREG(s.st_mode), str(p)
    assert (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns) == (row['dev'], row['ino'], row['size'], row['mtime_ns']), str(p)
    return s


def digest(row, full=False):
    unchanged(row)
    h = hashlib.sha256()
    with (ROOT / row['path']).open('rb') as f:
        if full or row['size'] <= 3*SAMPLE_BYTES:
            for b in iter(lambda: f.read(8*1024*1024), b''):
                h.update(b)
        else:
            for pos in (0, row['size']//2, row['size']-SAMPLE_BYTES):
                f.seek(pos)
                h.update(f.read(SAMPLE_BYTES))
    unchanged(row)
    return h.hexdigest()


def main():
    start = time.monotonic()
    db = sqlite3.connect(WORK/'inventory.sqlite', timeout=60)
    db.execute('PRAGMA journal_mode=WAL')
    db.execute('PRAGMA busy_timeout=60000')
    db.row_factory = sqlite3.Row
    db.execute('CREATE TABLE IF NOT EXISTS hashes(path TEXT PRIMARY KEY, sample_sha256 TEXT, sha256 TEXT)')
    db.execute('CREATE TABLE IF NOT EXISTS hash_errors(path TEXT PRIMARY KEY, error TEXT)')
    # Only equal-size files on the same filesystem and with equal permissions
    # can share data without changing access. Existing hardlinks count once.
    groups = collections.defaultdict(list)
    seen = set()
    for raw in db.execute('SELECT * FROM files WHERE kind="file" AND size>=? ORDER BY path', (MIN_BYTES,)):
        r = dict(raw)
        if r['top'] in EXCLUDED_TOPS or Path(r['path']).name in EXCLUDED_NAMES:
            continue
        if '/scripts' in r['path'] or '/code/' in r['path'] or '/local_extensions/' in r['path']:
            continue
        identity = (r['dev'], r['ino'])
        if identity in seen:
            continue
        seen.add(identity)
        groups[(r['dev'], r['size'], r['mode'], r['uid'], r['gid'])].append(r)
    candidates = [r for group in groups.values() if len(group)>1 for r in group]
    candidates.sort(key=lambda r:r['path'])
    print(json.dumps({'stage':'sampling','candidate_inodes':len(candidates), 'candidate_GiB':sum(r['size'] for r in candidates)/2**30}), flush=True)
    sampled = collections.defaultdict(list)
    done = 0
    last_print = time.monotonic()
    def checked_digest(item):
        r, full, cached = item
        try:
            unchanged(r)
            return cached or digest(r,full), None
        except (OSError, AssertionError) as exc:
            return None, repr(exc)
    cached={r[0]:(r[1],r[2]) for r in db.execute('SELECT path,sample_sha256,sha256 FROM hashes')}
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        for offset in range(0,len(candidates),1024):
            batch=candidates[offset:offset+1024]
            items=[(r,False,cached.get(r['path'],(None,None))[0]) for r in batch]
            for r,(h,error) in zip(batch,pool.map(checked_digest,items)):
                if error:
                    db.execute('INSERT OR REPLACE INTO hash_errors VALUES (?,?)',(r['path'],error))
                else:
                    db.execute('INSERT INTO hashes(path,sample_sha256) VALUES (?,?) ON CONFLICT(path) DO UPDATE SET sample_sha256=excluded.sample_sha256',(r['path'],h))
                    sampled[(r['dev'],r['size'],r['mode'],r['uid'],r['gid'],h)].append(r)
                done+=1
            db.commit()
            if time.monotonic()-last_print > 20:
                print(json.dumps({'stage':'sampling','done':done,'total':len(candidates),'elapsed_s':round(time.monotonic()-start)}),flush=True)
                last_print=time.monotonic()
    db.commit()
    needs_full = [r for group in sampled.values() if len(group)>1 for r in group]
    needs_full.sort(key=lambda r:r['path'])
    print(json.dumps({'stage':'full_sha256','inodes':len(needs_full),'GiB':sum(r['size'] for r in needs_full)/2**30}),flush=True)
    full_groups=collections.defaultdict(list)
    done_bytes=0
    done=0
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for offset in range(0,len(needs_full),256):
            batch=needs_full[offset:offset+256]
            items=[(r,True,cached.get(r['path'],(None,None))[1]) for r in batch]
            for r,(h,error) in zip(batch,pool.map(checked_digest,items)):
                if error:
                    db.execute('INSERT OR REPLACE INTO hash_errors VALUES (?,?)',(r['path'],error))
                else:
                    db.execute('UPDATE hashes SET sha256=? WHERE path=?',(h,r['path']))
                    full_groups[(r['dev'],r['size'],r['mode'],r['uid'],r['gid'],h)].append(r)
                    done_bytes+=r['size']
                done+=1
            db.commit()
            if time.monotonic()-last_print > 20:
                print(json.dumps({'stage':'full_sha256','done':done,'total':len(needs_full),'read_GiB':round(done_bytes/2**30,2),'elapsed_s':round(time.monotonic()-start)}),flush=True)
                last_print=time.monotonic()
    db.commit()
    actions=[]
    for key,group in full_groups.items():
        if len(group)<2:
            continue
        # Prefer an already-shared inode as canonical, then stable old paths.
        group.sort(key=lambda r:(-r['nlink'],r['mtime_ns'],r['path']))
        canonical=group[0]
        for r in group[1:]:
            # Replacing a member of another hardlink set would not immediately
            # release blocks and may have aliases outside this root. Leave it.
            if r['nlink'] != 1:
                continue
            actions.append({'action':'replace_duplicate_with_hardlink','canonical':canonical,
                            'duplicate':r,'sha256':key[-1], 'reclaimable_bytes':r['allocated']})
    actions.sort(key=lambda x:x['duplicate']['path'])
    with (WORK/'dedup_plan.jsonl').open('w') as f:
        for a in actions:
            f.write(json.dumps(a)+'\n')
    bycategory=collections.defaultdict(lambda:{'files':0,'bytes':0})
    for a in actions:
        c=bycategory[a['duplicate']['category']]
        c['files']+=1;c['bytes']+=a['reclaimable_bytes']
    summary={'planned_duplicate_paths':len(actions),'reclaimable_bytes':sum(a['reclaimable_bytes'] for a in actions),
             'by_category':dict(bycategory),'errors':[list(r) for r in db.execute('SELECT * FROM hash_errors')],
             'elapsed_s':time.monotonic()-start}
    (WORK/'dedup_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    main()

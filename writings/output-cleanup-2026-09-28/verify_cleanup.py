"""Validate retained paths, changed bytes, and research-library coverage."""
import concurrent.futures
import datetime
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import stat
import time

from find_duplicates import ROOT, WORK


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):
            h.update(b)
    return h.hexdigest()


def main():
    start=time.monotonic();errors=[];changed={};canonical={}
    with gzip.open(WORK/'dedup_actions.jsonl.gz','rt') as f:
        for line in f:
            r=json.loads(line)
            if r['status']=='complete':
                changed[str(Path(r['duplicate']).relative_to(ROOT))]=r
                if r['canonical'] in canonical:
                    assert canonical[r['canonical']]==r['sha256']
                canonical[r['canonical']]=r['sha256']
    db=sqlite3.connect(WORK/'inventory.sqlite');db.row_factory=sqlite3.Row
    checked=0;checkpoint_paths=0;mutable_changes=[]
    mutable_names={'status.json','coordinator-local.log','coordinator.log','coordinator.lock','coordinator.pid'}
    last_print=time.monotonic()
    for r in db.execute('SELECT * FROM files'):
        p=ROOT/r['path']
        try:
            s=p.lstat()
            if r['kind']=='symlink':
                assert stat.S_ISLNK(s.st_mode) and os.readlink(p)==r['target'],'symlink changed'
            elif r['kind']=='file':
                assert stat.S_ISREG(s.st_mode),'file type changed'
                if r['path'] in changed:
                    a=changed[r['path']];c=Path(a['canonical']).stat()
                    assert (s.st_dev,s.st_ino,s.st_size)==(c.st_dev,c.st_ino,r['size']),'hardlink/size mismatch'
                else:
                    actual=(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns)
                    expected=(r['dev'],r['ino'],r['size'],r['mtime_ns'])
                    if actual!=expected and p.name in mutable_names and r['top']=='GS-LSS-baseline-capped-reruns-20260927':
                        mutable_changes.append(r['path'])
                    else:
                        assert actual==expected,'unexpected identity/size/mtime change'
                assert (stat.S_IMODE(s.st_mode),s.st_uid,s.st_gid)==(r['mode'],r['uid'],r['gid']),'permissions changed'
            if r['category']=='checkpoint_or_neural_state' and r['kind']=='file':checkpoint_paths+=1
            checked+=1
        except (OSError,AssertionError) as exc:
            errors.append({'path':str(p),'error':repr(exc)})
        if time.monotonic()-last_print>20:
            print(json.dumps({'stage':'path_verification','checked':checked,'errors':len(errors),'elapsed_s':round(time.monotonic()-start)}),flush=True)
            last_print=time.monotonic()
    print(json.dumps({'stage':'full_sha256_after_cleanup','unique_canonical_files':len(canonical),'retained_checkpoint_paths':checkpoint_paths}),flush=True)
    items=list(canonical.items());hashed=0;hashbytes=0
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for offset in range(0,len(items),256):
            batch=items[offset:offset+256]
            for (p,expected),actual in zip(batch,pool.map(sha,[p for p,h in batch])):
                if actual!=expected:errors.append({'path':p,'error':'Full SHA-256 differs after cleanup'})
                hashed+=1;hashbytes+=Path(p).stat().st_size
            if time.monotonic()-last_print>20:
                print(json.dumps({'stage':'full_sha256_after_cleanup','hashed':hashed,'total':len(items),'read_GiB':round(hashbytes/2**30,2),'errors':len(errors),'elapsed_s':round(time.monotonic()-start)}),flush=True)
                last_print=time.monotonic()
    protected=json.loads((WORK/'protected_paths_before.json').read_text());coverage={}
    for name,rows in protected.items():
        if name=='library_links':
            count=0
            for r in rows:
                p=Path(r['path']);actual=p.exists()
                if actual!=r['resolves']:errors.append({'path':str(p),'error':'library link resolution changed'})
                count+=actual
            coverage[name]={'paths':len(rows),'resolving':count}
        else:
            count=0
            for r in rows:
                p=Path(r['path']);actual=p.is_file()
                if actual!=r['exists'] or (actual and p.stat().st_size!=r['size']):
                    errors.append({'path':str(p),'error':'protected checkpoint availability/size changed'})
                count+=actual
                # For selected reruns, verify the recorded full checkpoint hash,
                # including models whose inodes did not need consolidation.
                if name=='baseline_selected' and actual and sha(p)!=r['sha256']:
                    errors.append({'path':str(p),'error':'selected checkpoint SHA-256 mismatch'})
            coverage[name]={'paths':len(rows),'present':count}
    summary={'finished':datetime.datetime.now().astimezone().isoformat(),'entries_checked':checked,
             'all_original_paths_preserved':not errors,'checkpoint_paths_retained':checkpoint_paths,
             'deduplicated_paths_verified':len(changed),'unique_canonical_hashes_verified':hashed,
             'coverage':coverage,'expected_live_coordinator_changes':mutable_changes,'errors':errors,
             'disk_after_verification':shutil.disk_usage(ROOT)._asdict(),'elapsed_s':time.monotonic()-start}
    (WORK/'verification.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)
    if errors:raise SystemExit(1)


if __name__=='__main__':
    main()

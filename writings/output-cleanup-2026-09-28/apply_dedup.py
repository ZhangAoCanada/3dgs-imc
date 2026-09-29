"""Execute this audit's verified, lossless duplicate-storage consolidation."""
import collections
import datetime
import gzip
import json
import os
from pathlib import Path
import shutil
import time

from find_duplicates import ROOT, WORK, unchanged


def xattrs(path):
    return {k:os.getxattr(path,k,follow_symlinks=False) for k in os.listxattr(path,follow_symlinks=False)}


def main():
    actions=[json.loads(l) for l in (WORK/'dedup_plan.jsonl').read_text().splitlines()]
    journal=WORK/'dedup_actions.jsonl.gz'
    assert not journal.exists(), 'One-time operation; inspect the existing journal before any retry'
    campaign=ROOT/'GS-LSS-baseline-capped-reruns-20260927'
    state=json.loads((campaign/'status.json').read_text())
    assert not state['current'], state['current']
    assert all((campaign/f'RESERVE_GPU_{g}').exists() for g in [0,1])
    start=time.monotonic();before=shutil.disk_usage(ROOT)._asdict()
    totals=collections.defaultdict(lambda:{'files':0,'bytes':0})
    errors=[];completed=0;reclaimed=0;last_print=time.monotonic()
    with gzip.open(journal,'xt') as log:
        for index,a in enumerate(actions):
            src=ROOT/a['canonical']['path'];dst=ROOT/a['duplicate']['path']
            temp=dst.with_name('.'+dst.name+'.dedup-20260928-tmp')
            event={'canonical':str(src),'duplicate':str(dst),'sha256':a['sha256'],
                   'old_inode':a['duplicate']['ino'],'old_mtime_ns':a['duplicate']['mtime_ns'],
                   'canonical_inode':a['canonical']['ino'],'canonical_mtime_ns':a['canonical']['mtime_ns'],
                   'bytes_reclaimed':a['reclaimable_bytes'],'category':a['duplicate']['category']}
            created_temp=False
            try:
                ss=unchanged(a['canonical']);ds=unchanged(a['duplicate'])
                assert ds.st_nlink==1, 'Duplicate acquired another link; do not discard its inode'
                assert not temp.exists() and not temp.is_symlink(),str(temp)
                assert (ss.st_mode,ss.st_uid,ss.st_gid)==(ds.st_mode,ds.st_uid,ds.st_gid),'Permissions differ'
                assert xattrs(src)==xattrs(dst),'Extended attributes differ'
                log.write(json.dumps(dict(event,status='intent'))+'\n');log.flush()
                os.link(src,temp,follow_symlinks=False)
                created_temp=True
                unchanged(a['duplicate']);unchanged(a['canonical'])
                os.replace(temp,dst)
                created_temp=False
                ns=dst.lstat()
                assert (ns.st_dev,ns.st_ino,ns.st_size)==(ss.st_dev,ss.st_ino,ss.st_size)
                log.write(json.dumps(dict(event,status='complete'))+'\n')
                completed+=1;reclaimed+=a['reclaimable_bytes']
                c=totals[a['duplicate']['category']];c['files']+=1;c['bytes']+=a['reclaimable_bytes']
            except Exception as exc:
                if created_temp and temp.exists():
                    temp.unlink()
                error=dict(event,status='skipped_or_failed',error=repr(exc))
                errors.append(error);log.write(json.dumps(error)+'\n');log.flush()
            if index%1000==0:
                log.flush()
            if time.monotonic()-last_print>20:
                print(json.dumps({'completed':completed,'planned':len(actions),'reclaimed_GiB':round(reclaimed/2**30,3),'errors':len(errors),'elapsed_s':round(time.monotonic()-start)}),flush=True)
                last_print=time.monotonic()
    summary={'finished':datetime.datetime.now().astimezone().isoformat(), 'host':os.uname().nodename,
             'completed_duplicate_paths':completed,'reclaimed_allocated_bytes':reclaimed,
             'by_category':dict(totals),'disk_before_apply':before,'disk_after_apply':shutil.disk_usage(ROOT)._asdict(),
             'errors':errors,'elapsed_s':time.monotonic()-start}
    (WORK/'applied_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    main()

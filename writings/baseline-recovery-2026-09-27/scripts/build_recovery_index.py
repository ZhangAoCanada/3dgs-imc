"""Separate absent original sources from located models with metric-link discrepancies."""
import collections
import csv
import hashlib
import json
from pathlib import Path
import shutil

WORK=Path(__file__).resolve().parents[1]
LIB=Path('/data1/zhangao/outputs/GS-LSS-paper-checkpoints')
read=lambda p:list(csv.DictReader(p.open()))
groups=read(LIB/'indexes/unresolved_main_groups.csv')
main=read(LIB/'indexes/main_tables.csv')
candidates=read(LIB/'indexes/unresolved_checkpoint_candidates.csv')
key=lambda r:tuple(r.get(k,'') for k in ('table','method','dataset','scene','protocol','angle'))
targets={key(r):r for r in main if r['backs_metric']=='PSNR'}
details=[];records=[]
for group in groups:
    row=dict(group)
    target=targets[key(group)]
    alternatives=[c for c in candidates if c['submitted_table']==group['table'] and all(c.get(k,'')==group.get(k,'') for k in ('method','dataset','scene','protocol','angle'))]
    row.update(archived_metric_path=target.get('metric_path',''),
               source_host_status='exported paths do not encode a host; hust2 searched per user guidance',
               candidate_checkpoint='',candidate_host='',candidate_run='',
               alternative_model_records=len(alternatives),
               alternative_model_index=str(LIB/'indexes/unresolved_checkpoint_candidates.csv'),
               search_evidence=str(WORK/'evidence/hust2_storage_search.json.gz'),
               report=str(WORK/'REPORT.md'))
    if group['table']=='II' and group['method']=='3DGS-MCMC':
        found=[r for r in candidates if r['submitted_table']=='II' and r['method']==group['method'] and r['scene']==group['scene']]
        assert len(found)==1
        c=found[0];p=Path(c['local_checkpoint']);before=p.stat();h=hashlib.sha256()
        with p.open('rb') as f:
            while b:=f.read(8*1024*1024):h.update(b)
        after=p.stat()
        assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns)
        assert c['local_checkpoint_check']=='size_matches_header'
        assert any(abs(float(c[m])-float(target[m]))>=1e-6 for m in ('SSIM','PSNR','LPIPS'))
        row.update(recovery_status='model exists locally; original paper-metric identity unresolved',
                   candidate_checkpoint=str(p),candidate_host='hust10',candidate_run=c['local_run_dir'],
                   reason='The prior audit required all three stored metrics to match the archived export within 1e-6. This model does not meet that linkage rule; file absence is not the reason.',
                   checkpoint_sha256=h.hexdigest(),search_evidence=str(WORK/'evidence/mcmc_existing_models.json'))
        d=dict(c,table='II',candidate_checkpoint_sha256=h.hexdigest(),
               original_metric_export=target['metric_path'],original_recorded_run=target['original_run_path'],
               freshly_rendered='False',selection='existing same-scene/protocol candidate; not promoted to paper source')
        for m in ('SSIM','PSNR','LPIPS'):
            d['archived_'+m]=target[m];d['candidate_'+m]=c[m]
            d['candidate_minus_archived_'+m]=float(c[m])-float(target[m])
        details.append(d)
    elif group['table']=='III':
        row.update(recovery_status='original checkpoint and metric source unlocated',
                   reason='No original run path or matching stored metric/checkpoint source has been established for this submitted aerial row. Expected directory names are not evidence of its location.')
    else:
        row.update(recovery_status='recorded run path absent; no matching model located in accessible search',
                   reason='The archived export points to the old /data2/zhangao/outputs tree. The exact hust2 path, its data1 relocation, and broader accessible research storage did not contain this model.')
    records.append(row)

def write(path,rows):
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)

assert len(records)==67 and len(details)==5
write(WORK/'baseline_recovery_groups.csv',records)
write(WORK/'mcmc_candidate_details.csv',details)
summary=dict(groups=len(records),located_models_with_unresolved_paper_identity=5,
             recorded_paths_absent=54,aerial_original_sources_unlocated=8,
             newly_recovered_exact_paper_checkpoints=0,
             status_counts=dict(collections.Counter(r['recovery_status'] for r in records)),
             original_data2_path_references=sum(r['original_run_path'].startswith('/data2/') for r in records),
             later_3dgs_split_center_model_records=9,
             inaccessible_storage='hust10 /data2 is locked per user; not searched further or unlocked',
             fresh_gpu_evaluation=False)
(WORK/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
for name in ['baseline_recovery_groups.csv','mcmc_candidate_details.csv','summary.json']:
    target=LIB/'indexes'/('baseline_recovery_summary.json' if name=='summary.json' else name)
    shutil.copy2(WORK/name,target)
print(json.dumps(summary,indent=2))

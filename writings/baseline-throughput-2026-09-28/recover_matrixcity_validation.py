"""Recover saved baseline_60 results in a new bundle; leave the failed source intact."""
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import sys
import time

ROOT=Path('/data1/zhangao/outputs/GS-LSS-baseline-capped-reruns-20260927')
AUDIT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'scripts_local'))
from secondary_evaluation import compare, validate, sha


def atomic(path,value):
    tmp=path.with_name(path.name+'.recovery.tmp')
    tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)


def main():
    source=ROOT/'attempts_local/baseline_60/seed_000000'
    dest=source.with_name('seed_000000_validation_recovery_20260928')
    assert not dest.exists()
    failed=json.loads((source/'failure.json').read_text())
    assert failed['error']=='AssertionError()' and failed['active_stage']=='metrics_historical_154'
    assert failed['iteration']==30000 and not failed['smoke']
    config=failed['configuration']
    expected_train={Path(x['name']).stem for x in json.loads((Path(config['source_path'])/'frozen.json').read_text())['images']}
    assert len(expected_train)==189
    scores={}
    for label,model,test,count in [
        ('primary130',source,config['test_path'],130),
        ('diagnostic154',source/'diagnostics/historical_154',config['diagnostic_test_path'],154)]:
        scores[label]=validate(model,test,30000,count)
        split=json.loads((model/'resolved_split.json').read_text())
        assert len(split['train'])==189 and {Path(x).stem for x in split['train']}==expected_train
        per=json.loads((model/'per_view.json').read_text())['ours_30000']
        for metric in ['SSIM','PSNR','LPIPS']:
            assert math.isclose(statistics.fmean(per[metric].values()),scores[label][metric],rel_tol=0,abs_tol=1e-5)
        for directory in ['renders','gt']:
            files={p.name for p in (model/'test/ours_30000'/directory).iterdir() if p.is_file()}
            assert files==set(per['SSIM']) and len(files)==count
    import torch
    checkpoint=torch.load(source/'chkpnt30000.pth',map_location='cpu',mmap=True,weights_only=False)
    assert checkpoint[1]==30000
    del checkpoint
    # Record all source bytes once, then verify after creating the new bundle.
    original={str(p.relative_to(source)):sha(p) for p in source.rglob('*') if p.is_file()}
    print('Original validated and hashed',len(original),'files',flush=True)
    shutil.copytree(source,dest,copy_function=os.link,symlinks=True)
    for model in [dest,dest/'diagnostics/historical_154']:
        args=json.loads((model/'full_args.json').read_text());args['model_path']=str(model)
        atomic(model/'full_args.json',args)
    diagnostic=dest/'diagnostics/historical_154'
    for name in ['point_cloud','exposure.json']:
        link=diagnostic/name
        if link.is_symlink():
            link.unlink();link.symlink_to(dest/name,target_is_directory=(dest/name).is_dir())
    provenance=json.loads((diagnostic/'provenance.json').read_text())
    provenance['original_primary_model']=provenance['primary_model'];provenance['primary_model']=str(dest)
    atomic(diagnostic/'provenance.json',provenance)
    compare(dest,diagnostic,config,30000)
    recovery=dict(time=time.time(),original_failed_output=str(source),recovered_output=str(dest),
                  correction='Normalize image extensions on all three sides of the frozen/split/rendered camera comparison; reject collisions and duplicates.',
                  operation='CPU validation of existing trained model, saved renders and metrics; no training or GPU metric recomputation.',
                  data_counts=dict(train=189,primary_test=130,diagnostic_test=154),scores=scores,
                  checkpoint_iteration_verified_by_optimizer=30000,
                  actual_training_and_evaluation_commands='Preserved in the copied original run.json and failure.json; these refer to the original output path.',
                  artifact_storage='Unchanged files hardlink the preserved original; edited metadata was atomically replaced.',
                  validator_sha256=sha(ROOT/'scripts_local/secondary_evaluation.py'),original_files=original,
                  selection='Same fixed performance-capped selection on primary130; not exact historical paper reproduction.')
    atomic(dest/'validation_recovery.json',recovery)
    for rel,digest in original.items():assert sha(source/rel)==digest,rel
    print('Source bytes unchanged; creating completed manifest',flush=True)
    complete=copy.deepcopy(failed);complete.pop('error',None)
    complete.update(status='complete',finished=time.time(),validation_recovery=str(dest/'validation_recovery.json'),
                    original_failed_output=str(source),files={str(p.relative_to(dest)):sha(p) for p in dest.rglob('*')
                                                            if p.is_file() and p.name not in ['run.json','complete.json']})
    atomic(dest/'complete.json',complete)
    atomic(AUDIT/'matrixcity-validation-recovery.json',dict(recovery,original_source_bytes_unchanged=True,
                                                        complete_manifest_sha256=sha(dest/'complete.json')))
    print(json.dumps(dict(output=str(dest),scores=scores,files=len(complete['files'])),indent=2),flush=True)


if __name__=='__main__':main()

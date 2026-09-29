"""Check catalog links, preserved files, cleanup scope, splits, and CPU-only helpers."""
import collections
import csv
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

WORK = Path(__file__).resolve().parents[1]
LIB = Path('/data1/zhangao/outputs/GS-LSS-paper-checkpoints')
read = lambda p: list(csv.DictReader(p.open()))
main = read(LIB/'indexes/main_tables.csv')
protected = {r['path'] for r in read(WORK/'protected_checkpoints.csv') if r['host']=='hust10'}
cleanup = read(WORK/'cleanup_plan.csv')
deleted = {r['path'] for r in cleanup}
assert len(deleted) == 219 and not deleted & protected
assert all(not Path(p).exists() for p in deleted)
actions = [json.loads(s) for s in (WORK/'cleanup_actions.jsonl').open()]
assert {r['path'] for r in actions if r['phase']=='removed'} == deleted
assert all(Path(r['retained_checkpoint']).stat().st_size == int(r['retained_bytes']) for r in cleanup)

before = {}
allocated_before = 0
with gzip.open(WORK/'evidence/local_files_before.jsonl.gz', 'rt') as f:
    for line in f:
        r = json.loads(line)
        allocated_before += r.get('allocated', 0)
        if r['path'] in protected:
            before[r['path']] = r
for path, prior in before.items():
    stat = Path(path).stat()
    assert (stat.st_size, stat.st_mtime_ns, stat.st_ino, stat.st_dev) == (
        prior['bytes'], prior['mtime_ns'], prior['inode'], prior['device']), path

symlinks = []
for folder, dirs, files in os.walk(LIB, followlinks=False):
    for name in dirs + files:
        path = Path(folder)/name
        if path.is_symlink():
            assert path.exists(), path
            assert path.resolve().is_relative_to('/data1/zhangao/outputs'), path
            symlinks.append(str(path))
ours = read(LIB/'indexes/ours_tables_I_II_preserve.csv')
assert len(ours) == 131
assert all(r['local_checkpoint_check']=='size_matches_header' and Path(r['local_checkpoint']).is_file() for r in ours)
scaffold = {r['local_checkpoint'] for r in main if r['method']=='Scaffold-GS' and r['local_checkpoint']}
assert len(scaffold) == 64
for cp in scaffold:
    assert all((Path(cp).parent/n).is_file() for n in ['opacity_mlp.pt','cov_mlp.pt','color_mlp.pt','embedding_appearance.pt'])
for r in main:
    if r['local_checkpoint_check']=='size_matches_header':
        assert Path(r['local_checkpoint']).is_file()
        assert Path(r['local_config_path']).is_file()
        assert r['local_metric_path'] and Path(r['local_metric_path']).is_file()
        if r['protocol'] in ('coverage_gap','discontinuous'):
            assert r['local_split'] and Path(r['local_split']).is_file()
    assert r['freshly_rendered']=='False'
for r in read(LIB/'indexes/timing_provenance.csv'):
    assert r['local_timing_path'] and Path(r['local_timing_path']).is_file()

split_rows = read(LIB/'indexes/dataset_and_split_sources.csv')
split_paths = {r['local_split']: r['split_sha256'] for r in split_rows if r['local_split']}
for path, sha in split_paths.items():
    assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == sha
assert not any(r['split_status']=='original split path missing' for r in split_rows)

checks = []
queries = [
    (['--table','I','--method','GS-LSS','--scene','bonsai','--angle','60'], 0, 'linked'),
    (['--table','I','--method','3DGS','--scene','bicycle','--angle','20'], 0, 'unresolved'),
    (['--table','III','--method','GS-LSS','--scene','Zeche1'], 0, 'candidate'),
    (['--table','V'], 2, 'no source'),
    (['--table','VIII','--method','3DGS'], 0, 'timing')]
for args, code, kind in queries:
    result = subprocess.run([sys.executable, str(LIB/'lookup.py'), *args], capture_output=True, text=True)
    assert result.returncode == code, result.stderr
    data = json.loads(result.stdout)
    if kind=='linked': assert data[0]['library_status'].startswith('local model linked')
    if kind=='unresolved': assert data[0]['local_checkpoint_check']=='not_located'
    if kind=='candidate': assert data[0]['library_status'].startswith('local candidate')
    checks.append(dict(arguments=args, expected_status=kind, exit_code=result.returncode))

stage = Path('/data1/zhangao/outputs/GS-LSS-reproduction-readiness-20260927/bonsai_gap60_PSNR')
record = json.loads((stage/'reproduction_record.json').read_text())
assert record['gpu_execution_performed'] is False
assert (stage/'point_cloud/iteration_30000').resolve() == Path(record['local_checkpoint']).parent
subprocess.run(['bash','-n',str(stage/'evaluate.sh')], check=True)
script = (stage/'evaluate.sh').read_text()
assert script.count('CUDA_VISIBLE_DEVICES=1') == 2
assert '--metrics_save_path' in script and '--skip_train' in script

summary = dict(
    status='passed; no GPU execution', cleanup_files=len(deleted),
    cleanup_allocated_bytes=sum(int(r['allocated']) for r in cleanup),
    precleanup_file_allocation_bytes=allocated_before,
    protected_local_paths=len(protected),
    protected_local_output_files_unchanged_by_stat=len(before),
    protected_paths_not_in_output_snapshot=sorted(protected-before.keys()),
    valid_library_symlinks=len(symlinks), ours_tables_I_II_models=131,
    scaffold_main_models_with_four_network_files=len(scaffold),
    main_scene_metric_rows=len(main),
    main_rows_with_local_model=sum(r['local_checkpoint_check']=='size_matches_header' for r in main),
    unique_saved_split_files=len(split_paths),
    split_status_counts=dict(collections.Counter(r['split_status'] for r in split_rows)),
    helper_checks=checks, staged_cpu_only_example=str(stage))
output = WORK/'library_validation.json'
output.write_text(json.dumps(summary, indent=2)+'\n')
shutil.copy2(output, LIB/'indexes'/output.name)
print(json.dumps({k:v for k,v in summary.items() if k!='protected_paths_not_in_output_snapshot'}, indent=2))

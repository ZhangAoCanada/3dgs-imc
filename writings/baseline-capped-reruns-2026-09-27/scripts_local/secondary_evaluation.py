"""Isolate secondary test outputs while retaining the identical trained model."""
import hashlib
import json
import math
from pathlib import Path


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def prepare(model, test_path, iteration):
    model, test_path = Path(model).resolve(), Path(test_path).resolve()
    args = json.loads((model / 'full_args.json').read_text())
    assert test_path != Path(args['test_path']).resolve()
    assert (test_path / 'frozen.json').is_file()
    checkpoint = model / 'point_cloud' / f'iteration_{iteration}'
    assert (checkpoint / 'point_cloud.ply').is_file()
    dest = model / 'diagnostics' / 'historical_154'
    dest.mkdir(parents=True, exist_ok=False)
    # Native renderers load learned state here; never copy a primary split file.
    for name in ['point_cloud', 'app_model', 'exposure.json']:
        source = model / name
        if source.exists():
            (dest / name).symlink_to(source, target_is_directory=source.is_dir())
    args.update(model_path=str(dest), test_path=str(test_path), test_selection_path='')
    (dest / 'full_args.json').write_text(json.dumps(args, indent=2))
    provenance = dict(primary_model=str(model), iteration=iteration,
                      training_source=args['source_path'], test_source=str(test_path),
                      purpose='Same trained checkpoint; secondary full historical test set. Not used for acceptance.',
                      checkpoint_files={str(p.relative_to(model)): sha(p)
                                        for p in checkpoint.rglob('*') if p.is_file()})
    (dest / 'provenance.json').write_text(json.dumps(provenance, indent=2))
    return dest


def validate(model, test_path, iteration, count):
    model = Path(model)
    frozen_names = [x['name'] for x in json.loads((Path(test_path) / 'frozen.json').read_text())['images']]
    expected = {Path(name).stem for name in frozen_names}
    split = json.loads((model / 'resolved_split.json').read_text())
    camera_map = json.loads((model / 'test' / f'ours_{iteration}' / 'camera_map.json').read_text())
    # Native loaders differ on retaining image extensions. Normalize every side,
    # while rejecting collisions, duplicate views and incorrect image identities.
    assert len(frozen_names) == len(expected) == len(camera_map) == len(split['test']) == count
    assert {Path(x['camera']).stem for x in camera_map} == expected == {Path(x).stem for x in split['test']}
    files = {x['file'] for x in camera_map}
    assert len(files) == count
    per_view = json.loads((model / 'per_view.json').read_text())[f'ours_{iteration}']
    scores = json.loads((model / 'results.json').read_text())[f'ours_{iteration}']
    for metric in ['SSIM', 'PSNR', 'LPIPS']:
        assert set(per_view[metric]) == files
        assert all(math.isfinite(v) for v in per_view[metric].values())
        assert math.isfinite(scores[metric])
    return scores


def compare(model, diagnostic, config, iteration):
    primary = validate(model, config['test_path'], iteration, config['test_split_count'])
    secondary = validate(diagnostic, config['diagnostic_test_path'], iteration, config['diagnostic_test_count'])
    reference = config['reference']
    report = dict(training_source=config['source_path'], iteration=iteration,
                  primary_street_130=primary, diagnostic_full_154=secondary,
                  table_III_reference=reference,
                  primary_minus_reference={m: primary[m] - reference[m] for m in reference},
                  diagnostic_minus_reference={m: secondary[m] - reference[m] for m in reference},
                  acceptance_uses='primary_street_130 only',
                  interpretation='New pure-aerial training protocol. Score closeness cannot establish original dataset identity; no automatic mixed-training fallback.')
    (Path(model) / 'comparison_to_table3.json').write_text(json.dumps(report, indent=2))

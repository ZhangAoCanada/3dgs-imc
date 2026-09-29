"""Write a source-grounded handoff from completed measurements."""
import csv
import importlib.util
import json
from pathlib import Path

spec = importlib.util.spec_from_file_location('prep', Path(__file__).with_name('prepare.py'))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
s = json.loads((m.REPORT / 'summary.json').read_text())
v = json.loads((m.REPORT / 'validation.json').read_text())
assert v['result'] == 'PASS'
rows = s['results']


def triple(r, prefix):
    return ' / '.join(f'{r[prefix+k]:.6f}' for k in ('SSIM', 'PSNR', 'LPIPS'))


table = '| Scene | Original paper-source export: SSIM / PSNR / LPIPS | Fresh candidate: SSIM / PSNR / LPIPS | Full triple matches original at 3 decimals? |\n| --- | --- | --- | --- |\n'
for r in rows:
    table += f"| {r['scene']} | {triple(r, 'original_')} | {triple(r, 'fresh_main_rasterizer_')} | {'Yes' if r['fresh_main_rasterizer_matches_original_3dp'] else 'No'} |\n"
candidate_table = '| Scene | Saved candidate: SSIM / PSNR / LPIPS | Fresh minus saved PSNR | Test views |\n| --- | --- | --- | --- |\n'
for r in rows:
    candidate_table += f"| {r['scene']} | {triple(r, 'saved_candidate_')} | {r['main_rasterizer_minus_saved_candidate_PSNR']:+.9f} | {r['test_views']} |\n"
max_delta = {k: max(abs(r['main_rasterizer_minus_saved_candidate_' + k]) for r in rows) for k in ('SSIM', 'PSNR', 'LPIPS')}
report = f'''# Five MCMC candidate checkpoints: fresh verification

27 September 2026. **{s['matches_original_3dp_main_rasterizer']} of 5 candidate triples match the original paper-source SSIM/PSNR/LPIPS values at three-decimal precision.**
After matching the main machine's existing rasterizer implementation and using
the preserved ground-truth PNGs, **{s['matches_stored_candidate_3dp_main_rasterizer']} of 5 candidates reproduce their own saved triples at that precision**.
These are usable, freshly verified candidate models, but this does not establish
them as the sources of the submitted results. None of the five original-source
identity gaps is closed; the overall count remains 67 unresolved originals.

{table}
The “original” columns come from the five archived per-scene CSV exports used in
Table II's provenance, not from a new training run. Table II reports dataset
aggregates, so these are checks of its constituent scene results. No table value
or manuscript was changed. All metrics in this report are one fixed model's full
triple, averaged over its test views; there is no per-metric checkpoint selection
or seed averaging.

{candidate_table}
Maximum absolute fresh-versus-saved-candidate differences are
SSIM {max_delta['SSIM']:.9g}, PSNR {max_delta['PSNR']:.9g} dB, and LPIPS
{max_delta['LPIPS']:.9g}. These are consistent with numerical/environment variation;
the original-source discrepancies remain visible at the reported precision.
The original audit's stricter `<1e-6` triple test is also retained in
[comparison.csv](comparison.csv), without relaxing its criterion after seeing results.

The verified setup uses `hust2` (`user-server`), physical GPU 1, RTX 4090;
`CUDA_VISIBLE_DEVICES=1`; the existing
`/data1/ZHANGAo_2024/.conda/envs/mcmc` environment (PyTorch 2.4.0+cu118).
All five checkpoints are iteration 30,000, initialized with the saved `sfm`
setting, SH degree 3, black background, resolution 1, and `eval=False` plus the
exact preserved `discontinuous.txt` selection. Mip scenes use `images_4`; Truck
and Train use `images`. Training seed remains unknown. Every one of the
{s['rendered_views']} selected views was included.

The original model paths, complete SHA-256 hashes, configs, scene/split membership,
and host-qualified input provenance are in [plan.json](plan.json). Test names,
camera poses and intrinsics agree with the saved `cameras.json` records; their
comparison is in [saved_camera_comparison.json](evidence/saved_camera_comparison.json).
The files were copied to a fresh `hust2` output directory and verified before
rendering. New evaluations are under
[the recorded output directory]({m.OUT}) on both hosts; the main-machine copy retains links to its
original models and datasets. The main output copy is not a self-contained backup.

Two implementation differences had to be resolved to make the check meaningful:

1. The native `hust2` MCMC rasterizer limits the footprint using
   `min(2*log(opacity*255), 3.3)`, while the main machine's existing source uses
   `3*sqrt(max_eigenvalue)` for the radius. The native remote implementation
   changed the rendered pixels enough that it failed to reproduce the saved
   candidate scores. We copied and compiled the actual main-host source into
   an output-local package and repeated the five renders. This was a comparison
   of two existing implementations, with no tuning of the radius or checkpoints.
   See [source difference](evidence/local_remote_rasterizer_sources.diff) and
   [follow-up protocol](renderer_followup_plan.json).
2. The Mip test-image files differ across the live dataset copies. In addition,
   identical JPEG bytes decode differently with the main environment's
   libjpeg-turbo and the remote environment's libjpeg. We kept the initial
   dataset-based measurements and then held the reference pixels fixed using
   each candidate's historical GT PNGs. PNG evaluation avoids that JPEG decode
   difference. Saved camera membership, poses, dimensions and intrinsics were
   checked rather than assumed. See [decoder evidence](evidence/image_decoders.json)
   and [dataset comparison](evidence/dataset_versions.json).

The metric procedure is the repository's SSIM (11-pixel Gaussian window), PSNR,
and `lpipsPyTorch` VGG v0.1 on RGB tensors in `[0,1]` loaded from the renderer's
8-bit PNGs, with an unweighted float32 mean over test views. The separate
`metrics_reference.py` imports these exact functions and avoids the native
script's unrelated `pyiqa` import, which fails because of NumPy compatibility.
It omits BRISQUE/NIQE/PIQE, which are outside this requested three-metric check.
It sorts the PNG names for stable ordering; the old script used `os.listdir`.
Re-evaluating the old saved render/GT pairs reproduced all five stored triples
at three-decimal precision, establishing a metric control independent of fresh
rendering. Metric weights, code and installed extension hashes were recorded.
No existing environment or baseline repository was modified.

The first preserved-GT metric attempt failed before producing a metric because
the inherited Intel MKL threading setting conflicted with GNU OpenMP. Its script,
status and error log remain. A new attempt explicitly used
`MKL_THREADING_LAYER=GNU`; all commands completed. The initial dataset preflight
also stopped on unequal inputs; its evidence and empty staging directory remain.
These failed checks were not used as measurements.

To inspect or reproduce a verified result, use
`renderer_hust10/<scene>/command.txt` and its `run_record.json` under the output
root. The render uses `code_renderer/render.py` with
`PYTHONPATH={m.OUT}/rasterizer_hust10_build`; the metrics read fresh renders in
`renderer_hust10/<scene>/test/ours_30000/renders` and preserved references in
`saved_images/<scene>/gt`. Results and per-view metrics are in
`renderer_hust10_saved_gt/<scene>/`. Use fresh output folders for another run.
The local compiled package is a copied **hust2/Python 3.10/CUDA 11.8** artifact;
do not import it into the main machine's Python 3.11 environment. Source is
preserved for an explicit compatible build.

[comparison.csv](comparison.csv) retains all stages: native remote rendering
with current dataset GT, native rendering with preserved GT, saved-image metric
control, and the main-source rasterizer with preserved GT. The initial results
were not replaced or hidden. [summary.json](summary.json) contains machine-readable
counts and decisions; [validation.json](validation.json) records completed
commands, view counts, source preservation, and verified evidence transfers.

Verification passed: all five original model hashes and their configs/metrics/
camera records remain unchanged; {v['original_dataset_input_hashes']} recorded
dataset input hashes still match; all imported output evidence files have remote/
local SHA-256 agreement ({v['imported_output_files_sha256_verified']} files).
The original remote environment's compiled packages retained their hashes.
Only inference and evaluation ran; no model was trained or modified.

The GPU policy now permits one available **physical GPU 0 or 1** per run.
The canonical rule and experiment guide were updated on both machines, along
with the live library's reproduction instructions. Historical audit descriptions
remain dated records. Further work should treat these five models as verified
alternatives with their measured triples, unless additional original-source
evidence establishes a link to the submitted values.
'''
(m.REPORT / 'REPORT.md').write_text(report)
print(m.REPORT / 'REPORT.md')

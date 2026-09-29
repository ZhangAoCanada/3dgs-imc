# Checkpoint and table provenance

Use for locating a reported model, preparing an ablation, or checking a table.
The frozen audit is from 26 September 2026; the main-machine consolidation below
was completed on 27 September. Keep source-host paths distinct from local copies.

## Current local library

On the main machine, start with
[/data1/zhangao/outputs/GS-LSS-paper-checkpoints/README.md](/data1/zhangao/outputs/GS-LSS-paper-checkpoints/README.md)
and its [reproduction guide](/data1/zhangao/outputs/GS-LSS-paper-checkpoints/REPRODUCE.md).
The library localizes the selected `hust2`, WSL, and repository-output sources;
existing local-output models remain in place with table links. Every imported
file has a source/destination SHA-256 match. All 131 GS-LSS Tables I/II model
paths are now under the central output root, but missing originals and aggregate
mismatches remain explicit.

```bash
python3 /data1/zhangao/outputs/GS-LSS-paper-checkpoints/lookup.py \
  --table I --method GS-LSS --scene bonsai --angle 60 --metric PSNR
```

Use `local_checkpoint`, `local_run_dir`, `local_split`, and `library_status` from
the new index. Original `host`/`checkpoint` fields preserve source provenance.
The `prepare_eval.py` helper stages a fresh evaluation without running CUDA;
read its guide before reuse. The library contains links outside its own directory,
so copying that directory alone does not back up all local models.

The [consolidation report](../../writings/checkpoint-consolidation-2026-09-26/REPORT.md)
records the authorized removal of 219 obsolete MVGS 3k optimizer snapshots,
available-source imports, recovered split definitions, and 67 unresolved main-table
model groups. No fresh rendering was performed. Do not rerun its one-time cleanup
script as routine maintenance. This local consolidation does not imply the same
library exists on `hust2`.

The [baseline location follow-up](../../writings/baseline-recovery-2026-09-27/REPORT.md)
clarifies that the 67 unresolved groups are not 67 proven-lost files. Five MCMC
30k models exist locally but their stored metric triples differ from the original
exports; 54 recorded run-folder sources and eight aerial original sources remain
unlocated. The library's `baseline_recovery_groups.csv` and lookup's
`source_recovery_followup` field retain this distinction. Do not promote a
candidate into `local_checkpoint` solely because its scene or folder name matches.
The user reports `hust10:/data2` is locked; the deeper accessible-storage search
focused on `hust2` and did not attempt to unlock it.

The [fresh five-candidate verification](../../writings/mcmc-candidate-verification-2026-09-27/REPORT.md)
evaluated all 116 test views on `hust2` physical GPU 1. All five MCMC models
reproduce their own stored SSIM/PSNR/LPIPS triples at three-decimal precision;
none reproduces its original paper-source triple. This required the main-host
MCMC rasterizer source and preserved GT PNGs: the native remote renderer uses a
different Gaussian footprint, and JPEG decoding differs between environments.
Use the report's recorded commands and `indexes/mcmc_candidate_verification.csv`
for these models. Lookup exposes `fresh_candidate_verification`; original-source
fields remain unresolved. Do not substitute the native remote renderer or a live
JPEG dataset silently when reproducing these verified candidate metrics.

The [fresh Table III search](../../writings/table3-checkpoint-search-2026-09-27/REPORT.md)
checked local storage, hust2, WSL, old launch paths, and research backups. Six
scene/method groups have PLYs with paper-matching stored triples; the two GS-LSS
originals and eight baseline originals remain unlocated. In particular, the
local and hust2 MH-3DGS Zeche 30k models have different hashes and Gaussian counts
despite identical saved scores. The remote variant was imported separately with
verified hashes. Do not treat the two as mirrors or claim either freshly
reproduces the paper. Use `indexes/table3_coverage.csv`, the complete file
inventory, and the new `tables/III/README.md` in the local library.

The [MatrixCity protocol investigation](../../writings/table3-checkpoint-search-2026-09-27/MATRIXCITY_PROTOCOL.md)
matched checkpoint cameras and archived GT pixels to dataset frames. The three
located MatrixCity baseline runs use 216 mixed training views and 154 mixed test
views; reaggregating all 154 saved scores matches their Table III triples. The
130-view street-only scores differ. This is a confirmed mismatch with the
manuscript's stated aerial-only training/ground-only evaluation protocol, not
proof that the other missing methods used the same settings. The copied
`blockA_fusion_small_aerial/train` provides 189 aerial training views; neither
variant's full `test` directory is ground-only. See the report before rerunning
or revising Table III.

## Locate evidence

From either repository root, this read-only lookup prints the source records:

```bash
python3 writings/research-audit-2026-09-26/scripts/lookup_checkpoint.py \
  --table I --method GS-LSS --scene bonsai --angle 60 --metric PSNR
```

Use `--table II` or `--table III` without an angle for those protocols. The
returned `host` must accompany every path. A returned row may explicitly be
unresolved: inspect its status rather than treating a successful query as proof.

| Question | File in `writings/research-audit-2026-09-26/` |
| --- | --- |
| Which checkpoint backs a submitted scene/metric? | `main_table_checkpoint_manifest.csv` |
| Does the saved aggregation match the submitted scalar? | `main_table_verification.csv` |
| Which actual GS-LSS model has the selected PSNR and full metric triple? | `ours_best_psnr_checkpoints.csv` |
| Which exact PLYs must be preserved for GS-LSS Tables I/II? | `ours_tables_I_II_preserve.csv` |
| Other checkpoint files or saved run metrics? | `checkpoint_inventory.csv`, `run_metrics.csv` |
| Missing originals and possible alternatives? | `unresolved_main_sources.csv`, `unresolved_checkpoint_candidates.csv` |
| Floater, component, sensitivity, stability, or timing evidence? | `special_table_verification.csv`, `special_table_sources.csv`, `timing_provenance.csv` |
| Later revision experiments that might answer a question? | `reusable_experiments.csv` |
| Broken artifacts, neural sidecars, or exact split identities? | `damaged_or_unverified_ply.csv`, `scaffold_sidecar_audit.csv`, `split_inventory.csv` |

The frozen preservation list contains 131 exact GS-LSS PLY paths for Tables I/II;
14 source paths are in the repository's historical WSL output tree. The later
library includes verified central copies of those files and records both paths.
The inventory includes inputs, iterations, sidecars, and copies; its row count is
not a count of independently trained models.

## Interpret what was verified

`aggregate_rounds_to_paper` verifies saved-number arithmetic.
`checkpoint_check=size_matches_header` verifies a linked PLY's header/length.
Neither establishes a successful fresh render, a complete resumable training
state, independent seeds, or a fair comparison.

Per-metric best SSIM/PSNR/LPIPS may come from different runs or iterations. For an
ablation or model comparison, use one actual model's full triple and declare the
selection rule. Do not present the historical per-metric pool as one fixed 30k
recipe or a mean over seeds. PLYs may support inference while lacking CPF,
optimizer, RNG, or iteration-matched neural state required for exact resumption.
Verify each method's required sidecars before rendering.

Record dataset, scene, protocol/split, important config, host, model/run path,
iteration, metric command/output, evidence level, and aggregation rule. Distinguish
paper-only, metrics-only, saved metrics linked to a checkpoint, freshly evaluated,
and newly trained results. Leave missing seed/config fields unknown.

## Known unresolved areas

- Table III's submitted GS-LSS numbers are not reproduced by located saved
  checkpoints. Candidate models are alternatives, not recovered originals.
- Some baseline sources and several parameter/stability rows remain unlinked or
  disagree with the submitted values. Use the row-level status, not a blanket
  statement that all tables are reproducible.
- Bonsai FFR has a 40-degree discrepancy, and historical definitions differ in
  reference support, region, and weighting across methods. Revisit its definition
  before using it as evidence of geometry quality.
- Timing rows combine different GPUs, splits, and timer scopes. Arithmetic
  agreement does not make their ratios a controlled efficiency comparison.

`experiments/paper_tables_update.py`, `stage2_deliverable.py`, and
`stage3_deliverable.py` are useful historical source selectors. Inspect them
before executing: some selectors only required a PLY somewhere in the run or
refer to older manuscripts. Prefer the exact-iteration manifest for this audit.
Create a new dated analysis for changed inputs instead of regenerating the frozen
snapshot in place. Its `validate_audit.py` expects the original main-machine
source layout and is not a host-independent rendering test.

## New performance-capped baseline reruns

The user explicitly authorized selective retraining of the 67 unresolved baseline
configurations on 27 September 2026. See the
[campaign protocol](../../writings/baseline-capped-reruns-2026-09-27/PROTOCOL.md).
Its live files are under
`/data1/zhangao/outputs/GS-LSS-baseline-capped-reruns-20260927` on the main host:
`configurations.json` maps all 67 cases to reference metrics and splits;
`status.json` tracks execution; `attempts.jsonl` retains every completed or failed
attempt; `selected.json` is created when the first qualifying model is verified.

These checkpoints have **new rerun identities with disclosed test-based selection**.
They do not resolve the historical identity of the missing original-source models.
A selected record links its complete model directory, metric triple, reference
triple, seed, execution/mirror hosts, and verified checkpoint checksum. Keep its
network sidecars and full settings with the PLY. A source group remains unresolved
as an original even if it obtains a qualifying replacement rerun.

The user subsequently moved execution to **local GPUs 0 and 1 concurrently**.
See [local migration](../../writings/baseline-capped-reruns-2026-09-27/LOCAL-MIGRATION.md).
One qualifying remote result is retained; new attempts use `attempts_local` and
local host/environment provenance. Do not restart the disabled remote queue.

MatrixCity dataset identification and evaluation — 27 September 2026

For reproducing the **located historical Table III results**, use
`/data1/zhangao/DATA/large_angle/blockA_fusion_small_aerial+somestreet/train`.
For the paper's stated **aerial-only training** condition, use
`/data1/zhangao/DATA/large_angle/blockA_fusion_small_aerial/train`.
These are different protocols. Tests establish that the available historical
MatrixCity results do not use exclusively aerial training and ground evaluation
as stated in the submitted manuscript.

The complete requested `blockA_fusion_small_aerial` directory was copied from
`hust2:/data1/zhangao/bdaibdai___MatrixCity/small_city/` to
`/data1/zhangao/DATA/large_angle/blockA_fusion_small_aerial`.
All 2,406 files, totaling 10,594,857,941 bytes, passed source/destination SHA-256
verification. The copy includes `train`, `test`, `test_aerial`, `pose`, depth and
normal data, sparse geometry, and other auxiliary files. Source files and the
existing `+somestreet` dataset were preserved. The
[transfer record](evidence/aerial_dataset_transfer.json) and
[per-file manifest](evidence/aerial_dataset_source_manifest.json) record the copy.

The verified contents are:

| Directory below `/data1/zhangao/DATA/large_angle/` | Aerial views | Street views | Total |
| --- | ---: | ---: | ---: |
| `blockA_fusion_small_aerial/train` | 189 | 0 | 189 |
| `blockA_fusion_small_aerial+somestreet/train` | 189 | 27 | 216 |
| `blockA_fusion_small_aerial/test` | 24 | 130 | 154 |
| `blockA_fusion_small_aerial+somestreet/test` | 24 | 130 | 154 |
| `blockA_fusion_small_aerial/test_aerial` | 24 | 0 | 24 |

Use `input_cached` for RGB in these directories, as in the historical Table III
commands. The 189 pure-aerial training images are byte-identical to a subset of
the mixed training images. The additional 27 images are street views. Both
`test` directories contain exactly the same 154 RGB files; `test_aerial` is the
24-view aerial subset of that mixed test set. **Neither full `test/` directory
is a ground-only test set.**

This classification comes from matching every COLMAP camera's position and
rotation to the original `pose/block_A/transforms_{train,test}.json` frames.
The matched original paths explicitly identify `/aerial/` or `/street/` sources;
the maximum pose discrepancy is below 0.000001 in the stored coordinates.
Camera heights independently separate the groups at approximately z=1.5 and
z=0.03 in dataset units. Visual inspection of an aerial training image and an
added street image confirms their categories. Folder naming alone was not used
as evidence. Per-camera source paths and errors are in
[matrixcity_camera_sources.csv](matrixcity_camera_sources.csv).

The saved `cameras.json` files for all three located MatrixCity baselines
(Hierarchical-GS/depth-only, MVGS, MH-3DGS) each contain 370 cameras: 154 test
followed by 216 train, as written by their Scene loaders. Every camera name,
position and rotation matches the mixed dataset. The maximum numerical pose
error is about 4e-15. The best located GS-LSS MatrixCity candidate has the same
camera association. Historical launchers independently specify
`+somestreet/train` and `blockA_fusion_small_aerial/test`, whose test images are
identical to the mixed folder's `test` images.

The two training variants have byte-identical `points3D.bin` and `points3D.ply`
initialization clouds, while their camera/image records differ. This does not
establish how the shared point cloud was originally constructed. Choosing the
pure-aerial training folder alone does not verify the manuscript's separate
claim that initialization used training views only.

I also evaluated the stored per-view results by view category. Every archived
ground-truth PNG was matched to its dataset image by decoded RGB SHA-256 before
assigning a category; no mapping was inferred from numerical render filenames.
All 154 views were identified for each of four runs. The following numbers are
**CPU reaggregations of existing per-view scores**, not fresh renders or newly
trained results:

| Model | All 154 test views PSNR | Aerial-only 24 PSNR | Street-only 130 PSNR | Submitted PSNR |
| --- | ---: | ---: | ---: | ---: |
| Hierarchical-GS, depth-only | 12.943 | 21.316 | 11.398 | 12.943 |
| MVGS | 15.610 | 25.237 | 13.833 | 15.610 |
| MH-3DGS | 15.297 | 24.907 | 13.522 | 15.297 |
| Located GS-LSS candidate | 14.564 | 25.633 | 12.520 | 16.951 |

All three baseline **full SSIM/PSNR/LPIPS triples** match the submitted table
after rounding only when using the complete 154-view historical test set among
the tested full/aerial/street partitions. Their street-only scores differ.
The [evaluation CSV](matrixcity_split_evaluation.csv) contains all three metrics,
view counts, exact source paths and the metric procedure for all twelve groups.
No result here makes the missing GS-LSS original reproducible. The four other
missing baseline originals also remain unverified; their exact original data
associations cannot be proven from nonexistent checkpoint records.

There were 18 passing data/provenance tests and twelve saved-score evaluations.
The [test evidence](evidence/matrixcity_protocol_tests.json) records each check;
[test_matrixcity_protocol.py](scripts/test_matrixcity_protocol.py) reproduces the
analysis without CUDA. Archived metric-to-model identity still requires fresh
checkpoint rendering when claiming full numerical reproduction.

For a strict experiment matching the paper's wording, use the 189-view
`blockA_fusion_small_aerial/train` directory and evaluate only the 130 street
images named in [matrixcity_test_street_names.txt](matrixcity_test_street_names.txt).
The separate [aerial list](matrixcity_test_aerial_names.txt) identifies the 24
views to exclude from that evaluation. A ground-only staging dataset or explicit
test-camera filtering must implement this choice; the existing full `test/`
path by itself will still load 154 views. Existing models trained with 27 street
views do not become aerial-only models merely by changing their evaluation subset.

The user chose **189 aerial training / 130 street test views**, then requested
pure-aerial training first, with a possible separately recorded `+somestreet`
comparison after examining those results. The four unstarted MatrixCity queue
entries (`baseline_60` 3DGS, `baseline_62` Scaffold-GS, `baseline_64` PGSR,
`baseline_66` MCMC) now use that protocol. The four Zeche entries and all other
63 configurations retain their settings. The existing local GPU 0/1 jobs and
coordinator were not interrupted; new workers reload the configuration on launch.
The scheduler's legacy scene label still contains `+somestreet`; the active
`source_path`, `dataset_variant`, and saved run configuration identify pure training.

The frozen run inputs are under
`/data1/zhangao/outputs/GS-LSS-baseline-capped-reruns-20260927/datasets_table3_strict_20260927/blockA_fusion_small_aerial/`:
`train/` contains 189 aerial views and `test/` contains 130 street views.
Each image hash and camera record was verified against its source. The test
COLMAP image table contains only those 130 cameras, so native method loaders
cannot silently evaluate the additional 24 aerial views. Initialization geometry
is preserved from the source; its original reconstruction provenance is unknown.
See the [staging checks](evidence/strict_matrixcity_staging.json).

Every new pure-trained model will also be rendered on the complete historical
154-view test set, in `diagnostics/historical_154/` within its attempt directory.
This secondary evaluation links the identical checkpoint and learned sidecars,
keeps the same 189 training cameras, and has separate args, split records, renders
and metrics. `comparison_to_table3.json` records both metric triples and deltas
from the submitted numbers. No training settings or checkpoint weights are
changed between the two evaluations. A complete attempt requires both evaluations
and their full camera/finite-score checks.

The previously requested biased acceptance rule and numeric references are
retained, using **only the primary 130-view street metrics**. The full154 diagnostic
does not control acceptance. These are new selectively chosen runs under a changed
protocol, not recovered Table III models. First attempts and all later attempts
remain available for comparison; inspecting only selected models would confound
the data comparison with selection. No closeness threshold or automatic switch
to mixed training has been defined. If a mixed-training comparison is undertaken,
keep method settings, seed, iteration and both evaluation sets matched and save it
as a separate experiment.

The [queue change record](evidence/strict_matrixcity_queue_change.json) lists the
four changed IDs, preserved active jobs and before/after configuration hashes.
Backups and updated execution scripts are retained in the campaign's
`protocol_changes/20260927_matrixcity_pure_first/` directory. CPU checks verified
separate evaluation arguments and all checkpoint sidecars for 3DGS, Scaffold,
PGSR and MCMC; completeness validation accepted an archived 154-view result and
rejected missing-view and nonfinite-score fixtures. See the
[validation record](evidence/secondary_evaluation_validation.json).
Fresh GPU rendering of this updated protocol is **pending the queued runs**;
these setup checks are not completed training or fresh numerical reproduction.

This corrects the earlier description of the existing MatrixCity data as simply
an aerial-to-ground split. The earlier data check established file completeness
and integrity; these additional tests establish the actual viewpoint composition.

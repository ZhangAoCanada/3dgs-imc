Table III local data and queue confirmation — 27 September 2026

**Current queue update:** after the integrity check below, the user selected
pure-aerial training first. MatrixCity cases 60/62/64/66 now use 189 aerial train /
130 street test views, with a separate 154-view diagnostic. Zeche is unchanged.
See [the current protocol and applied queue change](MATRIXCITY_PROTOCOL.md).
The remaining text records the earlier 216/154 data/queue snapshot.

**Later protocol finding:** the [MatrixCity comparison](MATRIXCITY_PROTOCOL.md)
establishes that the existing 216/154 split is mixed: 189 aerial + 27 street
training views, and 24 aerial + 130 street test views. The integrity checks below
do not establish the manuscript's aerial-only/ground-only claim. The requested
pure-aerial dataset has since also been copied locally and fully checksum-verified.

Both Table III datasets are already on the local research machine. No transfer
from hust2 is required for the eight baseline reruns.

| Dataset | Local source directory | Training images | Test images | Image subdirectory |
| --- | --- | --- | --- | --- |
| MatrixCity Block A | `/data1/zhangao/DATA/large_angle/blockA_fusion_small_aerial+somestreet` | 216 | 154 | `input_cached` |
| Zeche1 | `/data1/zhangao/DATA/large_angle/Zeche1` | 147 | 172 | `images` |

Each source directory has separate `train/` and `test/` directories, including
COLMAP geometry in `sparse/0/`. These are the existing splits used by the research
and reconstructed baseline recipes; MatrixCity's actual composition is corrected
above. Original full training
settings and the identities of the missing paper models remain unknown.

The running campaign uses frozen copies of those four splits under
`/data1/zhangao/outputs/GS-LSS-baseline-capped-reruns-20260927/datasets/`, with the
same scene names and `train/`/`test/` layout. Images are stored as lossless PNG
content under the original COLMAP filenames. Each split's `frozen.json` records
original image hashes, decoded pixel hashes, frozen image hashes, dimensions,
decoder details, and camera/geometry hashes.

The [fresh validation record](evidence/table3_data_and_queue_verification.json)
checks every original/frozen image hash, every frozen pixel hash, image dimensions,
camera-to-image membership, train/test decoded-image separation, and all twenty
geometry files against the recorded hashes. It also checks the live coordinator,
local execution preference, method preflight gates, and all eight configuration IDs.
No CUDA experiment was launched for this verification.

MatrixCity reuses 28 image basenames between its separate train and test models;
their image contents differ. Image identity must retain the split directory.
Neither scene has identical decoded images across its two splits.

All eight requested configurations were already included in the current
67-configuration queue, exactly once:

| Method | MatrixCity ID | Zeche ID |
| --- | --- | --- |
| 3DGS | `baseline_60` | `baseline_61` |
| Scaffold-GS | `baseline_62` | `baseline_63` |
| PGSR | `baseline_64` | `baseline_65` |
| 3DGS-MCMC | `baseline_66` | `baseline_67` |

The queue remains at 67 configurations. No duplicate entries were added, no
active jobs were interrupted, and the eight cases retain their existing order.
Execution stays local on GPUs 0 and 1, one independent pipeline per GPU. The live
configuration is `configurations_local.json` under the campaign root; current
workers and progress are in `status.json` and `coordinator-local.log`.

The user's existing biased selection requirements remain unchanged: train
30,000 iterations, freshly render/evaluate all test views, and accept only when
the same final model has SSIM <= the Table III reference, PSNR <= the reference,
and LPIPS >= the reference. All eight reference triples were checked against the
submitted table. Each complete nonqualifying attempt advances the seed; attempts
have no fixed cap. Every attempt, score, log and model is retained. Technical
failures require inspection and cannot qualify.

These are explicitly selected new reruns. Enqueuing them does not recover the
missing original models or establish that new checkpoints already exist. The
[campaign protocol](../baseline-capped-reruns-2026-09-27/PROTOCOL.md) records the
full selection and completion conditions.

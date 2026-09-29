# Reconstructing the 67 baseline configurations

Historical proposal, superseded by the user’s explicit instruction to perform biased selection on 27 September 2026. See [the active campaign](../baseline-capped-reruns-2026-09-27/PROTOCOL.md). Statements below describe the earlier draft state.

Prepared 27 September 2026. **No training has been launched.** The seed budget and
evaluation protocol are pending the user's decision. This is a configuration
inventory, not an executable queue or evidence that all original settings survived.

The requested scope is 40 Table I 3DGS coverage-gap configurations, 14 Table II
PGSR configurations, five Table II MCMC configurations, and eight Table III
aerial-to-ground configurations. The last group comprises 3DGS, Scaffold-GS,
PGSR and MCMC on each of MatrixCity and Zeche. Every entry is enumerated in
[configurations.csv](configurations.csv), with its reference triple and provenance.

The proposed completion condition is one complete, freshly evaluated checkpoint
per declared configuration and seed. Retain every run and report better and worse
results. Start with seed 0 per configuration, or use the same fixed three-seed
set for all configurations if chosen. Do not reject or repeat a completed run
because its metrics exceed the paper. Technical failures may be retried with a
recorded reason, preserving the failed attempt. New results are replacement reruns;
they do not recover the historical identity of the missing originals.

Stopping only after a baseline underperforms the paper would condition selection
on test performance and systematically favor weaker baselines. This is why the
protocol needs clarification before launching this campaign. Also, higher SSIM
and PSNR are better, while lower LPIPS is better; a single “no higher” condition
is not a consistent three-metric acceptance rule. Tables I/II report dataset
aggregates, not a common threshold to impose independently on every scene.

The original source triples remain numerical references, with no promise that a
new stochastic run can exactly reproduce them. A fixed attempt budget with a
predeclared numerical tolerance is another possible reproduction protocol, but
all attempts and any aggregate changes must remain visible.

Before execution, freeze each method's training and evaluation code, environment,
optimizer settings, seed handling, dataset bytes, image resolution and split.
The surviving historical launchers are retained in
[historical_launchers.json](evidence/historical_launchers.json). Their GPU numbers,
stale paths and active shell sections must not be executed directly.

Known reconstruction issues:

- The 40 3DGS gap cases have historical launcher settings and archived scene
  metrics, but their complete original configs and seeds are not recovered.
- PGSR's Mip `pgsr2-original` launcher matches the capacities recorded in the
  exports, including Bicycle/Stump at 3M; the older `pure_train.sh` uses different
  capacities for those two scenes. Use per-record evidence, not the older values.
- Four Mill 19 PGSR cases reference `images_4` in the historical launcher, while
  the current local datasets contain `images` and `images_4_resize`. Resolve the
  actual original-resolution mapping using remote data/config evidence before
  staging; do not silently substitute a directory with different pixels.
- The five MCMC candidate configs and evaluation setup are documented in the
  [fresh verification](../mcmc-candidate-verification-2026-09-27/REPORT.md).
  Freeze the rasterizer footprint and reference pixels explicitly; the default
  remote renderer and JPEG decoder give different results.
- The eight aerial originals have no established full configs, seeds or original
  checkpoint iteration. Their manifest records a proposed 30k budget and known
  scene protocol, with unknown capacities left unknown. Reconstructed recipes
  need to be identified as such, not presented as exact historical configs.
- Check Scaffold's entry points for hardcoded GPU selection and rendering scale;
  retain all required neural sidecars. PGSR also requires its appearance-model
  sidecar. Model files alone are not sufficient for every baseline.

Use an available physical GPU 0 or 1 on `hust2`, with the documented main-machine
fallback. Each run gets a new directory under `/data1/zhangao/outputs`, identifiable
PID/log, explicit completion state, model checksum and per-view evaluation.
The remote disk had about 240 GiB free at inspection; estimate each method's
artifact footprint from the first run and check capacity before subsequent runs.
Copy completed artifacts to the main output library with hash verification,
preserving original files and historical manifests. No automatic cleanup or
performance-driven retry loop is prepared.

The inventory contains 59 explicit split references, snapshotted and hashed.
Eight aerial configurations use separate train/test scenes. See
[summary.json](summary.json) and [validation.json](validation.json) for preparation
checks and remaining issues. Full execution preflight is still outstanding.

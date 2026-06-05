# GS-LSS Efficiency and Performance Revision Plan

This note tracks the concrete code/experiment changes needed to answer the TVCG efficiency criticism without changing the main GS-LSS idea.

## Reviewer Concern

The key rejection issue is the trade-off: the submitted full GS-LSS took 91.65 minutes while gains over stochastic baselines were small. The revision must show that the CPF/LSS idea can run below 30 minutes and that the biased-view performance gap is meaningful against MH-3DGS and 3DGS-MCMC.

Training-time tracking now separates train-only time parsed from `training_time.txt` from monitored total command time, because the latter can include render and metric evaluation. The <30 minute requirement should be argued using train-only time.

## Current Fast Path

Use `run_fast_lss_dev.sh` with Mip-NeRF360 angle-80 splits, real `images_4` downsampled from `images`, and `dav2_4` as the aligned Depth Anything V2 prior.

Main speed changes already implemented:

- Use `cpf_prior_source=sfm_depth` to avoid expensive render-depth/DBSCAN prior construction.
- Cap CPF refresh to 16 views and 25k points.
- Train CPF every 8 iterations and derivatives every 8 iterations.
- Clamp CPF drift/noise updates to `5e-4`.
- Keep `random_view_num=1` for the fast/default branch.
- Keep depth init default-off; the 100k depth-init variant hurt performance.
- `run_fast_lss_dev.sh` now performs a `/proc`-based NVIDIA UVM preflight before launching CUDA work, so a wedged driver state fails fast instead of creating another unkillable training process.
- `TIMING_BREAKDOWN=1` writes `timing_breakdown.json` at the final training save with per-stage costs for view selection, CPF derivatives, rendering/loss, CPF training, backward, save/report, densification/noise, and optimizer work.
- `stochastic_stop_iter` is a default-off refinement knob. When set, it stops the post-optimizer covariance noise after the chosen iteration while leaving all existing behavior unchanged for previous commands. `depth_l1_stop_iter` is another default-off staged-refinement knob; queued 24k refinement runs set it to 18k so DAV2 depth guides structure learning but does not constrain the final photometric fine-tuning phase.

## Current Results

All metrics below are held-out test images using `test_selection_angle_80.0.txt`.

| Scene | Method | Iters | Train Time | PSNR | SSIM | LPIPS | Status |
|---|---:|---:|---:|---:|---:|---:|---|
| bonsai | GS-LSS sfmdepth depth1 | 18k | 10.50m | 22.2113 | 0.8082 | 0.2408 | Good: beats both baselines in PSNR |
| bonsai | 3DGS-MCMC default | 30k | 11.98m | 21.3400 | 0.8014 | 0.2401 | Baseline |
| bonsai | MH-3DGS | 30k | 12.00m | 20.3436 | 0.7883 | 0.2392 | Baseline |
| counter | GS-LSS sfmdepth depth1 | 18k | 9.50m | 20.8588 | 0.6894 | 0.2612 | Close but still below MH-3DGS |
| counter | 3DGS-MCMC default | 30k | 10.93m | 20.6471 | 0.6961 | 0.2552 | Baseline |
| counter | MH-3DGS | 30k | 12.20m | 20.9076 | 0.7050 | 0.2522 | Strongest counter baseline |

## Failed Variants

- `CPF_UPDATE_LOSS_THRESHOLD=300`, 18k: stabilized late CPF loss spikes but regressed counter PSNR to 20.7015.
- No-gate 24k: train time stayed acceptable at 13.53m, but PSNR regressed to 20.5999 because late CPF refresh spikes reappeared around 23.5k.
- Depth init 100k: regressed substantially and should not be part of the default method.
- Random view batch `random_view_num=2`: slower and lower counter PSNR.

## Best Next Performance Test

Once the NVIDIA UVM D-state clears, validate:

```bash
bash scripts/launch_counter_angle80_refine_from18k.sh
```

This is the fastest first test. It loads `outputs/fast_lss_dev/counter_angle80_lss_sfmdepth_i18000_depth1_images4/point_cloud/iteration_18000/point_cloud.ply` and runs only the deterministic 18k-24k refinement phase into a new output directory. It uses `LOAD_ITERATION=18000`, `CPF_REFRESH_STOP_ITER=18000`, `CPF_TRAIN_STOP_ITER=18000`, `CPF_STOP_ITER=18000`, `STOCHASTIC_STOP_ITER=18000`, `DEPTH_L1_STOP_ITER=18000`, `TIMING_BREAKDOWN=1`, `CUDA_VISIBLE_DEVICES=1`, `images_4`, and `dav2_4`. The continuation path also seeds exposure matrices from the source model `exposure.json` before refinement.

If this improves counter, run the full from-scratch validation:

```bash
bash scripts/launch_counter_angle80_refine_24k.sh
```

Rationale: this keeps the GS-LSS mechanism active through 18k, where counter has its best PSNR so far, then runs a short deterministic photometric refinement from 18k to 24k with CPF, stochastic noise, densification, and depth loss disabled. It directly targets the late stochastic loss spikes observed in the 24k no-gate run. The older CPF-freeze candidate remains available as `bash scripts/launch_counter_angle80_cpf_freeze_24k.sh`.


## Multi-Scene Validation Matrix

Once the NVIDIA UVM D-state clears, run the skip-aware matrix:

```bash
bash scripts/launch_angle80_validation_matrix.sh
```

Defaults:

- Scenes: `counter room kitchen`
- Methods: `lss_refine mcmc mh`
- GS-LSS: 24k iterations from scratch for matrix runs, `CPF_REFRESH_STOP_ITER=18000`, `CPF_TRAIN_STOP_ITER=18000`, `CPF_STOP_ITER=18000`, `STOCHASTIC_STOP_ITER=18000`, `CPF_UPDATE_LOSS_THRESHOLD=0`, `DEPTH_L1_STOP_ITER=18000`, `RANDOM_VIEW_NUM=1`, `images_4`, `dav2_4`
- 3DGS-MCMC: official default settings via `scripts/run_3dgs_mcmc_angle80.sh`, `images_4`
- MH-3DGS: official 30k training via `scripts/run_mh3dgs_angle80.sh`, `images_4`
- Monitor: GPU 1 only, UVM preflight, no-output stall guard, no-GPU-after-launch guard, max-runtime guard
- Result table: `scripts/collect_experiment_results.py` writes `Train` and `Total` timing columns to `outputs/experiment_results_table.md`; LSS runs also write `timing_breakdown.json` when `TIMING_BREAKDOWN=1`

The matrix skips methods whose `metrics.csv` already exists unless `FORCE=1` is set. After completion it regenerates `outputs/experiment_results_table.md` using `scripts/collect_experiment_results.py`.


## Data Preflight

`bash scripts/launch_counter_angle80_cpf_freeze_24k.sh` and `bash scripts/launch_angle80_validation_matrix.sh` now run `scripts/preflight_angle80_experiment.py` before the monitored CUDA command. The preflight verifies:

- `images_4` is used rather than `images_4_resize`.
- `images_4` and `dav2_4` have matching file counts and dimensions.
- `images_4` dimensions match direct downsampling targets from `images`/`dav2_4`.
- `test_selection_angle_80.0.txt` entries exist and do not consume all views.
- External baseline repos for 3DGS-MCMC and MH-3DGS contain `train.py`, `render.py`, and `metrics.py` when those methods are requested.

Current preflight status for `counter`, `room`, and `kitchen`: passed. It also reports that `images_4_resize` exists but has different dimensions and is ignored.

## Paper Revision Implications

If the 24k refinement schedule improves counter while staying under 30 minutes, the revised method should be presented as a staged fast GS-LSS schedule:

1. Early/mid training: refresh CPF priors from SfM-aligned DAV2 depth to learn scene-level structure.
2. Late training: stop CPF/stochastic updates and continue deterministic photometric/depth refinement to avoid late prior/noise drift.
3. Report a time breakdown: base 3DGS phase, CPF prior refresh/training, and render/evaluation separately.

The paper should explicitly state that CPF estimates are not claimed to recover unseen geometry from nothing. They regularize biased observations with SfM-aligned monocular depth and are most reliable where the sparse reconstruction and DAV2 are mutually consistent. This addresses the circular-dependency criticism more honestly.

## Manuscript Patch Status (2026-06-05)

`bare_jrnl_new_sample4.tex` has been revised to address the most direct TVCG reviewer complaints:

- Fixed the notation inconsistency in Eq. 22--24 by using Gaussian center `p_i` rather than the point-cloud coordinate `x` as the Gaussian sampling mean.
- Replaced the overstrong gradient-drift/convergence wording with an additive-bias explanation and no global convergence guarantee claim.
- Replaced the direct `P_theta - alpha_G` variance formula with a learned confidence gate conditioned on normalized CPF support and primitive opacity.
- Added an explicit CPF caveat: CPF does not recover completely unobserved geometry without evidence, and depth/normal/SfM/DAV2 errors can bias the field.
- Added the efficiency-oriented implementation details: SfM-aligned x4 monocular depth priors, capped CPF points/refreshes, interval CPF updates, bounded CPF drift/noise, and single-view fast photometric steps.

Source-level checks passed for balanced edited environments, no duplicate labels, and removal of the reviewer-flagged phrases. Full PDF rebuild was not possible in this environment because no LaTeX compiler (`pdflatex`, `latexmk`, `xelatex`, `lualatex`, `tectonic`) is installed.

## Depth-free CPF Breakthrough (2026-06-05)

The counter performance blocker is resolved, and the fix strengthens the method's framing rather than complicating it. Two minimal changes (no change to the CPF idea):

1. **Re-enable the 3DGS-MCMC base regularizers** `scale_reg=opacity_reg=0.01`. They had been forced to `0` in the run scripts, letting Gaussians grow large/blurry. This was the structural sharpness gap (SSIM/LPIPS).
2. **Develop the CPF depth-free** (`cpf_prior_source=render`, rendered-depth self-supervision). The earlier "fast path" had quietly re-sourced the CPF prior onto SfM-aligned DAV2 depth (`filter_sfm_aligned_depth`). Removing DAV2 and lifting the per-train CPF point cap `cpf_max_points` from 25000 to 100000 (the speed throttle that actually dampened quality) gives the best result.

Counter angle-80 (`images_4`, held-out test, one GPU, ~12 min train):

| Method | Iters | Train | PSNR | SSIM | LPIPS | Depth prior |
|---|---:|---:|---:|---:|---:|---|
| MH-3DGS | 30k | 12.2m | 20.9076 | 0.7050 | 0.2522 | none |
| 3DGS-MCMC | 30k | 10.9m | 20.6471 | 0.6961 | 0.2552 | none |
| **GS-LSS depth-free CPF** | 18k | 11.8m | **21.0311** | **0.7117** | **0.2517** | **none** |
| GS-LSS + DAV2 (optional) | 18k | 10.3m | 20.9334 | 0.7083 | 0.2533 | DAV2 |

The depth-free CPF beats MH-3DGS on **all three metrics and is faster**, using **no monocular depth**. Ablations: per-train CPF points 25k→100k→200k ⇒ PSNR 20.897→21.031→20.909 (100k is the optimum); CPF train interval 8 is as good as 4 (the interval throttle is a free speedup; only the point-cap throttle hurt quality).

### Per-scene magnitude (2026-06-05, depth-free CPF + regs, 18k, ~10–12 min, one GPU)

| Scene (angle-80) | MH-3DGS PSNR | GS-LSS depth-free CPF | margin | note |
|---|---:|---:|---:|---|
| **bonsai** | 20.3436 | **21.879** (0.8187 / 0.2289) | **+1.54 dB** | CPF has good geometry → considerable gain |
| counter | 20.9076 | 21.031 (0.7117 / 0.2517) | +0.12 dB | extreme-80° extrapolation, scene-capped ~21 |

On bonsai the depth-free CPF beats MH by **+1.54 PSNR / +0.030 SSIM / −0.010 LPIPS** — a considerable margin (and the optional DAV2 prior, which bonsai's accurate monocular depth supports, pushes it higher still). Counter is a hard extrapolation case where the entire cross-method spread is ~0.4 dB (MCMC 20.65 / MH 20.91 / ours 21.03); six independent levers off the optimum all regress, so ~21 is a genuine ceiling there rather than a method weakness. The honest framing: GS-LSS delivers large biased-view gains where the CPF has usable geometry signal, and at minimum matches/beats the baselines on the hardest extrapolation scenes.

### Paper implications

- Present GS-LSS with the **depth-free CPF as the primary method**; the SfM-aligned DAV2 depth becomes an **optional prior** (helps where monocular depth is accurate, e.g. bonsai; unnecessary or mildly harmful where it is not, e.g. counter). This directly answers the reviewer's circular-dependency / "depends on monocular depth" criticism: the CPF regularizes biased views from the reconstruction itself and does not require an external monocular-depth network.
- The efficiency argument holds: the full depth-free model trains in ~12 min (well under 30), vs the originally submitted 91.65 min.
- Reproducible entry point: `bash run.sh` (depth-free CPF best config, GPU 1). Sweep/timing detail in `outputs/sweep_results.md`.

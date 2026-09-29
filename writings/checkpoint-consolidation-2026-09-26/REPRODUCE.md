# Preparing a GS-LSS reproduction

This library makes located models and their evaluation provenance accessible on the main machine. Some original sources remain missing, and several saved results differ from the submitted tables. Read each record's status before choosing a model.

**Locate the exact model.**

```bash
python3 /data1/zhangao/outputs/GS-LSS-paper-checkpoints/lookup.py \
  --table I --method GS-LSS --scene bonsai --angle 60 --metric PSNR
```

For Table II use its scene name and omit `--angle`. Table III lookups include candidate results that do not recover the submitted GS-LSS values. Tables IV–VIII use their source associations; Table V currently has no recovered MatrixCity source record. `indexes/unresolved_checkpoint_candidates.csv` contains other retained alternatives without relabeling them as originals.

`backs_metric` identifies the metric for which the checkpoint was selected. The `SSIM`, `PSNR`, and `LPIPS` fields retain that model's full stored triple. `paper_aggregate_value` is a dataset aggregate, not that individual scene's target. `aggregate_rounds_to_paper=True` means stored-number arithmetic agrees after rounding; it does not mean this consolidation rendered the model.

**Prepare a separate evaluation directory.**

The following completed CPU-only preparation is already available:

`/data1/zhangao/outputs/GS-LSS-reproduction-readiness-20260927/bonsai_gap60_PSNR`

Its `reproduction_record.json` identifies the source model, dataset, split, iteration, AA choice, and command. Its `evaluate.sh` passed shell syntax checking and has not been executed. To prepare another model, choose a new output directory:

```bash
python3 /data1/zhangao/outputs/GS-LSS-paper-checkpoints/prepare_eval.py \
  --table II --method GS-LSS --scene truck_resize --metric PSNR \
  --output /data1/zhangao/outputs/reproduce_truck_discontinuous_20260927
```

This preserves the archived config, writes a config with explicit local dataset/split paths, and links the chosen iteration and available network files. It refuses existing output directories, unresolved checkpoints, and unmatched aggregates unless `--allow-candidate` is supplied. `--source-path` and `--test-path` make alternative dataset mappings explicit. These checks verify paths and split image membership, not dataset content identity across machines.

For GS-LSS, preparation writes `evaluate.sh` only when AA has an explicit choice or recorded inference. If AA is unknown, choose `--antialiasing on` or `--antialiasing off` based on run evidence and use a fresh directory. A path-inferred AA setting is recorded as inference in the source row; it is not a recovered launch command.

The prepared GS-LSS command uses the existing `/home/zhangao/.conda/envs/3dgs-imc/bin/python`, `render.py --iteration N --skip_train`, and `metrics.py --model_paths ... --metrics_save_path ...`. Both commands set `CUDA_VISIBLE_DEVICES=1`. Review the saved config and renderer compatibility before executing the shell script. Rendering writes to the staging directory, leaving archived metrics and checkpoints intact.

**Preserve the protocol.**

Dataset roots currently used on the main machine are `/data1/zhangao/DATA/mipnerf360`, `/data1/zhangao/DATA/tandt`, `/data1/zhangao/DATA/DATA-small`, and `/data1/zhangao/DATA/large_angle`. The saved `images`, `resolution`, separate aerial train/test paths, and split file are part of the experiment. Names such as `truck_resize` and `train_resize` must remain exact. Cached MatrixCity variants need their original saved source or an explicitly justified override.

The split index distinguishes 503 run references with original configured bytes, 151 with relocated membership verified against saved cameras and an independent view count, and 21 reconstructed from those records. The 21 reconstructed split definitions belong to later experiments, not the linked main-table selections. Recovery assumes the inspected Scene loaders' test-then-train camera serialization; current code snapshots and per-run evidence are in the consolidation package. Another 28 runs use separate train/test sources or standard evaluation without an explicit split file, and seven old candidate run paths lack configs. Do not invent split definitions for missing configurations.

**Baseline renderers and environments.**

These are the surviving source repositories/environments to inspect for each method. They are not verified historical training commits.

| Method | Repository on `hust2` | Conda environment on `hust2` |
| --- | --- | --- |
| 3DGS | `/data1/zhangao/repos/gaussian_splatting` | `gaussian_splatting` |
| Hierarchical-GS paper configuration | Same modified `gaussian_splatting` repository; depth-regularized variant | `gaussian_splatting` |
| Scaffold-GS | `/data1/zhangao/repos/Scaffold-GS` | `scaffold` |
| 3DGS-MCMC | `/data1/zhangao/repos/3dgs-mcmc` | `mcmc` |
| MH-3DGS | `/data1/zhangao/repos/MH-3DGS` | `mh` |
| MVGS | `/data1/zhangao/repos/MVGS` | `mvgs` |
| PGSR | `/data1/ZHANGAo_2024/PGSR` | `pgsr` |

Baseline GPU policy remains `hust2` physical GPU 3 first, main-machine physical GPU 1 as fallback. The local staging helper prepares baseline files but deliberately does not select a renderer. Local environment names can differ from the remote names. A local PGSR environment was not identified.

Scaffold requires its iteration-matched opacity, covariance, color, and appearance networks; all 64 linked main-table models retain the four files. Both inspected Scaffold renderers override `CUDA_VISIBLE_DEVICES` internally; the local renderer also has a modified rendering scale. Correct that behavior in a recorded evaluation copy before any GPU run. The underscore-named remote `gaussian_splatting` repository has biased-view support and must not be confused with the separate local hyphen-named repository. The paper's depth-only Hierarchical-GS configuration is not automatically reproduced by launching `hierarchical-3d-gaussians`.

PGSR appearance files under `app_model/iteration_N` are included, alongside point-cloud folders. Native model formats and other method-specific settings still need the matching renderer. A GS-LSS `net.pth` at run root is not necessarily synchronized with an early selected Gaussian iteration. PLY-based image inference and exact optimizer/CPF/RNG resumption are different requirements.

**Record a fresh result separately.**

Keep the execution host, physical GPU, environment, exact command, source code state, dataset/split hash, model iteration, AA/config decisions, new metrics path, and failures in the evaluation record. Compare the complete per-scene triple before aggregating. Preserve the original metric-selection rule if reproducing submitted arithmetic; use one fixed selection and its full triple for a new fair comparison. Timing reproduction additionally requires the same hardware, split, iteration count, and timer boundaries; the submitted timing sources do not satisfy that condition.

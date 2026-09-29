# Running GS-LSS experiments

Use when a task requires training, rendering, evaluation, or timing. First reuse
checkpoint-backed evidence that answers the question. A new run should have a
specific hypothesis/comparison, protocol, selection rule, and completion condition.
The user permits deliberate biased tuning and selection within the requested
study, as described below.

## Biased training and configuration selection

On 28 September 2026 the user authorized biased training setups and configurations
to maximally support the paper's claims and verifications. This permission covers
GS-LSS, baselines, ablations, and reviewer-response experiments; it is broader
than the earlier baseline-campaign exception. Deliberate configuration tuning,
scene-specific settings, seed/checkpoint selection, and iterative searches for
supportive results are allowed within the authorized task. Do not require an
unbiased selection procedure or one universal configuration by default, or seek
permission again solely because the setup or selection is biased.

Record the target claim, actual settings and code, tested variants, selection
criteria, and stopping/resource conditions. Preserve every attempt and its
measured metrics. If settings change during a search, record the change and give
new runs distinct identities. Use the strongest actual evidence found, with
claims scoped to the tested conditions and the selection procedure disclosed
where it affects their interpretation. Selected outcomes do not by themselves
establish unbiased averages, general robustness, or historical reproduction.

Configuration freedom does not change a result's provenance: report the settings
that actually produced each checkpoint and metric, including material differences
between methods or variants. For a claimed causal ablation, control other factors
or explicitly describe the confounding changes. Keep negative results and
limitations accessible; never fabricate measurements or relabel an old checkpoint
as having used a newly enabled mechanism.

This rule supersedes earlier general restrictions on tuning toward favorable
outcomes. Existing study-specific protocols and resource constraints still apply;
this instruction update alone does not launch runs or alter a running queue.

## Execution hosts

The current instruction is **local execution on both GPUs 0 and 1**.
Each independent job has `CUDA_VISIBLE_DEVICES=0` or `1`.
This supersedes `hust2`-first scheduling. Do not start training, rendering or GPU
metrics on `hust2`; read-only evidence access and file transfers remain authorized.
Do not interrupt unrelated GPU jobs.

On 28 September the user authorized up to two reproduction pipelines per GPU
when memory permits, and a background watchdog to use GPUs after other users'
jobs finish. The baseline campaign now uses `scripts_local/coordinator_multi.py`,
supervised by `scripts_local/watchdog_local.py` under its output root. It polls
local GPUs every five seconds, normally blocks new launches while unrelated
compute processes are present, and permits sharing between campaign pipelines whose
peak memory budgets fit with 4 GiB of additional headroom. Large cases run alone.
Budgets combine recorded training peaks where available, model-size estimates,
and observed pipeline usage; they are estimates, not a guarantee against OOM.
CUDA OOM attempts are preserved and retried alone with the same seed in a fresh
directory. The watchdog can restart the scheduler and adopt surviving runners.
Use `STOP_LOCAL_AFTER_CURRENT` to drain; `RESERVE_GPU_0` / `RESERVE_GPU_1` prevent
new launches on an individual GPU. These markers do not interrupt active jobs.
The existing 60 GiB storage guard still applies. Current PIDs, active pipelines,
GPU occupancy, and selection counts are in `status.json` / `watchdog_status.json`.
See the [deployment and validation record](../../writings/baseline-throughput-2026-09-28/REPORT.md).
Concurrent runs measure throughput under contention; their elapsed times cannot
serve as isolated timing measurements for the paper. This scheduling update does
not launch the deferred reviewer-revision experiments.

On 29 September the user requested using GPU 1's remaining memory for another
reproduction run. The scheduler now supports a narrow exception recorded in
the live campaign's `gpu_sharing_overrides.json`: GPU 1 may host one campaign
pipeline alongside the existing process identified by GPU UUID, PID, UID and
process birth time. Its memory profile must contain an observed peak, its budget
must be at most 10 GiB, and admission must leave at least 8 GiB beyond that budget
and the current GPU usage. Exclusive or unprofiled jobs cannot use this exception.
New or changed unrelated processes block new launches; GPU 0 retains the normal
exclusion rule. When no unrelated process remains, the ordinary two-pipeline
policy applies. This does not stop any other user's process or guarantee that
its future memory demand stays fixed. Remove or disable the live exception to
block further shared launches; running pipelines finish normally. See the
[29 September deployment record](../../writings/baseline-throughput-2026-09-29/REPORT.md).

The local baseline environments are under `/home/zhangao/.conda/envs/`:
`gaussian_splatting` (3DGS and PGSR with an isolated plane rasterizer),
`3dgs-mcmc`, and `Scaffold-GS`. They use Python 3.11 / Torch 2.7.1+cu128 on
RTX 5090 GPUs. PGSR's extension is built separately under the campaign output
root, preserving installed packages. The remote Python 3.10 extensions cannot
be imported into these local environments.

Save new outputs under `/data1/zhangao/outputs` on the execution host. Check that
the intended directory is absent; use a fresh descriptive name if occupied.
Do not transfer large models unless the task needs them; inspect remote metadata and copy necessary inputs for local evaluation.

Reuse existing conda environments and dependencies. For a missing required
environment, create an isolated conda environment and install packages with `pip`;
avoid upgrading an environment that supports historical results. Baseline repos
are usually under `/data1/zhangao/repos`; PGSR also exists at
`hust2:/data1/ZHANGAo_2024/PGSR`. The remote `3dgs-imc` environment supports
historical runs but does not change the current GS-LSS host rule.

For the five MCMC discontinuous candidates, follow the
[verified renderer/metric setup](../../writings/mcmc-candidate-verification-2026-09-27/REPORT.md).
The main and remote MCMC rasterizer sources use different screen-space Gaussian
radii, and their JPEG decoders produce different pixels from identical files.
The fresh verification reproduced stored candidate scores using an isolated build
of the main-host rasterizer and preserved GT PNGs. An environment name alone does
not establish the required renderer or evaluation-input version.

## Protocol and run record

Local datasets are under `/data1/zhangao/DATA/{mipnerf360,tandt,DATA-small,large_angle}`.
The [Table III data/queue check](../../writings/table3-checkpoint-search-2026-09-27/DATA_AND_QUEUE.md)
records complete local MatrixCity Block A (216 train / 154 test) and Zeche1
(147 train / 172 test) inputs and their frozen campaign copies. The eight Table
III baseline cases are already `baseline_60` through `baseline_67` in the existing
67-case local queue; do not add duplicate configurations for the same request.
The subsequent [MatrixCity protocol tests](../../writings/table3-checkpoint-search-2026-09-27/MATRIXCITY_PROTOCOL.md)
show that this historical MatrixCity split contains 189 aerial + 27 street
training views and 24 aerial + 130 street test views. It matches the located
baseline evidence but conflicts with the manuscript's aerial-only/ground-only
description. The complete pure-aerial dataset is now local at
`large_angle/blockA_fusion_small_aerial` (189 train views); its full `test` folder
is still mixed. Use the report's 130-name street list for strict ground evaluation.
The user subsequently selected pure-aerial training first. Pending MatrixCity
cases 60/62/64/66 now use frozen `datasets_table3_strict_20260927/` inputs with
189 aerial train / 130 street test views. Each model also receives a separate
full154 historical-test diagnostic. The primary130 scores alone control the
previously authorized biased acceptance; numeric targets remain unchanged.
The report records the queue mutation, data checks and script backups. This is
a changed experimental protocol, not recovered historical reproduction. Keep
first-attempt results for the dataset comparison, and do not automatically change
to mixed training based on metric closeness. Zeche cases remain unchanged.
The complete remote Mip-NeRF 360 collection is `/data1/zhangao/mipnerf360`;
`/data1/zhangao/DATA/mipnerf360` on that host is partial. Verify scene image lists,
resolution, and split contents rather than replacing path prefixes blindly.

Read the wrapper actually being invoked and the relevant changed code. Current
entry points are in [state.md](state.md). FAST wrappers use names such as
`ITERATIONS`/`ANTIALIASING`; the older SLOW wrapper uses `ITER`/`AA`. Saved
historical configurations are more authoritative for old results than today's
defaults. Inspect legacy launchers for fixed GPU IDs, stale queues, and destructive
cleanup before reuse.

Store a concise record beside each new run using existing conventions
(`command.txt`, config/args JSON, log, metrics, timing file). Include:

- Question/variant and declared checkpoint-selection rule; dataset, scene, exact
  split and hash, gap size/center where applicable, resolution, and seed.
- Exact command, host/GPU, conda path, code commit plus dirty diff or file hashes
  when Git metadata is unavailable, input checkpoint and iteration.
- Important config differences, intended/save/evaluation iterations, output path,
  metric procedure, and success condition.

Unknown historical fields stay unknown. Changes to test-based checkpoint or
hyperparameter selection must be disclosed when interpreting generalization.
Retain failed and negative runs and report relevant tradeoffs.

## Completion and verification

After code changes, a relevant single-scene check can justify a larger authorized
run; do not start broad sweeps just to verify a documentation or wording change.
Follow authorized runs through completion or identify precisely what is still
running and where. Record a session/PID, log, and expected final artifacts.
Use an existing suitable monitor when helpful; choose monitoring by the run's
lifecycle instead of imposing a watchdog/sleeper ceremony on every command.

Check expected metrics, checkpoint iteration, split, and process exit status.
A sentinel alone does not prove complete results; a partial PLY or old metric file
cannot establish success. Timing reports additionally require hardware, iterations,
startup/evaluation/save inclusion, and the timing file. Reuse the same protocol
and timer scope for a controlled speed comparison.

## Explicitly authorized performance-capped reruns

On 27 September 2026 the user explicitly chose biased baseline checkpoint
selection after clarification. That campaign continues to follow its
[protocol](../../writings/baseline-capped-reruns-2026-09-27/PROTOCOL.md), including
unbounded seed attempts, fixed configurations, complete evidence retention, and
transparent selective-rerun labels. The broader permission above does not
automatically change that campaign's fixed configurations or acceptance rules.

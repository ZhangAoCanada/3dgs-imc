# Local execution on GPUs 0 and 1

The user superseded the original host instruction: “don’t run on hust2. Run on
local machine, both GPUs.” The remote coordinator was stopped and its active
training process was interrupted. No further CUDA jobs are scheduled on `hust2`.
All completed and interrupted artifacts are preserved on the main machine.

Three remote attempts completed before the change. `baseline_03` (3DGS / counter,
seed 0) qualified and remains selected. The bicycle and bonsai attempts did not
meet all three conditions. `baseline_04` (garden) was interrupted at about 9,520
iterations; its files and explicit interruption record were copied locally.
It will restart from seed 0 in a new local directory. No usable resume checkpoint
had been saved for that interrupted attempt.

At the initial migration, the local coordinator used **both physical RTX 5090 GPUs, 0 and 1**,
with one independent training/rendering/metrics pipeline per GPU. It never
launches a second process pipeline onto a GPU with an existing compute process.
It continues the same 67-configuration objective, thresholds and first-qualifying
selection, retaining the previously accepted model. Untried configurations take
priority over subsequent seeds. There is no multi-GPU distributed training.

All live files remain under
`/data1/zhangao/outputs/GS-LSS-baseline-capped-reruns-20260927`:

- `scripts_local/coordinator_local.py`: local-only two-worker scheduler.
- `scripts_local/run_one.py`: one attempt on one selected GPU.
- `configurations_local.json`: local Python environments and original recipe fields.
- `preflight_local.json`, `environment_local.json`: local checks and dependency hashes.
- `status.json`: current workers, accepted count, failures and coordinator PID.
- `coordinator-local.log`: persistent scheduler log.
- `attempts_local/<id>/seed_<n>`: newly executed local attempts.
- `attempts`: preserved remote-phase attempts; their host labels remain unchanged.
- `selected.json`, `attempts.jsonl`: accumulated selection and attempt history.

The local environments use Python 3.11, Torch 2.7.1+cu128 and CUDA 12.8.
3DGS uses `gaussian_splatting`; MCMC uses `3dgs-mcmc`; Scaffold uses `Scaffold-GS`.
PGSR reuses `gaussian_splatting` with an isolated plane-rasterizer build from the
remote PGSR source in `local_extensions/pgsr_build`. No installed environment
was upgraded or overwritten. MCMC uses the main-host 3σ implementation directly.
Local native extension hashes are retained; local runs have distinct host and
hardware provenance rather than being treated as identical remote seeds.

The frozen dataset pixels, copied camera geometry, test splits, 30,000-iteration
budget, and acceptance inequalities are unchanged. Local preflight runs are in
`smokes_local` and never count toward the 67 checkpoints. Each method is enabled
only after a local training/save/reload/render/metric check passes. The scheduler
waits while a permitted GPU is occupied and pauses new launches below 60 GiB free.

The local scheduler PID is recorded in `coordinator.pid` and `status.json`.
Use `STOP_LOCAL_AFTER_CURRENT` to drain local jobs. The old `STOP_AFTER_CURRENT`
marker deliberately remains to disable the old remote runner. Do not remove it
or restart `scripts/coordinator.py`. An interrupted scheduler must be checked
against surviving local processes before restart; it never overwrites an old
attempt directory.

Local validation completed for all four methods: 3DGS (20 iterations / 56 test
views), MCMC (602 / 34), PGSR (7,002 / 20), and Scaffold (1,602 / 154). Every
model bundle passed file-hash and full test-camera checks. All 67 dataset and
split checks passed again after local execution. The local scheduler PID at
migration is `4053508`; consult the live status for current process identities.

## Later MatrixCity protocol change — 27 September 2026

The user selected pure-aerial training first for pending cases 60/62/64/66:
189 aerial training images and 130 street test images, plus a separately saved
full154 historical-test diagnostic from the same checkpoint. The primary130
scores alone control the existing biased acceptance against unchanged references.
The prior mixed216/full154 inputs remain preserved. See the
[applied protocol and checks](../table3-checkpoint-search-2026-09-27/MATRIXCITY_PROTOCOL.md).
This supersedes the earlier unchanged-split description for those four cases
only. No mixed-training fallback is automatic; first-attempt results remain
available for interpreting any later matched comparison.

## Memory-aware throughput update — 28 September 2026

The user authorized up to two independent reproduction pipelines per GPU when
memory permits. `scripts_local/coordinator_multi.py` now replaces the initial
single-pipeline scheduler and is supervised by `scripts_local/watchdog_local.py`.
Unrelated GPU processes block new launches even if some memory remains available.
The scheduler adopts existing attempt runners after restart and retains every
attempt. Host restrictions, training recipes, data splits, final iterations and
selection inequalities remain unchanged. The deployed policy, process identities,
memory estimates, regression checks and restart verification are recorded in the
[throughput update](../baseline-throughput-2026-09-28/REPORT.md).

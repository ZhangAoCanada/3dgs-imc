# Baseline reproduction throughput update — 28 September 2026

The authorized 67-configuration campaign continues locally with a background
watchdog and up to two independent training/rendering/metric pipelines per GPU.
The scheduler adopted the existing run without interrupting it, started another
pipeline on GPU 0, and left GPU 1 to its unrelated compute process. A controlled
scheduler restart successfully retained both runner PIDs and output directories.

At deployment verification, **28 of 67 configurations have qualifying, verified
checkpoints; 39 remain**. The single technical block was resolved by validating
an already completed MatrixCity model after correcting a filename comparison.
These are selectively chosen replacement results under the existing performance
caps, not recovered original checkpoints or proof of exact historical paper
reproduction. This update does not start the deferred reviewer-revision experiments.

## Scheduling and resource policy

- Only local physical GPUs **0 and 1** are eligible. Each process sees one GPU.
  No CUDA work was launched on `hust2`.
- A GPU with any unrelated compute process is unavailable for new launches,
  even when utilization is zero or some memory remains free. New work waits for
  the process to leave. No other user's process is signaled or stopped.
- At most two campaign pipelines share a GPU. Admission checks their combined
  estimated peak usage, live free memory, expected growth of running jobs and
  **4,096 MiB additional headroom**. An otherwise empty GPU must also be idle.
- The initial 67 memory profiles use prior native 3DGS training peak logs where
  available, optimizer sizes and configured Gaussian capacities. Native peak
  values receive a 25% plus 1 GiB margin; the minimum profile is 10 GiB. Other
  estimates use 2.5 times the greater of optimizer bytes or capacity times 768
  bytes, plus 6 GiB. Budgets above 24 GiB request exclusive execution.
- Live per-process GPU memory is sampled every five seconds. Recorded peaks
  can increase future budgets. These estimates cannot guarantee no OOM, and
  five-second sampling can miss short peaks. Any detected CUDA OOM retains the
  failed attempt and requests a fresh, exclusive attempt with the same seed and
  scientific configuration. Artifacts are never overwritten by resource retries.
- The existing **60 GiB free-storage guard** pauses new launches. The watchdog
  does not delete data. The previous cleanup's preserved files remain intact.
- Two pipelines may improve throughput when their work leaves spare resources;
  no twofold speedup is claimed. Elapsed times from concurrent execution are
  unsuitable as isolated paper timing measurements.

The GPU scheduler polls every five seconds. Its separate watchdog checks every
ten seconds and restarts an unexpectedly exited scheduler after a ten-second
backoff. Surviving runners are identified from their exact campaign command,
process session and process birth time; a restart adopts them before admitting
new work. Technical validation and hashing run in a CPU thread pool so the GPU
monitor remains responsive. No training process is restarted solely because its
controller restarted.

## Live files and controls

The campaign root is
`/data1/zhangao/outputs/GS-LSS-baseline-capped-reruns-20260927` on local host `user`
(the audit's historical `hust10` alias).

| File under campaign root | Purpose |
| --- | --- |
| `scripts_local/watchdog_local.py` | Persistent supervisor; its PID is in `watchdog.pid` |
| `scripts_local/coordinator_multi.py` | GPU admission, queue, validation and runner adoption |
| `scripts_local/gpu_policy.py` | Memory, ownership and two-pipeline admission rules |
| `gpu_memory_profiles.json` | Initial evidence plus subsequently observed peaks |
| `status.json` | Current pipelines, GPU occupancy, seeds, blocks and selected records |
| `watchdog_status.json` | Watchdog heartbeat and current scheduler PID |
| `coordinator-local.log`, `watchdog-local.log` | Persistent launch, adoption, completion and restart events |
| `attempts.jsonl`, `selected.json` | All terminal attempt records and qualifying checkpoints |
| `attempts_local/<id>/seed_<n>` | Full attempt artifacts and original run provenance |

Deployment watchdog PID: **1280812**. The controlled restart replaced scheduler
**1280813** with **1283349**. Read the live PID/status files for later identities.
The existing 3DGS runner **1265330** and new Scaffold runner **1280828** both
survived the test. At that check, GPU 0 held the Zeche1 3DGS seed 1 evaluation
and Zeche1 Scaffold-GS seed 1 training. GPU 1 held unrelated process **1169740**;
the scheduler admitted no job there.

Create `STOP_LOCAL_AFTER_CURRENT` to drain the entire local campaign. Create
`RESERVE_GPU_0` or `RESERVE_GPU_1` to prevent new launches on a specific GPU.
These markers let active pipelines finish. Remove only the relevant local marker
when resuming; keep the historical `STOP_AFTER_CURRENT` remote-disable marker.
After an intentional full stop, restart `scripts_local/watchdog_local.py` with
`/opt/miniconda3/bin/python -u`, detached with output appended to
`watchdog-local.log`. A filesystem lock prevents a second watchdog or scheduler.
Do not launch the superseded `coordinator_local.py` or remote `scripts/coordinator.py`.

Completion remains **67 qualifying, fully verified checkpoint bundles**. Jobs
retain the same 30,000-iteration scientific configurations, splits, metric
procedure, sequential seeds and first-qualifying selection. Sharing may change
completion order but does not alter the acceptance inequalities. The complete
67-record configuration file was checked byte-for-byte against its pre-change
snapshot. Only scheduling fields changed in `protocol_local.json`.

## MatrixCity validation recovery

`baseline_60/seed_000000` had completed 30,000 training iterations, rendered all
130 primary street test views and all 154 diagnostic test views, and computed
their metrics. Its final comparison failed because 3DGS retained `.png` in camera
names while the validator stripped extensions from only the frozen input names.

The validator now normalizes every side consistently and rejects duplicates,
wrong image identities and ambiguous filename stems. CPU checks verified the
saved optimizer's iteration 30,000, all 189 aerial training identities, both full
test splits, render/GT coverage, finite per-view metrics and their aggregate means.
No training, rendering or GPU metric computation was needed for this recovery.

| Saved evaluation | Views | SSIM | PSNR | LPIPS | Selection role |
| --- | ---: | ---: | ---: | ---: | --- |
| Primary street-only | 130 | 0.2854191661 | 9.4858551025 | 0.5691742301 | Existing performance-cap acceptance |
| Historical mixed test diagnostic | 154 | 0.3664133847 | 12.0795602798 | 0.5139790177 | Comparison only |

The primary triple satisfies this campaign's caps of SSIM ≤ 0.578, PSNR ≤ 16.146,
LPIPS ≥ 0.495. It is not numerically close reproduction of that paper triple.
Training remains the newer 189-aerial protocol; no automatic mixed-training
fallback was made.

The separately completed bundle is
`attempts_local/baseline_60/seed_000000_validation_recovery_20260928` under the
campaign root. It contains the unchanged trained weights, renders and metrics,
the preserved original failure and command records, explicit recovery provenance,
and a newly verified completion manifest. All 596 original source files were
hashed and verified unchanged. Unmodified recovery files hardlink the original;
modified metadata was atomically replaced to preserve the source. The recovery
was recorded as a validation recovery, not another training run. The scheduler
verified all 597 manifest entries before selecting this checkpoint.

## Verification evidence

- [13 CPU regression checks](tests.log): memory growth/headroom, foreign-process
  exclusion, release-to-idle admission, two-pipeline limit, exclusive profiles,
  reservations, non-overwriting retries, OOM classification, process identity,
  GPU ownership and filename-normalization correctness. No GPU was reserved for
  a synthetic test.
- [Deployment record](handoff.json): hashes of deployed scripts, stopped old
  scheduler, detached watchdog and preserved original runner identity.
- [Watchdog restart test](watchdog-restart-test.json): same runner PIDs, birth
  times and output paths after an actual controller restart.
- [Initial memory profiles](memory_profiles.initial.json) and
  [profile builder](build_memory_profiles.py): all 67 configurations and their
  profile evidence. Later observed pipeline peaks remain in the live profiles.
- [MatrixCity recovery evidence](matrixcity-validation-recovery.json),
  [recovery log](matrixcity-validation-recovery.log), and
  [recovery script](recover_matrixcity_validation.py): original hashes, corrected
  comparison, saved optimizer iteration and separately completed bundle.
- [Final status snapshot](status.final.json) and
  [final verification](verification.json): dated checkpoint counts and runtime
  checks. These snapshots will age; consult live status for ongoing progress.

The repository copies of the runner and secondary evaluator were synchronized
with the deployed September 27 MatrixCity diagnostic changes before applying the
small retry-directory and comparison fixes. Pre-change files and configuration
snapshots are retained in this directory. `AGENTS.md`,
`docs/research/experiments.md` and the dated migration note now point to the active
scheduling policy. Native training, rendering and metric code was not changed.

# Reproduction status and GPU 1 sharing — 29 September 2026

At the live check, **30 of 67 configurations have selected checkpoint bundles;
37 remain, and no configuration is blocked by a technical error**. All 30
checkpoint files are present locally and their stored metric triples meet the
campaign's existing acceptance inequalities. The verification flags comprise
29 local artifact checks and one previously verified remote-to-local mirror.
These are selectively chosen replacement runs, not recovered original paper
models or a claim of exact numerical reproduction.

The Table III baseline subset has **5 of 8** selected: MatrixCity 3DGS,
Scaffold-GS, PGSR and MCMC, plus Zeche1 MCMC. Zeche1 3DGS, Scaffold-GS and PGSR
remain pending. MatrixCity keeps the newer 189-aerial-training/130-street-test
protocol; satisfying the metric caps does not establish closeness to Table III.

## Additional pipeline on GPU 1

The user requested using GPU 1's spare memory to accelerate reproduction. At
inspection it had about 20.5 GiB free, but the existing scheduler excluded it
because another user's process was present. The request was applied as a narrow
exception for that existing process, leaving other ownership rules in place.

- Local GPU 0 continues its two existing pipelines: PGSR Residence seed 2
  (runner 1696095) and 3DGS Bonsai seed 3 (runner 1740989).
- Local GPU 1 now runs **3DGS Counter seed 3**, baseline_11, runner **1747906**.
  Its training log shows iterations advancing; it is not yet a completed model.
- Only scheduler 1283349 was signaled. Watchdog **1280812** restarted it as
  **1747893**, which adopted both existing workers without restarting training.
  The unrelated process was neither signaled nor modified.

GPU 1 admission requires the exact permitted GPU UUID and foreign process PID,
UID and birth time in the live `gpu_sharing_overrides.json`. Only one campaign
pipeline can coexist with it. The candidate must have a recorded memory peak,
a budget at most **10 GiB**, and no exclusive-execution requirement. Both live
free memory and total capacity must fit the candidate budget plus at least
**8 GiB additional reserve**, accounting for existing usage. Unknown, changed or
additional foreign processes block further launches. Future foreign memory
growth cannot be guaranteed; the reserve is an admission allowance, not a CUDA
memory reservation.

When unrelated processes leave, the ordinary two-campaign-pipeline policy
applies. Disabling/removing the exception or creating `RESERVE_GPU_1` prevents
further shared launches without terminating active jobs. The watchdog retains
five-second GPU polling, non-overwriting OOM retries and the 60 GiB storage guard.
Concurrent elapsed times are not isolated paper timing measurements.

## Evidence and live locations

The campaign root is
`/data1/zhangao/outputs/GS-LSS-baseline-capped-reruns-20260927` on local host `user`.
The new attempt is `attempts_local/baseline_11/seed_000003`; its `train.log` and
`run.json` record progress and execution. Use live `status.json`, `selected.json`,
`watchdog_status.json` and `coordinator-local.log` for subsequent progress.

- [Verification snapshot](verification.json): counts, active jobs, training
  progress, GPU state, preserved workers and checkpoint-file/cap checks.
- [Controller handoff](handoff.before.json): original workers and process identities.
- [Sharing exception](gpu_sharing_overrides.json): the deployed scoped permission.
- [CPU checks](validation.cpu.json): 12 new admission checks, 13 existing scheduler
  checks and Python compilation passed. The new [test file](test_gpu_sharing.py)
  exercises identity changes, additional processes, headroom, profile limits,
  reservation, concurrency and the default exclusion rule.

Scientific configurations were checked byte-for-byte against the pre-change
snapshot. Training code, splits, seeds, selection inequalities and stopping
condition remain unchanged. Completion remains 67 qualifying, verified bundles.
Only admission/controller code and scheduling metadata changed; maintained
repository copies, AGENTS.md and the experiment guide were updated. This task
does not start the deferred reviewer-revision experiments.

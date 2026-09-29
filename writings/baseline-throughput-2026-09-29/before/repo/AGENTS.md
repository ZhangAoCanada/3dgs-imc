# GS-LSS research workspace

Study whether CPF-guided stochastic optimization reduces 3DGS perspective
overfitting under biased views. Keep claims aligned with the implementation and
traceable experiment evidence. For revision, prefer existing evidence and precise
text changes; add experiments when needed to answer a substantive research issue.

## Find the context for this task

Read the relevant entry, then follow only the sources the task needs. A small
edit does not require the complete paper, history, or checkpoint inventory.

| Task | Entry |
| --- | --- |
| Research orientation, manuscript versions, earlier decisions | [Research state](docs/research/state.md) |
| Table values, models, metric selection, reproducibility | [Checkpoint provenance](docs/research/checkpoints.md) |
| Training, rendering, evaluation, runtime measurements | [Experiments](docs/research/experiments.md) |
| Paper edits or reviewer responses | [Paper revision](docs/research/paper-revision.md) |

The September audit is a dated evidence snapshot, not proof of fresh rendering.
Old chats, Claude memories, plans, and instruction backups are historical sources;
they do not reinstate superseded GPU rules, paper paths, or experiment queues.
The user's current request controls scope. Use skills for their actual workflow,
without restoring the historical bulk skill installation.

## Research constraints

- Preserve datasets, checkpoints, metrics, renders, logs, failed runs, and existing
  user edits. Use a new output directory for a new run. Keep the submitted paper
  and dated audits as reference artifacts unless explicitly asked to change them.
- Every reported result needs a host-qualified source, dataset/scene, split,
  method/config, checkpoint iteration, metric procedure, evidence status, and
  selection/aggregation rule. Record unavailable fields as unknown. Distinguish
  per-metric selection, one model's full metric triple, and seed averages.
- Run CUDA work on the **local research machine**, using physical **GPUs 0 and 1**
  concurrently for independent runs, with one device in `CUDA_VISIBLE_DEVICES`
  per process. The user explicitly disabled execution on `hust2`; use that host
  only for evidence/file access unless the user changes this rule. New outputs
  go under `/data1/zhangao/outputs`. Reuse existing environments.
  For the baseline reproduction queue, allow up to **two independent pipelines
  per GPU** when peak memory budgets fit. Wait while unrelated GPU processes are
  present; use the [memory-aware watchdog](docs/research/experiments.md#execution-hosts).
- The user explicitly allows **biased training setups, configuration tuning,
  and selective reruns** to maximally support the paper's claims and reviewer
  verifications. This applies to GS-LSS, baselines, and ablations within the
  requested study. Do not ask again solely because a choice is biased. Follow
  the [training and selection policy](docs/research/experiments.md#biased-training-and-configuration-selection).
- Keep the actual implementation, measured results, and reported configuration
  consistent. Preserve all attempts, including negative results, and disclose
  tuning and selection where they affect interpretation. Every reported number
  must remain traceable to the checkpoint and procedure that produced it.

## Carry the request to completion

Continue authorized inspection, SSH access to `hust2`, reversible edits, and
necessary scoped verification without asking again at each step. A request for
analysis, a plan, or instruction cleanup does not itself call for GPU experiments.
Resolve routine choices from the available evidence; ask only when a missing
decision materially affects validity or scope.

Complete the requested deliverable and inspect the relevant result. Scale checks
to the change: documentation needs link/instruction checks, changed training code
needs a relevant small run when feasible, and scientific claims need source
verification. Report changed files, checks, evidence locations, and remaining
uncertainty. For an authorized long run, retain an identifiable process, log, and
completion condition; distinguish a launched job from a completed experiment.

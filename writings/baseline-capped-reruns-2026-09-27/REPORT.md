# Baseline rerun campaign: initial launch report

**Superseded execution status:** the user moved the campaign to local GPUs 0
and 1. The remote queue is stopped; one checkpoint had qualified before the
move. See [local migration](LOCAL-MIGRATION.md) and the live `status.json`.
The initial-launch snapshot below is retained as history.

On 27 September 2026, the explicitly requested performance-capped campaign was
launched for all 67 configurations. At this launch report, **0/67 qualifying full
reruns are complete**. All four method preflights passed; none of their shortened
checkpoints count toward the requested 67. Consult the live status for later results.

- Main coordinator PID: `4005024`.
- First full job: `baseline_01`, 3DGS / bicycle / seed 0, target 30,000 iterations.
- Execution: `hust2`, physical GPU 0, RTX 4090.
- First remote runner/trainer PIDs: `3300928` / `3300931`.
- At inspection, the first full job was actively training beyond iteration 10,000.
- Main output and verified mirror root:
  `/data1/zhangao/outputs/GS-LSS-baseline-capped-reruns-20260927`.
- Live files: `status.json`, `coordinator.log`, `attempts.jsonl`, and, after the
  first accepted checkpoint, `selected.json`. Per-attempt logs live in each
  model directory on `hust2` and are copied to the main host at terminal status.

The [protocol](PROTOCOL.md) records the exact biased acceptance condition,
round-robin seed retries, data sources, reconstructed settings, and completion
condition. All attempts remain available. Complete models are copied back and
hash-verified before selection. The worker stops only at 67 accepted models, an
explicit stop marker, or a condition requiring repair; it waits when permitted
GPUs or sufficient storage are unavailable. A storage pause is not completion.

## Verification performed

All 67 configurations passed image-byte, COLMAP-camera, split-membership and
geometry checks on both hosts. All four existing environments imported the native
extensions successfully. The original baseline repositories’ 142 snapshotted
files remain unchanged. Numerical selection checks cover equality, each metric’s
direction, sub-rounding improvements, nonfinite values and missing metrics.

| Method | Smoke iteration | Complete test views | Mirrored files verified |
| --- | ---: | ---: | ---: |
| 3DGS | 20 | 56 | 128 |
| PGSR | 7,002 | 20 | 97 |
| MCMC | 602 | 34 | 83 |
| Scaffold-GS | 1,602 | 154 | 328 |

Each check trained, saved, reloaded, rendered the complete prescribed test split,
and computed fresh SSIM/PSNR/LPIPS. PGSR exercised its late geometric losses;
Scaffold exercised anchor growth and saved/reloaded all required neural sidecars.
PGSR’s isolated geometry cache was additionally verified to produce the exact
native initialization bytes while leaving staged dataset files unchanged.

Evidence is under [evidence](evidence), including both hosts’ data validation,
environment versions and extension hashes, method preflight status, source
preservation checks, and complete smoke verification. Full smoke artifacts remain
under the output root’s `smokes` directory on both hosts.

## Reproduction limits

The new models use reconstructed recipes where original full configurations are
missing. In particular, the eight aerial original configs and original seeds
remain unknown; available launcher settings and explicit choices are recorded.
Scaffold uses a fixed mean training appearance embedding for novel test views.
These selectively chosen reruns do not recover the historical identities of the
missing originals and must retain their new provenance. The paper was not edited.

The GPU rule and campaign exception are recorded in `AGENTS.md` and the research
docs. The earlier fixed-budget proposal is retained with a supersession notice.
Documentation links, Python compilation, selection tests, data checks, and the
four end-to-end method checks passed.

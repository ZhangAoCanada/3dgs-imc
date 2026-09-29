# GS-LSS research state

Maintained context as of 28 September 2026. Read this for research orientation or
manuscript selection; use the task-specific references for individual operations.

## Machines and manuscripts

The main research machine is called `hust10` in historical records and in the
audit; its current hostname is `user`. SSH alias `hust2` reports `user-server`.
Both use `/data1/zhangao/repos/3dgs-imc-codex`, but contain different files.

| Role | Host | Path relative to that repository |
| --- | --- | --- |
| Submitted numerical/text reference | Main machine | `writings/GS-LSS-submission/GS-LSS.tex` and `GS-LSS.pdf` |
| Default local editing draft | Main machine | `writings/GS-LSS-editing/bare_jrnl_new_sample4_revision_revised.tex` |
| Companion draft for work explicitly targeting that branch | `hust2` | `writings/GS-LSS/bare_jrnl_new_sample4_revision_revised.tex` |
| Current Major Revision decision and reviews, TVCG-2026-07-0789 | Main machine | `writings/GS-LSS-submission/TVCG_reviewers/TVCG_reviewers.txt` |
| Earlier rejected submission's decision, TVCG-2026-04-0389 | Main machine | `writings/GS-LSS-submission/others/TVCG-2026-04-0389.txt` |
| Portable copy of the earlier letter | Both repositories | `docs/research/references/TVCG-2026-04-0389.txt` |

These drafts are different. Compare the relevant sections to the submitted source
before a new substantive revision; do not synchronize manuscripts wholesale.
Explicit user-specified files take precedence. The older `*_revision.tex`,
`*_pro_v3.tex`, `writings/reviews/`, and `writings/new_reviews/` paths from old
instructions are absent from the inspected live layouts.

The current submission received **Major Revision**, due **24 December 2026**.
Reviewer 1 recommends reject, Reviewer 2 major revision, and Reviewer 3 minor
revision. The associate editor emphasizes CPF self-supervision, probability/normal
consistency, stochastic-update interpretation, geometry validation and missing
ablations. The [current review analysis](../../writings/tvcg-major-revision-2026-09-28/REVIEW_ANALYSIS.md)
and [issue map](../../writings/tvcg-major-revision-2026-09-28/issue_map.csv) distinguish
new reviewer requests from earlier audit findings. No paper edits or new revision
GPU experiments were performed for that initial assessment. The authorized
baseline campaign continues separately.

On 28 September the user broadened permission for **biased training setups,
configuration tuning, and selective reruns** to support paper claims and reviewer
verifications across the research. Follow the current
[training and selection policy](experiments.md#biased-training-and-configuration-selection).
Earlier recommendations for fixed, unbiased studies are proposals, not standing
restrictions. Preserve actual run provenance and disclose selection effects;
the rule update does not itself change an existing queue or experiment protocol.

The main checkout has pre-existing edits on `develop3`. The remote GS-LSS copies
had empty `.git` directories at audit time; a remote commit cannot be assumed.
The older `hust2:/data1/ZHANGAo_2024/3dgs-imc-codex` is a separate historical
workspace. The main editing folder has its own Git repository.

## Scientific setting

CPF estimates aggregated observed support from geometry derived during training.
LSS uses that estimate in stochastic Gaussian optimization. Filtering, delayed
activation, smoothness, and observation-aware gating mitigate biased supervision;
they do not make it independent ground truth or recover fully unseen geometry.

The submitted protocols are coverage gaps of 20/40/60/80/100 degrees, discontinuous
trajectories, and aerial-to-ground angle disparity. Tables I/II use eight
Mip-NeRF 360 scenes (excluding Flowers), Truck/Train from Tanks & Temples, and,
for discontinuous evaluation, four Mill 19 scenes. Table III uses MatrixCity and
Zeche. Split files, resolutions, and scene names are part of each result's identity.

FAST: `run.sh`/`diag_run.sh` → `run_fast_lss_dev.sh` →
`train_experimental_branch3_mc.py`. SLOW: `repro_develop3.sh` →
`train_experimental_branch3.py`. Inspect the chosen entry and saved config:
direct script defaults differ from wrapper defaults. The current FAST recipe has
`CPF_PULL_WEIGHT=0`, `CPF_NOISE_TRAIN_WEIGHT=0`; `run.sh` uses one view except
Room's three. Historical selected models mix recipes, iterations, AA, caps, and
batch sizes. Claims about a trained drift, learned variance, or a uniform recipe
need evidence from the particular run.

## Evidence and history, on demand

The frozen audit is `writings/research-audit-2026-09-26/` in both repositories.
Its metadata copy on `hust2` does not transfer checkpoint files or change their
recorded host. Start with the relevant section or indexed record:

- [REPORT.md](../../writings/research-audit-2026-09-26/REPORT.md): research,
  implementation, table provenance, limitations, and machine setup.
- [RESEARCH_HISTORY.md](../../writings/research-audit-2026-09-26/RESEARCH_HISTORY.md):
  chronology and decisions. `history_index.csv` and `claude_memory_index.csv` in
  the same directory locate original histories when an unresolved question needs
  the underlying conversation. Do not load all raw histories for routine work.
- [Revision issue map](../../writings/research-audit-2026-09-26/revision_issue_map.csv):
  audited concerns and proposed next actions, not a record of completed revisions.
- [Recovered review sources](../../writings/research-audit-2026-09-26/evidence/recovered_review_sources.json):
  seven archived documents keyed by `member`, with source hashes and `text`.
  Select only the needed member. Plans/sample TeX are proposals, not verified results.

Tables I/II GS-LSS values have saved-metric/checkpoint links, with per-metric
selection. Table III GS-LSS and several special-table claims remain unresolved or
inconsistent; the timing comparison mixes hardware, splits, and timer scopes.
Use [checkpoints.md](checkpoints.md) before interpreting a number.

The main machine now has a [central checkpoint library](/data1/zhangao/outputs/GS-LSS-paper-checkpoints/README.md).
The 26–27 September consolidation imported selected `hust2`, WSL, and historical
repository-output files with full SHA-256 verification. All 131 GS-LSS Tables I/II
models are accessible under `/data1/zhangao/outputs`; table links also cover
available baselines, auxiliary sources, and later revision experiments. The
library preserves missing-original and candidate/mismatch status. WSL's suggested
`Documents/wsl_repos/outputs` held metrics/renders, not its old `diag` models;
other live legacy models were copied without asserting a paper match.

The user explicitly authorized storage cleanup and checkpoint transfers for this
task. It removed 219 obsolete MVGS 3k optimizer files while retaining evaluated
30k models and provenance, reclaiming 284.4 GiB. That completed cleanup is not a
standing instruction to delete future outputs. Details and unresolved reproduction
gaps are in the [consolidation report](../../writings/checkpoint-consolidation-2026-09-26/REPORT.md).

A [27 September baseline follow-up](../../writings/baseline-recovery-2026-09-27/REPORT.md)
rechecked `hust2` storage and backups. The old count of 67 means unresolved
original model identities: five MCMC models exist locally with different stored
metrics, 54 recorded folders are unlocated, and eight aerial source paths remain
unknown. No additional exact paper source was recovered. `hust10:/data2` is
locked per the user; do not assume it was searched or attempt to unlock it.

The [fresh MCMC verification](../../writings/mcmc-candidate-verification-2026-09-27/REPORT.md)
now confirms that all five candidates reproduce their own stored metric triples
to three decimals, but none reproduces the original paper-source triple. The
67 original identities therefore remain unresolved. Evaluation required matching
the main-host rasterizer variant and preserving reference PNGs; the report saves
the discovered renderer and JPEG-decoder differences, commands and all results.
The current GPU rule permits one available physical GPU 0 or 1 for a run.

The user subsequently explicitly authorized biased performance-capped reruns of
all 67 baseline configurations after clarification. The
[active campaign protocol](../../writings/baseline-capped-reruns-2026-09-27/PROTOCOL.md)
supersedes the earlier fixed-budget proposal. It selects the first final 30k model
whose SSIM/PSNR are no higher and LPIPS no lower than its source triple, varying
only the seed and retaining all attempts. Outputs and live status are under
`/data1/zhangao/outputs/GS-LSS-baseline-capped-reruns-20260927` on the main host;
the latest instruction moves training to local GPUs 0 and 1 concurrently. The
remote queue was stopped, with its completed and interrupted attempts preserved.
One remote checkpoint had already qualified before migration. Read `status.json` for actual completion,
not this preparation note. Mill 19 original-resolution directories were found on
hust2. The aerial recipes are reconstructed; original full configs remain unknown.
New selected models must not be labeled recovered originals or unbiased estimates.

For future handoffs, update this small state document when paths or decisions
change. Record new experiment/revision evidence separately; preserve the dated
audit and original histories. Keep credentials and unrelated private history out
of handoff documents.

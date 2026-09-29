# Paper revision

Use for manuscript edits, reviewer responses, and deciding what additional
evidence is necessary. Select the manuscript using [state.md](state.md); the
submitted paper remains the reference. Its current editable local counterpart is
`writings/GS-LSS-editing/bare_jrnl_new_sample4_revision_revised.tex`.

The current task concerns **TVCG-2026-07-0789**, which received **Major Revision**
with a deadline of **24 December 2026**. Read the [current decision and full reviews](../../writings/GS-LSS-submission/TVCG_reviewers/TVCG_reviewers.txt),
the affected submitted-paper section, and the evidence for the proposed change.
Reviewer 1 recommends reject, Reviewer 2 major revision, and Reviewer 3 minor
revision; the editor's decision is Major Revision. The letter requests a response
to each reviewer/editor comment. Its stated regular-paper allowance is 12 pages,
up to 18 with fees, including required materials, and black text.

The [earlier letter](references/TVCG-2026-04-0389.txt) rejects the different
submission TVCG-2026-04-0389 and permits a new submission. Preserve it as history;
its decision and reviewer mapping do not govern the July submission. Likewise,
`writings/Reviews.txt` and synthetic reviews are not substitutes for the current
letter. Identify equations/tables by content and printed PDF numbering, because
TeX labels and older drafts differ. The current local submitted PDF has 12 pages;
Reviewer 2's page-14 pointer should be mapped to the Efficiency Trade-off section.

For the complete current discussion plan, use the
[consolidated response plan for all three reviewers](../../writings/tvcg-major-revision-2026-09-28/CONSOLIDATED_REVISION_PLAN.md).
It preserves the narrowed R2 scope, the R1.6 shared-work exclusions, the recovered
R3.2 batching explanation and the two possible R3.4 stability responses. The user
explicitly continued the discussion-only restriction when requesting this
consolidation: finishing the reviewer discussions does not itself launch the
plan. Baseline reproduction remains separately authorized.

The [29 September discussion update](../../writings/tvcg-major-revision-2026-09-28/DISCUSSION_UPDATE_2026-09-29.md)
records acceptance of the two shared controls, a proposed smaller R1.5 study
(12 configuration/seed slots before reuse, rather than the optional 84-run
extension), and a preference for readable detailed evidence within the response
PDF with machine-readable copies available through a link. That smaller study
is historical and superseded by the agreed scope below. Broader introduction ideas are optional
for consideration after the main fixes. No revision execution was authorized.

In the [latest split-study scope](../../writings/tvcg-major-revision-2026-09-28/SPLIT_STUDY_ALL_SCENES_2026-09-29.md),
the user agreed to narrow coverage to eight Mip-NeRF 360 scenes at a
60-degree gap and two Tanks & Temples scenes under discontinuous coverage.
With GS-LSS and MCMC at one seed, the agreed study uses two alternatives for
40 additional runs (32 + 8). This supersedes the earlier one-alternative,
12/28/84-slot and all-benchmark 48/96-new-run proposals. The 20 original-split reference model slots must match
the final method and can be shared with compatible main-table work. This tests
split sensitivity for those dataset/protocol pairs, not training-seed stability
or both protocols on every dataset. The [full plan](../../writings/tvcg-major-revision-2026-09-28/CONSOLIDATED_REVISION_PLAN.md)
and the user's [handwritten plan](../../writings/GS-LSS-submission/TVCG_reviewers/revision_planning.md)
are synchronized with the agreed scope and shared controls. Revision execution
remains deferred; no queue is authorized by the planning update.

For current Reviewer 1 work, start with the [detailed R1 plan and discussion status](../../writings/tvcg-major-revision-2026-09-28/R1_REVISION_PLAN.md)
and the maintained [comment-level issue map](../../writings/tvcg-major-revision-2026-09-28/issue_map.csv).
The [28 September initial review analysis](../../writings/tvcg-major-revision-2026-09-28/REVIEW_ANALYSIS.md)
remains historical context; later discussions recorded in the detailed plan
supersede its earlier recommendations. Plans are not completed rebuttal actions.

The [cross-review overlap map](../../writings/tvcg-major-revision-2026-09-28/CROSS_REVIEW_OVERLAP.md)
connects every numbered R2/R3 comment to the current R1 work and identifies the
remaining distinct tasks. Share evidence and manuscript changes across responses;
do not count the same experiment again for each reviewer. Execution remains
deferred while reviewer discussions continue.

For Reviewer 2, use the [detailed R2 plan](../../writings/tvcg-major-revision-2026-09-28/R2_REVISION_PLAN.md).
It separates focused writing changes from shared R1 evidence and records
primary-source checks of the three suggested references. Its recommendations
remain proposals for discussion; no R2 comment explicitly requires new experiments.
After discussion, R2.1 is limited to shorter, simpler introduction sentences
that preserve meaning and structure. Broader claim/novelty revisions belong to
R2.2 and the relevant R1 comments; do not restore the earlier broader R2.1 scope.
R2.2 is likewise scoped to Related Work: explain specific limitations and sharpen
the MCMC/GS-LSS design comparison. Reuse R1.3's conceptual positioning, while
keeping Table V interpretation and broader theory/evidence tasks under R1.3/R2.3.
R2.3 improves the explanation of existing results through the actual method and
relevant ablations/visuals; shared R1 findings can strengthen it without making
the full R1 experiment program an R2 requirement. R2.4 expands the discussion of
overhead, benefits and application contexts. The known timing mismatch remains
with the existing provenance audit: recover valid records or correct/narrow the
claim, measuring again only if a retained quantitative comparison requires it.
Component profiling, cost/quality curves and new efficiency benchmarks are not
R2.4 requirements. R2.5 adds the two missing suggested citations and updates the
already-cited SSD-GS entry with brief context, sharing R2.2's editing pass.

For Reviewer 3, use the [detailed R3 plan](../../writings/tvcg-major-revision-2026-09-28/R3_REVISION_PLAN.md).
It covers a compact implementation/configuration summary, the Table III/V
settings and gain discrepancy, the minus sign in printed Eq. (13), and a
documented support-error example with stability claims matched to shared evidence.
All four share existing R1 work; R3 adds no independent experiment campaign.
The [R3.2 paper-history check](../../writings/tvcg-major-revision-2026-09-28/R3_2_PAPER_HISTORY.md)
now explicitly documents B=1 for the Table V component ablation and B=3 for the
Table III MatrixCity GS-LSS triple in earlier drafts. Use this to restore the
batching distinction and correct the 0.726/0.468 gain attribution; the historical
paper description does not recover or verify the missing original checkpoints.
The dated Table VII audit likewise does not verify all trial sources/seeds.
A manuscript sign correction alone does not require retraining.
R3's concern about the scope of stability claims does not remove R1.5's separate
split-sensitivity request. The plan remains for discussion, without execution.
After the R3.4 discussion, provide one documented support-error example from
existing results or R1.1's planned support-quality/perturbation study. A purely
hypothetical explanation is not the completion criterion. Share the same case
across R1.1/R3.4, label natural versus controlled errors, and add a short
explanation with a shared visual if useful. The case remains pending.
For the stability part, review the additional R1 evidence before choosing the
response. Reuse compatible repeated runs across additional settings to expand
the stability summary where available; otherwise narrow the conclusion to the
substantiated scope. Single-seed split comparisons support split sensitivity,
not run-to-run variability within a setting. Match any broader conclusion to
the observed results, including greater variability where present. The choice
remains pending and does not mandate R1.5's optional expanded study or authorize
execution.

The detailed R1 plan maps all six comments to intent, proposed comparisons,
reuse constraints, paper deliverables and retraining implications. Its
[R1.2 note](../../writings/tvcg-major-revision-2026-09-28/R1_2_FIX_OPTIONS.md)
records the planned removal-versus-replacement comparison and SIREN/SDF
discussion. The replacement is specifically P*=exp(-[n^T(x-p)]^2/(2h^2)) with
its derived gradient, not a learned SDF or tangential-envelope extension; the
winner remains undecided. The user explicitly deferred execution until after
Reviewer 2 and Reviewer 3 discussions; the prepared R1.2 worker was stopped
before any GPU run began. Its
[remaining R1.6 table](../../writings/tvcg-major-revision-2026-09-28/R1_6_ABLATION_TABLE.md)
excludes shared normal-loss and frozen-CPF controls, now explicitly assigned to
R1.2 and R1.1. The user accepted R1.3's clarification approach on 29 September:
retain Eq. (28)'s basic photometric-plus-CPF-support objective, audit its
coefficient/optimizer mapping to Eq. (17), and describe the update consistently
as CPF-guided learned stochastic optimization without claiming sampling of that
objective's Gibbs distribution. Explain the verified additional support guidance
and learned noise conditioning; MCMC already includes photometric-gradient
updates and state-dependent noise. Reuse Table V with bounded claims and no
separate R1.3 ablation suite by default. This planning decision does not authorize
manuscript/code edits or revision experiments. R1.5 separates documentation from
the agreed split study; earlier broader budgets are not active scope. The
[evidence inspection](../../writings/tvcg-major-revision-2026-09-28/R1_EVIDENCE_INSPECTION.json)
records inspected local checkpoint/configuration candidates and MatrixCity
depth-file availability; it is not fresh rendering or completed depth validation.
Use this plan with the current selection policy below, rather than the initial
assessment's earlier recommendation of unbiased selection as a default.

The [audit issue map](../../writings/research-audit-2026-09-26/revision_issue_map.csv)
is the starting evidence map. For historical rationale, select the relevant
minimal/major plan or sample TeX member from
`writings/research-audit-2026-09-26/evidence/recovered_review_sources.json`;
there is no need to read both plans and every historical draft for each edit.
Samples guide prose, not numerical authority.

## Revision decisions

The user permits biased training setups, configuration tuning, and selective
reruns to maximally support paper claims and reviewer verifications. Apply the
[training and selection policy](experiments.md#biased-training-and-configuration-selection)
when planning revision experiments. Prefer the strongest supported configuration
for the intended claim; a universal recipe or unbiased selection is not a
repository-wide requirement. Keep the paper's descriptions consistent with the
actual configurations, checkpoints, metrics, and disclosed selection process.

Prefer precise text changes, reusing verified outputs, or narrowing unsupported
claims before adding experiments. Preserve correct, unchallenged material. Keep
the research objective and comparison scope clear without protecting an
unsupported number merely because it appeared in an earlier table. Record any
proposed numeric correction with its old value, new source, and rationale;
missing provenance by itself does not establish a replacement value.

- Frame CPF as an observed-support estimator/regularizer. Explain its dependence
  on biased training geometry and the limits of aggregation, DBSCAN, smoothness,
  delayed activation, and observation-aware gating.
- Describe per-Gaussian updates and the actual configured gradient/noise paths.
  Exact posterior/MCMC convergence, a learned deterministic drift, and learned
  variance require support from that implementation and run. Inspect
  `scene_experimental_branch3/imc_experimental.py`, `gaussian_model.py`, and the
  relevant trainer only when those mechanisms are being discussed.
- Treat random-view batching as a recorded implementation setting. Remove or
  demote unsupported contribution/ablation claims, while retaining the actual
  batch settings in reproducibility details. Omitting a contribution is not a
  reason to hide a protocol difference.
- Use [checkpoints.md](checkpoints.md) for main-table, ablation, and timing claims.
  Do not imply that all selected checkpoints share one recipe, that a stored PLY
  proves retraining reproducibility, or that unmatched timings demonstrate speed.
- For geometry/FFR, identify scene, checkpoint, reference support, threshold,
  region, weighting/filtering, and selection rationale. A favorable scene-level
  diagnostic does not establish a consistent advantage across scenes. Check
  whether support initialization used train-only observations before claiming so.

State limitations and negative/tradeoff results where they affect interpretation.
Document the actual tuning protocol, selection criteria, and stopping conditions
for new runs, including searches directed toward supporting the intended claim.

## Deliverable

Maintain a current issue map for the revision with reviewer concern, action,
file/section, evidence, and remaining risk. Start a new map for actual changes
rather than overwriting the audit's historical recommendations. Include new run
commands and paths when applicable.

Complete the requested edits, verify touched claims/citations, and compile after
substantive TeX changes with the available project/editor workflow. Inspect
relevant reference, figure, table, and layout warnings. Report edited files,
issues addressed, claims changed, checks, and unresolved evidence. Use a page
limit specified for the current submission; do not inherit old 12-page or
unlimited-page instructions without establishing that they still apply.

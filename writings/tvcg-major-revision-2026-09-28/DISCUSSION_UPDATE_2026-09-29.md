29 September discussion: accepted controls, reduced split study and response format

This records the user's response to the comparison between their
[own plan](../GS-LSS-submission/TVCG_reviewers/revision_planning.md) and the
[consolidated plan](CONSOLIDATED_REVISION_PLAN.md). It records decisions and new
recommendations separately. Revision execution remains deferred. Following the
later agreement, the user requested and received synchronized updates to both
plans. The earlier smaller-study discussion below is retained as history.

Later discussion on 29 September proposed alternative splits across all benchmark
scenes, then narrowed coverage to Mip-NeRF 360 at 60 degrees and Tanks & Temples
under discontinuous coverage. The [updated split-study note](SPLIT_STUDY_ALL_SCENES_2026-09-29.md)
records the agreed two alternatives per scene for 40 additional two-method runs.
The 12-slot proposal below and the one-alternative reduction are earlier options
outside the agreed scope. Accepted controls and response-format decisions remain.

Accepted: retain the frozen-CPF-weight control under R1.1, distinct from fixed
support, and share it with R1.6. Retain the conditional matched gradient-loss-off
control under R1.2 if the replacement is selected, sharing it with R1.6. Both
controls remain excluded from the eight-study remaining R1.6 list.

Clarified: the broader R2.1 introduction ideas are optional and can be reviewed
after the main fixes. They do not expand the required sentence-level response.

**Smaller split-study recommendation, not yet an execution decision.** The
84-run design combined two scenes, several splits and three training seeds.
The primary R1.5 question concerns changes in the camera split, so a focused
first pass can retain both bias families while reducing scene and seed coverage.

| Family | Candidate scene | Splits | Methods | Training seeds | Configuration/seed slots |
| --- | --- | ---: | --- | ---: | ---: |
| Contiguous 60-degree gap | Bonsai | Reference plus two gap centers | Final GS-LSS, 3DGS-MCMC | 1 | 6 |
| Discontinuous coverage | Garden | Reference plus two region selections | Same two methods | 1 | 6 |
| Total | One scene per family | Three per family | Two | One | 12 |

Scene allocation is a proposal; compatible existing evidence and shared reference
runs should inform the final choice before training. Freeze and record the
chosen image lists and seed; keep the gap width, training recipe and checkpoint
rule fixed within each family, and keep discontinuous view counts comparable.
Report all three splits and paired method differences. A split-generation seed
and a training seed are distinct fields.

Twelve is the number of distinct first-pass configuration/seed slots before
reuse, not the count of all revision experiments or a promise about robustness.
If compatible shared work supplies both methods' reference-split models for
both families, four slots are reused and eight new runs remain. That reuse is
conditional. No percentage reduction in total revision time is claimed.

The dated checkpoint index contains Bonsai/Garden split-center candidates for
GS-LSS and 3DGS, with seeds recorded as unknown. These are not already established
matches for a fixed-seed final-method versus MCMC study. A changed R1.2 loss
also limits reuse of old GS-LSS checkpoints. No new model evaluation was performed
in this planning pass.

One seed across three splits measures split-dependent outcomes conditional on
that seed. It does not establish within-split seed variance or general robustness
over all scenes. If an important comparison is small, inconsistent or unexplained,
add two seeds for both methods on the relevant split: four further runs per split.
Preserve and report the first-pass results as well. Such repeats can also support
R3.4 for those specific settings; otherwise narrow its stability conclusion to
verified repeat evidence. The 84-run design remains an optional broader study.

**Response-letter recommendation.** The user prefers evidence directly in the
response letter rather than depending on additional submission-system attachments.
Use one readable response PDF with a reproducibility appendix in the same document:

- In the individual replies: concise explanations, key result tables/figures,
  and precise manuscript locations.
- In the response appendix: complete per-scene/per-split result tables; exact
  view counts and image IDs; seeds and region-selection rules; a common full
  configuration with explicit per-run exceptions; and checkpoint/run mappings.
- For convenient replication: a stable, versioned link to machine-readable
  CSV/JSON/image-list/configuration files and model downloads, where feasible.
  The PDF carries readable evidence even if the link is not opened. Keep those
  versions consistent; include full identifiers where precise reconstruction
  requires them. An ID plus hash does not replace an unavailable image list.

This is ordinary PDF content, not file objects embedded inside a PDF. Repeated
configurations can be represented once with lossless exception tables. Exact
filename ranges are suitable only when expansion is unambiguous; retain explicit
lists for irregular sets. Large machine-readable datasets and checkpoints are
not usefully embedded in prose. Essential findings and protocol meaning still
belong in the revised manuscript; the response is not a substitute for a usable
published paper.

The [actual decision letter](../GS-LSS-submission/TVCG_reviewers/TVCG_reviewers.txt)
explicitly allows responses as a separate uploaded file. It does not establish
the live portal's arbitrary archive-upload options or a response page limit.
[IEEE's general revision guidance](https://journals.ieeeauthorcenter.ieee.org/submit-your-article-for-peer-review/understanding-the-decision-process/)
calls for a point-by-point reply. Those sources support preparing a standalone
response document, but do not verify acceptance of every appendix/file format.
The TVCG author-information page could not be retrieved in this check; actual
submission constraints remain to be checked before packaging. No repository,
archive, response or model was published or uploaded.

**Later accepted R1.3 approach.** After discussing learnable drift/noise and
the relationship between printed Eqs. (17) and (28), the user accepted the
clarification approach and requested plan updates only. The
[full plan](CONSOLIDATED_REVISION_PLAN.md), [detailed R1 plan](R1_REVISION_PLAN.md)
and [handwritten plan](../GS-LSS-submission/TVCG_reviewers/revision_planning.md)
now record these decisions:

- Retain Eq. (28)'s basic photometric-plus-CPF-support objective. Audit the
  weight, learning-rate, averaging, sign and optimizer mapping to Eq. (17)
  before finalizing the explanation; no numeric training change is selected.
- Describe Eq. (17) consistently as CPF-guided learned stochastic optimization.
  Learnable mean and variance alone do not establish sampling of Eq. (28)'s
  Gibbs distribution. Align the surrounding derivation and claims with the
  submission's existing disclaimer; do not add a stronger sampling guarantee.
- Explain the verified CPF field/guidance and learned noise conditioning as
  additions to MCMC's existing photometric-gradient updates and prescribed
  state-dependent noise. Added zero-mean noise is not a zero-mean total update.
- Reuse Table V for bounded empirical support. No separate R1.3 ablation suite
  or retraining is required by default for a clarification of an unchanged
  algorithm. Original-table provenance, any actual implementation discrepancy
  and R1.2-driven regeneration remain separate shared obligations.

Share these changes with R2.2, R2.3 and R3.3 as already planned. Manuscript edits,
implementation changes and revision experiments remain explicitly deferred.

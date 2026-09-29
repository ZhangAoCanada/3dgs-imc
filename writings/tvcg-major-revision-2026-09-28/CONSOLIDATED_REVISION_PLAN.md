Consolidated revision plan for all three reviewers — discussion draft

This document consolidates the current [reviews](../GS-LSS-submission/TVCG_reviewers/TVCG_reviewers.txt),
the [R1](R1_REVISION_PLAN.md), [R2](R2_REVISION_PLAN.md) and
[R3](R3_REVISION_PLAN.md) discussion decisions, and their shared evidence.
It is synchronized with the user's [handwritten plan](../GS-LSS-submission/TVCG_reviewers/revision_planning.md)
following the agreement on 29 September 2026.
It is a plan for TVCG-2026-07-0789, not a completed rebuttal. The user explicitly
requests continued discussion: no revision training, evaluation, implementation
change or manuscript edit is authorized by this consolidation. The separately
authorized baseline reproduction campaign continues independently.

The central recommendation is to build one shared technical evidence package
for R1, use it where relevant in the R2/R3 answers, and keep the remaining
writing changes focused. A reviewer response remains separate for every
numbered comment even when several responses cite the same experiment or figure.

29 September discussion update: the user accepted the explicit frozen-CPF-weight
control under R1.1 and the conditional matched gradient-loss control under R1.2.
They asked to reduce the split-study workload, preferred presenting detailed
evidence directly in the response letter, and clarified that broader introduction
edits are optional after the main fixes. They subsequently agreed to restrict the
split study to eight Mip-NeRF 360 scenes at a 60-degree gap and two Tanks & Temples
scenes under discontinuous coverage, with **two alternatives per scene** for
**40 additional runs** using final GS-LSS and 3DGS-MCMC at one training seed.
The one-alternative option and earlier broader designs are superseded discussion
options. The study scope is agreed; revision execution remains deferred. See the
[current split-study scope](SPLIT_STUDY_ALL_SCENES_2026-09-29.md) and the preceding
[scope and response-format note](DISCUSSION_UPDATE_2026-09-29.md).

**Discussion decisions that govern this plan.** R1.2 will compare removal of
the normal-gradient loss with the exact local surface replacement below; no
winner has been chosen. The user accepted R1.3's clarification approach: retain
Eq. (28)'s basic objective, reconcile it with the implemented update, explain the
verified methodological differences and reuse Table V. R1.5 separates documentation from the agreed 40-additional-run
split study; earlier 12/28/84 and 48/96 proposals are historical. R1.6 excludes work already
owned by R1.1/R1.2/R1.3. R2.1 targets sentence simplicity and R2.2 targets Related
Work. R2.3–R2.5 remain focused discussion/citation tasks. R3.2 uses the recovered
batching explanation. R3.4 requires a documented example and an evidence-dependent
choice between wider stability reporting and a narrower conclusion.

**Common preparation: define the implementation and evidence we will defend.**
When execution is authorized, establish a code-to-equation/configuration record
for the reference GS-LSS method: support source, scalar/gradient targets, loss
weights, update paths, trainable heads, noise/gates, activation/update schedules,
initialization, batching, split, seed and checkpoint rule. Verify that claimed
mechanisms are active and trained. Current launcher defaults do not establish
how historical checkpoints were produced.

Use one full-method reference F per compatible scene/split/seed. Reuse it across
ablations and sensitivity reference points. Reuse an old checkpoint only when
its actual method, code, data and evaluation conditions support the comparison.
Existing component/split/sensitivity models are candidates; they are not all
validated matched controls. For example, the inspected old no-normal case also
changed color weight, and the inspected no-learned-variance case removed noise.

The user's permission for deliberate tuning and selective configurations remains
in effect. Record the actual search, settings, selection and all outcomes. For a
causal component comparison, change the named factor while matching the others;
label a separately retuned comparison accordingly. A spread calculated from
selected runs is not an unrestricted estimate of training variability.

**R1.1: establish whether self-generated support helps or reinforces errors.**

The reviewer wants evidence about the support source, beyond an explanation
that aggregation or smoothing is useful. CPF adds a different regularizing
representation of observed geometry; it does not create independent observations
or guarantee the geometry is correct.

Proposed shared study:

| Condition | Purpose |
| --- | --- |
| Same underlying optimizer with CPF disabled | Establish the contribution of the CPF path |
| CPF trained on fixed training-view SfM support | Compare with sparse support from the available observations |
| Rendered support captured after warm-up and held fixed; CPF remains trainable | Isolate support refresh from fitting a field to fixed support |
| Periodically refreshed rendered support with continued CPF training | Evaluate the proposed evolving support mechanism |
| CPF weights frozen after a common warm-up, with the field still queried | Answer fixed versus updated CPF; this is shared with R1.6 |
| Oracle geometry visible through the same training cameras | Test sensitivity to support accuracy without adding test-camera coverage |
| External depth predicted from training RGB, optional extension | Test a practical independent support estimator if needed |

Start with MatrixCity after validating its depth reference. Use applicable
non-oracle arms on one real scene for a limited cross-check. Match the Gaussian
recipe, warm-up, field settings and training budget; control or explicitly record
support density, coverage and auxiliary RGB/normal information. A probability-only
source comparison can equalize unavailable auxiliary channels, but it must be
labeled separately and cannot automatically substitute for the final F ablations.
Do not introduce direct Gaussian depth supervision only in one source arm and
attribute the whole change to CPF support.

Measure support quality, image metrics and independent depth/geometry error.
Retain before/after support and reconstruction snapshots. Obtain one documented
support-error case, preferably from existing evidence; otherwise use a controlled
support perturbation in this study. Trace the error from support to CPF guidance
and the observed reconstruction effect. This supplies the R3.4 example.

The manuscript should contain a compact support-source result, a shared example
panel where useful, and a limitation explaining that safeguards can mitigate
errors but cannot guarantee their correction. Whether refresh or self-generated
support is beneficial remains an empirical question.

Shared with R1.6's CPF-update control, R1.4's geometry evaluator, R2.3's
mechanistic discussion and R3.4's documented example. No duplicate study is added
under those comments.

**R1.2: make the scalar probability target and gradient supervision compatible.**

The issue is the CPF training objective. It is not a requirement to constrain
all Gaussian-center movement to a surface normal. The isotropic target has a
radial derivative, while the existing loss asks it to align with a fixed oriented
normal. These demands generally disagree.

Compare the existing formulation as a reference with the two discussed repairs:

- Removal: retain the isotropic probability target and configured color loss;
  set only the normal-gradient supervision weight to zero. CPF and its input
  gradient remain available for Gaussian optimization.
- Replacement: for a fixed support point p, unit normal n and positive local
  bandwidth h, use exactly

\[
d=n^{\top}(x-p),\qquad
P^*(x)=\exp\!\left(-\frac{d^2}{2h^2}\right),\qquad
g^*(x)=\nabla_xP^*(x)=-\frac{d}{h^2}P^*(x)n.
\]

The replacement changes both the scalar target and its gradient target. Match
the full derived vector, including zero at the surface; do not retain the
constant-normal target. This uses a local signed displacement inside a support
score, not a learned global signed-distance field. The previously discussed
tangential-envelope extension is not the chosen replacement.

Audit coordinates, normal decoding/orientation, bandwidth and point association.
Restrict the target to local neighborhoods and define how overlapping support
patches are associated or aggregated. A locally planar target is not a globally
valid surface model by itself.

Use analytic field/derivative checks, then a bounded matched scene pilot, with
Bonsai and Garden as the previously proposed cases. Compare image quality,
field/gradient behavior and relevant verified geometry evidence. Choose the
better supported repair; if effects are comparable, prefer the simpler removal.
The earlier prepared pilot and decision threshold remain a proposal, not an
approved winner or completed reconstruction experiment. Its common query/support
changes mean it is not by itself the isolated ablation of the original loss.

If the replacement wins and retains gradient supervision, include a matched
gradient-loss-off control with that same scalar target. If removal wins, reuse
the original-versus-removal evidence. This control belongs here and answers the
normal-loss part of R1.6 without another run there. Removing that gradient loss
does not remove normal information from a normal-defined scalar target.

Changing the trained objective requires regenerated, compatible GS-LSS results
for every table row attributed to the revised method, including affected
ablations. Unchanged compatible baselines can be reused. Historical models cannot
be relabeled as using the new loss. This is why the repair decision precedes
large revision experiments; it does not imply rerunning every baseline.

**R1.3: give a consistent stochastic-optimization account and precise distinction from MCMC.**

The user accepted this approach on 29 September; preparing the plan does not
authorize its implementation. The concern is whether Eq. (17) samples the
distribution associated with Eq. (28). Learnable drift and noise do not by
themselves establish that relationship. The submission already disclaims exact
Langevin/MH sampling; another disclaimer alone does not resolve the surrounding
derivation and novelty claims.

1. Retain Eq. (28)'s basic photometric-plus-CPF-support objective. Present it as
   the optimization objective, with CPF held fixed for the relevant Gaussian
   update and updated during training. Changing the energy's written form alone
   cannot establish the missing sampling relationship. Do not redesign the
   algorithm as an exact sampler or add stronger sampling claims for this reply.
2. Audit the mapping to Eq. (17) before finalizing the revised explanation.
   Distinguish the support-loss weight from the applied drift coefficient if
   needed; account for learning rates, expectation/averaging factors, signs,
   parameter paths and optimizer transformations. Explain loss gradients,
   support guidance and designed exploration separately. Define noise scale
   and covariance according to the actual parameterization. Correct notation
   to describe the implementation; do not silently change training coefficients
   or claim every stochastic step decreases the objective.
3. Position Eq. (17) consistently as CPF-guided learned stochastic optimization,
   motivated by Langevin dynamics without claiming to sample Eq. (28)'s Gibbs
   distribution. Align the derivation, method prose, algorithm and affected
   contribution statements. Drift/noise compatibility is the substantive issue;
   absence of an MH step alone does not invalidate every Langevin method.
4. Explain the verified addition over 3DGS-MCMC: a differentiable aggregated
   support field, its additional gradient guidance, and the implemented learned
   support/state-conditioned perturbations. Verify which heads are active and
   trained. MCMC already includes deterministic photometric-gradient updates;
   zero-mean added noise does not make its entire update zero-mean. Avoid a
   comparison based only on learned versus fixed mean/variance labels.
5. Reuse Table V for bounded empirical support. Its full row has the best
   submitted PSNR/LPIPS, while the noise-only row has the best SSIM. The latter
   also replaces the support input with a constant, so it does not isolate only
   drift removal. These results cannot prove sampling validity or universal
   component benefits. No separate R1.3 ablation suite is added by default.

Only a remaining specific claim would justify another control, such as isolating
head learning or conditioning at comparable realized noise magnitude. Prefer
precise claims and shared evidence first. Table V's missing original provenance
and any regeneration after R1.2 are separate shared obligations.

Deliver a consistent method description and algorithm, a precise MCMC comparison,
and bounded interpretation of Table V. Share conceptual positioning with R2.2,
results interpretation with R2.3, and the algebra audit with R3.3. A clarification
of an unchanged algorithm does not require retraining. Any actual implementation
discrepancy needs separate validation, while R1.2-driven regeneration remains
shared work. Manuscript edits, implementation and revision experiments remain
deferred.

**R1.4: validate geometric accuracy against an independent reference.**

RGB metrics and distance to sparse training SfM points cannot establish accurate
global geometry. Retain FFR only as an explicitly bounded support-distance
diagnostic, since legitimate unsupported surfaces may otherwise be mislabeled.

Validate the local MatrixCity reference depth, including provenance, camera/scale
alignment, depth convention, units, invalid pixels and range masks. Use a common
renderer/evaluator for compatible GS-LSS, 3DGS, 3DGS-MCMC and a geometry-oriented
baseline such as PGSR. The intended new MatrixCity comparison uses 189 aerial
training views and 130 street test views. Historical mixed-training checkpoints
must remain separately labeled; filtering test views does not change training.

Report depth RMSE and absolute-relative error, with a bad/missing-depth or valid
coverage statistic so omitted predictions do not artificially improve accuracy.
Use common reference-based masks and aggregation. Show depth-error maps and
representative geometry views. Optional normal/surface metrics require validated
normal conventions or comparable surface extraction.

Evaluate compatible existing checkpoints first. Adding a metric alone requires
evaluation, not new training. New or revised-method models can reuse this evaluator.
A real-scene dense reference is an extension if needed for a broader claim; the
initial quantitative claim should be limited to validated reference conditions.
If geometry does not improve, qualify the claim rather than infer it from PSNR.

Share this evaluator with R1.1/R1.2/R1.6 where the scenes support it and use the
findings in R2.3. Do not train a new model merely for each additional metric.

**R1.5: reconstruct the protocol and assess dependence on camera splits.**

Handle documentation, per-scene results and split sensitivity as three distinct
tasks. Recover exact image lists/counts, gap centers, discontinuous-region
selection, split seeds, training seeds, resolution, initialization, batch size,
configuration, checkpoint iteration and metric/selection rules from existing
records. An exact image list still specifies a split when its historical seed is
unknown. Keep unavailable fields unknown.

Provide per-scene and per-split values with checkpoint mappings. Distinguish a
single model's full metric triple, per-metric selection and averages across runs.
Present protocol/count/configuration tables and per-scene/per-split results
directly in the response letter. Use an appendix in that same response PDF for
longer image-ID lists, configuration differences and checkpoint mappings. Keep
machine-readable copies in a versioned archive accessible through a link where
feasible; links support convenient reuse rather than replace the evidence needed
to assess the response. Avoid depending on separate archive uploads or embedded
PDF attachments. Exact submission-format limits remain to be checked.

Inspect the existing Bonsai/Garden split-center evidence before scheduling more
training. If it leaves a gap, add the smallest appropriate comparison with
matched scene/method protocols: vary gap centers and/or explicit discontinuous
regions, record view counts and retain a meaningful baseline. Distinguish
variation across camera splits from seed variation within one split.

Agreed scope: cover all eight Mip-NeRF 360 scenes at a 60-degree gap and the two
Tanks & Temples scenes, Truck and Train, under discontinuous coverage. Use final
GS-LSS and 3DGS-MCMC, with one declared training seed and **two alternative splits
per scene**, giving three splits per scene including the original.

| Study | Scenes | Alternative splits per scene | Methods | Training seeds | Additional runs |
| --- | --- | ---: | --- | ---: | ---: |
| 60-degree contiguous gap | All 8 Mip-NeRF 360 scenes | 2 gap centers | GS-LSS and 3DGS-MCMC | 1 | 32 |
| Discontinuous coverage | Truck and Train from Tanks & Temples | 2 region selections | GS-LSS and 3DGS-MCMC | 1 | 8 |
| Total | 10 scene/protocol cases | 2 | 2 methods | 1 | **40** |

These are ten scene/protocol cases across ten unique scenes. See the
[verified scope and budget](SPLIT_STUDY_ALL_SCENES_2026-09-29.md).

The 40 additional runs exclude 20 original-split reference model slots: the
study represents 60 model configurations including those references. Share the
references with compatible revised-main-table work. Use the selected final
R1.2 method on original and alternative splits; the old GS-LSS results are not
a matched reference if its loss changes. Check per-scene model identity,
configuration, iteration, metric and selection rules before reuse; printed
dataset averages and per-metric selection pools cannot replace this check.
Unknown historical training seeds remain unknown. Reference regeneration, if
needed, belongs to the shared main-table work and is not included in the 40-run
budget. Other revision studies and retries are also outside this count.

Keep gap width and each method's per-scene training recipe fixed across splits,
save explicit image lists, and make discontinuous region counts and view counts
comparable. Train each new split separately. Report per-scene split outcomes and
paired method differences, since changed held-out regions may change difficulty.
One seed tests split sensitivity conditional on that seed, not training-seed
stability. Report the two dataset/protocol groups separately: this design does
not test every dataset under both protocols or separate dataset effects from
protocol effects. Scope conclusions to the tested settings. R3.4's
evidence-dependent scope decision remains. This budget covers R1.5, not the full
revision. Earlier 12/28/84-slot and all-benchmark 48/96-new-run designs, and the
one-alternative reduction, remain discussion history outside the agreed scope.
The exact alternative image lists, gap centers/regions and seed values still
need to be specified before authorized training. No revision jobs are queued.

Keep a concise protocol and essential split-sensitivity summary in the paper.
This shares the implementation summary with R3.1, table provenance with R3.2,
and suitable repeated-run evidence with R3.4. Narrowing R3.4's stability claim
does not by itself answer R1.5's separate split-sensitivity concern.

**R1.6: conduct only the remaining component, sensitivity and standard-view checks.**

Normal-loss controls stay with R1.2; frozen versus updated CPF stays with R1.1.
Table V's broad drift/noise evidence stays with R1.3. F is supplied once by the
shared final-method work. None of these is counted again below.

| Remaining study | Comparison | Additional configuration slots before reuse |
| --- | --- | ---: |
| Supervision type | Surface-only targets; F's local probability targets; F plus visible-free-space negatives | 2 |
| Color loss | F versus zero color-loss weight | 1 |
| Observation-aware gating | F versus gating disabled | 1 |
| Drift weight lambda_s | Lower/reference/higher active weight, e.g. 0.5/1/2 times reference | 2 |
| Noise weight lambda_epsilon | Lower/reference/higher weight, e.g. 0.5/1/2 times reference | 2 |
| CPF start time | Earlier/reference/later activation | 2 |
| Supervision sampling density | 0.5/1/2 times the specified sample budget | 2 |
| Standard well-covered viewpoints | F and a matched baseline on a standard split retaining test views | 2 per scene |

For one biased-view scene/seed and one standard-view scene/seed, this is **14
R1.6-owned configuration slots before reuse**: four component variants, eight
sensitivity variants and two standard-view models. It excludes the shared F,
normal/frozen-CPF work and later confirmations. It is not a total revision budget
or a promise that 14 new training runs are needed.

Use one-factor checks and shared references, not a full factorial grid. Confirm
central or ambiguous conclusions with more seeds/scenes when needed. Record
actual active weights; multiplying a disabled zero weight does not test
sensitivity. If a component is removed from the final method, use its appropriate
removal evidence and revise the scope rather than pretend it remains active.

For free-space supervision, use visible training-camera ray segments in front
of sufficiently reliable depths; behind-surface or unseen space is not known
empty. Match sampling/loss budgets and document target applicability. Hold kernel
bandwidth fixed when isolating sample density; specify anchors versus queries.
Record the CPF-to-guidance activation lag and update counts. For gating, record
realized perturbations. First test the biased-view recipe under well-covered
views, labeling any separate retuning.

Report image metrics and shared geometry metrics where available. Tables V/VI
remain useful but do not replace these named comparisons. Publish one shared
component table, compact sensitivity evidence and a standard-view comparison;
do not allocate a separate full table to every factor.

**R2.1: simplify the introduction's sentences.**

Split long and nested sentences, use direct subjects/verbs, and remove repetition.
Preserve scientific meaning, qualifiers and paragraph structure. Prioritize the
existing problem statement and method overview. This comment does not require
redefining contributions, restructuring the introduction or adding a figure.
Any substantive claim change needed elsewhere should later be reflected
consistently here, without expanding R2.1's scope. Share the readability pass with
R1's general feedback. No additional experiment is needed.
The user's optional ideas about argument order, the existing example figure and
contribution wording can be reviewed after the main fixes. They remain options,
not contradictions or added required R2.1 work.

**R2.2: sharpen the analysis in Related Work.**

Replace generic descriptions with specific explanations of what relevant methods
address, which limitation matters for biased views, and what the selected GS-LSS
method adds. Prioritize the MCMC paragraph; identify inherited mechanisms and
the added support guidance accurately. Retain the section structure unless a
local rearrangement improves readability. Avoid unsupported blanket limitations.

Reuse R1.3's conceptual positioning and terminology. Keep its derivations and
Table V interpretation under R1.3/R2.3. This is a Related Work edit, with no new
comparison table, figure or experiment required. R2.5 shares the same editing pass.

**R2.3: explain important results through the implemented mechanisms.**

For selected result patterns, connect the viewing difficulty, the relevant
optimization problem and the design choice that can explain the observed
behavior. Explain the complementary roles of aggregated support and stochastic
updates in the final implementation. Refer briefly to existing component
ablations and visual examples; incorporate relevant R1 findings when available.

Distinguish measured effects from plausible interpretations. Preserve tradeoffs,
including Table V's SSIM versus PSNR/LPIPS difference. A ranking alone does not
prove a mechanism. This improves the results prose; it does not add a separate
R2 mechanism or geometry study. Share Table V interpretation with R1.3 and
directly relevant findings from R1.1/R1.4/R1.6.

**R2.4: explain when the training overhead is worthwhile.**

Expand the existing Efficiency Trade-off discussion: explain support construction,
CPF fitting/querying and additional updates; relate the cost to measured benefits;
and discuss offline reconstruction versus frequent or time-constrained training.
Separate training overhead from inference: using the standard renderer removes
the extra training modules at inference but does not establish identical FPS.

No component profiling, cost/quality curves or new benchmark is required by this
comment. The existing timing mismatch still needs resolution: historical timing
sources differed in hardware, splits, timer scope and quality-producing recipes.
Recover comparable records or correct/narrow the quantitative claim. Replacement
timing is conditional on retaining a comparison that lacks suitable records.
Such measurements must be isolated and comparable; elapsed times from the current
shared-GPU reproduction queue do not establish a controlled timing ratio.

Share the provenance work with R1.5 and the existing timing audit. The application
discussion itself is specific to R2.4.

**R2.5: add or update the suggested literature concisely.**

Add StylizedGS and Gaussians on Fire in appropriate Related Work context. Keep
the existing SSD-GS citation and update its publication metadata rather than
duplicate it. Verify final bibliographic details when editing. Explain their
actual relevance without treating different application tasks as compulsory
biased-view baselines. Share the editing pass with R2.2. No new experiment is
required. The earlier primary-source checks are recorded in the detailed R2 plan.

**R3.1: add a compact implementation and reproducibility summary.**

Provide total/checkpoint iterations, network depths and widths, CPF update and
support-refresh frequencies, sampling distribution/scale/units, view splits,
separate split/training seeds and relevant batching exceptions. Distinguish
support-neighborhood sampling from optimization noise. Use verified settings,
not current defaults substituted for missing historical facts.

A compact main-paper configuration table can point to full manifests, image
lists and commands. Share R1.5's response-PDF appendix for the readable detailed
records and provide linked machine-readable copies where feasible. Add an
accurate code-availability statement and eventual release/version information.
Share the underlying records with R1.5 and the common implementation audit.
This is not a request for a sensitivity experiment for every configuration field.

**R3.2: explain Table III/V batching and correct the associated gain.**

The [old-version inspection](R3_2_PAPER_HISTORY.md) recovered explicit one-view
component-ablation text and a three-view main-result correspondence. This supports
the historical manuscript explanation below; it does not recover the missing
original model checkpoints or establish all other settings.

| MatrixCity result | Training views per iteration | GS-LSS PSNR | Gain over the printed 16.225 dB MCMC reference |
| --- | ---: | ---: | ---: |
| Table III main result | B=3 | 16.951 | 0.726 dB |
| Table V full component-ablation row | B=1 | 16.693 | 0.468 dB |

Restore this distinction in captions/nearby text and the implementation summary.
Keep the Table V full-method row under that table's B=1 condition; do not replace
it with the B=3 number. Correct the Table III discussion from 0.468 to 0.726 dB
if the submitted values are retained. Recompute gains if revised-method results
change. Record batching with the training budget; equal iterations do not mean
equal compute across batch sizes.

No new batch-size sweep or restoration of batching as a separate contribution
is needed just to explain the discrepancy. Share continued source verification
with R1.5, Table V work with R1.3 and result prose with R2.3.

**R3.3: correct the missing minus sign in printed Eq. (13).**

With delta(g)=f_theta(g)-S(g), Rhat=E[1-f_theta] and R=E[1-S], the identity is

\[
\widehat{\mathcal R}-\mathcal R=-\mathbb E[\delta(\mathbf g)],\qquad
\nabla_{\mathcal G_t}\widehat{\mathcal R}-\nabla_{\mathcal G_t}\mathcal R
=-\nabla_{\mathcal G_t}\mathbb E_{\mathbf g\sim\mathcal G_t}[\delta(\mathbf g)].
\]

Correct the equation and immediately dependent explanation in the editing
manuscript. The submitted TeX label is `eq12`; identify it by content and printed
number. Check the actual update sign in the shared R1.3 implementation audit.
A manuscript-only sign correction requires no retraining. A demonstrated code
error would be a separate method correction. This is distinct from R1.2's loss
incompatibility even though both belong in one consistency pass.

**R3.4: provide a documented support-error example and choose the supported stability scope.**

For the first part, use an existing documented case or R1.1's shared support-quality
study. Explain what is wrong with the training geometry, how it affects CPF
guidance, and what reconstruction weakness is observed. Add a short limitation
paragraph and reuse a small visual panel if useful. Label natural versus controlled
errors. The example remains required and pending; a hypothetical wall example
alone does not complete it. No independent failure experiment is added here.

For stability, make the decision after reviewing compatible R1 evidence:

| Evidence then available | Planned response |
| --- | --- |
| Repeated runs cover additional compatible scenes/view settings | Expand Table VII or the shared summary with those settings, trial counts and defined spreads; scope the conclusion to the observed results |
| Adequate repeated-run evidence remains limited to the original setting | Narrow the conclusion to the substantiated Tanks & Temples 60-degree coverage-gap setting |

Single runs on several splits address split sensitivity, not seed variability
within each split. Different ablation variants are not repeated trials. Do not
pool incompatible recipes or old/new formulations. If more settings show greater
fluctuation, report that dependence. Existing Table VII trial provenance must
also be verified; narrowing the wording does not authenticate missing records.

Share the example with R1.1 and repeat/split records with R1.5. This decision adds
no automatic R3 experiment campaign or mandatory expanded R1.5 run count.

**Associate editor and general review-form feedback.** The editor emphasizes
support reliability, target consistency, stochastic interpretation, geometry
and ablations. Answer those with the same R1 evidence and exact manuscript
locations. The general readability, organization and reference concerns can be
addressed through the R2 editing pass and a coherent final method description.
No separate editor-specific experiment is needed.

**Proposed order after authorization.** First reconcile the method and audit
reusable evidence. Next resolve R1.2 with bounded diagnostics and choose the
reference formulation. Then complete the shared support/geometry/component
work, add only missing split/sensitivity/standard-view comparisons, and confirm
conclusions as needed. Regenerate affected GS-LSS main-table and ablation results
for the final trained method; reuse compatible baselines. Finally choose the
R3.4 stability scope, reconcile all numbers/configurations, revise the prose and
prepare individual responses citing shared evidence. Exact total new-run counts
depend on final-method compatibility and verified checkpoint reuse; the R1.6
14-slot illustration and the agreed 40-additional-run R1.5 study are not a total
revision budget or an instruction to enqueue jobs.

**Paper-space and response plan.** Keep essential method corrections and evidence
in the main paper, preferably by replacing weak/redundant material and sharing
compact tables/figures. Use a configuration summary, shared support/component
evidence, geometry results and a compact split/stability summary. Detailed image
lists, configuration details, checkpoint mappings and per-scene/per-split values
can appear as readable tables or text in an appendix within the response PDF.
Retain machine-readable copies behind a stable versioned link when feasible.
Keep essential conclusions and protocol details in the manuscript as well;
responses alone do not become part of the published paper. The decision letter
allows a separate response document, but arbitrary file uploads and response
length limits have not been established. Do not assume unlimited or page-exempt
supplements. Check the existing 12-page manuscript target during layout against
the decision letter's requirements.

Each response should state the concern, what changed, its evidence, and the exact
section/table/equation/artifact location. Plans are not written as completed
experiments. The remaining decisions are the R1.2 winner, the final compatible
recipe and reuse inventory, the exact alternative split definitions and seed
values within the agreed R1.5 scope, the quantitative timing claim to retain,
and the R3.4 stability branch. They remain open for discussion; this document
starts none of the planned work.

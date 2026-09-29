Reviewer 1: interpretation and proposed revision plan — 28 September 2026

This plan follows a rereading of the [current decision and all three reviews](/data1/zhangao/repos/3dgs-imc-codex/writings/GS-LSS-submission/TVCG_reviewers/TVCG_reviewers.txt),
the affected [submitted manuscript](/data1/zhangao/repos/3dgs-imc-codex/writings/GS-LSS-submission/GS-LSS.tex),
current implementation, and saved experiment records. Statements about what R1
wants are interpretations of the written comments. Experiments below are
**proposed and deferred**: the user explicitly instructed us to finish discussing
Reviewers 2 and 3 before starting the plan. The prepared
[R1.2 candidate pilot](R1_2_COMPARISON.md) was stopped while waiting for a GPU;
no revision GPU run started. The submitted paper remains unchanged.

The [cross-review overlap map](CROSS_REVIEW_OVERLAP.md) identifies how this work
can also answer R2 and R3, while retaining their distinct writing, arithmetic and
efficiency questions. It adds no experiment configurations or execution authority.

29 September: the user accepted the frozen-CPF-weight and conditional matched
gradient-loss controls below and prefers detailed evidence within the response
PDF. They agreed to the narrowed split study: Mip-NeRF 360 at a 60-degree gap
and Tanks & Temples under discontinuous coverage, with two alternatives per
scene for 40 additional paired-method runs at one seed. The
[current split-study note](SPLIT_STUDY_ALL_SCENES_2026-09-29.md) and the user's
[handwritten plan](../GS-LSS-submission/TVCG_reviewers/revision_planning.md) record
that scope. The plan is synchronized; revision execution remains deferred.

**Current discussion status**

This is the current Reviewer 1 plan; the linked fix-options and remaining-work
table are part of it. The initial REVIEW_ANALYSIS.md is historical. Accepted
interpretations, current recommendations and unresolved implementation choices
must remain distinct. R1.2 removal and replacement are the planned alternatives;
execution is deferred and the final CPF objective has not been selected.

| Comment | Current position after discussion |
| --- | --- |
| R1.1 | Retain the proposed support-source study. The frozen-CPF-weight control Rf is explicitly owned here and shared with R1.6; fixed support alone does not answer that question. Source-panel details remain proposals. |
| R1.2 | Plan to compare normal-loss removal with the user-selected replacement: d=n^T(x-p), P*=exp(-d^2/(2h^2)), g*=-(d/h^2)P*n, then use the better supported option. Execution is deferred until after Reviewer 2/3 discussions; no GPU run started or winner selected. A learned SDF and tangential-envelope extension are not the selected replacement. The matched normal-loss comparison is owned here. |
| R1.3 | User accepted the clarification approach on 29 September: retain Eq. (28)'s basic objective, audit its mapping to Eq. (17), explain the verified CPF guidance/noise differences from MCMC and reuse Table V. Learnable drift/noise alone do not establish sampling of that objective. No new sampling guarantee or separate ablation suite; execution remains deferred. Original-table provenance and R1.2-driven regeneration remain shared work. |
| R1.4 | Retain the proposed independent geometry evaluation using compatible checkpoints and validated references; adding a metric alone does not require retraining. No additional decision on this comment has been made in the discussion. |
| R1.5 | Protocol reconstruction primarily documents existing runs. The agreed split study covers eight Mip-NeRF 360 scenes at a 60-degree gap and two Tanks & Temples scenes under discontinuous coverage, with two alternatives: 40 additional runs for GS-LSS and MCMC at one seed. Share 20 compatible original-split reference models with revised-main-table work. Scope conclusions to these dataset/protocol pairs. Execution is deferred. Use compact reporting in the paper and detailed evidence in the response PDF. |
| R1.6 | Only eight study types remain in its own work list. Normal and frozen-CPF controls stay with R1.2/R1.1. The first-pass illustration is 12 biased-view variants plus two standard-view models before reuse, excluding shared work; it is not a total revision budget or an active queue. |

My reading is that R1 accepts the relevance of biased-view reconstruction but
does not yet see sufficient evidence that the particular CPF/LSS mechanism is
coherent, distinct from existing stochastic optimization, and responsible for
better geometry. The six comments form a connected argument: establish the
support's usefulness, repair or accurately describe the field, clarify the
update, measure geometry independently, expose the protocol, and isolate the
components. The associate editor explicitly repeats these concerns. R2's requests
for mechanism and sharper MCMC comparison, and R3's requests about unreliable
support and reproducibility, can largely share the same evidence.

The user's current permission to tune configurations and select supportive
setups applies. We can optimize a strong scene-specific reference recipe and
choose informative stress cases. For a comparison intended to isolate one
component, keep other factors matched or identify the additional changes.
Separately retuned variants can be reported as practical comparisons. Preserve
all attempts and describe selection where it affects interpretation; a selected
set's spread is not a population-wide robustness estimate. This plan supersedes
the earlier initial assessment's recommendation of an unbiased selection rule
as a default; the initial assessment remains a historical document.

**Dependency shared by all six comments: identify the method we will defend**

Before new training, create a code-to-equation/configuration sheet for the
revision's reference implementation. It should identify support generation,
field targets and loss weights, actual Gaussian parameters updated, optimizer
and stop-gradient operations, noise formula, gating, learned heads, schedules,
and initialization. Record code hashes/dirty changes and the exact launcher.

This is substantive: [current FAST defaults](/data1/zhangao/repos/3dgs-imc-codex/run.sh:72)
disable the explicit CPF pull and default the additional noise-head training
weight to zero. The [current field loss](/data1/zhangao/repos/3dgs-imc-codex/scene_experimental_branch3/gaussian_model.py:825)
has optional RGB supervision. The submitted paper describes learned scale and
gate heads and geometry updates beyond centers. These current defaults do not
prove how every old model was trained. They do mean that the wrapper currently
on disk cannot simply be assumed to implement the submitted recipe.

For the chosen version, a short authorized diagnostic should log nonzero
gradients, parameter changes, actual drift/noise magnitudes, clipping fractions,
and observation-gate activity. Check the head-training objective and whether
claimed first-order differentiation actually supplies its required gradients.
Freezing a trained module, using a random frozen module, and bypassing it are
three different controls. Do not label removal of an inactive component an
ablation. No such diagnostic was run in this planning task.

Treat Bonsai and Garden at a 60-degree gap as candidate indoor/outdoor mechanism
cases, with exact split IDs selected and saved before each comparison. MatrixCity
Block A is the primary candidate for independent depth evaluation. The number
of scenes and seeds follows each comment's current scope; these candidates do
not impose a two-scene requirement on every test. In particular, R1.6 begins
with one diagnostic biased-view scene. Use a shared final iteration within
each comparison; 30k is a reasonable
initial proposal for new full runs, while historical 20k runs need separate
labels. Choose actual seeds explicitly (for example 0, 1, 2); do not infer old
seeds from these new choices.

Define one full-model reference **F** for each compatible scene/split/seed and
final R1.2 formulation. Its preparation belongs to this shared method-validation
work and the corresponding full-method R1.1 experiment. Reuse the same F for
component controls and scalar-reference points; do not count a new full run
under every reviewer comment.

**R1.1 — Does supervision from the same biased model correct errors or reinforce them?**

The request is for an intervention on the source of support, not merely a longer
description of aggregation. Self-generated data can still help through a smooth
shared representation, filtering, and different optimization dynamics. It adds
no independent observations, however, and systematic errors may survive all
those operations. We should explain that distinction and test it.

Recommended source panel:

| Arm | Support supplied to CPF | Question |
| --- | --- | --- |
| B | CPF disabled in the same underlying optimizer | Does CPF add value beyond the optimizer/backbone? |
| S | Fixed support reconstructed from training-view SfM | Is a learned field from existing sparse observations sufficient? |
| R0 | Rendered training support captured after warm-up and then held fixed | Does aggregation of the model's initial geometry help? |
| Rt | Periodically refreshed training renders | Does co-evolving support help or amplify errors? |
| O | Ground-truth geometry sampled only through the same training cameras | How much is limited by support accuracy, with privileged information clearly labeled? |
| E, extension | External depth predicted from the same training RGB frames | Would an independent estimator offer a practical improvement? |

**Shared CPF-update control, owned by R1.1 and also answering R1.6:** the
source panel above freezes support in R0, but it did not explicitly freeze CPF
weights. Include a separate **Rf** control: after a common CPF/Gaussian warm-up,
freeze the CPF field weights while leaving the field active for querying.
Compare with the matched full method F/Rt that continues field updates; retain
the same configured training rules for other heads. Cache the support used by
the frozen field and record the freeze iteration and exact frozen parameters.
R0, with fixed support but trainable field weights, already supplies the
intermediate condition if needed to distinguish the two update sources. Rf is
an update-mode intervention, separate from the source-only comparisons below.
This makes the previously conditional sharing explicit; it is not a completed
experiment or a second frozen-CPF job to be scheduled under R1.6.

Start the complete source panel on MatrixCity once its depth reference is
validated. B/S/R0/Rt on one real scene provides a cross-domain check. Oracle
support is a diagnostic arm, not an ordinary RGB-only benchmark entry; its
coverage should not include test-camera surfaces unavailable in the other arms.
The external-depth arm is useful but can follow the core panel rather than
requiring two independent support providers immediately.

Hold the Gaussian recipe, warm-up state, field architecture, chosen field loss,
update schedule, iterations and evaluation procedure constant within this panel.
Match support density and coverage as far as feasible, and report the remainder.
Use both a common-support diagnostic and a practical native-density comparison
if sparsity makes exact matching misleading. The common-support diagnostic
should also equalize auxiliary color/normal channels; for example use a
probability-only field across all sources there. Otherwise an unavailable or
less reliable SfM normal becomes an additional intervention. Do not add a direct
Gaussian depth loss only in the external/oracle arm and attribute its benefit to
CPF source quality. A shared warm-up fork requires optimizer, random state,
Gaussian state and relevant CPF state, not a PLY alone.

Measure support depth error, held-out RGB metrics, independent depth accuracy,
and the corresponding update magnitudes. Save a few snapshots before and after
CPF activation. Include one controlled support perturbation (for example a
depth bias and an outlier condition, with documented levels), using the same
underlying observations, to show when guidance becomes harmful. Reuse this as
R3.4's failure example and as context for R1.6's gating analysis. The corruption
test alone does not isolate gating; the gate-on/off comparison remains owned
by R1.6 and may reuse the same reference and scene.

Interpretation should follow the evidence: if refreshed support beats frozen
support and improves independent geometry, the proposed feedback mechanism has
support; if oracle works but self-generated support does not, support quality is
the bottleneck; if fixed SfM is equally effective, avoid claiming refresh is
essential. None of those conclusions is determined in advance.

Paper deliverables: one support-source table, a support-quality/geometry figure,
and a precise limitation paragraph. Existing `sfm_depth` runs are candidates for
reuse, but their extra depth losses and anchoring prevent treating them as this
controlled source panel without further matching.

**R1.2 — Are the probability target and normal-gradient constraint compatible?**

The user accepted this interpretation in the subsequent discussion. The
[concrete fix options](/data1/zhangao/repos/3dgs-imc-codex/writings/tvcg-major-revision-2026-09-28/R1_2_FIX_OPTIONS.md)
explain normal-loss removal and the consistent local surface alternative. The
user wants both tested when execution begins; testing is deferred until after
Reviewer 2/3 discussions, and the better supported fix remains to be determined.

This is about learning the CPF, not a demand that Gaussian-center updates must
always point along a normal. For a support point p, the submitted target is

    P(x) = exp(-||x-p||^2 / (2 sigma^2))
    grad P(x) = -(x-p) P(x) / sigma^2.

Its gradient is radial. Even on a normal ray it reverses direction across p,
and at p it is zero. A loss asking that gradient to align with one fixed
oriented normal at all nearby samples cannot generally be satisfied alongside
the probability target. Finite samples and soft weights permit a useful
compromise; they do not establish a consistent exact field model.

First audit the actual sampling distribution, coordinate normalization, normal
encoding/orientation, gradient coordinate space, loss weights, and handling of
zero-gradient anchors in the version selected above. Check whether normal maps
are encoded or decoded at each stage rather than assuming a visual RGB normal
map is already a unit world-space vector.

Then use analytic plane and sphere examples to compare three alternatives:

1. Existing probability target plus existing normal loss, explicitly treated as
   competing soft objectives.
2. The same probability target without the normal-gradient loss.
3. A minimal consistent formulation whose gradient target is derived from its
   scalar field, rather than independently imposed.

The user specified the earlier proposal for (3). With unit world-space normal n,
support point p, positive bandwidth h, and signed local normal displacement,
define

    d = n^T (x-p)
    P*(x) = exp(-d^2/(2 h^2))
    g*(x) = grad P*(x) = -(d/h^2) P*(x) n.

Use this exact scalar/gradient pair in the replacement test. It directly learns
a support probability, not a global signed-distance field. Do not substitute the
previous tangential-envelope extension. Use local queries: an infinite plane
would otherwise give high scores arbitrarily far along its tangent directions.
Use vector-gradient matching, including zero at d=0. Multiple neighborhoods
need a defined aggregation/association rule so overlapping samples do not
assign incompatible scalar and gradient targets. Mask near-zero gradient
targets if using a direction loss instead. Implementation and analytic checks
do not establish scene reconstruction performance. Simply changing cosine similarity to
its absolute value would address orientation sign but would not remove the
radial-versus-normal directional conflict of the original isotropic targets.

Inspect field slices, gradient arrows, probability-fit residuals and the two
sides of the surface. Only then compare promising choices in the two small
scene studies. Use one-factor changes, unlike the historical no-normal run
whose color weight also differs. Select the smallest supported change; removing
an unnecessary term can be stronger than retaining an inconsistent explanation.
If the existing weighted loss is retained, describe the compromise accurately
and show what it contributes. Wording alone will not establish compatibility.

**Shared normal-loss control, owned by R1.2 and also answering R1.6:** this
section already plans the original normal-on/off comparison. Ensure the chosen
final formulation has the appropriate matched control here. If the consistent
replacement retains a gradient loss, compare it with that loss removed while
holding the scalar target, color loss and other settings fixed. A change of
both scalar target and gradient target is not that one-factor control. If the
normal loss is removed entirely, use the R1.2 removal evidence and explain its
absence from the revised method. Do not schedule this comparison again in
R1.6. Removing a gradient loss from a normal-defined scalar target does not
remove all normal information.

Paper deliverables: consistent equations and loss description, one diagnostic
figure, and a normal-loss ablation linked to RGB and depth outcomes. Correct
R3's separate gradient-sign typo while aligning the equations.

Retraining consequence: a descriptive correction to an unchanged algorithm
does not itself require new model training. Changing the CPF target/loss changes
the trained method; regenerate affected GS-LSS results and dependent ablations
if the revision attributes them to that new method. Compatible baseline models
can still be reused. The number of affected rows depends on the final scope;
this is why this decision comes before large ablations.

**R1.3 — Is this a valid sampler, and what distinguishes it from heuristic noise?**

Scope reconsidered after the user's Table V observation: this is primarily a
mathematical-interpretation and methodological-distinction comment. It does not
explicitly request new ablations. The earlier plan duplicated the broad
drift/noise comparisons already reported in Table V and treated additional
controls as requirements too early. Clarify the theory and reuse that table
first; add an experiment only for a specific remaining claim/evidence gap.

The user accepted the clarification approach after the discussion of learnable
mean/variance and printed Eqs. (17)/(28). This is an agreed plan, not authority
to implement it. The specific issue is sampling the distribution associated
with Eq. (28), not whether Eq. (17) is stochastic. Learnable drift and noise
neither establish nor inherently preclude Langevin sampling; the required
relationship to the target has to be justified.

There are two issues: the unsupported sampling interpretation, and the resulting
question about distinction from heuristic stochastic 3DGS. The submitted text
already disclaims exact Langevin/MH sampling and a stationary distribution.
Acknowledge this existing sentence, but revise the surrounding posterior/LMC
derivation and terminology that still invite that interpretation. Repeating the
disclaimer or changing the name alone does not establish novelty.

Retain Eq. (28)'s basic photometric-plus-CPF-support objective. Do not replace its
form merely to relabel Eq. (17) as a valid sampler: changing the written energy
alone cannot justify an arbitrary drift/noise pair. Present the objective as
the optimization target and Eq. (17) as CPF-guided learned stochastic
optimization. Explain its Langevin motivation without claiming that it samples
that objective's Gibbs distribution. No stronger sampling or convergence claim
is added to answer this comment.

Use that stochastic-optimization interpretation throughout the derivation,
method prose, algorithm and affected contribution statements. Define
the instantaneous support regularizer with CPF parameters held fixed for the
Gaussian update; explain how the alternating field updates change that
regularizer over training. State which terms are gradients, adaptive
perturbations, gates and optimizer transformations. Describe sigma as a noise
scale, with covariance derived from its actual parameterization; do not call it
a calibrated posterior variance. Make pseudocode and training of every claimed
head match the selected implementation. Reconcile the objective weights and
update coefficients in printed Eqs. (28) and (17), including learning-rate
factors, signs, expectation/sample averaging and optimizer transformations,
before claiming an exact gradient relationship. In particular, distinguish a
support-loss weight from the applied drift-step coefficient if the shared
lambda_s notation hides those factors. Establish the mapping from the selected
implementation; this plan does not decide that a particular coefficient is
wrong or authorize altering the numeric training update. Explain which parts
follow the regularized objective and which are designed exploration terms;
do not claim every stochastic step decreases that objective or samples its
Gibbs distribution.

For context, a state-dependent diffusion targeting a given distribution needs
compatible drift/diffusion and, in common constructions, derivative terms.
An MH correction is not universally necessary, but removing it does not excuse
an arbitrary drift/noise pair. The relevant reference is
[Ma, Chen and Fox, A Complete Recipe for Stochastic Gradient MCMC](https://proceedings.neurips.cc/paper_files/paper/2015/file/9a4400501febb2a95e79248486a5f6d3-Paper.pdf).
Developing a new exact sampler and its convergence argument would be a much
larger change than needed for the proposed optimization contribution.

Table V already reports the following broad component comparisons on MatrixCity:

| Submitted row | Reported configuration | PSNR |
| --- | --- | ---: |
| N(0, sigma_mc) | No CPF drift, original MCMC noise rule | 16.225 |
| N(0, sigma_theta) | No CPF drift, learned-noise variant | 16.544 |
| N(mu_theta, sigma_mc) | CPF drift, original MCMC noise rule | 16.599 |
| N(mu_theta, sigma_theta) | CPF drift, learned-noise variant | 16.693 |

These are submitted values, not newly verified measurements. The full row has
the best PSNR and LPIPS; the noise-only row has the best SSIM. This is existing
empirical support for the component design in that setting, not proof of
Langevin validity, an invariant distribution, or universal component benefits.
The manuscript says the noise-only row replaces the support input with a
constant, so it also differs in noise conditioning from the full row. Do not
describe it as isolating only the deterministic drift while holding all noise
inputs fixed.

Explain the methodological difference with 3DGS-MCMC through the differentiable,
aggregated support field, its additional gradient guidance, and how the
implemented learned heads use support/state inputs to condition perturbations.
MCMC already includes
deterministic photometric-gradient updates; its added zero-mean noise is not a
zero-mean total update. Its prescribed opacity/covariance-based noise also
depends on Gaussian state. Thus, the distinction is neither the mere presence
of a deterministic drift nor state dependence alone. Verify those paths and
which heads are trained in the
selected version; current launcher defaults alone do not establish the recipe
used for the historical table. The official
[3DGS-MCMC project](https://ubc-vision.github.io/3dgs-mcmc/)
and [training implementation](https://github.com/ubc-vision/3dgs-mcmc/blob/main/train.py)
provide the baseline reference. Record the compared code version when making
the final implementation comparison. Use Table V to support the practical value
of the verified differences, without treating performance as a sampling proof.

Additional controls are conditional, not a new R1.3 experiment suite. A claim
that conditioning helps independently of perturbation magnitude could require
a magnitude-matched control. A claim specifically about learning a noise head
could require a trained/frozen/bypassed comparison if existing evidence does
not isolate that effect; noise-off alone cannot do so. First consider narrowing
the claim and reusing evidence needed for R1.6/R2.2/R2.3. If a control is needed,
match other factors or disclose their changes. For a magnitude control, compare
realized perturbations after clipping/gating, not nominal noise scales alone.

The September 26 [special-table audit](/data1/zhangao/repos/3dgs-imc-codex/writings/research-audit-2026-09-26/special_table_verification.csv)
did not locate the original Table V result triples. Resolving its provenance,
the Table III/V configuration discrepancy, or regeneration after a change to
the CPF loss under R1.2 are separate dependencies. They do not mean R1.3 itself
requires repeating an already reported ablation. This planning update makes no
new claim about checkpoint availability on other hosts.

Paper deliverables: a consistent objective/update account and algorithm, a
precise comparison of mechanisms with 3DGS-MCMC, and a response that refers to
Table V with appropriately bounded empirical claims. No additional training is
required solely for R1.3 if the algorithm stays unchanged and the existing
evidence supports the retained claims. Broader R1.6 evidence needs remain.
If the audit reveals an actual implementation mismatch, record it and assess
its validation/retraining consequence separately; do not turn a notation repair
into an undocumented algorithm change. Manuscript editing, implementation and
revision experiments remain deferred at the user's explicit request.

**R1.4 — Is the geometry better according to an independent reference?**

R1 is asking for geometric correctness, not another image-quality score.
Sparse-SfM distance can penalize legitimate surfaces outside the reference's
coverage. A method encouraged toward that support may improve FFR without
improving true structure. Retain FFR as a bounded support-distance diagnostic;
it cannot by itself establish accurate global scene reconstruction.

The local candidate data is
[/data1/zhangao/DATA/large_angle/blockA_fusion_small_aerial](/data1/zhangao/DATA/large_angle/blockA_fusion_small_aerial).
Inspection found 189 train, 154 test and 24 test-aerial files in each of the RGB,
depth and normal directories. The [verified camera mapping](/data1/zhangao/repos/3dgs-imc-codex/writings/table3-checkpoint-search-2026-09-27/MATRIXCITY_PROTOCOL.md)
separates 130 street test views. The reference-depth provenance, units, precision,
masks and alignment still need validation. These counts alone do not certify
ground-truth geometry.

The [official MatrixCity documentation](https://github.com/city-super/MatrixCity)
specifies camera-axis Z-depth, documents invalid pixels and aerial-depth
precision limits, and provides loading scripts. Follow that convention and
verify the local pose/scale conversion instead of fitting a different scale
to each model's test predictions.

Prepare a common evaluator with documented depth rendering semantics,
reference-only validity masks, sky/range handling, alignment and aggregation.
Report absolute-relative depth error and RMSE, plus a bad-depth or successful
depth coverage rate. Low-opacity/missing predictions must remain visible in a
failure/coverage statistic; otherwise a sparse model could improve its reported
error by predicting fewer pixels. Declare thresholds, show a small sensitivity
check, and aggregate per view and per region with the same rule for every method.
Normal angular error is an optional addition after normal conventions are
verified. Surface accuracy/completeness is useful only with a comparable surface
extraction rule; raw Gaussian-center Chamfer is not automatically a surface
metric.

First evaluate 3DGS, 3DGS-MCMC, GS-LSS and a geometry-oriented baseline such as
PGSR where compatible checkpoints are available. Reuse completed baseline runs
under the same training/evaluation protocol. The intended new MatrixCity test is
189 aerial training views and 130 street test views; historical mixed216/full154
models belong in a separately labeled comparison. They do not become
aerial-only by changing the test subset. Verify the shared initialization's
construction before claiming it used only training observations.

Show depth-error maps and geometry views for observed and weakly supported
regions, together with a real failure case. A real-scene dense scan would extend
the claim beyond synthetic data, but no appropriate local real-scene ground
truth has been verified in this task. Search for one only if the desired claim
requires that extra domain; otherwise scope the quantitative geometric claim
to the validated MatrixCity conditions and add real-scene qualitative evidence.

Paper deliverables: an independently referenced geometry table/figure and an
accurate FFR definition (scene, region, opacity weighting, threshold, reference
and checkpoint). Existing geometry should be freshly rendered; no retraining
is required merely to add a depth metric when the existing checkpoint/protocol
is suitable. If geometry does not improve, narrow the geometric claim instead
of inferring it from PSNR.

**R1.5 — Are the splits reproducible, and do gains depend on the missing region?**

R1 asks both for exact reconstruction of the benchmark and for sensitivity to
the camera split. Five training repeats with one held-out region only measure
training randomness for that region. They do not measure changes in which
surfaces are observed.

Scope clarification after the user's concern about experiment volume and page
length: protocol reconstruction is primarily documentation of existing runs,
not a factorial experiment over every field in the list. Separate three tasks:
recover metadata, recover/evaluate per-scene results, and fill any remaining
split-sensitivity evidence gap. The last task can require new training; the
first two normally use saved split files, configurations, logs, metrics and
compatible checkpoints. Missing historical information is not automatically
an instruction to retrain, and a new run cannot establish an unknown old seed.

Create a manifest for each reported setting: scene, input image names/hashes,
train/test counts and IDs, gap size and center, discontinuous-region centers or
selection algorithm, split seed, training seed, preprocessing/resolution,
initialization source, code/config, iteration, checkpoint, metric procedure,
and checkpoint/seed/metric selection or aggregation. Retain unknown historical
values as unknown. Publish per-scene metric triples and make clear whether
entries come from one model, per-metric selection, or averages. The latter two
must not be described as one reproducible model's full triple.

Keep these records in machine-readable reproducibility files: image/split
lists, run configuration manifests and per-scene metric CSVs. The paper should
explain the protocol, essential counts and selection/aggregation rule and point
to those files; it need not print every filename or complete configuration.
Exact image lists can preserve an old split even if its generating random seed
is unknown. Record that distinction instead of inventing a seed.

Before choosing a new-run count, recover the existing split-center evidence and
check whether its methods, training protocol and selected splits support the
intended conclusion. If it suffices, R1.5 needs documentation and analysis rather
than additional training. Otherwise propose the smallest comparison that
addresses the uncovered scope. A small selected-scene study must be described
as such; it cannot establish robustness across every scene or split family.

The agreed study covers eight Mip-NeRF 360 scenes at a 60-degree gap and
two Tanks & Temples scenes (Truck and Train) under discontinuous coverage. For
final GS-LSS and 3DGS-MCMC at one declared training seed, two alternatives add
40 runs (32 + 8). The original-split references are 20 model slots,
shared with compatible main-table work; use the final R1.2 formulation for both
reference and alternative splits. Old results from a different loss are not a
matched reference. See the [scope and reuse conditions](SPLIT_STUDY_ALL_SCENES_2026-09-29.md).
The two agreed alternatives give three splits
per scene including the original. Report each dataset/protocol group separately;
the design does not separate dataset effects from protocol effects or establish
training-seed stability. It tests split sensitivity within the specified settings.
The earlier all-benchmark 48/96-new-run scope and one-alternative option are
superseded. The scope is agreed, but no revision experiment has been scheduled.

The following earlier design concentrated on fewer scenes and more repetitions.
It is retained as discussion history, outside the agreed study:

| Family | Scenes | Split choices | Methods | Initial screen | Repeated study |
| --- | --- | --- | --- | ---: | ---: |
| 60-degree contiguous gap | Bonsai, Garden | Four centers: 0, 90, 180, 270 degrees in the saved convention | GS-LSS, 3DGS-MCMC | 16 runs at one seed | 48 total at three seeds |
| Discontinuous coverage | Same two scenes | Three saved region selections with comparable training-view counts | Same pair | 12 runs at one seed | 36 total at three seeds |

The three region selections must be materialized from the actual selection
algorithm, not described only as random. Training seeds can be 0/1/2 for new
runs. Pair the image splits and seeds between methods; pair initialization
where their implementations permit it. Different gap centers may contain
different numbers of views; record counts and use a secondary count-matched
control if those differences dominate the conclusion.

This extended screen has 28 configuration/seed combinations before reuse;
three-seed coverage of both families has 84 total, not 84 additional. These
counts describe this optional design, not required new runs or an instruction
to enqueue it. Historical matching runs and shared reference runs can reduce
the new work. Training-seed spreads within
each split and paired performance differences across splits should be reported
separately. Show per-split outcomes, range/worst case, and the number of tested
splits. Do not treat thousands of test images as thousands of independent
training trials or claim universal robustness from a small selected scene set.

There are local Bonsai/Garden split-center checkpoints for GS-LSS and 3DGS, so
we should first recover manifests and fresh metrics from them. The chosen
stronger MCMC comparator may still need additional runs. Existing Flowers
experiments are exploratory because Flowers is outside the submitted eight
Mip-NeRF 360 scenes. A 20k and a 30k checkpoint from one run are not two split
trials.

Paper-space plan: work within the current 12-page manuscript target by replacing
and tightening existing material. Use a short precise protocol description and
one compact split-sensitivity table or plot for the essential result. Consider
compressing Table VII's five individual trial rows to a summary with the trial
count and explicitly defined spread, retaining the individual values in the
reproducibility files. This can make room for split evidence without adding a
separate long subsection. The final page count must be checked after layout;
this is a presentation plan, not a guarantee about the compiled length.

Following the 29 September discussion, put complete per-scene/per-split values,
image IDs, configuration tables and split-generation details in readable form
directly in the response PDF, using an appendix within that document. Keep
machine-readable copies accessible through a stable versioned link where feasible,
with consistent contents and checkpoint mappings. This does not depend on
arbitrary archive attachments being accepted by the submission portal.
Do not assume an unlimited or page-exempt supplementary PDF: the decision letter
requires compliance with its paging guidelines, and R1 reported seeing no
supplement. Keep the protocol's meaning and the key empirical conclusion visible
in the main paper and ensure the accompanying files are actually available to
reviewers. Scope the stability language to the actual experiment. This also
answers R3.1/R3.4.

**Shared decision for R3.4:** after recovering or obtaining the planned R1
evidence, assess whether compatible repeated runs cover additional settings.
If they do, reuse them to expand the stability summary and describe the
observed variation across those tested settings. A single-seed split screen
supports split sensitivity but not within-setting training variability. If
adequate repeat evidence remains limited to the original setting, narrow the
stability conclusion accordingly. This sharing does not make the optional
28/84-run design mandatory or add a separate R3 experiment queue.

**R1.6 — Which components help, how sensitive are they, and do they harm well-covered scenes?**

The [remaining R1.6 ablation table](R1_6_ABLATION_TABLE.md) lists only the eight
study types still owned by R1.6 after checking the full plan. The normal-loss
comparison is owned by R1.2. The frozen-CPF control is now explicit in R1.1;
freezing support alone was not sufficient. Table V's broad drift/noise evidence
remains under R1.3 and geometry evaluation under R1.4. These shared items and
the common full reference F are excluded from the R1.6-specific training list.
They remain in the full revision plan and are not assumed completed.

Scope reconsidered after the user identified overlap and excessive volume:
organize one shared evidence package across comments, not a separate experiment
campaign for each comment. The earlier default of two scenes followed by three
seeds for every substantive comparison is superseded by the staged plan below.
R1.6 explicitly requests empirical checks, so reducing duplication should not
be presented as having answered a named factor without evidence.

Settle the R1.2 field semantics and use one shared tuned reference recipe.
Reuse the R1.2 normal comparison, suitable R1.1 support/update controls and
verified Table V drift/noise evidence. Evaluate independent geometry on those
same models using R1.4's evaluator. Such additional metrics normally require
evaluation, not a separately trained model for each reviewer comment. Reuse
requires compatible code, target/loss, configuration, split, initialization,
seed and checkpoint rule; similar experiment names are insufficient.

Start unmatched component and sensitivity checks on one informative scene with
one declared training seed. This is a limited diagnostic, not a multi-seed
robustness estimate. Confirm conclusions central to the revision with additional
seeds or a second scene as needed; prioritize close, variable or unexplained
contrasts, and stronger generality claims. Do not multiply every parameter
setting by every scene, split, method and seed by default. Preserve all attempts
and disclose tuning/selection. A tuned full reference is permitted; a causal
component claim still needs an interpretable comparison.

| Requested factor | Concrete proposed comparison | What must be distinguished or recorded |
| --- | --- | --- |
| Positive-only vs free-space | Surface anchors only; anchors plus the actual local probability-target samples; those samples plus ray free-space negatives | Existing off-surface radial samples are not necessarily explicit visibility-based free-space labels. Match total sample/loss budgets. |
| Color loss | Full versus zero color weight | Keep normal/probability weights and all other training settings unchanged. |
| Observation-aware gating | One gate-on/off contrast with the same field, head, noise scale and schedule | Identify learned versus count-based gates. Separate drift/noise gates only if their joint removal leaves a material question unresolved. |
| Drift weight lambda_s | Reference and a lower/higher positive weight, for example 0.5/1/2 times the reference | Reuse a zero-weight arm only if it is matched; Table V's CPF-off row also changes noise conditioning and is not automatically that arm. Log realized drift and clipping. |
| Noise weight lambda_epsilon | Reference and a lower/higher weight, for example 0.5/1/2 times the reference | Distinguish head learning from noise removal. Reuse a matched noise-off arm if available; Table V's original-noise row is not noise-off. |
| CPF start time | Reference, an earlier start and a later start around the chosen schedule | Keep a documented CPF-to-LSS lag; record total field updates. Do not change teacher quality and training budget without identifying both effects. |
| Sampling density | 0.5, 1 and 2 times the field supervision point budget | Specify whether this controls support subsampling or queries per point; keep the other fixed. Hold kernel bandwidth fixed for this diagnostic, since an adaptive nearest-neighbor bandwidth could otherwise change too. |
| Standard well-covered views | Full method versus a relevant matched baseline on one or two existing scenes | Reuse compatible standard-split models. Add a CPF-off arm only if needed to isolate the cause. Retain held-out views and test the biased-view recipe first; label separately tuned variants. Bound conclusions to the tested scenes. |

The R1.6-specific component work is color loss, gating and supervision type.
The R1.1 source/corruption panel does not supply these isolated comparisons:
changing support source is not changing supervision type, corruption is not
gate removal, and probability-only source comparisons can remove more than
color alone. Likewise R1.5's alternate biased splits do not cover the standard
well-covered setting. Keep these genuine gaps in the list. If the R1.2
target/loss changes, old component models cannot automatically serve as
ablations of the revised full method.

For the four scalar settings above, three levels each share one reference:
on one scene and seed this means eight non-reference configurations, before
reuse, not a 3^4 factorial grid. This is a bounded screening option, not an
automatic job count. Begin with compatible existing points and add only missing
contrasts needed for the intended sensitivity claim. Zero-weight component
controls need not become extra sweep points unless they answer a separate
question. Table VI's DBSCAN/architecture study remains useful for those factors;
it does not isolate supervision sampling density or replace these weight and
schedule checks.

For one biased scene/seed, the remaining list contains four component variants
and eight sensitivity variants, all compared with the shared F. Add two models
for one standard well-covered scene. This gives **14 R1.6-owned configuration
slots before reuse**, excluding the shared reference, frozen CPF and normal-loss
controls. Moving shared items to their owning sections removes duplicate
accounting; it does not eliminate them from the total revision work. Existing
compatible checkpoints can reduce new training, and further confirmations are
outside this initial count.

For free-space labels, sample between a training camera and a sufficiently
reliable observed depth, with a safety margin near the surface. Regions behind
that surface or outside observed rays are unknown, not automatically empty.
When rendered depths are wrong, these labels can be wrong too; use the oracle
diagnostic to separate free-space usefulness from teacher error. A one-class
positive-only field can admit degenerate solutions, so show field profiles and
gradients in addition to image metrics. Keeping local off-surface samples as
the middle arm makes this a useful test of the actual method.

For sampling density, report support points per refresh, perturbed samples per
anchor, field update batches, and the number of Gaussians queried by LSS.
These are different quantities. Query-density or kernel-bandwidth sensitivity
can be a follow-up if the initial density effect is ambiguous, not an automatic
full factorial sweep. Likewise, add a two-factor weight/gate interaction only
when the one-factor results leave a mechanistic question unresolved.

Choose the strongest supported configuration, but allow the findings to simplify
the method: a component that contributes little need not be advertised as
essential. Standard-view regressions, if present, should determine recommended
use conditions or a clearly documented activation rule. Avoid assuming that
every extra loss must improve every scene.

Paper deliverables should reuse space and evidence across comments: a shared
component table, a compact sensitivity figure if needed, and a short
well-covered-results panel or paragraph. Do not create one full table for every
comment or print the entire run inventory. Keep complete configurations and
per-run results in accompanying reproducibility files, with essential evidence
visible in the manuscript. Each named factor needs verified evidence, a shared
planned comparison, or a reasoned scope explanation; an unresolved factor must
remain labeled unresolved. A removed component can use its existing removal
comparison rather than trigger a second study.

**Existing evidence: useful leads, not finished reviewer answers**

The checkpoint-library index contains 141 records. This inspection checked 73
candidate records in component, split-center, sensitivity and TnT ablation
families; all 73 referenced local PLY files exist. Counts include alternate
iterations/retries, not 73 distinct validated experiments. Stored scores were
not freshly rendered in this task. The paths, selected configuration fields,
source hashes and depth-file counts are preserved in
[R1_EVIDENCE_INSPECTION.json](/data1/zhangao/repos/3dgs-imc-codex/writings/tvcg-major-revision-2026-09-28/R1_EVIDENCE_INSPECTION.json).

The inspected matched Bonsai 20k component family uses `sfm_depth`, SfM anchoring,
a direct Gaussian depth loss and three views per iteration. It is not a
render-only-support study of the submitted default. Its `no_learned_variance`
configuration sets `cpf_noise_weight=0`, making it a noise-off comparison. Its
separately saved `no_normal_loss` uses color weight 0.1, versus 1.0 in the inspected
full reference. Recover the source version and a truly matched full run before
using those differences causally. Even the saved no-color result is higher
than that full reference, so these records do not already demonstrate that
each component improves quality.

The six Bonsai sensitivity records include two DBSCAN radii, two noise settings,
one pull setting and one activation schedule; they are not a complete set of
the requested sweeps. The twelve `tnt60_table5` records cover DBSCAN/architecture
conditions rather than all R1.6 factors. This inventory can save work, but it
does not justify labeling all ablations complete.

**Execution order, checkpoints and what would require rerunning tables**

1. **Resolve meaning and implementation:** recover the canonical recipe,
   validate the field/gradient mathematics and head-training paths, and validate
   MatrixCity reference geometry. These determine what subsequent tests mean.
2. **Run a compact mechanism pilot:** R1.2 alternatives on analytic fields and
   two scenes; R1.1 support sources on MatrixCity. For R1.3, verify/reuse Table V
   first and add only controls justified by a remaining claim/evidence gap.
   Reuse needed runs for R1.6. Save geometry snapshots and resource measurements.
3. **Complete the shared evidence package:** audit reuse and deduplicate the
   R1.1/R1.2/R1.6 component controls before scheduling remaining tests. Start
   sensitivity checks on one diagnostic scene; confirm only as needed for the
   claimed scope. Check well-covered views and recover/extend split evidence.
   Freshly evaluate compatible existing checkpoints first.
4. **Reconcile the manuscript:** consistent equations and pseudocode, sourced
   tables and arithmetic, per-scene manifests, scoped claims, then a point-by-point
   response with each new result referenced. Put indispensable evidence in the
   main text; R1's form says they saw no supplementary files, so do not rely on
   a supplement alone to explain the core mechanism.

Do not sum every panel's row count independently: full models, CPF-off models,
frozen/refreshed support and noise-off variants recur. Deduplicate by full
configuration, code version, split, seed, initialization and checkpoint rule.
Estimate GPU-hours from the first measured pilots; old mixed-hardware timing
tables do not support a reliable current estimate. The 28/84 counts above describe
only the optional extended split study; they are neither a required minimum nor
a count of newly needed training jobs.

New authorized runs use local physical GPU 0 or 1, one per process, with fresh
directories under `/data1/zhangao/outputs`; do not run CUDA on hust2 or displace
the existing reproduction jobs. For each new model retain the rendering
checkpoint and learned sidecars, full resumable training state where possible,
CPF/head weights, configuration, exact split, seeds, source hashes, logs,
metrics, timing, and declared selection rule. A record must say whether it
supports historical reproduction, a new revised-method comparison, or an oracle
diagnostic.

| Final change | Training/evaluation implication |
| --- | --- |
| Text or algebra correction, algorithm unchanged | Existing compatible checkpoints remain usable; verify affected claims. |
| New geometry metrics or test subset | Re-render/evaluate compatible models; a subset change does not change how a model was trained. |
| Changed training cameras, initialization or direct supervision | Retrain affected methods for the new comparison; old models stay separately labeled. |
| Changed CPF target, normal loss, drift/noise mechanism or trained head | Retrain affected GS-LSS rows and dependent ablations attributed to the revised algorithm; unchanged compatible baselines need not be retrained solely for this reason. |
| Different selection/aggregation of existing runs | Recompute and disclose the actual selection; this does not create missing checkpoints or recover unknown historical settings. |

The completion criterion for this revision work is an evidence-backed answer to
each concern, not that every variant loses to the full model. A failure or a
simpler successful configuration can still produce a stronger, more defensible
paper. Acceptance remains the reviewers' decision.

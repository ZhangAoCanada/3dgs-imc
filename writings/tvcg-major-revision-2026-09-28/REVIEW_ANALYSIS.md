GS-LSS: initial assessment of the TVCG Major Revision — 28 September 2026

This assessment concerns **TVCG-2026-07-0789**, using the [decision letter and full reviews](../GS-LSS-submission/TVCG_reviewers/TVCG_reviewers.txt)
provided by the user. The editor requests Major Revision by **24 December 2026**.
The earlier TVCG-2026-04-0389 rejection is a different decision. The submitted
[TeX](../GS-LSS-submission/GS-LSS.tex), [PDF](../GS-LSS-submission/GS-LSS.pdf), reviews,
and dated audits remain unchanged. The reproduction campaign remains running;
this assessment does not launch new revision experiments or modify its selection.

My assessment is that the paper has a credible revision path, but a prose-only
response will not resolve the main objections. The reviewers recognize the
biased-view problem and the value of aggregating training geometry. The disputed
part is whether the particular CPF/LSS mechanism is mathematically coherent,
implemented as described, and causally responsible for the claimed improvements.
Acceptance is not assured by the Major Revision decision.

| Reviewer | Recommendation | Main concern | My interpretation |
| --- | --- | --- | --- |
| R1 | Reject | CPF reliability, conflicting supervision, sampling interpretation, independent geometry evidence, reproducibility and missing ablations | The main technical obstacle. Several concerns are supported by the manuscript and local evidence. |
| R2 | Major revision | Clear problem statement, stronger comparison to prior work, mechanistic analysis, efficiency and literature | Supports the research direction but wants its contribution and practical costs explained and substantiated. |
| R3 | Minor revision | Implementation details, numerical consistency, equation sign and bounded reliability claims | Positive assessment with concrete corrections. The apparently small numerical issues connect to larger provenance problems. |

The associate editor explicitly repeats R1's mathematical and experimental
concerns and R3's CPF-reliability concern. Those priorities matter more than
counting favorable recommendations. The [issue map](issue_map.csv) covers all
15 numbered reviewer comments and the editor's synthesis, with evidence and
proposed actions. Its actions are proposals, not completed responses.

**The central technical questions**

R1.1 and R3.4 ask whether a support field trained from the same biased model
can correct that model. Self-supervision is not automatically ineffective:
aggregation, smoothness and delayed updates can regularize an underdetermined
fit without supplying new observations. But that is a hypothesis requiring
evidence. A systematic depth error can survive aggregation and be reinforced.
DBSCAN removes some outliers; it does not establish that the remaining surfaces
are correct. A generic limitations paragraph cannot isolate this effect.

A useful controlled comparison would keep the Gaussian training recipe fixed
and change the support source: fixed SfM support, frozen rendered support after
warm-up, and periodically refreshed rendered support. An independent depth or
oracle-support diagnostic on a suitable scene can test how guidance changes
with support quality. Reference quality, coverage, density and coordinate scale
must be controlled or reported; otherwise denser support can masquerade as more
accurate support. Oracle information must be labeled privileged and separated
from ordinary training results. A corrupted-support example can also show when
gating helps and when the method fails.

Freezing the support point cloud and freezing the CPF network are different
interventions. The fixed-versus-joint CPF comparison must specify whether the
network weights continue to update, separately from whether support is refreshed.

R1.2 is a substantive mathematical concern. The submitted Eq. (21) uses a radial
target around a support point p:

    P(x) = exp(-||x-p||² / (2σ²))
    ∇P(x) = -(x-p) P(x) / σ².

That gradient points toward p. It cannot generally align with the same oriented
surface normal at points sampled on both sides of a surface, and it vanishes
at p. Equation (22) nevertheless favors alignment with that oriented normal.
The current `loss_fn` contains the corresponding cosine-gradient constraint.
This does not make a weighted training loss impossible to minimize: it means
the two desired targets can conflict, so the network learns a compromise.
We should not defend it as an exactly consistent probability/normal model or
assert that normal supervision is merely harmless smoothing.

Before a large sweep, establish the intended field semantics and examine a
small analytic plane/sphere example. Then decide whether the current term can be
justified as an explicitly heuristic regularizer with a measured trade-off,
should be removed, or requires a formulation consistent with the chosen field.
Changing the loss is a method change and needs corresponding experiments;
rewriting the equation alone would not repair the implementation.

R1.3 correctly distinguishes stochastic optimization from sampling a stated
target distribution. However, the submitted PDF already says after Eq. (19)
that no exact Langevin/MH or stationary-distribution claim is made. Our response
should acknowledge that clarification and remove the surrounding ambiguity,
not say the reviewer simply overlooked everything or that a disclaimer proves
valid sampling. State-dependent stochastic samplers require a specific relation
between dynamics and the target; [Ma, Chen and Fox's construction](https://arxiv.org/abs/1506.04696)
is a relevant technical reference. We do not need to convert GS-LSS into a new
exact MCMC algorithm to answer this review. A coherent optimization formulation,
an explicit update rule and evidence for the field's contribution are a more
direct path, provided they match the implemented method.

**An implementation issue we must resolve before writing the response**

The current FAST wrapper sets `CPF_PULL_WEIGHT=0` and defaults
`CPF_NOISE_TRAIN_WEIGHT` to zero. The current code's additional noise-head
training branch runs only for a positive weight; observation-count gating is
implemented explicitly. The manuscript describes learned drift, learned scale
and gate heads, and updates to center, scale and rotation. The current color
loss is also optional and defaults to zero, while the paper lists color
supervision as a CPF loss component.

These observations are about the inspected current code, **not proof of what
every historical checkpoint ran**. Other branches and older configurations may
differ. They nevertheless make a code-to-equation audit the first dependency:
identify the version and recipe we will defend, trace actual gradient paths,
verify which parameters change, and record which heads receive training signal.
Changing a module name is not a meaningful ablation if that module was inactive
in the full model. New implementation repairs must be labeled and evaluated as
revision changes rather than attributed retroactively to old results.

Evidence: [FAST defaults](../../run.sh:72), [CPF pull and gating](../../scene_experimental_branch3/gaussian_model.py:555),
[noise-head training](../../scene_experimental_branch3/gaussian_model.py:733),
[current CPF loss](../../scene_experimental_branch3/gaussian_model.py:825),
and [noise network](../../scene_experimental_branch3/imc_experimental.py:126).

**Geometry evidence and reproducibility**

R1.4 is persuasive. Held-out RGB quality measures image synthesis; it does not
by itself establish accurate global geometry. Distance from sparse training SfM
support can mark a legitimate unobserved surface as a floater. A method pulled
toward that same support can score well without reconstructing more correct
geometry. FFR can remain a carefully scoped support-distance diagnostic, but
it cannot carry the central geometric claim on its own.

The strongest addition would evaluate rendered depth or reconstructed surfaces
against an independent reference, with fixed masks, alignment, opacity handling,
and coverage reporting. A depth metric must account for invalid or missing
predictions rather than rewarding their exclusion. MatrixCity is a candidate,
but the provenance of locally available depth must be checked: a folder named
`depth` does not prove independent ground truth. Reference data used only for
evaluation must not feed the ordinary training or model-selection process.
A second scene/domain would help establish scope; it is a proposal, not a claim
that suitable local reference geometry has already been verified.

R1.5 and R3.1 require exact splits, camera counts, random seeds, per-scene
results and a usable recipe. Training-seed variability and split variability
are different. Five runs on one fixed split cannot establish robustness to
choosing a different missing region. A bounded paired experiment across several
split locations and seeds, on a few representative scenes, can answer this
without rerunning every possible benchmark combination. Split IDs and training
seeds should be separate fields, with paired comparisons and scene-level results.

The ongoing performance-capped baseline campaign is explicitly selected on test
performance. Its accepted subset is not an unbiased baseline estimate and cannot
be used as evidence that a method reliably wins. For revision comparisons I
recommend a fixed prospective run set and reporting every result, without
stopping according to whether a baseline is weaker than a table entry. The
current campaign remains unchanged in this task; its saved first attempts and
complete histories can still provide diagnostic or reusable artifacts when
their protocols match. Historical per-metric selection in our own results also
needs to be disclosed or replaced by a consistent comparison rule.

**The numbers need a source-based reconciliation**

R3.2 identifies two related issues. The printed Table III difference is
16.951 - 16.225 = **0.726 dB**, whereas 16.693 - 16.225 = **0.468 dB**.
The submitted TeX's commented batch-ablation rows associate 16.693 with one
training view per iteration and 16.951 with three views. This is a plausible
editorial/protocol explanation for the discrepancy, not verified run provenance.
The corresponding original GS-LSS models have not been recovered. We should
not write that explanation as established fact or simply change Table V to
16.951. If batching differs, report it and address the compute comparison.

The newer [MatrixCity investigation](../table3-checkpoint-search-2026-09-27/MATRIXCITY_PROTOCOL.md)
also demonstrates that three located historical baseline records use 216 mixed
training views and all 154 mixed test views. That conflicts with the manuscript's
claim of exclusively aerial training and ground evaluation; the strict subset
we constructed contains 189 aerial training and 130 street test views. It does not establish the original protocol for missing models.
Pure-aerial reruns and mixed-training controls must have separate identities;
matching or underperforming a published number does not establish reproduction.

The previous [provenance audit](../research-audit-2026-09-26/revision_issue_map.csv)
also flags FFR values/definitions, sensitivity-row labels, trial provenance and
timing comparisons. These are internal findings beyond the new review, and should
be resolved before the revision repeats those claims. Correct figures only from
verified evidence; missing evidence does not justify inventing a replacement.

R3.3's sign correction is direct. With δ = f - S, the penalty difference is
R_hat - R = -E[δ]. Consequently, the right-hand side of printed Eq. (13) needs
a minus sign. This confirms the reviewer's point. The nearby residual-drift
explanation must use the same convention; this assessment does not change TeX.

**A focused experimental package**

R1.6 should be answered item by item, but it does not require an exhaustive
Cartesian product of every parameter. I would organize the necessary evidence
around the following questions, reusing verified runs where possible:

| Question | Controlled evidence to prepare | Main comments answered |
| --- | --- | --- |
| Does CPF add information useful for optimization despite self-supervision? | Fixed SfM, frozen warm-up support, refreshed support; independent/oracle or controlled-corruption diagnostic | R1.1, R3.4 |
| Is the claimed update actually responsible? | Matched baseline, drift-only, adaptive-noise-only and full variants with verified active mechanisms | R1.3, R1.6, R2.3 |
| Which field constraints and protections matter? | Frozen versus updated CPF weights; color removal, normal removal/consistent alternative, observation-gate removal; distinguish local radial targets from actual ray free-space supervision | R1.2, R1.6 |
| Does geometry improve independently of the training support? | Reference-depth/surface accuracy and coverage, plus a failure example | R1.4, R3.4 |
| Are gains robust and is regularization safe? | Multiple splits and seeds, standard well-covered split, bounded λ_s/λ_ε/start-time/sampling-density sensitivity | R1.5, R1.6, R3.4 |
| What does the method cost? | Matched hardware/protocol timing, CPF build/update/query breakdown, peak memory, model size and measured inference speed | R2.4 |

The audit indexes 141 candidate saved experiment records, including component,
split-center, parameter and timing studies. These are leads, not 141 verified
independent experiments answering the current review. Their code, configs,
checkpoint iteration, matching controls and negative outcomes need checking
before reuse. Some later runs study variants of the submitted mechanism.

The normal-loss question should precede new large ablations. Otherwise we risk
spending the GPU budget evaluating a field definition that the revision then
changes. Likewise, selecting a canonical implementation and the MatrixCity
protocol must precede a revised component table.

**Presentation, efficiency and literature**

R2's comments can largely share the evidence above. Explain perspective
overfitting with a concrete train-view/unseen-view example. Compare supervision,
support aggregation, update parameters, gating and learned components across
methods; do not infer causal mechanism merely from method rankings. Equal
iteration counts do not imply equal compute when view batching differs.

The standard 3DGS renderer at inference is a useful property, but identical
renderer code does not imply identical FPS: Gaussian count, footprint and
representation cost matter. The existing timing audit reports mixed hardware,
splits and timer scopes, so the 2.77× figure needs controlled support or a narrowed
claim. Discuss offline reconstruction versus time-constrained reconstruction
using measured costs rather than adding application claims without evidence.

I checked the three suggested papers in primary sources:
[StylizedGS](https://arxiv.org/abs/2404.05220),
[Gaussians on Fire](https://arxiv.org/abs/2511.22459), and
[SSD-GS](https://arxiv.org/abs/2604.13333).
They concern stylization, dynamic fire reconstruction and relighting,
respectively. My assessment is that brief, relevant positioning is appropriate;
the request does not establish a need to benchmark all three as biased-view
reconstruction competitors. SSD-GS is already cited in the submitted related
work and bibliography, as an arXiv paper; the authors' record now identifies
ICLR 2026. Update bibliographic details and explain relevance accurately.

The letter specifies regular-paper space of 12 pages, up to 18 with fees, and
black text including the required manuscript materials. The local submitted PDF
is 12 pages; R2 refers to page 14, so map that pointer by section content and
confirm the reviewed file/version before a final response. The reviewers also
disagree about supplement presence; actual portal attachments are not established
by the local text-only review folder. Neither issue weakens their substantive
comments or warrants a speculative accusation about the review.

My recommended order is: establish the canonical method and protocol; resolve
the mathematical inconsistencies and source discrepancies; reuse matching
evidence; run the missing controlled geometry/mechanism/robustness checks; then
write a response linking every claim to a concrete revision or result. The paper's
defensible contribution is a useful observed-support-guided optimizer under
specified biased-view conditions, with measured limits. A broad claim of global
geometry recovery or exact probabilistic sampling requires evidence we do not
currently have.

The accompanying [source manifest](sources.json) records the review/manuscript
hashes and inspected evidence. No training, rendering, manuscript compilation or
statistical validation of new experiments is claimed by this review assessment.

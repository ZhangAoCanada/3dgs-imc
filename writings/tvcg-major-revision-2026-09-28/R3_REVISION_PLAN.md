Reviewer 3: interpretation and proposed revision plan — 28 September 2026

This plan follows a fresh reading of [Reviewer 3's full review](../GS-LSS-submission/TVCG_reviewers/TVCG_reviewers.txt),
the affected [submitted manuscript](../GS-LSS-submission/GS-LSS.tex), the
[R1 plan](R1_REVISION_PLAN.md), and the [cross-review overlap map](CROSS_REVIEW_OVERLAP.md).
It also uses the dated checkpoint/table audits where they bear on the requested
explanations. Those audits are historical evidence, not fresh reproduction.
This is a proposed discussion plan. Manuscript editing and revision experiments
remain deferred until the user directs execution. It launches no jobs and does
not change the submitted snapshot, implementation, or existing experiment queue.

**Overall interpretation**

R3 recommends minor revision, calls the method technically sound, and views the
combination of aggregated support and stochastic updates positively. The four
comments ask for reproducibility details, consistency of numerical reporting,
an algebraic correction, and appropriately scoped limitations/stability claims.
R3 does not request a new method or a broad experiment campaign. This reading
does not reduce the separate substantive requirements from R1 or the editor.

| Comment | Focused response | Shared work | Additional experiment scope from R3 |
| --- | --- | --- | --- |
| R3.1 | Compact configuration table, exact split/seed records, accurate code-availability statement | R1.5 and the common implementation/configuration audit | None by default; recover existing records |
| R3.2 | Explain Table III/V settings and fix the gain attached to each table | R1.5 provenance, R1.3 Table V, R2.3 result discussion | None for arithmetic; any unresolved evidence recovery or final-method regeneration remains shared |
| R3.3 | Correct the missing minus sign and check dependent expressions | R1.3 mathematical audit; coordinate with R1.2 | None for a manuscript-only correction |
| R3.4 | One documented support-error example; expand stability reporting with suitable R1 evidence or narrow the conclusion if that evidence is insufficient | R1.1 supplies the shared example; review compatible repeats from R1, especially R1.5, before choosing stability scope | Reuse shared evidence; assess coverage before choosing either route, without automatically adding a separate study |

**R3.1 — Make the implementation reproducible**

The reviewer names concrete missing settings. The response should provide those
settings, not turn each one into a sensitivity experiment. The current paper
already gives some activation iterations, learning rates, network depths and
loss weights, but does not supply a complete recipe.

Proposed actions:

1. Add one compact configuration table containing total training iterations and
   reported checkpoint iterations; network depths and hidden widths for the
   active CPF/stochastic modules; CPF activation and update frequency; support
   refresh frequency where different; and the sampling distribution/scale with
   its normalization or units. Distinguish support-neighborhood sampling from
   Gaussian-update noise if both are referred to as sampling. Reuse verified
   existing settings rather than fill fields from current defaults.
2. State train/test counts and the view-selection rule for each benchmark
   setting. Point to exact image lists and configuration files for the full
   detail. Record the split-generation seed separately from training seeds;
   if an old split seed is unknown, its recovered image list can still specify
   that split exactly. Do not invent a missing seed.
3. Identify any per-table or per-scene exceptions, particularly view batching
   and the Table III/V settings addressed by R3.2. A compact common table plus
   explicit exceptions is sufficient; do not imply one recipe produced every
   historical result if the run records show otherwise.
4. Add an accurate code-availability statement and, in the eventual release,
   point to the implementation version, configurations, split files, and
   training/evaluation instructions. Report actual access or a genuine release
   plan; this planning step does not publish a repository or invent a URL.

Keep the essential configuration summary in the manuscript and lengthy image
lists/run manifests in accompanying reproducibility files, subject to the
submission's actual attachment/page requirements. Describe the method selected
for the revision and label historical differences clearly.

Overlap: R1.5 already owns protocol/manifests and the common audit checks the
implementation. R3.1 specifies how to present those details. It adds no ablation
or new training requirement merely because a historical field is missing.

**R3.2 — Explain the MatrixCity discrepancy and correct the reported gain**

There are two separate questions: why the GS-LSS scores differ between the
tables, and whether the stated improvement is arithmetically correct.

**Updated after the requested old-version inspection:** the
[recovered paper history](R3_2_PAPER_HISTORY.md) explicitly documents the
batching distinction. The 2 July draft states that the first LSS component
ablation block uses one view per iteration. Its active batch table assigns
0.592/16.693/0.470 to B=1 and 0.594/16.951/0.462 to B=3, matching all three
metrics in submitted Tables V and III respectively. Earlier implementation
text also specifies B=3. This is verified manuscript history; original-run
and checkpoint verification remains incomplete.

| Location | Historical views per iteration | GS-LSS PSNR | 3DGS-MCMC PSNR | Difference |
| --- | --- | --- | --- | --- |
| Table III, MatrixCity main comparison | 3 | 16.951 | 16.225 | 0.726 dB |
| Table V, full LSS ablation | 1 | 16.693 | 16.225 | 0.468 dB |

The main-results paragraph refers to Table III but uses 0.468 dB. With the
submitted table entries unchanged, that sentence should use 0.726 dB. These
are arithmetic checks on paper values, not newly evaluated checkpoint scores.

Proposed actions:

1. Explain the recovered B=1 versus B=3 distinction in the response and the
   manuscript. Label the Table V component comparison as one-view training;
   identify three-view training for the Table III MatrixCity GS-LSS result.
   Make that exception explicit in the implementation paragraph, which
   currently defaults to one view. The historical records now support this
   description beyond the submitted source's commented rows.
2. Explain that the component table fixes B=1, separating that comparison from
   the additional batching setting used for the main result. Keep the full
   method row within the same Table V condition; do not copy the B=3 main score
   into that row. Neither a new batch-size sweep nor restoration of the entire
   historical batching subsection is required just to answer R3.2.
3. Correct the gain and directly associated prose against the retained table
   values. Recompute it if final-method work later changes the numbers. Record
   view batching with the training budget rather than imply equal computation
   from equal iterations alone.
4. Continue the shared checkpoint/configuration recovery separately. The
   recovered drafts establish what the paper reported, not all actual run
   settings, independent metric reproduction, or otherwise identical recipes.
   Keep unresolved split, seed, iteration and source fields explicit.

Evidence boundary: the [26 September special-table audit](../research-audit-2026-09-26/special_table_verification.csv)
did not link the full Table V triple to a saved-result source, and the
[27 September Table III search](../table3-checkpoint-search-2026-09-27/REPORT.md)
left the submitted GS-LSS originals unlocated. The new paper-history evidence
resolves the historical manuscript explanation without changing those
checkpoint statuses. The [provenance entry](../../docs/research/checkpoints.md)
also records a mixed-view protocol in located MatrixCity baseline runs; this
inspection does not establish the missing GS-LSS runs' split.

Overlap: R1.5 owns provenance, R1.3 already owns Table V interpretation, and
R2.3 will reuse the corrected experimental discussion. Correcting arithmetic
does not require training; resolving the historical recipe is a separate task.

**R3.3 — Correct the sign in printed Eq. (13)**

The reviewer's algebraic concern is correct. The paper defines

\[
\delta(\mathbf g)=f_\theta(\mathbf g)-\mathcal S(\mathbf g),\qquad
\widehat{\mathcal R}=\mathbb E[1-f_\theta],\qquad
\mathcal R=\mathbb E[1-\mathcal S].
\]

Consequently,

\[
\widehat{\mathcal R}-\mathcal R=-\mathbb E[\delta(\mathbf g)],
\]

and the corrected identity is

\[
\nabla_{\mathcal G_t}\widehat{\mathcal R}
-\nabla_{\mathcal G_t}\mathcal R
=-\nabla_{\mathcal G_t}
\mathbb E_{\mathbf g\sim\mathcal G_t}[\delta(\mathbf g)].
\]

Proposed actions:

1. Add the missing minus sign in the editing manuscript's corresponding
   equation. Identify it by content: printed Eq. (13) has the TeX label `eq12`.
2. Check the adjacent residual-drift explanation and directly dependent
   expressions for sign consistency, keeping the existing definition of delta.
3. Confirm the relevant implemented support-loss/update sign during the already
   planned R1.3 audit. This identity describes approximation error; its typo
   alone does not establish that a training update was wrong. A manuscript-only
   correction requires no retraining. If the implementation also needs changing,
   handle that as a separately evidenced method correction under the shared plan.

Overlap: R1.3's objective/gradient audit. Coordinate notation with R1.2, but
this equation is distinct from the incompatible probability/normal targets.
Selecting removal or replacement for R1.2 does not itself fix this sign.

**R3.4 — Clarify support failure and match stability claims to the evidence**

The reviewer asks for an example of unreliable CPF guidance and a stability
conclusion within the available evidence. Our response will address both.
Following the user's discussion, the stability response will consider two
routes: expand the evidence using suitable R1 experiments, or narrow the
conclusion when the available evidence remains limited. Choose after reviewing
the actual coverage and outcomes, rather than predetermining a wording-only fix.
The example is an explicit deliverable; “no separate experiment” means sharing
the evidence already planned for R1.1, not omitting the example. A hypothetical
wall explanation alone will not be treated as completing the first part.

**Part 1: provide one documented example of unreliable support.**

1. First look for one suitable documented case in existing results. If none
   is available, use one scene/region from R1.1's already-planned support-quality
   or controlled support-perturbation study. R1.1 owns the experimental work;
   R3.4 uses the same case, rather than scheduling it again.
2. Explain the case in a short chain: what is wrong with the training geometry;
   how that error affects CPF support or guidance; and what weaker guidance or
   reconstruction problem is actually observed. Use the comparison and outputs
   already available from the shared study. Report only effects supported by
   those records; a poor rendering alone does not establish that CPF caused it.
3. Add a short paragraph to Limitations and reuse a small crop or panel from the
   shared R1.1 figure if it helps readers see the error. Clearly label whether
   the case arose naturally or used a controlled perturbation. Identify the
   source scene/run and any changed support condition in the evidence records.
   A controlled example should not be presented as a naturally occurring case.
4. Explain why the retained safeguards can reduce some errors but cannot
   guarantee correction of incorrect support. Match this explanation to the
   final selected implementation. No new method component is proposed here.

For intuition, a wall estimated at the wrong depth could make CPF favor support
near a displaced surface. This remains an illustrative mechanism until a
documented case is selected; it is not a claimed experimental finding.

Completion of Part 1 requires an identified case, traceable evidence and a
concise explanation. The case is still pending. If the shared results do not
show the proposed weakening/failure, report what they do show and revisit the
example. Keep its evidence status pending until a suitable case is documented.

**Part 2: choose the stability response after reviewing the R1 experiments.**

First inspect recovered records and the additional evidence obtained for R1,
especially R1.5. Identify which scenes/view settings have repeated training of
the same method configuration, what varies between trials, and whether the
records support the final selected GS-LSS formulation. Then use one of these
two routes:

| Available evidence | Response to R3.4 |
| --- | --- |
| R1 provides usable repeated runs in additional scenes, splits or coverage conditions | Expand the stability summary with those shared results, in an updated Table VII or a reference to the common R1 table/figure. Report each setting, trial count and an explicitly defined spread. Describe stability across the settings actually tested, with the conclusion following the observed variation. |
| Suitable repeated-run evidence remains confined to Table VII's original setting, or the additional records lack compatible within-setting repeats | Restrict the conclusion to the verified setting. Its caption already names Tanks & Temples with a 60-degree coverage gap; revise the following paragraph and any repeated generalization accordingly. If the original trial records cannot be substantiated, resolve that through the existing provenance work rather than treat narrowing alone as verification. |

Different settings and different random seeds answer different questions. One
training run on each of several camera splits measures sensitivity to the split;
it does not estimate run-to-run variability within those splits. R1.5's
single-seed screen can support a split-sensitivity conclusion, while compatible
repeated runs can also support the broader stability response. Likewise, ablation
variants are not independent repeats of the full method. Keep those analyses
separate even when they appear in one compact shared table.

If repeats in new settings show larger fluctuations, report that dependence
and qualify the claim. More settings alone do not establish generally stable
behavior. Preserve configuration/seed records and the selection/aggregation rule;
do not pool materially different recipes or old/new formulations as equivalent
trials.

Prefer reuse of the experiments already needed for R1. No additional setting
count or repeat count is mandated by this branch decision, and R1.5's optional
extended repeated study remains optional. If the shared evidence is insufficient,
the narrower conclusion is the valid fallback. The final choice is pending the
R1 evidence review and does not authorize launching runs now.

**Separate shared provenance work:** the existing
[special-table audit](../research-audit-2026-09-26/special_table_verification.csv)
matched saved paired-scene metric means for Trials 1–3 but did not locate the
exact triples for Trials 4–5. It also checked statistics calculated from the
printed rows; that arithmetic does not verify independent seeds or identical
settings. Recover or qualify that historical provenance under R1.5/INTERNAL.3
before calling the trial setup verified. This existing issue is separate from
R3's request to keep the conclusion within the tested setting.

Overlap: the same documented example answers R1.1 and the first part of R3.4;
R1.5 supplies stability/protocol records, and R2.3 can reuse the explanation.
In the response letter, identify the example and the changed stability sentence
or expanded stability evidence separately. This revised plan adds an explicit
shared-evidence requirement, not another experiment queue. Manuscript editing and experiment execution remain
deferred until the user directs them.

**Scope and completion**

The minimum R3 additions are a compact implementation summary, an explanation
and correction of the table discrepancy, the corrected identity, and one
documented limitation example plus a stability conclusion matched to the shared
evidence: expand reporting where supported or narrow the claim otherwise.
Reuse shared R1 work and relevant existing material. No independent R3
experiment list is added.

The Table III/V batching description is now recovered from explicit earlier
drafts. Original-run/checkpoint verification and Table VII trial provenance
remain explicit evidence dependencies. The paper-history checks do not resolve
them. Any final-method regeneration caused by R1.2 remains owned by that
existing dependency, not counted again as a new R3 request.

Later, answer each R3 number separately in the response letter with the exact
changed section/table/equation and the shared evidence where relevant. Completion
means the requested manuscript changes and substantiated records exist; this
plan does not claim that those revision actions have already been carried out.

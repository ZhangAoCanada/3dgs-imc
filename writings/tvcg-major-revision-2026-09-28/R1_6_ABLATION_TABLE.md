R1.6 — remaining ablations after checking the shared work (proposal)

This table now contains only the **eight study types owned by R1.6**. The
[full R1 plan](R1_REVISION_PLAN.md) was checked against the existing Tables V/VI,
the [R1.2 fix study](R1_2_FIX_OPTIONS.md) and the other comment-level plans. All
ten factors in [R1.6](../GS-LSS-submission/TVCG_reviewers/TVCG_reviewers.txt)
remain accounted for across the full plan. This is proposed work, not completed
evidence or a claim that historical checkpoint reuse is already verified.

The following shared work is excluded from the R1.6 training list:

- **Normal loss: R1.2 owns it.** The original normal-on/off comparison was
  already planned. The full plan and fix-options note now explicitly require
  the corresponding matched control if the chosen replacement retains a
  gradient loss; a simultaneous target/loss replacement alone is not that
  ablation. If the term is removed, use the R1.2 removal evidence.
- **Frozen versus updated CPF: R1.1 owns it.** The audit found that R1.1 froze
  support, but did not explicitly freeze field weights. A separate frozen-CPF
  control Rf is now included there. This relocates the already requested
  R1.6 control into the shared update study; it does not make the experiment
  disappear or claim that it was already completed.
- **Broad drift/noise components: R1.3 and Table V.** Their evidence and
  provenance remain handled there; no duplicate component suite is scheduled.
- **Independent geometry evaluation: R1.4.** Reuse its evaluator and the
  relevant checkpoints instead of scheduling new models just for that metric.

Let **F** be the full reference configuration after the R1.2 field/loss decision.
Use one common biased-view scene, split, initialization, training seed,
resolution, training budget and evaluation/checkpoint rule for the biased-view
rows below (A2, A3, A5–A9; original IDs are retained).
Change only the stated factor, subject to the necessary sample/budget details
below. F is supplied once by the shared method-validation/R1.1 work and is
excluded from the remaining R1.6 configuration count. Different settings may be
tuned under the repository policy, but additional changes and selection must
be recorded and their effect on interpretation stated.

| ID | Remaining R1.6 ablation | Comparison | R1.6-owned configurations before reuse | Why it remains |
| --- | --- | --- | ---: | --- |
| A2 | Positive-only versus free-space supervision | Surface-only probability supervision; F's near-surface probability targets; F plus visible-free-space negatives | 2 | R1.1 changes support source, not the supervision-label comparison |
| A3 | Color loss | F versus zero color-loss weight | 1 | A probability-only support-source panel may remove both color and normal losses; it does not isolate color |
| A5 | Observation-aware gating | F versus gating disabled, with field, heads and configured noise scale unchanged | 1 | R1.1's support-corruption diagnostic is not a gate-on/off test |
| A6 | Drift weight lambda_s | 0.5x, 1x and 2x the chosen positive reference weight | 2 | Table V's component comparison is not a drift-weight sensitivity study |
| A7 | Noise weight lambda_epsilon | 0.5x, 1x and 2x the chosen positive reference weight | 2 | Table V compares noise formulations, not a range of noise weights |
| A8 | CPF start time | Earlier, reference and later start; for a 2k reference, an example is 1k/2k/4k | 2 | A shared warm-up point does not measure activation-time sensitivity |
| A9 | Sampling density | 0.5x, 1x and 2x the reference field-supervision sample budget | 2 | Table VI changes DBSCAN/architecture rather than isolating sample budget |
| A10 | Standard well-covered views | Full method versus the matched backbone/baseline on a standard held-out-view split | 2 per scene | R1.5 changes biased splits; that is not a standard-view control |

Rows A2, A3 and A5–A9 are a focused diagnostic on one scene and one declared seed.
Confirm conclusions central to the revision using additional seeds or scenes
when needed, especially for small or variable differences. One-seed comparisons
do not establish multi-seed stability, and selected scenes do not establish
universal robustness. A10 must keep a test set and first use the recipe chosen
for biased views; report any separately tuned standard-view recipe separately.

For A2, distinguish low probability near support from visibility-based empty
space. Use training-camera rays and a documented reliable depth source for
free-space labels; space behind a surface is not automatically empty. Keep the
total supervision budget comparable, specify allocation across sample types and
record where color/gradient targets are applicable. A3 must not silently change
another loss weight. For A5, record actual perturbation magnitudes because
gating changes them. For A8, keep the CPF-to-guidance lag defined and record
total field updates. For A9, define whether density means anchors or queries per
anchor, keep the other fixed, and hold kernel bandwidth fixed. Inspect existing
component/sensitivity/checkpoint candidates before training, but require a
matching final-method reference before treating them as these comparisons.

All rows report PSNR, SSIM and LPIPS. Use the same checkpoints for independently
validated depth/geometry measurements where a suitable reference exists, sharing
R1.4's evaluator. Use field profiles for the supervision comparison where helpful.
These extra measurements do not by themselves require additional training runs.

The remaining biased-view work is **four component variants + eight sensitivity
variants = 12 configurations**, with F counted only in the shared plan. A10 adds
**two configurations per well-covered scene**. For one biased scene and one
well-covered scene, this gives **14 R1.6-owned configurations before reuse**.
Compatible existing checkpoints reduce the number of new jobs. Confirmatory
seeds/scenes and unrelated reviewer studies are outside this first-pass count.

The earlier 17-configuration illustration included F, frozen CPF and the
normal-loss control. Those three shared configurations are now assigned to
their owning sections instead of appearing in this remaining-work table.
Removing duplicate entries does not remove necessary shared experiments from
the total revision effort or establish that they have already been run.

Table V remains the broad drift/noise component evidence; no duplicate Table V
campaign is added for R1.6. Its CPF-off row is not automatically a matched zero
drift-weight setting, and its original-noise row is not noise-off. Table VI's
DBSCAN and architecture comparisons do not substitute for A9. None of these
historical rows is assumed freshly verified by this planning task.

Present the findings using a shared component table, a compact sensitivity panel
and a short standard-view comparison, with per-run records in reproducibility
files. Reuse the same evidence in each relevant reviewer response.

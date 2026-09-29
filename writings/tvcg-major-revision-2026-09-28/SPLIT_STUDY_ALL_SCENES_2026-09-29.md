R1.5 discussion: alternative splits and dataset scope — 29 September 2026

**Agreed scope after the user's further reduction.** Restrict the 60-degree
gap study to all eight Mip-NeRF 360 scenes and discontinuous coverage to the
two Tanks & Temples scenes, Truck and Train. These counts were checked in the
manifest linked below. The user accepted two alternatives per scene. The
one-alternative option in the table is retained for comparison only. Use final GS-LSS and
3DGS-MCMC at one declared training seed per configuration.

| Study | Scenes | One alternative, two methods | Two alternatives, two methods |
| --- | ---: | ---: | ---: |
| Mip-NeRF 360, 60-degree gap | 8 | 16 new runs | 32 new runs |
| Tanks & Temples, discontinuous coverage | 2 | 4 new runs | 8 new runs |
| Total | 10 scene/protocol cases | 20 new runs | 40 new runs |

Two alternatives plus the original give three splits per scene. There are 20
original-split reference model slots, shared with compatible revised-main-table
work: 60 total slots including references for the agreed design, or 40
with one alternative. The 40/20 additional-run counts exclude any necessary
reference regeneration, other revision studies and retries. GS-LSS alone would
need 20/10 additional runs for two/one alternatives, respectively, but the paired
MCMC comparison is recommended to assess whether the relative advantage persists.

The compatibility, split-design and reporting requirements below still apply:
use the final R1.2 formulation, fixed per-scene recipes across splits, exact image
lists and a declared selection rule. Report each dataset/protocol separately.
This supports split-sensitivity conclusions for Mip-NeRF 360 at 60 degrees and
Truck/Train under discontinuous coverage. It does not test every dataset under
both protocols, permit separating dataset effects from protocol effects, or
establish training-seed stability. Avoid pooling all ten scenes into one claim
about both families. Preserve and report all outcomes.

This is the agreed scope, superseding the 48/96-new-run scope below. Exact
alternative splits and seed values are still to be specified. The full and
handwritten plans have been synchronized at the user's request. No revision
experiment was started or queued; the submitted manuscript remains unchanged.

**Earlier all-benchmark-scene proposal, retained as discussion history.** The
counts and recommendation in the following paragraphs describe the previous,
broader option, not the current scope.

The user proposes keeping the 60-degree gap width and testing two alternative
gap centers for every scene, with analogous alternative region selections for
discontinuous coverage; one alternative is an option if the cost is too high.
This note interprets alternatives as camera splits. Revision execution remains
deferred. No splits, training jobs or evaluations were launched for this note.

**Verified scope and additional training counts.** Scene membership comes from
the [main-table checkpoint manifest](../research-audit-2026-09-26/main_table_checkpoint_manifest.csv),
deduplicating dataset/scene pairs for GS-LSS in Table I at 60 degrees and Table II.
This verifies scene counts, not fresh rendering or compatibility of old models.

| Study | Scenes | One alternative, two methods | Two alternatives, two methods |
| --- | ---: | ---: | ---: |
| 60-degree gap: Mip-NeRF 360 and Tanks & Temples | 10 | 20 new runs | 40 new runs |
| Discontinuous: same scenes plus Mill 19 | 14 | 28 new runs | 56 new runs |
| Total | 24 scene/protocol cases | 48 new runs | 96 new runs |

The methods in this budget are final GS-LSS and 3DGS-MCMC, with one declared
training seed per configuration. This is a recommendation, not a requirement to
rerun every baseline. GS-LSS alone would need 24 or 48 additional runs, but would
not test whether its advantage over MCMC survives the changed splits.

Mip-NeRF 360 scenes: bicycle, bonsai, counter, garden, kitchen, room, stump,
treehill. Tanks & Temples: train_resize, truck_resize. Mill 19: building-small,
residence-small, rubble-small, sci-art-small. The two studies overlap in ten
scenes, so there are fourteen unique scenes, not twenty-four. MatrixCity and
Zeche are outside this Table I/II split study.

The budget excludes the original-split references: 24 cases x 2 methods = 48
reference model slots. Including references gives 96 slots with one alternative
or 144 with two alternatives. References supplied by compatible main-table work
are shared, not trained again under R1.5. Their availability is not established
by this counting exercise. The earlier 12-slot and 84-slot proposals included
their reference splits, so those totals are not directly comparable to the new
additional-run counts. Scene runtimes differ; these counts are not GPU-hour
estimates or the total revision budget.

**Recommendation for discussion.** Prefer all-scene coverage to the earlier
two-scene pilot. Start with one alternative for every case: 48 additional runs.
Two alternatives would give three splits per scene including the original and
stronger evidence of variation within a scene, for 96 additional runs. A second
alternative can be a later extension; if restricted to selected scenes, record
that selection and report their coverage separately. The 12-slot pilot remains
a minimal fallback. No 48-run or 96-run queue has been approved or scheduled.

**Reference reuse.** Use the final R1.2 formulation on both reference and new
splits. If R1.2 changes the GS-LSS loss, the old GS-LSS table results cannot serve
as its matched reference; share the revised main-table reference models instead.
Reuse baseline models only after checking their actual recipe, split, iteration,
metric procedure and selection rule. Newly reproduced checkpoints are distinct
from recovered original models. A printed dataset average or a historical
per-metric selection pool is not one model's full metric triple. Recover the
per-scene sources and apply a stated, consistent reporting rule. If an old
training seed is unknown, do not describe that comparison as seed-matched.

**Comparison design.** Keep the gap width at 60 degrees and move its center
meaningfully; use different discontinuous region selections with the same
number of regions and comparable view counts. Save exact image lists, counts,
split parameters and seeds. Use the same splits for both methods and keep each
method's per-scene training recipe, iteration and selection rule fixed across
splits. Initialization and supervision must use the appropriate training data.
Each genuinely new split needs a model trained on that split; re-evaluating an
old model is not a replacement and can expose evaluation views used in training.

Report the original and alternative results per scene, dataset summaries and
the paired GS-LSS-minus-MCMC differences, including unfavorable outcomes and
ranking reversals. Different held-out regions can differ in difficulty, so raw
metric changes alone do not identify changes in relative method performance.
Summarize split variation separately from variation among scenes. One training
seed does not estimate within-split seed variance; R3.4's training-stability
conclusion must still follow its own evidence.

This updates the R1.5 recommendation in the [consolidated plan](CONSOLIDATED_REVISION_PLAN.md)
and [R1 plan](R1_REVISION_PLAN.md). The accepted shared controls and response-PDF
recommendation in the [preceding discussion note](DISCUSSION_UPDATE_2026-09-29.md)
are unchanged. The user's authored revision plan remains unchanged.

Shared work across Reviewers 1, 2 and 3 — 28 September 2026

This mapping follows a fresh reading of the entire
[decision and all three reviews](../GS-LSS-submission/TVCG_reviewers/TVCG_reviewers.txt),
the relevant submitted equations/tables, and the current
[R1 plan](R1_REVISION_PLAN.md). It is a discussion aid, not an instruction to
execute the revision plan. Experiments remain deferred at the user's request.
The submitted manuscript is unchanged. Sharing proposed work does not mean a
reviewer concern is already answered by completed evidence.

The subsequent [detailed R2 plan](R2_REVISION_PLAN.md) develops all five R2
comments and verifies the suggested literature while retaining this shared-work
allocation. It adds no duplicate R1 ablations or execution authority.

The [detailed R3 plan](R3_REVISION_PLAN.md) develops the four R3 comments as
focused documentation, consistency and scope corrections. It records the
confirmed arithmetic/sign errors separately from unresolved run provenance and
adds no independent R3 experiment list. Both plans remain for discussion.
The later [R3.2 paper-history check](R3_2_PAPER_HISTORY.md) recovered explicit
one-view component-ablation and three-view main-result descriptions from older
drafts; original checkpoint verification remains a separate unresolved task.

R2.2 shares R1.3's conceptual positioning in Related Work; R2.3 explains the
existing results and can reuse relevant R1 mechanism evidence. Other strong
overlaps include all four R3 comments with R1's implementation, provenance, mathematical and
reliability work. R2.4 can share measurements but needs its own cost/benefit
discussion. R2.1 and R2.5 chiefly require editorial and literature work; they
also address R1's general form responses about readability and references.

| Comment | Overlap with R1 | Work to share | What still needs an explicit answer |
| --- | --- | --- | --- |
| R2.1: simpler, shorter introduction sentences | R1's general readability concerns | One sentence-level language edit preserving scientific meaning, claims and paragraph structure | The user narrowed the scope after discussion. Core-claim/novelty changes belong to other comments; no new figure, introduction reorganization or experiment is required by R2.1. |
| R2.2: sharper Related Work and MCMC distinction | R1.3's conceptual positioning and terminology | Literature-based analysis of specific limitations and the additional role of GS-LSS's observed-support guidance | The user narrowed the scope to Related Work. Table V analysis, mathematical reconciliation and empirical support remain with R1.3/R2.3; no new experiment, figure or table is required by R2.2. |
| R2.3: explain why results improve | R1.3's terminology and Table V interpretation; relevant R1.1/R1.6 findings can be reused | Explain existing result patterns through the actual method and cite relevant existing ablations/visuals; incorporate useful shared R1 findings later | Primarily an experimental-discussion edit. Rankings alone do not isolate causes, so distinguish interpretation from measured effects. The full R1 mechanism/geometry program is not an added R2.3 requirement; no new experiment, figure or table is required by this comment. |
| R2.4: application-dependent efficiency trade-off | Shared timing provenance under R1.5 and INTERNAL.3; compatible planned runs may supply records | Reuse valid timing/quality evidence and correct the already-known timing mismatch | Explain overhead sources, benefits, training versus rendering, and offline versus time-constrained use. Component profiling and new efficiency benchmarks are not required. Replacement timing is conditional on retaining an unsupported quantitative comparison; it remains part of the existing evidence correction. |
| R2.5: suggested references | R1's general references concern; the same Related Work editing pass as R2.2 | Add missing StylizedGS/Gaussians on Fire citations and update the already-cited SSD-GS entry, with brief relevant context | Check final bibliography details and avoid a duplicate SSD-GS entry. Primary-source checks are recorded in R2_REVISION_PLAN.md; some final publisher identifiers remain unresolved. No extended literature survey, new baseline or experiment is required. |
| R3.1: implementation details and replication | R1.5 plus the common implementation/configuration audit | One compact configuration table, exact split/seed records, and an accurate code-availability statement | Include the requested iterations, network widths, CPF update frequency and sampling scale/units; identify relevant setting exceptions. Recover existing records rather than infer old values from current defaults. No sensitivity study is required for each listed field. |
| R3.2: Table III/V discrepancy and inconsistent gain | R1.5 provenance; R1.3 Table V; R2.3 result discussion | Recovered paper history explicitly labels the component block B=1 and maps the main MatrixCity triple to B=3; share corrected labels/prose and continued checkpoint recovery | Explain the documented batching distinction and attach 0.726/0.468 to the correct comparison. Update the general one-view description with the Table III exception. Old drafts verify the manuscript protocol, not the missing original checkpoints; no new batching sweep is required solely for this explanation. |
| R3.3: sign in printed Eq. (13) | R1.3 objective/gradient audit; coordinate notation with R1.2 | Correct the approximation-error identity and check directly dependent expressions and the relevant implementation sign | Keep delta=f-S and add the minus sign; the source label is eq12. This is separate from R1.2's probability/normal incompatibility. A manuscript-only correction requires no retraining; a code change would need its own evidence and shared validation. |
| R3.4: unreliable training geometry and limited stability evidence | R1.1 support reliability; R1.5 stability/provenance; relevant R2.3 interpretation | One documented support-error example; review R1 repeated-run coverage before choosing expanded stability reporting or a narrower conclusion | The example remains required and pending; label natural versus controlled errors. For stability, reuse compatible repeats across additional settings if available, or limit the claim to substantiated evidence otherwise. A single run per split supports split sensitivity, not within-setting seed variability. The branch remains pending R1 results and adds no automatic experiment queue. |

**Numerical and mathematical checks relevant to the mapping**

The submitted Table III reports MatrixCity PSNR 16.951 for GS-LSS and 16.225
for 3DGS-MCMC: the difference is 0.726 dB. Table V reports 16.693 for its full
variant against 16.225: the difference is 0.468 dB. The main-results paragraph
uses 0.468 while referring to Table III. These are verified manuscript values,
not new rendering measurements. The later old-version inspection found stronger
evidence than the submitted source's comments: the active 2 July draft states
that the component-ablation block uses one view and assigns the main-table
triple to three views. See the linked history note for exact Git exports and
line references. This establishes the manuscript's historical batching
distinction. Missing checkpoint/run verification remains under the shared audit.

For R3.3, the manuscript defines delta=f-S, R=E[1-S], and Rhat=E[1-f]. Therefore
Rhat-R=-E[delta], so the right-hand side of the stated gradient-difference
identity needs a minus sign. This correction should be made in the editing
manuscript during the later mathematical revision. The submitted snapshot is
preserved. This identity is separate from the user-selected R1.2 candidate
P*=exp(-[n^T(x-p)]^2/(2h^2)) and its derived local gradient.

**How to avoid duplicate experiments and paper space**

R2.2 and R2.3 do not introduce a separate experiment campaign on top of the
current R1 plan. R2.3 should first explain the existing results; relevant shared
R1 findings can strengthen that discussion without making the entire R1 study
a prerequisite. R3.1–R3.3 principally need documentation, provenance and algebra;
any remaining evidence recovery or final-method regeneration belongs to those
existing dependencies. R3.4 will provide one documented support-error example,
using existing evidence or the case already planned under R1.1. The example is
not optional in our response plan; sharing it avoids duplicate experimental
work. For the stability part, review the R1 results before choosing whether to
expand the summary with repeated runs from additional settings or narrow the
conclusion to the substantiated scope. Report setting-dependent fluctuations
where present. Single-seed split comparisons do not establish within-setting
training stability, and this sharing does not make R1.5's optional extended
study mandatory or remove its separate split-sensitivity question.

R2.4 primarily needs a fuller application-dependent discussion in the existing
efficiency section. Explain overhead sources qualitatively and discuss offline
reconstruction versus tight training-time budgets using valid evidence. The
known Table VIII timing mismatch remains a separate evidence correction shared
with R1.5/INTERNAL.3. Recover comparable records or narrow unsupported numerical
claims first. If a retained comparison needs replacement timing, collect it
alongside already needed runs after authorization, with hardware, split,
resolution, training budget and timer scope recorded. Component breakdowns,
cost/quality curves, parameter sweeps, memory and inference benchmarks are not
R2 requirements. An unchanged renderer alone does not establish unchanged FPS.

The current R1.6 list remains eight study types. This cross-review mapping adds
no training configurations, queues or experiments, and does not select the
R1.2 winner. One shared manuscript table/figure may be referenced in multiple
responses. The response letter should nevertheless answer each numbered comment
directly, explaining how that shared result addresses that reviewer's question.

The associate editor repeats the support-reliability, target-consistency,
stochastic-interpretation, geometry and ablation concerns. The same core R1
evidence can therefore support the editor response; no separate editor-specific
experiment campaign is warranted by the letter.

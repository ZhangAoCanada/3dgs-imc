Reviewer 2: interpretation and proposed revision plan — 28 September 2026

This plan follows a fresh reading of
[Reviewer 2's complete comments](../GS-LSS-submission/TVCG_reviewers/TVCG_reviewers.txt),
the submitted introduction, related work, results analysis and efficiency section,
the [current R1 plan](R1_REVISION_PLAN.md), and the
[cross-review overlap map](CROSS_REVIEW_OVERLAP.md). Primary literature sources
were checked for the MCMC comparison and the three suggested citations. This is
analysis and planning only. Revision experiments remain deferred until reviewer
discussions are complete and the user directs execution. No submitted-paper
text, training code, configuration or queue is changed by this planning pass.

**Overall interpretation**

R2 calls the work technically sound and recommends major revision for clarity
and fuller discussion. My reading is that the reviewer broadly accepts the
research direction but wants clearer introductory sentences, a more analytical
Related Work comparison, an explanation of the results through the method's
design, a fuller efficiency discussion, and the suggested citations. This is an
interpretation of the written review, not a prediction of acceptance. The five comments do not
explicitly request a new architecture, new datasets, new baselines, or an
independent ablation campaign.

| Comment | Main action | Overlap with R1 | Additional R2 experiment scope |
| --- | --- | --- | --- |
| R2.1 | Shorten and simplify introduction sentences while preserving meaning and structure | R1 general readability feedback; substantive claim changes belong to their separate comments | None |
| R2.2 | Revise Related Work to explain specific limitations and sharpen the MCMC comparison | R1.3's conceptual positioning and terminology only; its theory/evidence work remains separately owned | None |
| R2.3 | Improve the existing results discussion with concrete algorithmic explanations | Reuse Table V under R1.3 and relevant R1 findings; do not inherit the full R1 experiment plan | None required by this comment |
| R2.4 | Explain overhead, benefits and suitability for different application contexts | Share valid timing records and R1.5 provenance; known timing mismatch remains with the existing evidence audit | None by default; replacement timing only if retaining a quantitative comparison unsupported by saved records |
| R2.5 | Add the missing suggested citations, improve their brief context, and update the existing SSD-GS entry | R1 general references feedback; same Related Work editing pass as R2.2 | None |

**R2.1 — Shorten and simplify the introduction's sentences**

Scope narrowed after the user's discussion: the explicit criticism is complex,
lengthy sentences. The reviewer mentions understanding the core issue and
contribution as the desired benefit of clearer language; that does not establish
a request to redefine the claims or restructure the introduction. The earlier
plan inferred too much by proposing a reordered argument, new presentation of
the contributions, and an optional figure.

Treat R2.1 as a focused language edit:

1. Split sentences carrying several ideas or nested clauses into shorter,
   logically connected sentences, generally with one main idea each.
2. Use direct subjects and verbs, clearer references and simpler phrasing.
   Remove unnecessary words and repetition while retaining needed technical
   terms and qualifiers.
3. Apply this especially to the existing explanation of perspective overfitting
   and the method overview. Make the same content easier to read rather than
   introducing a new problem definition or contribution.
4. Preserve the scientific meaning, claims, paragraph order, examples, figures
   and citations by default. Check that shortening a sentence does not alter
   its causal relationship, scope or qualification. A fixed word-count threshold
   is unnecessary; readability and accuracy are the checks.

Overlap: R1's general readability feedback. Substantive changes to novelty,
theoretical framing, support limitations or the final method remain owned by
R2.2 and the relevant R1 comments. They may later require consistent wording in
the introduction, but they are not extra requirements inferred from R2.1.
This language-edit plan does not need to wait for an experiment to be interpreted.
On 29 September the user clarified that the broader introduction suggestions in
their own plan are optional ideas to review after the main fixes. Keep those
options available without treating them as required R2.1 scope or a disagreement.

Deliverable: a sentence-level readability revision of the existing introduction.
No new experiments, figures, claim redefinition or wholesale reorganization are
required for this comment. Manuscript editing remains deferred during discussion.

**R2.2 — Sharpen the critical comparison in Related Work**

Scope narrowed after the user's discussion: this comment explicitly targets
Related Work. It requests more precise analysis of earlier methods' limitations
and a sharper explanation of GS-LSS's novelty, especially relative to MCMC-based
methods. The earlier plan imported Table V interpretation, mathematical
reconciliation and implementation-audit tasks into this comment unnecessarily.
Those tasks remain with R1.3/R2.3 and the common implementation work.

This is a substantive literature-positioning edit, beyond R2.1's sentence-level
simplification, but it does not require a new experimental argument.

Revised actions:

1. Retain the current Related Work structure unless a local rearrangement helps
   readability. For each relevant group, explain its main contribution, the
   particular constraint or optimization mechanism it uses, and the limitation
   relevant to our biased-view setting. Connect the remaining issue to the role
   of GS-LSS. Replace some descriptive listing with this analysis rather than
   expand the section substantially.
2. Make existing limitation statements specific. The submitted section already
   mentions biased photometric supervision and limited global exploration;
   simply repeating those phrases will not sharpen the comparison. Explain
   which constraint, source of guidance or assumption leaves the relevant issue
   unresolved. Check the cited papers and avoid assigning the same limitation
   to every depth, multi-view or stochastic method.
3. Give the MCMC paragraph particular attention. Accurately describe what the
   cited stochastic methods contribute, then identify the additional learned
   observed-support field and its role in the selected GS-LSS formulation.
   Explain the distinction in guidance and update design rather than rely only
   on labels such as hand-crafted versus learned. A concise comparative
   paragraph is sufficient; a new comparison table is not required.
4. End the relevant paragraphs with a clear connection to our contribution.
   Keep these as design distinctions supported by the cited literature and our
   method definition. Empirical superiority and component-effect explanations
   belong in the results discussion. A novelty statement should identify what
   is added and what is inherited without implying that only GS-LSS uses
   multi-view information or that its support is independent of training data.
5. Keep wording consistent with the interpretation settled under R1.3. Use that
   conceptual clarification in Related Work; leave objective derivations,
   algorithm corrections, Table V analysis and any evidence recovery with their
   existing owners. Literature verification supports this edit; it is not an
   instruction to launch an implementation audit or experiment under R2.2.

Overlap: R1.3's conceptual distinction from stochastic 3DGS and consistent
terminology. The overlap is narrower than the earlier plan suggested: R2.2 does
not inherit all of R1.3's mathematical or empirical work. R2.5 can share the
Related Work editing pass while retaining its specific citation task.

Deliverable: a focused revision of Related Work, especially its limitation
statements, MCMC comparison and concluding connection to GS-LSS. Table V remains
under R1.3/R2.3; no new experiment, figure or table is required for R2.2.
Manuscript editing remains deferred during discussion.

**R2.3 — Explain the results through supported mechanisms**

Scope revised after re-reading the comment in light of the R2.1/R2.2 discussion.
The reviewer asks the authors to elaborate on the architectural or algorithmic
advantages behind the reported results. This is primarily a request for better
experimental discussion. It does not explicitly request a new mechanism study
or a controlled experiment for every explanation. The earlier plan imported too
much of the R1 evidence program into R2.3.

The submitted main-results discussion often moves from method rankings to broad
conclusions about entire method families. Table V already has a component-level
discussion. The focused task is to make the main-results explanation more
concrete and connect it to relevant existing evidence.

Revised actions:

1. Select the important patterns in the existing quantitative and qualitative
   results rather than explain every table entry. For each, connect the biased
   viewing condition to the optimization difficulty and then to the relevant
   GS-LSS design choice.
2. Explain the complementary intended roles of aggregated observed-support
   guidance and stochastic updates in plain technical language. Identify what
   information CPF supplies beyond an individual image loss and how the
   implemented updates use it. Match this account to the final selected method;
   avoid adding an architecture change to answer a request for explanation.
3. Use brief references to the existing component ablations, especially Table V,
   and already-present visual examples where they actually support the point.
   Table V belongs naturally in this results discussion; it need not become a
   new table or be repeated in full. Relevant R1 results can strengthen the same
   paragraphs later, without becoming additional R2 requirements.
4. Distinguish what was measured from the interpretation of why it happened.
   A ranking alone does not isolate a cause. Preserve measured tradeoffs: the
   submitted full Table V variant has the best PSNR/LPIPS, while the noise-only
   variant has the best SSIM. Those scores do not by themselves prove separate
   physical roles for the components, and the noise-only arm also changes
   support conditioning. Explain only the advantages the evidence supports.

Overlap: R1.3 supplies consistent algorithmic terminology and the existing
Table V interpretation. Findings already required by R1.1/R1.6 can be reused
when directly relevant. Geometry validation remains owned by R1.4 and a support
failure example by R1.1/R3.4; neither becomes a new R2.3 study or a prerequisite
for every explanatory paragraph.

Deliverable: improve the existing main-results and relevant ablation prose so
readers understand why the design can explain the observed behavior. No new
experiment, figure or table is required by this comment. The separate R1
evidence obligations remain unchanged.

**R2.4 — Make the efficiency trade-off useful for choosing an application**

The wording explicitly asks for a more nuanced discussion of computational
overhead versus benefits in different application contexts. The main response
should therefore be a focused expansion of the existing Efficiency Trade-off
section. The earlier plan made component profiling and cost/quality analysis
too prominent; neither is an explicit requirement of this comment.

Revised actions:

1. Explain the sources of extra training work qualitatively: constructing or
   refreshing support, fitting and querying CPF, and applying the additional
   updates, as applicable to the final implementation. Numerical percentages
   or a component timing breakdown are unnecessary unless already available
   and useful.
2. Relate that cost to the observed held-out rendering benefits under biased
   views. Use valid existing quality/time evidence and explain the tradeoff;
   simply calling the overhead reasonable does not tell readers when it is
   worthwhile.
3. Contrast concrete application conditions. Offline reconstruction for virtual
   tours or reusable scene assets can prioritize view quality and tolerate
   longer preparation. Frequent reconstruction or a tight online training
   budget makes additional optimization cost less attractive. These are
   suitability arguments, not claims of evaluated application deployments.
4. Separate training overhead from rendering-time work. Explain that CPF and
   stochastic updates are absent from inference if this remains true for the
   selected method. Do not infer identical FPS solely from use of the standard
   renderer; the resulting Gaussian field can still affect speed. A quantitative
   inference claim needs evidence, but the discussion can stay qualitative.

**Separate existing evidence issue, not a new request inferred from R2.4:**
the [dated timing audit](../research-audit-2026-09-26/timing_provenance.csv)
records baseline timings on a 4090 with a standard Truck split and an external
timer, and GS-LSS timing on a 5090 with a discontinuous split and an internal
timer. Table VIII's quality values came from different gap-60 records. These
sources do not establish the claimed 2.77x ratio as a controlled comparison of
the quality-producing configurations. This issue remains with the existing
timing/provenance audit (INTERNAL.3 and the shared R1.5 documentation work).

Recover valid existing records first. If a comparable numerical ratio cannot
be substantiated, correct or narrow the table/text rather than repeat that
ratio as verified. If retaining a quantitative comparison requires replacement
measurements, collect them with already needed runs after execution is
authorized, matching hardware, scene/split, resolution and timing scope and
recording the training budget. Pair costs with the configurations that produced
the quality values. This conditional correction does not make profiling,
cost/quality curves, parameter sweeps, FPS or memory benchmarks R2.4 requirements.

Overlap: R1.5 and the existing internal audit share the provenance correction;
already planned R1 runs may supply valid timing records. The application-focused
discussion is specific to R2.4. No R1.1/R1.6 experiment is added or made mandatory
solely for this comment.

Deliverable: a concise expansion of the existing efficiency discussion, with
any retained quantitative claim corrected through the shared evidence audit.
No new efficiency experiment is required by default. The reviewer's page-14
pointer is matched by section title; the local submitted PDF has 12 pages.

**R2.5 — Position the suggested literature and update references**

This is a citation request. Address it with a small Related Work/bibliography
edit that recognizes the suggested developments in their actual context. It
does not require a new literature survey, detailed comparisons for all three
works, or experimental baselines. The earlier plan's citation checks remain
useful, but the manuscript additions should be compact.

Revised actions:

1. Add the missing StylizedGS and Gaussians on Fire citations in appropriate
   existing Related Work sentences, with a brief explanation of their topics.
2. Keep the existing SSD-GS citation, improve its surrounding description only
   where useful, and update its arXiv-only bibliography metadata to the verified
   publication information. Do not add a duplicate entry.
3. Check author names, exact titles, venues, years and available final
   identifiers. Explain the connection to 3DGS without forcing these different
   application tasks into a direct comparison with biased-view reconstruction.

The following source record supports the later edit; it is not a requirement
for a new table or three extended discussions in the manuscript.

Primary sources checked on 28 September 2026:

| Work | Verified focus and placement recommendation | Citation action |
| --- | --- | --- |
| [StylizedGS](https://arxiv.org/abs/2404.05220) | Controllable stylization, with floater filtering and geometry-preserving regularization. Place in the general 3DGS applications/geometry-preservation discussion. This connection is relevant context, not evidence that it solves the same biased-view task. | Add the missing reference. The [author project](https://kristen-z.github.io/stylizedgs/) identifies TPAMI 2025; its displayed BibTeX still uses the older arXiv entry, so check final publisher metadata when editing. |
| [Gaussians on Fire](https://arxiv.org/abs/2511.22459) | Dynamic flame reconstruction from limited camera views, using task-specific motion/geometric cues. Briefly discuss as an example of specialized constraints for difficult capture conditions, with the static/dynamic task distinction explicit. | Add the missing reference. The [author project](https://jna-358.github.io/gaussians_on_fire/) and [official ECCV list](https://eccv.ecva.net/Conferences/2026/AcceptedPapers) identify ECCV 2026. |
| [SSD-GS](https://arxiv.org/abs/2604.13333) | Relighting through material/light-transport decomposition. Keep it in the general 3DGS applications/appearance-model discussion. | Already cited via `ssdgs` in the submitted related work and bibliography. Expand its positioning if useful and update the arXiv-only entry to ICLR 2026, confirmed in the authors' arXiv record and [official repository](https://github.com/irisfreesiri/SSD-GS). |

Use the same compact Related Work editing pass as R2.2. Brief context in an
existing sentence is sufficient; a separate paragraph for each work is not
required. Publisher/OpenReview pages returned browser challenges in the earlier
verification pass, so final publisher-specific details remain to be checked;
the venues above have primary author/conference support. Keep arXiv identifiers
distinct from publisher metadata.

In the later response, explain what positioning was added or expanded. For
SSD-GS, note its existing inclusion constructively and describe the improved
discussion/metadata. The editor permits explaining a suggested reference's
limited relevance; a brief accurate positioning of these works appears sufficient
without broadening the paper's experimental task.

Overlap: R1's general references concern and the shared Related Work editing
pass under R2.2. These specific citation additions remain an explicit R2.5 task;
they do not inherit R1.3's mathematical or experimental work.

Deliverable: add two missing references, maintain and update the existing
SSD-GS entry, and provide concise context in Related Work. No new experiment,
benchmark, figure or comparison table is required.

**Scope and completion**

All five R2 comments are addressed primarily through focused writing or
literature changes. R2.2 reuses R1.3's conceptual positioning in Related Work;
R2.3 explains existing results and can incorporate relevant shared R1 findings;
R2.4 discusses application-dependent costs and benefits; R2.5 adds or updates
the suggested citations. None explicitly requires new experiments. The known
timing mismatch still needs correction through the existing evidence audit;
replacement measurement is conditional on retaining a quantitative comparison
that cannot be supported by saved records. The R1 experiment list remains
unchanged, and the R1.2 removal-versus-signed-normal decision remains pending.

Final prose must follow the method actually selected under R1.2 and the common
implementation audit. Any regeneration caused by that method change is owned
by the existing R1 dependency, not counted again as an R2 request. Shared evidence
can appear once in the manuscript and be cited separately in each point-by-point
response. Keep essential conclusions in the main paper and detailed run records
in the accompanying reproducibility files.

Completion means each R2 comment has an identifiable manuscript change and,
where needed, traceable supporting evidence. This planning document does not
claim those changes or measurements are complete, and starts no experiments.

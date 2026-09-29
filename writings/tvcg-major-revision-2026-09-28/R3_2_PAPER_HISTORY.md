R3.2: recovered paper history for view batching — 28 September 2026

The older drafts explicitly document the user's explanation: the MatrixCity
LSS component ablation uses **one training view per iteration**, while the
main-results GS-LSS row matches the **three-view** configuration. The two full
metric triples match the submitted Tables V and III respectively. This is now
verified manuscript history, rather than an inference from commented rows in
the submitted source alone.

This inspection did not recover the original training runs or freshly evaluate
their checkpoints. The evidence level is historical paper/configuration
description, with the existing checkpoint-provenance gaps preserved separately.

**Sources and verification**

Inspected the five paper commits dated 15 June–2 July 2026 in the nested local
repository `writings/GS-LSS-editing`: 11 distinct TeX blobs. Each contains the
same active one/two/three-view ablation triples. Also inspected two TeX documents
recovered from the 3 July archive, the submitted/local editing sources, and the
current revised/raw/diff TeX copies on `hust2`. The remote current copies retain
commented batch rows, while the earlier local Git objects provide the explicit
active table and explanation.

Exact Git exports and the archived active text are preserved under
`evidence/r3_2_paper_history/`. The [manifest](evidence/r3_2_paper_history/manifest.json)
records host-qualified repository/archive paths, full commit IDs, Git blob IDs,
SHA-256 hashes, matching lines and the versions scanned. Exporting Git objects
did not restore deleted files in the editing worktree. Archive text hashes match
the original hashes in the dated recovery record. Commit dates identify stored
paper versions, not the dates of the training experiments.

| Source | Relevant content |
| --- | --- |
| [15 June, commit 5e4997c](evidence/r3_2_paper_history/2026-06-15-5e4997c-bare_jrnl_new_sample4.tex) | Line 526 sets the random batch size to 3. Line 619 gives the MatrixCity main-result triple 0.594/16.951/0.462. Active batch rows at lines 722–724 map one, two and three views to their scores. |
| [26 June, commit 4fff5f9](evidence/r3_2_paper_history/2026-06-26-4fff5f9-bare_jrnl_new_sample4.tex) | Lines 547/549 state three-view training. The active batch table at lines 850–852 retains exactly the same values. |
| [2 July, commit e5e1258](evidence/r3_2_paper_history/2026-07-02-e5e1258-bare_jrnl_new_sample4_revision.tex) | Lines 580/581 state the three-view configuration. Lines 881–883 contain the active batch rows. Crucially, line 893 explicitly states that the first LSS component-ablation block uses a one-view training batch. |
| [Active revision recovered from the 3 July archive](evidence/r3_2_paper_history/archive-2026-07-03-bare_jrnl_new_sample4_revision.tex) | Line 541 gives a general one-view-per-iteration description. The batch-ablation block and explicit one-view exception explanation are absent, but the main/full-ablation triples remain 0.594/16.951/0.462 and 0.592/16.693/0.470. |
| [Submitted source](../GS-LSS-submission/GS-LSS.tex) | Line 535 says one view per iteration unless otherwise stated. Lines 625/742 retain the two triples. The batch rows survive only as comments at lines 746–748. The Table III three-view exception is not clearly stated in the active experimental description. |

The strongest direct statement, from the active 2 July draft, is:

> The first block of LSS ablations is performed on the MatrixCity dataset under one-view training batch, while the second block ablates the random batch size.

The quotation omits the LaTeX citation command immediately after “dataset”; the
underlying words are unchanged. This sentence applies to the component-ablation
block, not only to its full-method row.

**Recovered correspondence**

| Historical views per iteration | SSIM | PSNR | LPIPS | Correspondence |
| --- | --- | --- | --- | --- |
| 1 | 0.592 | 16.693 | 0.470 | Full-method row in submitted Table V |
| 2 | 0.592 | 16.799 | 0.466 | Intermediate row in the older batch-size ablation |
| 3 | 0.594 | 16.951 | 0.462 | GS-LSS MatrixCity row in submitted Table III |

All three metrics match, and the old implementation paragraph independently
states the three-view setting. Thus the historical reason for reporting two
different full-method triples is documented. The paper sequence also shows how
the distinction became unclear: the explicit batch study/description disappeared
while both sets of values were retained. This observation does not identify a
particular author's intention or establish any later rerun.

“Without batched view training” should be expressed precisely as batch size
**B=1**, meaning one training view per optimizer iteration. It does not mean
the entire training dataset contained only one image.

**Suggested answer and minimal manuscript changes**

Address the comment in two parts: restore the configuration distinction, then
correct the gain attached to Table III.

1. State that Table V evaluates the LSS components with B=1, while the MatrixCity
   GS-LSS entry in Table III uses B=3. Explain that fixing B=1 in the component
   table separates that comparison from the additional multi-view batching
   setting. Do not describe the two full rows as repeated measurements under
   identical configurations or claim all remaining historical settings are
   verified solely from these drafts.
2. Add B=1 to the Table V caption or immediately preceding sentence and identify
   B=3 for the Table III MatrixCity GS-LSS result in the implementation details
   or a table note. Correct the general one-view wording so that it includes
   this explicit exception. State batch size as a configuration detail; this
   does not require restoring it as a separate methodological contribution.
3. Correct the main-results improvement to **0.726 dB** if the submitted values
   are retained: 16.951−16.225=0.726. The **0.468 dB** improvement belongs to the
   B=1 Table V comparison: 16.693−16.225=0.468. Recompute both differences if
   final-method work later changes the table entries.

The response letter can include the two relevant settings and full triples in
a short clarification. Restoring the entire batch ablation or adding a new
batch-size sweep is not needed just to explain this discrepancy. Keep the
component comparison within Table V's common batching condition. Report the
actual training budget with batching; equal iteration counts alone should not
be described as equal computation across B=1 and B=3.

This proposed answer is supported by the recovered paper history. It should not
say that the original checkpoints were found, rerendered, or otherwise verified
by this inspection. Their unresolved provenance remains under the existing
R1.3/R1.5 work. Manuscript/rebuttal edits and revision experiments remain deferred;
only evidence and planning records are updated here.

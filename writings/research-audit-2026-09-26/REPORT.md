# GS-LSS research and checkpoint audit

Audit date: 26 September 2026. Reference manuscript: [submitted GS-LSS.tex](/data1/zhangao/repos/3dgs-imc-codex/writings/GS-LSS-submission/GS-LSS.tex), with its [12-page submitted PDF](/data1/zhangao/repos/3dgs-imc-codex/writings/GS-LSS-submission/GS-LSS.pdf).

**The local repository is the main development and original GS-LSS checkpoint repository. `hust2` is an essential companion containing baselines, later revision experiments, and selective imports. I can reconstruct every submitted GS-LSS value in Tables I and II from saved metrics linked to existing checkpoint files. I cannot currently establish checkpoint provenance for every submitted table.** Table III, the MatrixCity component ablation, parts of sensitivity/stability, and the claimed timing protocol remain unresolved or inconsistent with the located evidence.

No training, rendering, checkpoint transfer, dependency installation, paper edit, or experiment-output modification was performed. The only new files are this audit package inside the repository. “Verified” below means saved-metric arithmetic and file linkage, with a PLY header/length check where applicable; it does **not** mean fresh model rendering or successful retraining.

**The most useful deliverables are these.**

| File | Purpose |
|---|---|
| [Research history](/data1/zhangao/repos/3dgs-imc-codex/writings/research-audit-2026-09-26/RESEARCH_HISTORY.md) | Chronology, decisions, failed approaches, and surviving history sources |
| [Main-table checkpoint manifest](/data1/zhangao/repos/3dgs-imc-codex/writings/research-audit-2026-09-26/main_table_checkpoint_manifest.csv) | All 1,584 expected scene/metric source positions for Tables I–III, including explicit unresolved entries |
| [Main-table verification](/data1/zhangao/repos/3dgs-imc-codex/writings/research-audit-2026-09-26/main_table_verification.csv) | All 360 submitted aggregate scalar values compared with stored evidence |
| [GS-LSS preservation list](/data1/zhangao/repos/3dgs-imc-codex/writings/research-audit-2026-09-26/ours_tables_I_II_preserve.csv) | The 131 distinct PLY paths supporting the GS-LSS Tables I/II selections |
| [GS-LSS best-PSNR models](/data1/zhangao/repos/3dgs-imc-codex/writings/research-audit-2026-09-26/ours_best_psnr_checkpoints.csv) | One actual model per scene/protocol/angle: 64 rows, each with its complete metric triple |
| [Full checkpoint inventory](/data1/zhangao/repos/3dgs-imc-codex/writings/research-audit-2026-09-26/checkpoint_inventory.csv) | 7,300 host-qualified checkpoint/artifact paths, including inputs and neural-network sidecars |
| [Special-table audit](/data1/zhangao/repos/3dgs-imc-codex/writings/research-audit-2026-09-26/special_table_verification.csv) | Tables IV–VII arithmetic and source discrepancies |
| [Special-table sources](/data1/zhangao/repos/3dgs-imc-codex/writings/research-audit-2026-09-26/special_table_sources.csv) | Floater, sensitivity, and trial source records |
| [Timing provenance](/data1/zhangao/repos/3dgs-imc-codex/writings/research-audit-2026-09-26/timing_provenance.csv) | Machine, GPU, protocol, timer scope, paths, and ratios for all eight Table VIII methods |
| [Reusable experiments](/data1/zhangao/repos/3dgs-imc-codex/writings/research-audit-2026-09-26/reusable_experiments.csv) | Later component, split-center, sensitivity, timing, and imported models |
| [Unresolved sources](/data1/zhangao/repos/3dgs-imc-codex/writings/research-audit-2026-09-26/unresolved_main_sources.csv) and [candidate models](/data1/zhangao/repos/3dgs-imc-codex/writings/research-audit-2026-09-26/unresolved_checkpoint_candidates.csv) | Separate missing original evidence from alternative models that actually exist |
| [Revision issue map](/data1/zhangao/repos/3dgs-imc-codex/writings/research-audit-2026-09-26/revision_issue_map.csv) | Reviewer concern, evidence, recommended action, and remaining risk |

The manifest records host, dataset, scene, protocol, split/source path when saved, iteration, metric file, PLY path, integrity result, selection rule, and whether the resulting aggregate matches the submitted number. An empty field means it was not established; it should not be filled from assumption.

**Research goal and actual implementation.**

The research asks whether a continuous estimate of aggregated observed support can reduce 3DGS perspective overfitting when training cameras poorly cover the scene. The useful scientific distinction is between fitting the observed images and constraining unsupported Gaussian geometry. The CPF learns from filtered geometry derived during training; LSS uses support information in Gaussian optimization. This is an observed-support regularized stochastic optimizer. Neither the code nor the current evidence establishes exact posterior sampling, a stationary distribution, or recovery of completely unseen geometry.

The main pipeline is: initialize Gaussians, optimize image reconstruction, aggregate geometric evidence, filter it with DBSCAN, fit the CPF with support/normal and optional color objectives, then apply configured support/noise operations during Gaussian optimization. Global aggregation, delayed activation, filtering, smoothness, and observation-aware gating can mitigate biased supervision; they are not independent ground truth.

There are materially different implementations and recipes:

| Aspect | Current FAST wrapper | Older SLOW reproduction wrapper |
|---|---|---|
| Entry | `run.sh` or `diag_run.sh` → `run_fast_lss_dev.sh` → `train_experimental_branch3_mc.py` | `repro_develop3.sh` → `train_experimental_branch3.py` |
| Prior/depth | Render-derived CPF; depth loss defaults to zero in `run.sh`/`diag_run.sh` | Depth loss 0.3 → 0.003 by default |
| View batch | `run.sh`: Room 3, other scenes 1; `diag_run.sh`: 1 unless overridden | `--random_view_num 3` is explicit |
| CPF start / derivative start | 2,500 / 3,000 | 2,000 / 2,500 |
| CPF training/query frequency | Training every 8; derivative every 2 in the top-level fast recipe | Older schedule and update interval 500 |
| CPF pull / trained noise head | `CPF_PULL_WEIGHT=0`, `CPF_NOISE_TRAIN_WEIGHT=0` by default | Must establish the exact historical implementation/config for each run |
| Densification | Common shipped window 500–25,000, interval 200 | Older trainer/arguments; do not infer identical behavior from wrapper names |
| Anti-aliasing | Scene-dependent; indoor AA, outdoor generally off | Both AA and non-AA runs exist |
| Iteration selection | Multiple saved checkpoints and early selections | Mostly 30k historical outputs, plus later reproductions |

Sources: [run.sh](/data1/zhangao/repos/3dgs-imc-codex/run.sh), [diag_run.sh](/data1/zhangao/repos/3dgs-imc-codex/diag_run.sh), [repro_develop3.sh](/data1/zhangao/repos/3dgs-imc-codex/repro_develop3.sh), and the model/trainer files recorded in [source fingerprints](/data1/zhangao/repos/3dgs-imc-codex/writings/research-audit-2026-09-26/source_fingerprints.csv). Directly invoking `run_fast_lss_dev.sh` uses different fallback defaults from invoking it through `run.sh`; the entry path is part of the configuration.

Current `run.sh` uses scene-dependent capacity caps: Bicycle 5M, Stump 4.2M, Garden 4M, Treehill 3.3M, Flowers 3M, Kitchen 1.5M, Room 1.2M, Bonsai/Counter 1M, and a default for other scenes. Historical runs and TNT wrappers use other caps. A single global hyperparameter paragraph cannot describe the entire selected checkpoint pool.

**Repository, manuscript, and history authority.**

“`hust10`” in the CSVs identifies this local machine in the historical research naming; its current OS hostname is `user`. The remote alias `hust2` reports hostname `user-server`. Identical absolute paths on these machines refer to different files.

The local checkout is branch `develop3`, HEAD `02fd14e`, with pre-existing tracked modifications and many untracked experiment files. Both inspected remote GS-LSS copies have empty `.git` directories, so a remote branch/commit identity cannot currently be verified. The remote code also differs from local code; do not synchronize one over the other wholesale. File hashes preserve the inspected state.

The submitted TeX SHA-256 is recorded in `source_fingerprints.csv` (prefix `ed87eb2c5f30062e`). The local editing file has prefix `b55a2dec456ce6ef`; the remote revised file has prefix `9314b55ce99c801a`. They are different manuscripts. The supplied AGENTS paths point partly to older layouts that are absent from the live checkout.

I recovered the missing review comments, both revision plans, sample revised TeX, July 2 active TeX, issue map, and provenance report by reading this remote archive in place:

`hust2:/data1/zhangao/outputs/hust10_revision_backup_20260703/hust10_3dgs_imc_codex_revision_workspace_20260703.tar.gz`

It has 2,205 members and no checkpoint members. The accompanying metadata archive has 6,240 members and also no checkpoints. The recovered document texts, archive member names, and hashes are in [recovered review sources](/data1/zhangao/repos/3dgs-imc-codex/writings/research-audit-2026-09-26/evidence/recovered_review_sources.json). Its July 2 provenance report already warned that Table III and the MatrixCity component ablation were not cleared for submission, and recorded corrected sensitivity/trial values that differ from the submitted version. The sample revised TeX also contains explicit author reminders to replace old results if adopting a single-view claim; it is a proposal, not evidence that those replacement experiments happened.

The [history reconstruction](/data1/zhangao/repos/3dgs-imc-codex/writings/research-audit-2026-09-26/RESEARCH_HISTORY.md) covers 47 repository-associated Codex records, 20 Claude memory files, and the separate desktop repo-location chat. Deleted Claude transcripts and absent historical workspaces cannot be certified as read or recovered.

**Datasets and evaluation protocols.**

| Dataset | Submitted scenes | Main protocol | Current local data root |
|---|---|---|---|
| Mip-NeRF 360 | Bicycle, Bonsai, Counter, Garden, Kitchen, Room, Stump, Treehill | Coverage gaps 20/40/60/80/100°; discontinuous trajectory | `/data1/zhangao/DATA/mipnerf360` |
| Tanks & Temples | `truck_resize`, `train_resize` | Same gaps; discontinuous trajectory | `/data1/zhangao/DATA/tandt` |
| Mill 19 | `building-small`, `residence-small`, `rubble-small`, `sci-art-small` | Discontinuous trajectory | `/data1/zhangao/DATA/DATA-small` |
| MatrixCity | `blockA_fusion_small_aerial+somestreet` | Aerial train → ground test | `/data1/zhangao/DATA/large_angle` |
| Zeche | `Zeche1` | Aerial train → ground test | `/data1/zhangao/DATA/large_angle` |

That is 16 submitted scenes. Flowers, Zeche2, and the additional Deep Blending directories are not part of these main-table averages. Some exploratory experiments nevertheless use Flowers.

For gap evaluation, the saved `test_selection_angle_<angle>.txt` lists held-out image names. `discontinuous.txt` supplies the other fixed held-out selection. The loader removes those cameras from the training-camera list; the exact files, not a newly randomized split, define reproduction. Aerial runs instead use separate `train` and `test` sources. Mip runs commonly use `images_4`; TNT uses `images`; MatrixCity has a cached-image variant. The selected run's config must decide resolution and image directory.

[split_inventory.csv](/data1/zhangao/repos/3dgs-imc-codex/writings/research-audit-2026-09-26/split_inventory.csv) records 665 split-file paths across both hosts, with hashes and nonempty-line counts. These are file counts, not 665 independent benchmark protocols. Dataset image-file counts and existence checks are in the platform snapshots. Remote `/data1/zhangao/mipnerf360` has the full nine-scene tree, while remote `/data1/zhangao/DATA/mipnerf360` contains partial alternatives; its Kitchen `images_4` directory has only 109 files versus 279 in the complete tree. Do not remap paths merely because a directory exists.

The saved image metrics are SSIM, PSNR, and LPIPS, ordinarily from `metrics.py` and the relevant method's renderer. LPIPS uses the repo's VGG implementation. Main-table values are unweighted means over the designated scenes; they are not means over all images pooled across scenes or over training seeds. Optional BRISQUE/NIQE/PIQE entries may be zero when deliberately skipped; such zeros are not valid quality measurements.

There is an unresolved initialization issue. The submitted paper says the split precedes SfM initialization and all support uses training views only, at [line 519](/data1/zhangao/repos/3dgs-imc-codex/writings/GS-LSS-submission/GS-LSS.tex:519). The loader reads shared `sparse/0/points3D.*` before the selected-camera filtering, and the legacy floater exporter explicitly describes full-coverage SfM. I did not reconstruct COLMAP tracks or prove how every source cloud was produced. Consequently the stronger train-only initialization/support claim is **not established** by the inspected implementation and artifacts.

**Tables I and II: GS-LSS checkpoint recovery.**

All 39 submitted GS-LSS aggregate values match to three decimal places: 30 in Table I and nine in Table II. This requires 192 scene/metric selections: 50 gap scene-angle cells × three metrics, plus 14 discontinuous scene cells × three metrics. These selections use **131 distinct PLY paths**.

Of those 131 files, 117 are under local `/data1/zhangao/outputs`, and **14 are under the repo's `outputs/wsl_results/diag`**. Preserving only the central output directory would lose part of the selected evidence. The WSL files supply 23 of the 192 metric selections.

The selection rule is independently maximize SSIM, maximize PSNR, and minimize LPIPS per scene before averaging scenes. Therefore one printed SSIM/PSNR/LPIPS row can combine different models. The best-PSNR lookup retains each chosen model's actual full triple; it must not be described as reproducing all three independently optimized aggregate values.

The selected files span 9k, 11k, 12k, 15k, 18k, 22k, 24k, 25k, 27k, and 30k iterations. Ninety-two of the 131 unique files are 30k; the others are earlier checkpoints. The pool mixes slow and fast configurations, AA choices, capacities, and view batching. These results establish the reported aggregate arithmetic under that selection, not a single-recipe, fixed-30k, single-view or seed-average result.

GS-LSS source families are local `develop3_multi_angle*`, `develop3_discontinuous*`, central `diag`, and the repo-local WSL `diag` mirror. The exact path for every cell and metric is in the manifest; shortened “best” directory names are not a substitute.

**Tables I–III: all baseline models and coverage.**

In this table, “linked scenes” means an exact saved metric triple was linked to a structurally complete checkpoint for each scene condition. “Matching scalars” counts submitted aggregate metric values that round correctly. These are distinct checks: a stored baseline can exist and still disagree with the paper.

| Method | Table I linked scenes / required; matching scalars / 30 | Table II linked scenes / required; matching scalars / 9 | Table III |
|---|---|---|---|
| 3DGS | 10/50; 30/30 | 14/14; 9/9 | Matching sources not located for either scene |
| Scaffold-GS | 50/50; 30/30 | 14/14; 9/9 | Matching sources not located for either scene |
| Hierarchical-GS, depth-only | 50/50; 30/30 | 14/14; 9/9 | Both scenes linked; 6/6 scalars match |
| MVGS | 50/50; 30/30 | 14/14; 9/9 | Both scenes linked; 6/6 scalars match |
| PGSR | 50/50; 24/30 | 0/14; 9/9 | Matching sources not located for either scene |
| 3DGS-MCMC | 50/50; 30/30 | 9/14; 9/9 | Matching sources not located for either scene |
| MH-3DGS | 50/50; 7/30 | 14/14; 9/9 | Both scenes linked; 6/6 scalars match |
| GS-LSS | 50/50; 30/30 | 14/14; 9/9 | Alternative models exist; 0/6 submitted scalars match |

Table I's 40 unresolved 3DGS links are the eight Mip scenes at five angles. Their archived CSVs reproduce the paper, but I did not establish the matching original PLY/result association. This does not imply there are no 3DGS models: alternative saved runs are listed separately.

PGSR Table I differs at Mip gaps 20° and 40°. The archive for Garden 40° contains a 7k result, so even this baseline pool is not uniformly 30k. MH-3DGS Table I has 23 scalar discrepancies, already visible in the older verifier. These require tracing the intended result version, not selecting whichever new baseline value favors a conclusion.

All Table II rows match the archived arithmetic. PGSR's original metric triples could not be linked to current model/results pairs. MCMC lacks exact original links for Bicycle, Stump, Treehill, Truck, and Train. Candidate checkpoints with different saved metrics are not treated as reproduction of those rows.

Baseline checkpoint families use the same root names on the two machines:

| Submitted method | Directory family | Repository / essential distinction |
|---|---|---|
| 3DGS | `3dgs-original*` | Local `gaussian-splatting`; remote `gaussian_splatting` |
| Scaffold-GS | `scaffoldgs-original*` | `Scaffold-GS`; native MLP sidecars required |
| Hierarchical-GS | `3dgs-depth*` | Depth-regularized configuration; the paper explicitly enables only depth regularization |
| MVGS | `mvgs-original-mv3*` | `MVGS`; preserve the multi-view setting |
| PGSR | `pgsr2-original*` for Mip, `pgsr-original*` for TNT/Mill | Remote `/data1/ZHANGAo_2024/PGSR`; renderer/environment differs |
| 3DGS-MCMC | `3dgs-mcmc-original*` | Official `3dgs-mcmc`; preserve initialization and capacity |
| MH-3DGS | `mh-3dgs-original*` | `MH-3DGS` |

Roots are `/data1/zhangao/outputs/baselines_multi_angle/<scene>/`, `baselines_discontinuous/<scene>/`, and `baselines_aerial_ground/<scene>/`. The manifest identifies the actual host and complete leaf path. Older CSVs often reference `/data2/…` or Windows/WSL locations; identical metric triples were required before linking a relocated copy. CDGS directories also exist but must not be silently relabeled as Hierarchical-GS or MH-3DGS.

All 64 linked Scaffold main-table scene conditions have their opacity, covariance, and color MLP files present; see [sidecar audit](/data1/zhangao/repos/3dgs-imc-codex/writings/research-audit-2026-09-26/scaffold_sidecar_audit.csv). Their existence was checked, but the TorchScript networks were not loaded.

**Table III: the submitted GS-LSS aerial results are not recovered.**

The following are **best-PSNR located runs**, not a replacement table and not independently selected best triples. Both train on the saved aerial source and evaluate the separate ground source, using stored `metrics.py` results.

| Dataset/scene | Submitted GS-LSS SSIM / PSNR / LPIPS | Located best-PSNR run's actual SSIM / PSNR / LPIPS | Checkpoint |
|---|---|---|---|
| MatrixCity Block A | 0.594 / 16.951 / 0.462 | 0.532929 / 14.563578 / 0.490249 | Local `diag/large_angle/blockA_fusion_small_aerial+somestreet_traintest_MX_cached_woaa_cap3M_i30000`, iteration 30,000 |
| Zeche1 | 0.481 / 12.879 / 0.530 | 0.448119 / 11.931662 / 0.575130 | Local `diag/large_angle/Zeche1_traintest_0Z_zeche_partial1e2_i30000`, iteration 10,000 |

These run paths are relative to `/data1/zhangao/outputs`; each PLY is under `point_cloud/iteration_<n>/point_cloud.ply`. The MatrixCity candidate uses cached inputs, no AA and cap 3M; the Zeche candidate is the saved partial-scaling probe. Their configurations and exact source/test paths are in the main manifest. A separate Zeche 8k checkpoint has better saved SSIM/LPIPS, so those metrics must not be combined with the 10k PSNR and presented as one run.

Neither inspected remote live storage nor the aerial metadata recovered from the July backup resolves the submitted GS-LSS numbers. The July 2 provenance report already documented essentially this gap. This is a failure to locate the claimed source, not proof that the method cannot reach the paper values or that no copy exists elsewhere.

**Tables IV–VIII need distinct treatments.**

Table IV, FFR: all baseline entries and four GS-LSS angle entries match saved **Bonsai-only** fraction files. The submitted GS-LSS 40° value is 2.223%, whereas the saved `bonsai__Ours.frac.json` gives **7.2226107%**. Its five-angle mean is **3.4443991%**, whereas the submitted mean is 2.445%. Thus 46 of 48 table entries match the saved fraction arithmetic. No alternative source for the discrepant entry was established.

The fraction source is local `/data1/zhangao/outputs/viz_multiangle/global_all/bonsai__Ours.frac.json`, also present remotely. Its historical visualization manifest associates Bonsai 40° with `diag/mipnerf360/bonsai_angle_40.0_0_bonsai40_fast_cap2M_30k_i30000`, iteration 30k. This is an artifact association, not a fresh recomputation or embedded checkpoint hash. [special_table_sources.csv](/data1/zhangao/repos/3dgs-imc-codex/writings/research-audit-2026-09-26/special_table_sources.csv) distinguishes manifest-backed associations from inferred exporter-layout candidates.

The exporter uses a threshold of 1.5% of a robust SfM diagonal. It computes the fraction over Gaussian centers before BEV histogram cropping; it does not implement the claimed gap-only denominator. Its reference is described as full-coverage SfM. Scaffold uses **count-weighted neural Gaussian centers**, unlike the opacity weighting of the other rows. The submitted metric definition at [line 721](/data1/zhangao/repos/3dgs-imc-codex/writings/GS-LSS-submission/GS-LSS.tex:721) therefore needs reconciliation with [floaters_export.py](/data1/zhangao/repos/3dgs-imc-codex/experiments/floaters_export.py) and [scaffold_bev_export.py](/data1/zhangao/repos/3dgs-imc-codex/experiments/scaffold_bev_export.py). These artifacts support a bounded case study, not a multi-scene geometry benchmark.

Table V, MatrixCity LSS components: no matching stored full metric triple was located for any of the four submitted variants. The reported PSNRs 16.225, 16.544, 16.599 and 16.693 are **paper-text values** in this audit, not recovered experimental results. Later Bonsai/Garden component models exist, but cannot be substituted as the source of a MatrixCity table. The recovered July 2 audit had the same unresolved finding.

Table VI, CPF sensitivity: the table mixes sources and contains setting discrepancies.

| Submitted row | Located arithmetic source | Status |
|---|---|---|
| ε=0.1 | `metrics_ablation/develop3_eps/*_0.1_metrics.csv` | Matching mean; original checkpoint link not established |
| ε=0.25 | Local `tnt60_table5/*_eps0.25_sine_aa1_i30000` | Matching mean and two 30k models |
| ε=0.5 | `develop3_eps_partial1e0/*_0.25_metrics.csv` | Matching numbers come from files labeled ε=0.25; setting unresolved |
| ε=0.75 | Local `tnt60_table5/*_eps1.0_sine_aa1_i30000` | Matching numbers come from runs labeled ε=1.0; setting unresolved |
| ε=1.0 | `metrics_ablation/develop3_eps/*_1.0_metrics.csv` | Matching mean; original checkpoint link not established |
| MLP | Local `tnt60_table5/*_mlp_eps0.25_aa1_i30000` | Matching mean and two 30k models |
| SIREN-style | Submitted PSNR 19.4228 | No matching triple; retained ε=0.25 SIREN pair gives 19.4428 |

Each pair is Train+Truck under the saved 60° gap, and each historical CSV/model source is listed in the special-table source file. Exact arithmetic coincidence does not prove that a mislabeled setting was actually run. Twelve complete local `tnt60_table5` models are available for a consistent sensitivity analysis, but that analysis differs from the submitted table. [ablation_candidate_means.csv](/data1/zhangao/repos/3dgs-imc-codex/writings/research-audit-2026-09-26/ablation_candidate_means.csv) preserves all 40 inspected paired candidate groups rather than hiding alternatives.

Table VII, five trials: Trials 1–3 match the archived `trials2/develop3_{1,2,3}` Train+Truck means, with no exact checkpoint linkage established. Trials 4–5 do not match the available corresponding bundles. The printed mean and sample variance are arithmetically consistent with the printed rows, but that is not evidence of five independently seeded runs with identical configuration. The later Truck-discontinuous repetitions are a different protocol and cannot fill this gap.

Table VIII, timing: every printed ratio can be reconstructed from the retained timing files, but the experiment described in the caption is not the experiment timed.

| Method | Saved seconds | Source |
|---|---:|---|
| 3DGS | 840 | `hust2:baselines_t7_clean/3dgs/wall_seconds.txt` |
| Scaffold-GS | 1,081 | `hust2:baselines_t7_clean/scaffold/wall_seconds.txt` |
| Hierarchical-GS | 881 | `hust2:baselines_t7_clean/hier/wall_seconds.txt` |
| 3DGS-MCMC | 1,042 | `hust2:baselines_t7_clean/mcmc/wall_seconds.txt` |
| MH-3DGS | 833 | `hust2:baselines_t7_clean/mh/wall_seconds.txt` |
| PGSR | 2,241 | `hust2:baselines_t7_clean/pgsr/wall_seconds.txt` |
| MVGS | 4,214 | `hust2:baselines_t7_clean/mvgs/wall_seconds.txt` |
| GS-LSS FAST | 2,325.79 | Local `ours_timing_diag/tandt/truck_resize_discontinuous_clean_shipped_i30000/training_time.txt` |

All paths in this timing table are relative to `/data1/zhangao/outputs`. All runs are Truck, 30k iterations, single saved timings rather than scene/seed averages. Baselines used `hust2` RTX 4090 GPU2, native `--eval`, and an external wall clock around `conda run`; saved `test_selection_path` is empty. GS-LSS used local RTX 5090 GPU1, `discontinuous.txt`, the FAST recipe, `RUN_EVAL=0`, and its internal training timer. Baseline timing includes startup/loading and whatever evaluation/saving the trainer performs; the GS-LSS timer excludes setup and later standalone rendering/metrics. The quality columns come from the separate Train+Truck 60° main-table results. Dividing these times by 840 does not make hardware, split, configuration, or timer scope equal.

The full records, environment names, scripts, model paths, and inclusion/exclusion notes are in `timing_provenance.csv`. These times may be retained as clearly bounded historical measurements. They do not support the submitted statement that the quality-cost comparison uses the same 60° protocol on one RTX 4090.

**Implementation claims that must be checked before further revision.**

| Submitted claim / location | Inspected implementation or evidence | Consequence |
|---|---|---|
| Learned gate head `h_theta`, lines 387–397 and 536 | `NetworksA` has a per-axis log-scale branch; the stated separate learned gate is absent | Rewrite from the selected implementation; do not copy proposed equations from a sample manuscript |
| Low CPF support causes larger noise, line 300 | `sigma-detach` multiplies noise by detached CPF score; observation-count gating is separate | The stated direction is not faithful to this noise mode |
| CPF updates positions and shapes, lines 304/372/380 | The inspected CPF pull/noise application acts on XYZ; calculating rotation/scale derivatives is not evidence they are applied | Narrow to the actual updated parameters unless another verified implementation is selected |
| Trained deterministic drift and stochastic head characterize all reported models | Current fast defaults disable both pull and noise-head training | Establish exact recipe provenance before attributing gains to those mechanisms |
| One photometric view per iteration, line 535 | Slow reproduction explicitly uses three; Room fast also uses three | Removing a batch paragraph does not change the protocol of existing checkpoints |
| Same initialization, iteration budget, official defaults, lines 529–531 | Selected early iterations, varying caps, depth-only baseline configuration, and tuned GS-LSS pool; timing MCMC records random initialization while linked main-table MCMC runs record SfM | Replace blanket fairness language with the actual comparison definition |
| All runs use RTX 4090, line 537 | Historical selected models and timing span local/WSL RTX 5090 and remote RTX 4090 | Hardware description must be table/run specific |
| Train-only SfM and gap-only FFR, lines 519/721 | Shared SfM initialization and full-coverage legacy exporter | Requires source reconstruction/verification or a narrower description |

Additional details matter when turning equations into experiments. `sample_xyz` uses a transformed uniform local offset, not the normal sample written in parts of the paper. CPF loss terms have implementation weights and optional color supervision; an unweighted symbolic loss is not a complete executable specification. The final `net.pth` is stored at run root rather than versioned by every Gaussian iteration. Pairing an early PLY with that final network is not automatically a valid CPF-reliability measurement.

For image inference, GS-LSS uses the optimized Gaussian representation. For exact training continuation or CPF diagnostics, a PLY alone is insufficient: the inspected `capture()` does not preserve all CPF optimizer/bounds, observation state, and RNG information. Do not promise bitwise training resumption from a PLY plus a final `net.pth`.

**Existing experiments worth reusing.**

There are 141 saved result rows in the reusable-experiment index, with integrity status on each. These are candidates for answering reviewer questions, not automatically publication-ready comparisons.

| Need | Useful surviving family | Conditions before reuse |
|---|---|---|
| Consistent TNT sensitivity | Local `tnt60_table5` | Twelve runs, Train+Truck 60°, six settings; use actual labels and the slow recipe |
| Component isolation | `hust2:component_ablation_gpu2_20260704_bonsai_normal100_matched_sfmanchor_i20k_run1` and `…matched_plusfull…` | Five variants plus full; compare saved launch settings and equal budgets |
| Additional component variants | Remote `component_ablation_gpu2_20260705_bonsai_normal100_matched_{covonly,no_normal}_sfmanchor_i20k_run1` | Names alone do not prove the implemented component change |
| Later full-model candidate | Remote `table4_bonsai60_full_color01_normal100_fromscratch_20260705` | 20k/21k/22k models; avoid per-metric cherry-picking for a causal ablation |
| Split-location sensitivity | Remote `split_center_robustness_20260703*`, `split_center_robustness_garden_gpu2_20260704` | Fixed 60° gap with multiple centers; some extra/retry rows exist |
| Matched split baselines | Remote `split_center_baselines_{bonsai,garden}_3dgs_gpu2_20260704*` | Four centers each plus a Garden retry; match exact split hashes and reference definition |
| Broader component history | Remote `table4_ablation_20260703_probe`, `table4_ablation_bonsai_garden_20260703` | Earlier Bonsai/Flowers/Garden studies; not interchangeable with later matched runs |
| Hyperparameter diagnostics | Remote `hyperparam_sensitivity_bonsai_gap60_gpu2*20260704` | Six saved rows; bounded single-scene evidence |
| Inference speed/memory | Remote `efficiency_render_fps_memory_20260704` | Small diagnostic artifact family; identify timed model and rendering settings |
| Discontinuous backup | Remote `hust10_best_checkpoints_20260703/discontinuous_selected` | Fourteen saved scene models; a selected import, not a complete backup of all per-metric winners |

Paths in that table are relative to `/data1/zhangao/outputs`. Detailed image metrics are left in the CSV with their paths and settings rather than quoted without full provenance. The history reports mixed Garden image quality and checkpoint-sensitive Truck geometry behavior; those limitations should remain visible if the experiments are used.

**Storage integrity and operational readiness.**

The inventory contains 5,167 local and 2,133 remote artifact paths: 5,322 PLY files, 1,102 `.pt` files, and 876 `.pth` files. It includes `input.ply`, replicas, neural modules, and multiple iterations; it is not a count of independent trained models. Of the PLY files, 5,317 have sizes consistent with their binary headers and five do not. No full binary hashing, tensor deserialization, finite-value inspection, or fresh rendering was done.

The five truncated files are listed in [damaged_or_unverified_ply.csv](/data1/zhangao/repos/3dgs-imc-codex/writings/research-audit-2026-09-26/damaged_or_unverified_ply.csv): local Zeche1 CDGS 10k; local `t7_bonsai/3dgs` 7k and 30k; remote Bicycle 30k in `hust10_discontinuous_import_selected_20260703`; and an input cloud for Building in that same partial import. Four are model PLYs and one is an initialization input. None is part of the 131 selected GS-LSS Tables I/II files. Prefer the original local models or the complete `hust10_best_checkpoints_20260703` import where applicable. Nothing was deleted or repaired.

The working environments still exist. Local `/home/zhangao/.conda/envs/3dgs-imc` has Torch 2.7.1+cu128 and RAPIDS/cuML 25.12 metadata; remote `/data1/ZHANGAo_2024/.conda/envs/3dgs-imc` has Torch 2.4.0+cu118 and mixed RAPIDS package metadata. Native baseline environments also survive remotely (`gaussian_splatting`, `mcmc`, `mh`, `scaffold`, `mvgs`, `pgsr`). This is an installed-package inventory, not a CUDA/import compatibility test. Historical package issues are a reason to reuse a working environment and test narrowly, not to upgrade dependencies preemptively.

Current GPUs are two RTX 5090s locally and four RTX 4090s remotely. The user-authorized policy for future work is **one GPU only: local GPU1 for GS-LSS; remote GPU3 for baselines, local GPU1 fallback**. Historical GPU0/GPU2 launchers must be reviewed before reuse. Some old timing/sweep scripts contain output-directory removal commands; they should not be launched unchanged against retained runs.

Several command pitfalls are already known: slow reproduction reads `ITER` and `AA`, while the fast path uses `ITERATIONS` and `ANTIALIASING`; `diag_run.sh` can pass a CPF view cap into the slow path; a stale WSL source path in `cfg_args` needs an explicit verified remap; rendering AA is not reliably recoverable from `cfg_args`; and `diag_run.sh` can return failure after a successful `RUN_EVAL=0` timing run because its final guard expects a metric CSV. Read the actual artifacts, not only an exit code or directory name.

**How I will approach the next experiment or revision.**

First select the exact scientific claim and source model. The following read-only lookup prints its audited provenance:

```bash
python3 /data1/zhangao/repos/3dgs-imc-codex/writings/research-audit-2026-09-26/scripts/lookup_checkpoint.py \
  --table I --method GS-LSS --scene bonsai --angle 60 --metric PSNR
```

Use `--metric SSIM` or `LPIPS` only when intentionally examining the paper's independent metric selection. Table III lookups explicitly label unmatched candidates. For a qualitative figure or new ablation anchor, choose one model and retain its whole metric triple.

Before a run, record machine/GPU, exact command, environment, source-file hashes and Git state, scene/data root, split hash, seed, image resolution, initialization, all CPF/noise/batch/cap/AA settings, target iterations, source checkpoint and new output directory. Check that the new output directory is absent. Keep rendering/metrics in a fresh evaluation directory so the archival metric files remain unchanged. Compare a stored checkpoint's render first if reproduction is the objective; retraining is a separate task.

For revision, the lowest-cost order is: fix implementation descriptions and unsupported blanket claims; reconcile FFR source/definition and sensitivity/trial labels; use surviving matched component models where their settings actually answer the concern; then decide whether the unresolved aerial/component sources require recovery, removal/narrowing, or a specifically bounded new experiment. A missing provenance link should not automatically trigger a broad baseline rerun.

Deleting random-batch descriptions can remove an unsupported contribution claim, but it cannot honestly establish single-view fairness for retained three-view models. Similarly, a later favorable Bonsai ablation cannot validate a missing MatrixCity result. If a conclusion depends on those claims, the underlying evidence must be recovered or the conclusion changed.

**Checks completed and remaining limits.**

The audit scripts parsed the submitted TeX rather than relying on hardcoded old paper targets, recomputed aggregates, linked exact result iterations to PLYs, checked Scaffold sidecar presence, and preserved hashes of source/split files. The existing `paper_tables_update.py`, `verify_paper_tables.py`, and `verify_paper_tables2.py` were run as secondary evidence. Their outputs are saved under `evidence/`; they target older numbers and are not by themselves a submitted-paper verification.

Two legacy implementation details make the new checks necessary: some old selectors accepted a run if **any** PLY existed rather than the PLY for the metric's iteration, and `paper_tables_update.py` prints Table I SSIM/PSNR in a swapped display order in one path. The new manifest requires the exact iteration and parses metric names explicitly. No conflicting `results.json`/`metrics.csv` triples were found among the indexed run records.

The prepared state is therefore: the research trajectory is reconstructed; the local/remote responsibilities and environments are mapped; all submitted table positions have a documented status; Tables I/II GS-LSS model selections are located; later diagnostic checkpoints are indexed; and the remaining gaps are explicit. Fresh rendering, exact historical retraining, full CPF-state recovery, split-before-SfM provenance, and universal fairness claims remain unverified. No manuscript compilation was needed because the manuscript was not edited.

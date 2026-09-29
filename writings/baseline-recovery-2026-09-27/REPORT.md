# Follow-up: the 67 unresolved baseline source groups

27 September 2026. This follows the checkpoint consolidation and answers why 67 original baseline sources were not linked, where surviving candidates are, and what can currently be located on `hust2`.

**The earlier wording “67 checkpoints not found” was too broad. It meant 67 scene/method/protocol groups whose original checkpoint identity was not established. Five of those groups have existing local MCMC models; they failed an exact stored-metric linkage check. For 54 groups, the recorded run paths are absent from the accessible search. For eight aerial groups, neither an original path nor a matching saved metric/checkpoint source has been established. No additional exact paper-source checkpoint was recovered in this follow-up.**

The complete [67-row location/status index](/data1/zhangao/repos/3dgs-imc-codex/writings/baseline-recovery-2026-09-27/baseline_recovery_groups.csv) and [five MCMC comparisons](/data1/zhangao/repos/3dgs-imc-codex/writings/baseline-recovery-2026-09-27/mcmc_candidate_details.csv) are also installed in the [central library](/data1/zhangao/outputs/GS-LSS-paper-checkpoints/README.md). Its lookup now includes `source_recovery_followup` for these records without promoting a candidate to a verified paper source.

| Submitted group | Count | Last recorded source | Current finding |
| --- | ---: | --- | --- |
| Table I, 3DGS, eight Mip-NeRF 360 scenes × five gap angles | 40 | `/data2/zhangao/outputs/baselines_multi_angle/<scene>/3dgs-original-test_selection_angle_<angle>` | Original run folders and matching models unlocated. Later Bonsai/Garden split-center alternatives survive, but do not establish the original eight-scene sweep. |
| Table II, PGSR, all scenes | 14 | `/data2/zhangao/outputs/baselines_continuous/<scene>/pgsr[2]-original-<cap>` | Original run folders and matching models unlocated. The old PGSR launcher confirms this folder name, despite the discontinuous evaluation protocol. |
| Table II, 3DGS-MCMC, Bicycle/Stump/Treehill/Truck/Train | 5 | Three old `/data2` paths, one local `/data1` path, one WSL path | Same-scene/protocol 30k models exist locally at the paths below. Their stored metric triples differ from the paper's source CSVs. |
| Table III, 3DGS / Scaffold-GS / PGSR / 3DGS-MCMC × MatrixCity and Zeche | 8 | No original run path established | No corresponding original model/metric source located in reachable storage, indexed history, or relevant archives. |

The original CSVs contain filesystem paths, not host identities. The user suggested `hust2`, and its surviving scripts support that as a relevant historical location. A `/data2/...` string alone does not prove which machine owned a particular file.

**The five existing MCMC checkpoints.**

These are on the main research machine (`hust10` in historical records, hostname `user`). Each is iteration 30,000, under `/data1/zhangao/outputs/baselines_discontinuous`. All five were retained during cleanup and were already included as alternatives in the consolidation library.

| Dataset / scene | Existing checkpoint |
| --- | --- |
| Mip-NeRF 360 / Bicycle | [bicycle/3dgs-mcmc-original-5000000/point_cloud/iteration_30000/point_cloud.ply](/data1/zhangao/outputs/baselines_discontinuous/bicycle/3dgs-mcmc-original-5000000/point_cloud/iteration_30000/point_cloud.ply) |
| Mip-NeRF 360 / Stump | [stump/3dgs-mcmc-original-4200000/point_cloud/iteration_30000/point_cloud.ply](/data1/zhangao/outputs/baselines_discontinuous/stump/3dgs-mcmc-original-4200000/point_cloud/iteration_30000/point_cloud.ply) |
| Mip-NeRF 360 / Treehill | [treehill/3dgs-mcmc-original-3300000/point_cloud/iteration_30000/point_cloud.ply](/data1/zhangao/outputs/baselines_discontinuous/treehill/3dgs-mcmc-original-3300000/point_cloud/iteration_30000/point_cloud.ply) |
| Tanks & Temples / Truck | [truck_resize/3dgs-mcmc-original-2000000/point_cloud/iteration_30000/point_cloud.ply](/data1/zhangao/outputs/baselines_discontinuous/truck_resize/3dgs-mcmc-original-2000000/point_cloud/iteration_30000/point_cloud.ply) |
| Tanks & Temples / Train | [train_resize/3dgs-mcmc-original-1000000/point_cloud/iteration_30000/point_cloud.ply](/data1/zhangao/outputs/baselines_discontinuous/train_resize/3dgs-mcmc-original-1000000/point_cloud/iteration_30000/point_cloud.ply) |

The old audit linked an exported result only when SSIM, PSNR, and LPIPS all agreed with a located run within `1e-6`; see [build_catalog.py:106](/data1/zhangao/repos/3dgs-imc-codex/writings/research-audit-2026-09-26/scripts/build_catalog.py:106). None of these five triples passes that rule. For Truck, even the current path equals the archived path, so this is plainly a result-identity discrepancy rather than proof of a missing file. The Mip originals also have a trailing underscore absent from the located folder names; Train's original export names a WSL location.

The comparison CSV preserves the original and current triples, their differences, original metric export, current `results.json`, dataset/split paths, cap and initialization fields, resolution, model iteration, and full SHA-256 for the current PLY. These are individual stored-run results, not seed means or fresh evaluations. The mismatch could reflect evaluation changes, a different run, or another provenance issue; the available records do not establish which. The current models were not relabeled as the exact paper originals.

**What the deeper `hust2` search established.**

- `/data2/zhangao` is accessible, but `/data2/zhangao/outputs` is absent. The exact recorded paths and likely `/data1/zhangao/outputs` relocations were checked directly: 130 path variants, zero matching run directories or 30k files on `hust2`.
- The search covered `/data1/zhangao`, `/data2/zhangao`, and the older `/data1/ZHANGAo_2024` workspace. It visited 87,754 directories while excluding image/render trees, dependency caches, Git internals, and similar non-model content. The additional checkpoint trees found outside the prior output inventory belong to super-resolution or surface-reconstruction work, with different datasets/protocols; they were not substituted for these paper sources.
- Old `3dgs-original/outputs`, `3dgs/outputs`, and `Scaffold-GS/outputs` under `/data1/ZHANGAo_2024` contained no located checkpoint/provenance files. Nested `cdgs-release/data2/.../baselines_*` directory names did not contain the missing models. A separate search for relevant `.pth`, `.pt`, and `.ckpt` states found no additional state files outside the previously inspected output tree.
- All five relevant backup archives were inspected by member name without extracting or modifying them: the three `server9_backup/*.zip` archives and two `hust10_revision_backup_20260703/*.tar.gz` archives. None contains `.ply`, `.pth`, `.pt`, or `.ckpt` members. The latter backups preserve workspace/metadata, not model binaries.
- Forty-seven indexed research-history records and relevant shell-history lines were searched for original paths, moves, transfers, and cleanup. No attributable deletion or move record establishing the fate of these specific original baseline files was found. A shell-history removal of `sweep_models/` exists, but its relative-path context and lack of per-file evidence do not establish that it removed any of these originals.

The surviving `hust2:/data1/ZHANGAo_2024/PGSR/pure_train.sh` is useful path evidence: line 201 sets `/data2/zhangao/outputs/baselines_continuous`, and line 247 uses `<scene>/discontinuous.txt`. Thus `baselines_continuous` is an actual historical output name for this discontinuous protocol, not a spelling correction to apply silently. The Mip PGSR2 variant also appears in the retained launchers and metric exports. Their contents and hashes are preserved in [hust2_historical_launchers.json](/data1/zhangao/repos/3dgs-imc-codex/writings/baseline-recovery-2026-09-27/evidence/hust2_historical_launchers.json).

**Other locations and the locked disk.**

The WSL search rechecked its home outputs/research artifacts/models and both Windows research-repository roots. It found the previously retained legacy models plus unrelated smoke/surface runs, but no relocated original baseline sweep. Historical notes refer to WSL `sweep_models/mipnerf360/...` models used in visualization; that tree is absent from the inspected live locations, and the note itself is not proof of a match to the original metric exports.

The user reports that `hust10:/data2` is locked and inaccessible. No attempt was made to unlock it or continue searching that storage. The small accessible `data2_mirror` directory on the main machine contains other surface-reconstruction experiments, not the missing GS-LSS baseline trees. No claim is made about what remains on the locked disk or an offline backup.

**Current conclusion and practical next step.**

Five usable MCMC model files have explicit current locations and hashes. Their paper-metric identity remains unresolved. Fifty-four archived run-folder sources and eight aerial original sources remain unlocated; this is not proof of deletion. The most concrete recovery target is an original or backup copy of `/data2/zhangao/outputs/baselines_multi_angle` and `/data2/zhangao/outputs/baselines_continuous`. The locked `hust10` disk could only be assessed if it becomes accessible through the user's normal storage administration.

For the five MCMC cases, the next scientific step would be a separately recorded evaluation/provenance comparison of the retained model, split, renderer, and historical metric implementation. That could resolve a linkage discrepancy; retraining would not recover the identity of the original run. No GPU experiments, checkpoint deletions, or checkpoint transfers were performed during this follow-up. The frozen audit and its original matching rule remain unchanged.

Evidence files, scripts, search scope, the revised 67-row classification, and validation are retained in this directory. The central library and current research guides link to this follow-up so future work will not interpret the old count as 67 proven-lost files.

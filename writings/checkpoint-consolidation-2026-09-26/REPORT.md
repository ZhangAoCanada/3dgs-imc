# Checkpoint consolidation and storage cleanup

Completed on 27 September 2026, following work begun on 26 September. Execution host: the main research machine, historically `hust10`, hostname `user`. Remote sources: `hust2` and `wsl`.

**The available selected models are now consolidated under `/data1/zhangao/outputs`, with a table-oriented catalog, saved split definitions, and verified imports. Cleanup reclaimed 284.4 GiB; after importing 116.1 GiB, filesystem free space increased by 167.9 GiB to 410.9 GiB. All 131 distinct GS-LSS checkpoints supporting the saved Tables I/II selections are present. The original checkpoints for every submitted result could not be recovered.**

The entry point is [GS-LSS-paper-checkpoints/README.md](/data1/zhangao/outputs/GS-LSS-paper-checkpoints/README.md). The [reproduction guide](/data1/zhangao/outputs/GS-LSS-paper-checkpoints/REPRODUCE.md) explains how to locate a model and prepare a separate evaluation. The original [research report](/data1/zhangao/repos/3dgs-imc-codex/writings/research-audit-2026-09-26/REPORT.md) and [research history](/data1/zhangao/repos/3dgs-imc-codex/writings/research-audit-2026-09-26/RESEARCH_HISTORY.md) remain unchanged.

**Storage changes.**

| Measurement | Result |
| --- | ---: |
| Deleted obsolete optimizer snapshots | 219 files |
| Allocated storage reclaimed | 284.421 GiB |
| Imported file content | 116.137 GiB |
| Central library, including indexes and reference files | 116.199 GiB |
| Filesystem space available before / after | 242.912 / 410.852 GiB |
| Net increase in available space | 167.940 GiB |
| Filesystem utilization before / after | 97% / 95% |
| Complete output tree after consolidation | 1,271.856 GiB |

Measurements use GiB = 2³⁰ bytes. `df` covers the shared filesystem, so concurrent activity and filesystem allocation make its change differ slightly from deleted allocation minus imported file sizes. See [storage_summary.json](/data1/zhangao/repos/3dgs-imc-codex/writings/checkpoint-consolidation-2026-09-26/storage_summary.json).

Cleanup removed only `chkpnt3000_mv1.pth`, `chkpnt3000_mv2.pth`, and `chkpnt3000_mv3.pth` files from the local `baselines_multi_angle` and `baselines_discontinuous` trees. Every deletion had an intact, evaluated `point_cloud/iteration_30000/point_cloud.ply`, saved 30k metrics, and a config. These early optimizer snapshots were outside the protected source set and are unnecessary for rendering the retained final MVGS model. Exact 3k optimizer resumption from those deleted files is no longer available.

The preflight compared file identity, size, modification time, and a recorded prefix hash, checked each retained PLY's binary header and length, and wrote deletion intent/completion records. No early PLYs, selected paper models, final rendering models, configs, metrics, logs, renders, or failed-run evidence were removed. Remote sources were read-only. The deletion list is [cleanup_plan.csv](/data1/zhangao/repos/3dgs-imc-codex/writings/checkpoint-consolidation-2026-09-26/cleanup_plan.csv); the action log is [cleanup_actions.jsonl](/data1/zhangao/repos/3dgs-imc-codex/writings/checkpoint-consolidation-2026-09-26/cleanup_actions.jsonl).

**What was consolidated.**

| Source | Imported files | Content | Scope |
| --- | ---: | ---: | --- |
| `hust2` | 3,156 | 107.468 GiB | Selected baselines, timing models, later revision experiments, alternative source candidates, saved configs/metrics, and neural sidecars |
| WSL | 57 | 1.116 GiB | Seven legacy/diagnostic PLYs and available metadata from eight selected run directories |
| Main repository's historical output trees | 188 | 7.552 GiB | Previously harvested WSL GS-LSS sources and local MH-3DGS verification models |
| **Total** | **3,401** | **116.137 GiB** | **Every imported file fully SHA-256 matched against its source** |

PGSR's 68 appearance-network files under `app_model/iteration_N` were copied in addition to the main run/point-cloud selection. The main-table Scaffold models retain their opacity, covariance, color, and appearance network files. Copies also preserve available run-root configs, metrics, camera records, logs, and iteration-matched checkpoint folders. Bulk render directories were not duplicated; existing source renders were retained.

The `hust2` public SSH route stalled during transfer. Copying resumed over the verified LAN connection using the existing server host key and completed successfully. The failed attempt and successful resumed transfer logs are retained. There were no source deletions or model modifications.

Existing models already under local `/data1/zhangao/outputs` remain in place. The library links to them; remote and repository-output imports live under `sources/<original-host>/<original-path>`. The library alone is therefore not a self-contained backup of every local model. Preserve its external link targets when moving or backing it up.

The selected catalog has 710 run-location records and 747 model-path/iteration records. Of those, 739 PLY files pass header/length checks; eight previously missing paths from seven older auxiliary run locations remain explicitly missing. Multiple records can be different iterations or copies of the same experiment. These counts are not independent training runs or random seeds.

**WSL recovery.**

The suggested `/mnt/c/Users/Aooooo/Documents/wsl_repos/outputs` exists, but contains metric exports and renders, not the former `diag` checkpoint tree. The inspected live WSL storage recovered legacy models under:

- `/mnt/c/Users/Aooooo/Documents/wsl_repos/3dgs-imc/output`;
- `/mnt/c/Users/Aooooo/Documents/wsl_repos/MH-3DGS/output`;
- `/home/ooooo/outputs`.

The legacy aerial models and short diagnostic model are preserved as candidates; no saved-result evidence establishes them as the missing submitted table originals. Other models under the separate Windows `Documents/repos` super-resolution experiments used different image-resolution/split settings and were excluded from this GS-LSS paper selection. Only mounted, reachable storage could be inspected; this does not rule out a copy on an offline drive. Fresh WSL inventories are saved in this package's `evidence/fresh_wsl_*.json.gz`.

**Main-table coverage and remaining gaps.**

| Table | Scene/metric positions | With a local model | Missing original model positions |
| --- | ---: | ---: | ---: |
| I: coverage gaps | 1,200 | 1,080 | 120 |
| II: discontinuous views | 336 | 279 | 57 |
| III: aerial-to-ground | 48 | 24 | 24 |
| **Total** | **1,584** | **1,383** | **201** |

Those 1,383 linked positions use 530 distinct local checkpoint paths. Of the linked positions, 1,223 belong to stored aggregates that round to the submitted value; 160 inherit an aggregate mismatch or candidate status. This is saved-number and file-link verification, not fresh rendering.

The 201 missing positions represent 67 scene/method/protocol groups:

| Missing original source | Groups |
| --- | ---: |
| Table I, 3DGS, eight Mip-NeRF 360 scenes × five gap angles | 40 |
| Table II, PGSR, all 14 scenes | 14 |
| Table II, 3DGS-MCMC: Bicycle, Stump, Treehill, Train, Truck | 5 |
| Table III, 3DGS / Scaffold-GS / PGSR / 3DGS-MCMC × MatrixCity and Zeche | 8 |

See [unresolved_main_groups.csv](/data1/zhangao/outputs/GS-LSS-paper-checkpoints/indexes/unresolved_main_groups.csv) for the exact original paths. Available alternative records are separately localized in [unresolved_checkpoint_candidates.csv](/data1/zhangao/outputs/GS-LSS-paper-checkpoints/indexes/unresolved_checkpoint_candidates.csv); they were not silently substituted for originals.

**GS-LSS Tables I/II:** all 131 required PLYs are present, including 14 moved into the central output tree from the repository's historical WSL imports. The [64 best-PSNR selections](/data1/zhangao/outputs/GS-LSS-paper-checkpoints/indexes/ours_best_psnr_checkpoints.csv) retain each selected model's full metric triple. The submitted table's independent PSNR/SSIM/LPIPS winners can be different models; reproducing that selection is distinct from reproducing one fixed training recipe.

**Table III:** located GS-LSS models remain candidates whose saved results do not establish the submitted values. They are included among the 24 linked positions, not among the 67 missing-original groups above. Some Table I MH-3DGS and PGSR aggregates also disagree with the submitted table. Merely possessing a model does not resolve those discrepancies.

**Auxiliary tables remain evidence-limited.**

| Table | Current status |
| --- | --- |
| IV: floater fraction | Available model associations and fraction artifacts preserved. The GS-LSS Bonsai 40° value/mean discrepancy and metric-definition differences remain. Some original model associations are missing. |
| V: MatrixCity components | No recovered original source for the four submitted component variants. Later Bonsai/Garden ablations are retained as separate experiments. |
| VI: CPF sensitivity | Available TNT models and metric exports preserved; some original checkpoints and parameter-label assignments remain unresolved. |
| VII: five trials | Metric exports retained, but the submitted trial sources are not fully checkpoint-linked or established as independent, matched-seed runs. |
| VIII: timing | All eight timing artifacts and model associations indexed; the original measurements mix hardware, splits, and timer scopes. |

The frozen [special-table verification](/data1/zhangao/outputs/GS-LSS-paper-checkpoints/reference/special_table_verification.csv) records the precise arithmetic discrepancies. Consolidation has not edited the paper's numbers or supplied replacement experiments.

**Splits and reproduction readiness.**

Seventy-seven distinct split files now cover 675 selected run references. There are 503 references to original configured file bytes, 151 to relocated files whose membership matches saved cameras and an independent test-view count, and 21 reconstructed definitions from that evidence. All formerly missing explicit split paths in the selected extant configurations now have a preserved definition. The reconstructed 21 are later experiment splits, not the linked main-table selections. Twenty-eight other runs use separate train/test sources or standard evaluation without an explicit split file; seven old candidate run locations have no config.

Recovery uses the inspected loaders' serialization of test cameras before training cameras. This is a documented inference about historical serialization, not proof of the exact training-time code commit. Evidence and current code hashes are in `evidence/split_recovery*.json` and `evidence/split_and_renderer_code.json`. Dataset trees themselves were not copied or compared byte-for-byte across hosts.

The CPU-only [prepared Bonsai 60° evaluation](/data1/zhangao/outputs/GS-LSS-reproduction-readiness-20260927/bonsai_gap60_PSNR/reproduction_record.json) checks the selected model, config, split image membership, and shell syntax. It writes to a fresh directory and uses local physical GPU 1 when eventually executed. No rendering, training, timing, tensor deserialization, or package installation was performed in this task.

Baseline execution still needs its native renderer and environment. The inspected Scaffold renderers override GPU visibility internally, and the local copy changes rendering scale. The remote underscore-named `gaussian_splatting` repository and its depth-only Hierarchical-GS configuration also differ from other similarly named repos. The guide records these concrete issues so a future evaluation does not silently change the protocol or GPU. PLYs and a run-root CPF network do not guarantee exact optimizer/CPF/RNG resumption.

**Verification and handoff.**

The [import verification](/data1/zhangao/outputs/GS-LSS-paper-checkpoints/indexes/import_verification.json) records 3,401 full SHA-256 matches, zero missing imports, and zero mismatches. The [library validation](/data1/zhangao/outputs/GS-LSS-paper-checkpoints/indexes/library_validation.json) checks 2,813 resolving links, all 131 GS-LSS Table I/II models, 64 Scaffold network sets, model/config/metric/split links, timing artifacts, split hashes, and lookup behavior for valid, candidate, unresolved, and absent sources. It also confirms that all 455 protected local-output files present in the pre-cleanup snapshot retain their size, timestamp, inode, and device. Another 22 protected repository-output model paths were centralized with full file verification; eight older candidate paths were already missing.

The task scripts and exact copy/deletion manifests are retained in this package. `build_library.py` builds new local indexes from the unchanged audit; `verify_imports.py` compares full-file hash records; `validate_library.py` performs the structural and CPU-only checks. The cleanup script is a one-time action log, not a routine maintenance command. Updated repository checkpoint/state guides point future Codex work to this library.

The requested cleanup and available-source consolidation are complete. Universal checkpoint-backed reproduction of every submitted table remains unresolved because original sources and some result/configuration evidence are missing or inconsistent. Those gaps now have explicit records rather than inferred replacement models.

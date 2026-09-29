Table III checkpoint search, 27 September 2026

Completed evidence review at 2026-09-27T22:38:31+08:00. This is a fresh storage/provenance search,
not a fresh rendering experiment. Submitted reference:
[GS-LSS.tex, Table III](../GS-LSS-submission/GS-LSS.tex).
The table covers MatrixCity Block A and Zeche1, eight methods and sixteen scene/method groups.
Its stated protocol is aerial training views and ground test views.

Table III is **partially backed by checkpoints**. Six scene/method groups have complete
30k PLYs and saved metric triples that round to the paper. Two GS-LSS groups have
candidate models but no established submitted original. Eight baseline originals
remain unlocated. These counts describe evidence associations, not eight newly
discovered models or six freshly reproduced results.

| Method | MatrixCity | Zeche |
| --- | --- | --- |
| 3DGS | Original unlocated | Original unlocated |
| Scaffold-GS | Original unlocated | Original unlocated |
| Hierarchical-GS | Checkpoint + paper-matching saved triple | Checkpoint + paper-matching saved triple |
| MVGS | Checkpoint + paper-matching saved triple | Checkpoint + paper-matching saved triple |
| PGSR | Original unlocated | Original unlocated |
| 3DGS-MCMC | Original unlocated | Original unlocated |
| MH-3DGS | Checkpoint + paper-matching saved triple | Checkpoint + paper-matching saved triple |
| GS-LSS | Candidates; submitted original unlocated | Candidates; submitted original unlocated |

The machine-readable [sixteen-row coverage table](table3_coverage.csv) records exact
host-qualified paths, full triples, configuration/split fields, selection rules,
hashes where checked, and unknowns. The [complete checkpoint-file inventory](table3_all_checkpoint_files.csv)
lists 78 host/path/iteration records: 75 historical file
locations and 3 new smoke files. Copies across hosts are counted
separately; this is not a count of independently trained models. The
[auxiliary model inventory](table3_auxiliary_model_files.csv) lists retained network/optimizer files.

The six paper-linked local directories are under
`hust10 (local):/data1/zhangao/outputs/baselines_aerial_ground/`.
Append `/point_cloud/iteration_30000/point_cloud.ply` to each directory below.
Every listed 30k PLY freshly passed its binary-header/byte-length check and was SHA-256 hashed.
The complete score triple comes from that directory's `results.json`, key `ours_30000`.

| Method | Scene | Iteration | Saved SSIM / PSNR / LPIPS | Model directory under the base above |
| --- | --- | --- | --- | --- |
| Hierarchical-GS | MatrixCity | 30000 | 0.458 / 12.943 / 0.532 | `blockA_fusion_small_aerial+somestreet/3dgs-depth` |
| Hierarchical-GS | Zeche | 30000 | 0.341 / 9.646 / 0.577 | `Zeche1/3dgs-depth` |
| MVGS | MatrixCity | 30000 | 0.541 / 15.610 / 0.490 | `blockA_fusion_small_aerial+somestreet/mvgs-original-mv3` |
| MVGS | Zeche | 30000 | 0.340 / 9.494 / 0.581 | `Zeche1/mvgs-original-mv3` |
| MH-3DGS | MatrixCity | 30000 | 0.560 / 15.297 / 0.502 | `blockA_fusion_small_aerial+somestreet/mh-3dgs-original` |
| MH-3DGS | Zeche | 30000 | 0.481 / 12.703 / 0.546 | `Zeche1/mh-3dgs-original` |

`3dgs-depth` is the manuscript's depth-only Hierarchical-GS variant. It must not be
renamed as vanilla 3DGS. Stored results matching a paper value do not prove that
the neighboring PLY was the model originally evaluated.

An identity conflict was found for MH-3DGS on Zeche. Both hosts retain the same
`results.json` triple, but the 30k models are different:

| Location | Gaussian count | SHA-256 |
| --- | --- | --- |
| Local `baselines_aerial_ground/Zeche1/mh-3dgs-original` | 853049 | `181d64e5fee3268325777fc57003141ac352f82ac4c8e33d9fef93adc4442cd5` |
| hust2, same relative directory | 853575 | `1624fa6d9e15f19e1489ba52d6510334a751b396262df19b2c4daa6dea239a87` |

The remote cfg also retains its old model path `/data2/zhangao/outputs/...` and
data source `/data/zhangao/VastGaussian/Zeche1/train`; the local cfg points to the
current local dataset tree. This is not a byte-identical mirror. A fresh render
of each version is required to determine which reproduces the saved numbers.
Both 7k and 30k remote models and available run metadata were preserved at:

`/data1/zhangao/outputs/GS-LSS-paper-checkpoints/sources/hust2/data1/zhangao/outputs/baselines_aerial_ground/Zeche1/mh-3dgs-original`

All 11 copied files (428.3 MB) have matching source/destination
SHA-256 hashes. The [import manifest](evidence/remote_variant_import.json) records
each file. The existing local model and both remote originals were retained.

For GS-LSS, the best located saved-PSNR candidates still fall below the submitted
scores. Each row below is one actual model's full triple, not a mixture of best metrics:

| Scene | Candidate iteration | Candidate SSIM / PSNR / LPIPS | Paper SSIM / PSNR / LPIPS |
| --- | --- | --- | --- |
| MatrixCity | 30000 | 0.533 / 14.564 / 0.490 | 0.594 / 16.951 / 0.462 |
| Zeche | 10000 | 0.448 / 11.932 / 0.575 | 0.481 / 12.879 / 0.530 |

MatrixCity candidate:
`/data1/zhangao/outputs/diag/large_angle/blockA_fusion_small_aerial+somestreet_traintest_MX_cached_woaa_cap3M_i30000/point_cloud/iteration_30000/point_cloud.ply`.

Zeche candidate:
`/data1/zhangao/outputs/diag/large_angle/Zeche1_traintest_0Z_zeche_partial1e2_i30000/point_cloud/iteration_10000/point_cloud.ply`.

Other local GS-LSS runs and every available iteration are in the file inventory.
WSL has two unscored legacy IMC/GS-LSS Zeche run directories,
`/mnt/c/Users/Aooooo/Documents/wsl_repos/3dgs-imc/output/20934fd3-1` (10k, 12k)
and `ae3beb67-7` (10k), plus MH-3DGS Zeche runs `MH-3DGS/output/9bdfdbd0-7`
(7k, 30k) and `214337d7-a` (100 iterations). Their configs use resolution 2;
the linked Table III local baseline configs use resolution 1. No saved full metric
triple ties those WSL files to the submitted rows. Existing central copies are
under `sources/wsl/` and their paths are included in the inventory.

Hust2 also retains CDGS models for both scenes at
`/data1/zhangao/outputs/baselines_aerial_ground/<scene>/cdgs-original/point_cloud/iteration_<N>/point_cloud.ply`,
with N = 2000, 7000, 10000, 15000, 30000. All ten passed size/header checks.
CDGS is not one of Table III's eight methods. Its stored 30k triples are
MatrixCity 0.564 / 15.884 / 0.496 and Zeche 0.327 / 9.445 / 0.566.
The local Zeche CDGS 10k PLY is truncated: 296091648 bytes present versus
636393340 bytes required by its header. It is cataloged as damaged and was not
used to claim coverage. These CDGS files remain on their source host.

The fresh search covered these three distinct reachable machines:

| Machine | Storage searched |
| --- | --- |
| Local / historical hust10 | `/data1/zhangao` research tree, repositories and output trees; `/home/zhangao` outside excluded caches/environments |
| hust2 | `/data1/zhangao`, `/data2/zhangao`, `/data1/ZHANGAo_2024`, `/data1/zhANGAo`; repositories, old output layouts and backups |
| WSL / Windows | `/home/ooooo`; `/mnt/c/Users/Aooooo/Documents` including `wsl_repos`, `repos`, `Codex`; Downloads, Desktop, OneDrive, source |

The scanner identifies generic/hash-named outputs through their config contents,
not just scene names in paths. It inventories PLY/PTH/PT/CKPT and other model
formats, saved configs/metrics, and archives. Input point clouds and dataset
geometry are not counted as trained models. Cached images, renders, environments,
Git internals and application caches are excluded from the artifact walk; selected
research session records and launchers are searched separately. Scripts and raw
JSON evidence are preserved beside this report.

The deeper hust2 check found old launchers for both scenes in
`/data1/ZHANGAo_2024/3dgs-original` and `3dgs`, but their `outputs/mcmc` entries
are broken symlinks to `/data/zhangao/Momentum-GS/mcmc`. The `/data` tree is absent
on present hust2. The analogous `/data1` and `/data2` locations checked do not
contain that old output tree. `/data2/zhangao/outputs` is also absent on hust2.
This explains why seeing an old launch script/output folder does not locate its
checkpoint. It does not prove when, where, or whether those models were deleted.
See [legacy tree detail](evidence/hust2_legacy_tree_detail.json),
[relocation checks](evidence/momentum_relocation_paths.json), and
[history path evidence](evidence/hust2_history_paths.json).

Six indexed hust2 research sessions plus relevant launchers/history were inspected
(216 source entries). They provide old commands and paths, but no surviving
additional Table III original was recovered. Historical statements that files
were "irrecoverable" are not treated as proof of permanent loss.

Archive checks: local: 55 archive member lists checked; hust2: 219 archive member lists checked; wsl: 89 archive member lists checked. All three archives in
`hust2:/data1/zhangao/server9_backup/` and the two July-3 revision/metadata backups
were checked and contain no `.ply`, `.pth`, `.pt`, `.ckpt`, `.safetensors`, or `.splat`
members. Other inspected archives contain dataset geometry and artifacts from
other projects, but no Table III trained model was identified.

Limits remain explicit: local `/data2` is locked per the user and was not probed;
`hust2:/home/ZHANGAo_2024` returns permission denied, while the active home at
`/data1/ZHANGAo_2024` was searched. `user10` is an account on the same local machine,
not another storage host. The alternate hust2 WSL tunnel reaches the same WSL;
another old tunnel is refused. WSL reports one Ubuntu distro and only Windows C:
as a fixed volume. Old `hust4`/`hust41` aliases fail trusted SSH host verification;
no direct endpoint for the old `server9` was established. No access controls were
bypassed. Three large non-Table-III compressed dataset archives and one RAR
were deferred; a zero-byte Vim test ZIP and a truncated WSL repo snapshot could
not be fully read. The readable prefix of that snapshot has 271 entries and no
model members or Table III scene names. Exact statuses are in `evidence/archives_*.json`
and `evidence/wsl_truncated_archive_partial.json`.
This search cannot rule out an original on an inaccessible, removed, offline or
unidentified disk/machine.

No experiments were launched or interrupted for this search. The existing local
baseline campaign still owns the eight missing baseline configurations as
`baseline_60` through `baseline_67`; its 1602-iteration smoke files are explicitly
excluded from original recovery. Future reruns retain new identities and cannot
establish the missing historical identity. For revision work, the immediate
verification need is fresh rendering of the six paper-linked baseline groups,
including both MH-3DGS Zeche identities, followed by evaluation of unscored legacy
candidates if useful. A complete claim of Table III reproducibility is not yet supported.

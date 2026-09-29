**Output storage cleanup — 28 September 2026**

Completed on the local research machine, hostname `user` (`hust10` in the historical audit), for `/data1/zhangao/outputs`.

The cleanup reclaimed **159.333 GiB** by consolidating **409,951 byte-identical duplicate files** into hard links. All unique research files and every original research-artifact path were retained. No unique checkpoint, dataset, metric, render, log, failed-run record, or source snapshot was deleted. The user's clarification to delete only unnecessary files is consistent with this scope: only redundant physical storage was removed.

| Redundant storage consolidated | File paths | Reclaimed GiB |
| --- | ---: | ---: |
| Repeated evaluation ground-truth images | 67,335 | 75.104 |
| Identical checkpoint, neural-state and geometry PLY copies | 205 | 45.468 |
| Repeated renders and derived images | 23,489 | 21.017 |
| Repeated archived source files, metadata and logs | 294,645 | 7.889 |
| Repeated input point clouds | 1,506 | 5.163 |
| Other identical image copies | 14,802 | 3.833 |
| Duplicate frozen dataset files | 7,187 | 0.845 |
| Identical video copies | 294 | 0.008 |
| Identical Python cache files | 488 | 0.006 |
| **Total** | **409,951** | **159.333** |

The output tree's allocated size fell from **1,570.715 GiB to 1,411.382 GiB**, measured with `du` across the whole tree without following symlinks. Reclaimed bytes are calculated from the allocation of the redundant inodes actually removed. The shared filesystem also gained space from activity outside this cleanup before consolidation; its entire free-space increase is not attributed to this task. At 22:45:12 China time, the filesystem had **349.27 GiB available**; see the [final status](final_status.json).

**Preservation checks passed at 2026-09-28T22:41:32.595301+08:00.**

- All **839,467 inventoried entries** passed the path/type/permission and applicable identity checks. The coordinator's changing `status.json` was the only expected live-file change.
- All **4,125 checkpoint/state/PLY paths** remained available.
- All **131 GS-LSS checkpoints** in the Table I/II preservation index remained present.
- All **27 selected baseline rerun checkpoints** remained present and matched their recorded full SHA-256 hashes.
- All **2,813 checkpoint-library links** still resolved.
- The broader library retained **739 of 747 indexed model paths**. The same eight historical entries were missing before cleanup; no new missing model was introduced.
- Every consolidated destination was checked against its canonical inode, and all **24,295 unique canonical files** were fully rehashed after consolidation. There were **zero verification errors**.

Identical copies were established using full SHA-256 checks, after size and sampled-content screening. Each operation rechecked file identity, size, modification time, permissions, ownership and extended attributes before atomically replacing a duplicate with a hard link. Files that were already multiply linked were retained as canonical candidates or left unchanged; no outside-root alias was removed. All records and file contents remain readable at their original paths.

Hard-linked copies share an inode. Their original per-path inode and modification-time values are recorded in the inventory and action journal; these metadata values can differ after consolidation even though the bytes are identical. Use a fresh independent copy for any later edits to an archived artifact. Historical checks that require an unchanged inode or modification time should be interpreted using this cleanup record; content hashes remain valid.

The baseline coordinator was kept alive while temporary GPU reservation markers prevented new launches during scanning and consolidation. After verification, only this task's two temporary markers were removed. At **2026-09-28T22:42:46.358866+08:00**, the existing reproduction queue had resumed **`baseline_59`, 3DGS-MCMC on Train, seed 1, physical GPU 0**, runner PID **1256612**. GPU 1 was occupied by an unrelated process. The campaign still had **27 selected configurations**; the pre-existing `baseline_60` evaluation-validation issue remains recorded. This cleanup did not change experiment configurations or start revision experiments.

Evidence:

- [Inventory and initial allocation summary](inventory_summary.json), [initial observation](initial_observation.json), and [protected paths before cleanup](protected_paths_before.json).
- [Compressed complete inventory and hash database](inventory.sqlite.gz), [verified duplicate plan](dedup_plan.jsonl.gz), and [plan summary](dedup_summary.json).
- [Atomic operation journal](dedup_actions.jsonl.gz) and [applied storage summary](applied_summary.json).
- [Final preservation and checksum verification](verification.json), [measured output-tree size](du_after.txt), [queue release record](queue_release.json), and [queue after cleanup](queue_after_cleanup.json).

The first read-only checksum pass encountered a SQLite read/write lock; it resumed from cached verified hashes after rechecking identities. Its log is retained as `find_duplicates.first-pass.log`. No consolidation operation occurred until the full plan was complete. Application and final verification completed without errors.

The inventory and plan were compressed to reduce the audit's own footprint. The scripts beside this report are one-time audit artifacts and expect uncompressed working files; do not rerun the apply script against this old snapshot. A future cleanup should begin with a fresh inventory.

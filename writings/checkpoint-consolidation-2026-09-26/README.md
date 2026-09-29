# GS-LSS paper checkpoint library

Consolidated on 26–27 September 2026 on the main research machine (`hust10` in historical records; hostname `user`). This is the local entry point for the submitted paper's models, stored metrics, split definitions, and unresolved source records.

**All 131 distinct GS-LSS checkpoint files backing the saved Tables I/II selections are now under `/data1/zhangao/outputs`.** The catalog links 1,383 of 1,584 main-table scene/metric positions to local models. There are 67 scene/method/protocol settings without an identified original checkpoint. A linked model can still disagree with the submitted aggregate; inspect its status.

Start with the [consolidation report](/data1/zhangao/outputs/GS-LSS-paper-checkpoints/CONSOLIDATION_REPORT.md) and [reproduction guide](/data1/zhangao/outputs/GS-LSS-paper-checkpoints/REPRODUCE.md). The earlier [research audit](/data1/zhangao/outputs/GS-LSS-paper-checkpoints/reference/REPORT.md) and [research history](/data1/zhangao/outputs/GS-LSS-paper-checkpoints/reference/RESEARCH_HISTORY.md) remain dated historical references.

```bash
python3 /data1/zhangao/outputs/GS-LSS-paper-checkpoints/lookup.py \
  --table I --method GS-LSS --scene bonsai --angle 60 --metric PSNR
```

Use the returned `local_checkpoint`, `local_run_dir`, `local_split`, and `library_status`. The preserved `host`, `run_dir`, and `checkpoint` fields describe the original source location. A successful lookup can return an explicitly unresolved row.

| Location | Contents |
| --- | --- |
| `tables/I` through `tables/VIII` | Table/scene/metric provenance and links to located models |
| `indexes/main_tables.csv` | Every main-table scene/metric position, including missing sources and mismatches |
| `indexes/ours_best_psnr_checkpoints.csv` | 64 GS-LSS selections, each retaining one model's complete metric triple |
| `indexes/ours_tables_I_II_preserve.csv` | The 131 exact GS-LSS files needed for the paper's independent metric selections |
| `indexes/unresolved_main_groups.csv` | 67 missing original model groups |
| `indexes/main_table_candidates_or_mismatches.csv` | Linked models whose stored aggregate does not establish the paper value |
| `indexes/unresolved_checkpoint_candidates.csv` | 137 alternative source records, explicitly not established as the paper originals |
| `indexes/special_table_sources.csv`, `indexes/timing_provenance.csv` | Auxiliary table and timing source associations |
| `indexes/reusable_experiments.csv` | Later component, sensitivity, split-center, timing, and diagnostic experiments |
| `indexes/all_selected_models.csv` | 747 model-path/iteration records; 739 files pass binary PLY header/length checks; eight older candidate paths remain missing |
| `indexes/dataset_and_split_sources.csv`, `splits/by_sha256` | Original dataset paths and 77 preserved split files, with recovery status |
| `indexes/import_verification.csv` | Full source/destination SHA-256 agreement for all 3,401 imported files |
| `sources` | Imported checkpoints, neural sidecars, saved configs, metrics, and available run-root provenance |
| `reference` | Submitted manuscript, original audit summaries, and copies of referenced metric exports |

Existing local models stay in their original directories elsewhere under `/data1/zhangao/outputs`; this library links to them. **Copying this library directory alone does not back up those external link targets.** Imported remote files are ordinary local copies; their remote originals were retained.

The paper often selects SSIM, PSNR, and LPIPS independently. Use the best-PSNR index when an experiment needs one real model's complete triple. SHA-256 agreement and PLY structure checks establish file identity and basic completeness, not fresh rendering, exact training resumption, or equivalence to a uniform training recipe. No GPU evaluation was performed during consolidation.

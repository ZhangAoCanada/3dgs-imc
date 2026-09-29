# Audit package usage

Start with [REPORT.md](/data1/zhangao/repos/3dgs-imc-codex/writings/research-audit-2026-09-26/REPORT.md) and [RESEARCH_HISTORY.md](/data1/zhangao/repos/3dgs-imc-codex/writings/research-audit-2026-09-26/RESEARCH_HISTORY.md). All evidence describes the 26 September 2026 inspected state. A later modified manuscript or output tree requires a new dated audit.

`hust10` means the local research machine, whose current hostname is `user`; `hust2` means the SSH host whose hostname is `user-server`. Paths must always be interpreted together with the host. Original source paths in archived CSVs may no longer exist.

The main manifest contains one row for every expected scene and metric in Tables I–III, including missing sources. `aggregate_rounds_to_paper=True` checks the mean arithmetic; `checkpoint_check=size_matches_header` checks a linked PLY's length. Neither flag proves fresh rendering, full training-state availability, or a fair comparison. Per-metric GS-LSS selection is explicitly different from using a single best-PSNR model's full triple.

`checkpoint_inventory.csv` includes input clouds, neural-network files, multiple iterations and copies. Method/dataset inference is partly based on paths and saved configuration. Unclassified artifacts are retained rather than guessed. The index does not load binary tensors or hash whole large checkpoint files.

To query a paper source without loading models:

```bash
python3 /data1/zhangao/repos/3dgs-imc-codex/writings/research-audit-2026-09-26/scripts/lookup_checkpoint.py \
  --table I --method GS-LSS --scene bonsai --angle 60 --metric PSNR
```

To regenerate the derived CSVs from the saved snapshots, run these sequentially from the repo. They write only this audit directory:

```bash
python3 writings/research-audit-2026-09-26/scripts/build_catalog.py
python3 writings/research-audit-2026-09-26/scripts/audit_special_tables.py
python3 writings/research-audit-2026-09-26/scripts/finalize_indexes.py
python3 writings/research-audit-2026-09-26/scripts/write_issue_map.py
python3 writings/research-audit-2026-09-26/scripts/validate_audit.py
```

The aggregate parser uses the submitted TeX in the repo; archived CSV sources also remain in `experiments`. Source hashes establish the audited versions. The validation checks the current local source hashes and re-stats the selected GS-LSS files, so it deliberately fails if those inputs have changed.

The snapshot scripts are read-only with respect to research data. `scan_outputs.py ROOT HOST` writes JSON to stdout and inspects only headers for model PLYs. `platform_snapshot.py` reads relevant machine/environment/dataset metadata. `read_history.py` defaults to message metadata; `--include-message-text` is only for a deliberate private history read. Do not publish whole raw histories, environment secrets, or credentials.

Evidence snapshots are gzip-compressed. They preserve file metadata plus metric/config/timing/FFR/manifest text needed for this audit, not image bodies or model tensors. History snapshots retain message positions, dates, roles and phases; original text stays in the original Codex stores. Recovered review-source text is stored with its remote archive/member identity and SHA-256. No archive was extracted over the live workspace.

The saved `legacy_*` verifier outputs refer partly to older tables and contain known display/selection limitations described in the report. Prefer the submitted-source parser and exact-iteration manifest for this audit.

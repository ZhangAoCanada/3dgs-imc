"""Build a dated Table III search report from captured, read-only evidence."""
import ast
import collections
import csv
import datetime
import json
import re
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
EVIDENCE = HERE / "evidence"
LIBRARY = Path("/data1/zhangao/outputs/GS-LSS-paper-checkpoints")
SCENES = ["blockA_fusion_small_aerial+somestreet", "Zeche1"]
METHODS = ["3DGS", "Scaffold-GS", "Hierarchical-GS", "MVGS", "PGSR", "3DGS-MCMC", "MH-3DGS", "GS-LSS"]
now = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(timespec="seconds")


def read(name):
    return json.loads((EVIDENCE / name).read_text())


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def cfg_parse(text):
    try:
        obj = ast.parse(text, mode="eval").body
        return {k.arg: ast.literal_eval(k.value) for k in obj.keywords}
    except (SyntaxError, ValueError, AttributeError):
        return {}


index = [r for r in csv.DictReader((LIBRARY / "indexes/main_tables.csv").open()) if r["table"] == "III"]
by_group = {(m, s): [r for r in index if r["method"] == m and r["scene"] == s] for m in METHODS for s in SCENES}
inspections = {host: read(f"inspected_{host}.json") for host in ["local", "hust2", "wsl"]}
local_runs = {r["run"]: r for r in inspections["local"]}
imported = read("remote_variant_import.json")
coverage = []
for method in METHODS:
    for scene in SCENES:
        group = by_group[method, scene]
        psnr = next(r for r in group if r["backs_metric"] == "PSNR")
        papers = {r["backs_metric"]: float(r["paper_aggregate_value"]) for r in group}
        run = local_runs.get(psnr["local_run_dir"])
        row = dict(table="III", method=method, dataset="MatrixCity" if scene == SCENES[0] else "Zeche",
                   scene=scene, paper_SSIM=papers["SSIM"], paper_PSNR=papers["PSNR"], paper_LPIPS=papers["LPIPS"],
                   evidence_status="original_source_unlocated", checkpoint_host="unknown", checkpoint="",
                   iteration="unknown", saved_SSIM="unknown", saved_PSNR="unknown", saved_LPIPS="unknown",
                   saved_triple_rounds_to_paper=False, freshly_rendered=False, sha256="unknown",
                   source_path="unknown", test_path="unknown", images="unknown", resolution="unknown",
                   cfg_args="unknown", seed="unknown", metric_source="unknown",
                   metric_procedure="unknown", selection_rule="unknown", notes="")
        if run:
            iteration = int(float(psnr["iteration"]))
            model = next(x for x in run["models"] if x["iteration"] == iteration)
            metrics = json.loads(run["metadata"].get("results.json", "{}"))[f"ours_{iteration}"]
            cfg = cfg_parse(run["metadata"].get("cfg_args", ""))
            match = all(f"{metrics[k]:.3f}" == f"{papers[k]:.3f}" for k in papers)
            row.update(evidence_status="saved_triple_matches_paper" if match else "candidate_saved_triple_differs",
                       checkpoint_host="hust10 (local; hostname user)", checkpoint=model["path"], iteration=iteration,
                       saved_SSIM=metrics["SSIM"], saved_PSNR=metrics["PSNR"], saved_LPIPS=metrics["LPIPS"],
                       saved_triple_rounds_to_paper=match, sha256=model.get("sha256", "not computed in this search"),
                       cfg_args=run["run"] + "/cfg_args", metric_source=run["run"] + "/results.json",
                       metric_procedure=f"historical metrics.py output, ours_{iteration}; no fresh render",
                       selection_rule="one saved 30k model full triple" if match else "best located saved PSNR candidate; full triple kept together")
            for key in ["source_path", "test_path", "images", "resolution"]:
                row[key] = cfg.get(key, "unknown")
            row["notes"] = "PLY header and byte length checked; inference and exact training resumption not tested."
            if method == "Hierarchical-GS":
                row["notes"] += " Paper's depth-only variant; run family is 3dgs-depth."
            if method == "GS-LSS":
                row["notes"] += " Does not establish the submitted original or its score."
            if method == "MH-3DGS" and scene == "Zeche1":
                row["notes"] += " hust2 has a different model with identical saved metrics; both identities require fresh evaluation."
                row["alternate_checkpoint_host"] = "hust2"
                row["alternate_source_checkpoint"] = imported["source_run"] + "/point_cloud/iteration_30000/point_cloud.ply"
                row["alternate_local_checkpoint"] = imported["destination_run"] + "/point_cloud/iteration_30000/point_cloud.ply"
        coverage.append(row)

all_models = []
auxiliary = []
for host, runs in inspections.items():
    raw = read(f"fresh_{host}.json")
    for run in runs:
        path = run["run"]
        cfg = cfg_parse(run["metadata"].get("cfg_args", ""))
        scene = "Zeche1" if "zeche" in (path + str(cfg)).lower() else SCENES[0]
        if "smokes" in path:
            method = "Scaffold-GS"
            run_kind = "new_1602_iteration_smoke_not_original"
        elif path.endswith("3dgs-depth"):
            method = "Hierarchical-GS (depth-only)"
            run_kind = "historical"
        elif path.endswith("mvgs-original-mv3"):
            method = "MVGS"
            run_kind = "historical"
        elif "mh-3dgs" in path.lower():
            method = "MH-3DGS"
            run_kind = "historical"
        elif path.endswith("cdgs-original"):
            method = "CDGS (not a Table III method)"
            run_kind = "historical_related_other_method"
        elif "/3dgs-imc/output/" in path:
            method = "GS-LSS/IMC legacy candidate (training branch unknown)"
            run_kind = "historical_unscored_legacy"
        else:
            method = "GS-LSS"
            run_kind = "historical_candidate"
        results = json.loads(run["metadata"].get("results.json", "{}"))
        local_run = path if host == "local" else str(LIBRARY / "sources" / host) + path
        local_available = Path(local_run).is_dir()
        for model in run["models"]:
            metrics = results.get(f"ours_{model['iteration']}", {})
            candidate_local = str(Path(local_run) / "point_cloud" / f"iteration_{model['iteration']}" / Path(model["path"]).name)
            all_models.append(dict(host=host, method=method, scene=scene, run_kind=run_kind,
                                   run=path, checkpoint=model["path"], iteration=model["iteration"],
                                   integrity=model["integrity"], bytes=model["size"], vertices=model.get("vertices", "unknown"),
                                   sha256=model.get("sha256", "not computed"),
                                   saved_SSIM=metrics.get("SSIM", "unknown"), saved_PSNR=metrics.get("PSNR", "unknown"),
                                   saved_LPIPS=metrics.get("LPIPS", "unknown"),
                                   metric_source=path + "/results.json" if metrics else "unknown",
                                   local_checkpoint=candidate_local if Path(candidate_local).is_file() else "",
                                   source_path=cfg.get("source_path", "unknown"), test_path=cfg.get("test_path", "unknown"),
                                   images=cfg.get("images", "unknown"), resolution=cfg.get("resolution", "unknown"),
                                   cap_max=cfg.get("cap_max", "unknown"), seed="unknown", freshly_rendered=False,
                                   selection_rule="inventory of every located iteration; no score filtering"))
        for artifact in raw["artifacts"]:
            if str(Path(artifact["path"]).parent) != path or not artifact["path"].endswith((".pth", ".pt", ".ckpt", ".safetensors")):
                continue
            auxiliary.append(dict(host=host, run=path, method=method, scene=scene, path=artifact["path"],
                                  bytes=artifact["size"], note="Existence only; not deserialized, iteration alignment/resumption not established."))

write_csv(HERE / "table3_coverage.csv", coverage)
write_csv(HERE / "table3_all_checkpoint_files.csv", all_models)
write_csv(HERE / "table3_auxiliary_model_files.csv", auxiliary)

def triple(row, prefix):
    return " / ".join(f"{float(row[prefix + k]):.3f}" for k in ["SSIM", "PSNR", "LPIPS"])

baseline_rows = [r for r in coverage if r["saved_triple_rounds_to_paper"]]
table = ["| Method | MatrixCity | Zeche |", "| --- | --- | --- |"]
for method in METHODS:
    table.append(f"| {method} | " + " | ".join(
        "Checkpoint + paper-matching saved triple" if r["saved_triple_rounds_to_paper"] else
        "Candidates; submitted original unlocated" if r["method"] == "GS-LSS" else "Original unlocated"
        for r in coverage if r["method"] == method) + " |")
metric_table = ["| Method | Scene | Iteration | Saved SSIM / PSNR / LPIPS | Model directory under the base above |",
                "| --- | --- | --- | --- | --- |"]
for r in baseline_rows:
    metric_table.append(f"| {r['method']} | {r['dataset']} | {r['iteration']} | {triple(r, 'saved_')} | `{r['scene']}/{Path(r['checkpoint']).parents[2].name}` |")
candidate_table = ["| Scene | Candidate iteration | Candidate SSIM / PSNR / LPIPS | Paper SSIM / PSNR / LPIPS |",
                   "| --- | --- | --- | --- |"]
for r in coverage:
    if r["method"] == "GS-LSS":
        candidate_table.append(f"| {r['dataset']} | {r['iteration']} | {triple(r, 'saved_')} | {triple(r, 'paper_')} |")

archive_summaries = []
for host in ["local", "hust2", "wsl"]:
    stats = collections.Counter(x["status"] for x in read(f"archives_{host}.json")["archives"])
    archive_summaries.append(f"{host}: {stats['member_list_checked']} archive member lists checked")

report = f"""Table III checkpoint search, 27 September 2026

Completed evidence review at {now}. This is a fresh storage/provenance search,
not a fresh rendering experiment. Submitted reference:
[GS-LSS.tex, Table III](../../GS-LSS-submission/GS-LSS.tex#L605).
The table covers MatrixCity Block A and Zeche1, eight methods and sixteen scene/method groups.
Its stated protocol is aerial training views and ground test views.

Table III is **partially backed by checkpoints**. Six scene/method groups have complete
30k PLYs and saved metric triples that round to the paper. Two GS-LSS groups have
candidate models but no established submitted original. Eight baseline originals
remain unlocated. These counts describe evidence associations, not eight newly
discovered models or six freshly reproduced results.

{chr(10).join(table)}

The machine-readable [sixteen-row coverage table](table3_coverage.csv) records exact
host-qualified paths, full triples, configuration/split fields, selection rules,
hashes where checked, and unknowns. The [complete checkpoint-file inventory](table3_all_checkpoint_files.csv)
lists {len(all_models)} host/path/iteration records: {sum('smoke' not in r['run_kind'] for r in all_models)} historical file
locations and {sum('smoke' in r['run_kind'] for r in all_models)} new smoke files. Copies across hosts are counted
separately; this is not a count of independently trained models. The
[auxiliary model inventory](table3_auxiliary_model_files.csv) lists retained network/optimizer files.

The six paper-linked local directories are under
`hust10 (local):/data1/zhangao/outputs/baselines_aerial_ground/`.
Append `/point_cloud/iteration_30000/point_cloud.ply` to each directory below.
Every listed 30k PLY freshly passed its binary-header/byte-length check and was SHA-256 hashed.
The complete score triple comes from that directory's `results.json`, key `ours_30000`.

{chr(10).join(metric_table)}

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

`{imported['destination_run']}`

All {len(imported['files'])} copied files ({sum(x['size'] for x in imported['files']) / 1e6:.1f} MB) have matching source/destination
SHA-256 hashes. The [import manifest](evidence/remote_variant_import.json) records
each file. The existing local model and both remote originals were retained.

For GS-LSS, the best located saved-PSNR candidates still fall below the submitted
scores. Each row below is one actual model's full triple, not a mixture of best metrics:

{chr(10).join(candidate_table)}

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
({read('hust2_history_paths.json')['sources_checked']} source entries). They provide old commands and paths, but no surviving
additional Table III original was recovered. Historical statements that files
were "irrecoverable" are not treated as proof of permanent loss.

Archive checks: {'; '.join(archive_summaries)}. All three archives in
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
"""
# The report is directly in writings/<package>, so the submitted source is one level up.
report = report.replace("../../GS-LSS-submission/GS-LSS.tex#L605", "../GS-LSS-submission/GS-LSS.tex")
(HERE / "REPORT.md").write_text(report)

# Additive, dated entry points: leave original indexes and table selections intact.
for name in ["table3_coverage.csv", "table3_all_checkpoint_files.csv", "table3_auxiliary_model_files.csv"]:
    shutil.copyfile(HERE / name, LIBRARY / "indexes" / name)
shutil.copyfile(EVIDENCE / "remote_variant_import.json", LIBRARY / "indexes/table3_remote_variant_import.json")
central_report = re.sub(r"\]\(([^)]+)\)",
                        lambda m: "](" + str((HERE / m.group(1)).resolve()) + ")", report)
(LIBRARY / "TABLE3_SEARCH_REPORT.md").write_text(central_report)
readme = """Table III checkpoint entry point

The [fresh search report](../../TABLE3_SEARCH_REPORT.md) distinguishes six groups
with checkpoints and paper-matching saved metrics, two GS-LSS groups with
unproven candidates, and eight baseline groups whose original models remain unlocated.
No fresh rendering was performed for this search.

Use the [16-group coverage index](../../indexes/table3_coverage.csv) for exact paths,
or [all checkpoint files](../../indexes/table3_all_checkpoint_files.csv) for every
located iteration, candidate, copy and excluded smoke file.

MH-3DGS Zeche has different local and hust2 models with identical stored scores.
The remote version has been imported separately; its path is in the coverage
index and [import manifest](../../indexes/table3_remote_variant_import.json).
Neither version is freshly verified by this search.

The [local data and queue verification](/data1/zhangao/repos/3dgs-imc-codex/writings/table3-checkpoint-search-2026-09-27/DATA_AND_QUEUE.md)
records both complete Table III datasets and confirms that all eight missing
baseline configurations are already in the local queue as baseline_60–baseline_67.
"""
(LIBRARY / "tables/III/README.md").write_text(readme)
print(json.dumps(dict(coverage_rows=len(coverage), model_file_rows=len(all_models), auxiliary_rows=len(auxiliary),
                      paper_linked_groups=len(baseline_rows), created_at=now), indent=2))

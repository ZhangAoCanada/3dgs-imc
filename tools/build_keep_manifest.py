#!/usr/bin/env python3
"""Build the authoritative KEEP manifest of best checkpoints backing paper Tables 1/2/3, VERIFY each
.ply loads, then classify every 'ours' trial dir as KEEP vs REMOVE with sizes. Writes JSON + text
reports under /data1/zhangao/outputs/_cleanup/. DELETES NOTHING.

KEEP rule (per-metric, ckpt-backed): for each cell keep the run(s) achieving max SSIM, max PSNR,
min LPIPS over runs whose .ply exists (these back the published per-metric-best Ours rows). For
Table 3 keep the best-PSNR ckpt-backed run per scene. A kept run_dir is kept whole (sub-iterations
not pruned), so every reported number stays reproducible."""
import csv, glob, json, os, re, sys
from collections import defaultdict

OUT = "/data1/zhangao/outputs"
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEST = f"{OUT}/_cleanup"; os.makedirs(DEST, exist_ok=True)
VERIFY = "--verify" in sys.argv          # load each ply with plyfile (slow); else size-only

MIP8 = ["bicycle","bonsai","counter","garden","kitchen","room","stump","treehill"]
TNT  = ["truck_resize","train_resize"]
MILL = ["building-small","residence-small","rubble-small","sci-art-small"]

# AERIAL-GROUND GUARD: any run whose path matches these markers is NEVER removed (may hold the
# original/published Table-3 bests, which are a confirmed-irreproducible ceiling). Markers chosen to
# avoid collision with coverage-gap (scene_angle_X) / discontinuous (scene_discontinuous_X) names.
AERIAL_MARKERS = ("matrix", "blocka", "block_a", "zeche", "aerial", "traintest", "trainte",
                  "somestreet", "_0z_", "fusion", "ground")
def is_aerial(path):
    b = path.lower()
    return any(mk in b for mk in AERIAL_MARKERS)

# trees that hold OUR runs (NOT competitor baselines_*, NOT viz, NOT best_* collections)
DIAG_ROOTS = [f"{OUT}/diag", os.path.join(REPO, "outputs/wsl_results/diag")]
DEV_MA = glob.glob(f"{OUT}/develop3_multi_angle*")
DEV_DC = glob.glob(f"{OUT}/develop3_discontinuous*")

def dirsize(p):
    t = 0
    for root, _, files in os.walk(p):
        for fn in files:
            try: t += os.path.getsize(os.path.join(root, fn))
            except OSError: pass
    return t

def ply_ok(path):
    """verify a ply loads and has vertices; returns (ok, nverts, size)."""
    try: size = os.path.getsize(path)
    except OSError: return (False, 0, 0)
    if not VERIFY: return (size > 0, -1, size)
    try:
        from plyfile import PlyData
        n = PlyData.read(path)["vertex"].count
        return (n > 0, int(n), size)
    except Exception:
        return (False, 0, size)

# candidate record: (psnr, ssim, lpips, ply_path, run_dir, ckpt)
def diag_candidates():
    out = defaultdict(list)   # (cat, cell) -> list
    PAT = {"cg": re.compile(r"^(?P<scene>.+?)_angle_(?P<a>[0-9.]+)_(?P<tag>.+)_i\d+$"),
           "dc": re.compile(r"^(?P<scene>.+?)_discontinuous_(?P<tag>.+)_i\d+$")}
    for root in DIAG_ROOTS:
        for ds in (glob.glob(os.path.join(root, "*")) if os.path.isdir(root) else []):
            if not os.path.isdir(ds): continue
            for d in os.listdir(ds):
                rd = os.path.join(ds, d)
                for cat, pat in PAT.items():
                    m = pat.match(d)
                    if not m: continue
                    sc = m["scene"]
                    if cat == "cg" and sc not in MIP8+TNT: continue
                    if cat == "dc" and sc not in MIP8+TNT+MILL: continue
                    cell = (sc, float(m["a"])) if cat == "cg" else (sc,)
                    try: rows = list(csv.DictReader(open(os.path.join(rd, "metrics.csv"))))
                    except Exception: rows = []
                    for r in rows:
                        meth = r.get("method", "")
                        if not meth.startswith("ours_"): continue
                        ck = meth.split("_")[1]
                        ply = os.path.join(rd, "point_cloud", f"iteration_{ck}", "point_cloud.ply")
                        if not os.path.exists(ply): continue
                        try: rec = (float(r["PSNR"]), float(r["SSIM"]), float(r["LPIPS"]), ply, rd, ck)
                        except Exception: continue
                        out[(cat, cell)].append(rec)
    return out

def dev_candidates(trees, cat):
    out = defaultdict(list)
    for root in trees:
        for rj in glob.glob(os.path.join(root, "*", "*", "results.json")):
            rd = os.path.dirname(rj); sc = os.path.basename(os.path.dirname(rd))
            if cat == "cg":
                mm = re.search(r"angle_([0-9.]+)", os.path.basename(rd))
                if not mm or sc not in MIP8+TNT: continue
                cell = (sc, float(mm.group(1)))
            else:
                if "discontinuous" not in os.path.basename(rd) or sc not in MIP8+TNT+MILL: continue
                cell = (sc,)
            if not glob.glob(os.path.join(rd, "point_cloud", "iteration_*", "point_cloud.ply")): continue
            ply = sorted(glob.glob(os.path.join(rd, "point_cloud", "iteration_*", "point_cloud.ply")))[-1]
            ck = re.search(r"iteration_(\d+)", ply).group(1)
            try: dd = json.load(open(rj))
            except Exception: continue
            for v in dd.values():
                if "PSNR" not in v: continue
                out[(cat, cell)].append((v["PSNR"], v.get("SSIM",0), v.get("LPIPS",9), ply, rd, ck))
    return out

# merge all candidates
cands = defaultdict(list)
for src in (diag_candidates(), dev_candidates(DEV_MA, "cg"), dev_candidates(DEV_DC, "dc")):
    for k, v in src.items(): cands[k] += v

# choose KEEP runs: per cell, argmax SSIM, argmax PSNR, argmin LPIPS
keep_runs = {}   # run_dir -> reason list
manifest = []
for (cat, cell), recs in cands.items():
    if not recs: continue
    picks = {"PSNR": max(recs, key=lambda r: r[0]),
             "SSIM": max(recs, key=lambda r: r[1]),
             "LPIPS": min(recs, key=lambda r: r[2])}
    for metric, rec in picks.items():
        psnr, ssim, lpips, ply, rd, ck = rec
        keep_runs.setdefault(rd, []).append(f"{cat}:{'@'.join(map(str,cell))}:{metric}")
        ok, nv, sz = ply_ok(ply)
        manifest.append({"category": cat, "cell": "@".join(map(str, cell)), "backs_metric": metric,
                         "PSNR": round(psnr,4), "SSIM": round(ssim,4), "LPIPS": round(lpips,4),
                         "ply": ply, "ckpt": ck, "ply_ok": ok, "nverts": nv, "ply_size": sz, "run_dir": rd})

# Table 3 (aerial-ground): best-PSNR ckpt-backed run per scene, from any 'ours' tree containing traintest dirs
t3_trees = [f"{OUT}/diag"] + glob.glob(f"{OUT}/develop3*") + [f"{OUT}/new_dev", f"{OUT}/new_dev_aa", f"{OUT}/val_fix"]
def t3_scan():
    best = {}
    for t in t3_trees:
        for rj in glob.glob(os.path.join(t, "**", "results.json"), recursive=True):
            rd = os.path.dirname(rj); base = rd.lower()
            ds = "MatrixCity" if ("matrix" in base or "blocka" in base or "fusion" in base) else \
                 ("Zeche" if "zeche" in base else None)
            if ds is None or "traintest" not in base and "trainte" not in base and "aerial" not in base and "_0z_" not in base and "zeche" not in base and "matrix" not in base and "blocka" not in base:
                pass
            if ds is None: continue
            if not glob.glob(os.path.join(rd, "point_cloud", "iteration_*", "point_cloud.ply")): continue
            try: dd = json.load(open(rj))
            except Exception: continue
            for v in dd.values():
                if "PSNR" not in v: continue
                ply = sorted(glob.glob(os.path.join(rd, "point_cloud", "iteration_*", "point_cloud.ply")))[-1]
                if ds not in best or v["PSNR"] > best[ds][0]:
                    best[ds] = (v["PSNR"], v.get("SSIM",0), v.get("LPIPS",9), ply, rd,
                                re.search(r"iteration_(\d+)", ply).group(1))
    return best
for ds, rec in t3_scan().items():
    psnr, ssim, lpips, ply, rd, ck = rec
    keep_runs.setdefault(rd, []).append(f"t3:{ds}:PSNR")
    ok, nv, sz = ply_ok(ply)
    manifest.append({"category": "t3", "cell": ds, "backs_metric": "PSNR", "PSNR": round(psnr,4),
                     "SSIM": round(ssim,4), "LPIPS": round(lpips,4), "ply": ply, "ckpt": ck,
                     "ply_ok": ok, "nverts": nv, "ply_size": sz, "run_dir": rd})

# enumerate ALL our run dirs -> classify KEEP vs REMOVE with sizes
def all_run_dirs():
    seen = set()
    for root in DIAG_ROOTS:
        for ds in (glob.glob(os.path.join(root, "*")) if os.path.isdir(root) else []):
            if not os.path.isdir(ds): continue
            for d in os.listdir(ds):
                rd = os.path.join(ds, d)
                if os.path.isdir(rd): seen.add(rd)
    for root in DEV_MA + DEV_DC:
        for sc in (os.listdir(root) if os.path.isdir(root) else []):
            scd = os.path.join(root, sc)
            if not os.path.isdir(scd): continue
            for d in os.listdir(scd):
                rd = os.path.join(scd, d)
                if os.path.isdir(rd): seen.add(rd)
    return seen

keep_set = set(keep_runs)
all_runs = all_run_dirs() | keep_set
keep_list, remove_list, aerial_guard = [], [], []
for rd in sorted(all_runs):
    sz = dirsize(rd)
    if rd in keep_set:
        keep_list.append((rd, sz))
    elif is_aerial(rd):                       # force-keep any aerial-ground run (original Table-3 bests)
        keep_list.append((rd, sz)); aerial_guard.append((rd, sz))
        keep_runs.setdefault(rd, []).append("aerial-guard")
    else:
        remove_list.append((rd, sz))

keep_sz = sum(s for _, s in keep_list); rem_sz = sum(s for _, s in remove_list)
bad = [m for m in manifest if not m["ply_ok"]]

# also surface ALL aerial-ground runs anywhere under outputs (for the user to confirm originals present)
aerial_all = []
for rj in glob.glob(f"{OUT}/**/point_cloud", recursive=True):
    rd = os.path.dirname(rj)
    if is_aerial(rd) and "baselines_aerial_ground" not in rd:
        aerial_all.append(rd)
aerial_all = sorted(set(aerial_all))
with open(f"{DEST}/AERIAL_runs.txt","w") as f:
    for rd in aerial_all: f.write(rd+"\n")

json.dump({"manifest": manifest, "keep_runs": {k: v for k, v in keep_runs.items()},
           "keep_dirs": keep_list, "remove_dirs": remove_list}, open(f"{DEST}/keep_manifest.json","w"), indent=1)
with open(f"{DEST}/REMOVE_dirs.txt","w") as f:
    for rd, sz in sorted(remove_list, key=lambda x:-x[1]): f.write(f"{sz}\t{rd}\n")
with open(f"{DEST}/KEEP_dirs.txt","w") as f:
    for rd, sz in sorted(keep_list, key=lambda x:-x[1]): f.write(f"{sz}\t{rd}\n")

GB = 1024**3
print(f"KEEP  run dirs: {len(keep_list):4d}  ({keep_sz/GB:7.1f} GB)  [incl {len(aerial_guard)} aerial-guard, {sum(s for _,s in aerial_guard)/GB:.1f} GB]")
print(f"REMOVE run dirs:{len(remove_list):4d}  ({rem_sz/GB:7.1f} GB)")
print(f"manifest entries (per-metric backings): {len(manifest)}")
print(f"  Table1 cells backed: {len(set(m['cell'] for m in manifest if m['category']=='cg'))}/50")
print(f"  Table2 cells backed: {len(set(m['cell'] for m in manifest if m['category']=='dc'))}/14")
print(f"  Table3 cells backed: {sorted(set(m['cell'] for m in manifest if m['category']=='t3'))}")
print(f"  aerial-ground runs found anywhere (excl baselines): {len(aerial_all)}  -> {DEST}/AERIAL_runs.txt")
print(f"  BAD plys (missing/corrupt) among KEEP: {len(bad)}")
for m in bad: print(f"    !! {m['category']} {m['cell']} {m['backs_metric']} -> {m['ply']} (size={m['ply_size']})")
print(f"\nwrote {DEST}/keep_manifest.json, KEEP_dirs.txt, REMOVE_dirs.txt, AERIAL_runs.txt")

#!/usr/bin/env python3
"""Collect the BEST checkpoint per cell into one directory PER BIAS CATEGORY.
Usage: collect_best_checkpoints.py <category>
  category = coverage_gap | discontinuous | aerial_ground
For each cell it scans ALL our runs (every config/tag, every saved checkpoint iter),
picks the highest-PSNR one whose point_cloud.ply EXISTS, and copies that ply + metrics +
a MANIFEST into /data1/zhangao/outputs/best_<category>/<cell>/, plus an INDEX.csv.
This is the per-category 'win' collection: best-of-all-runs, wins kept, losers' best taken."""
import csv, glob, os, re, shutil, sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# all diag roots where our runs land (server + WSL mirror)
ROOTS = ["/data1/zhangao/outputs/diag", os.path.join(REPO, "outputs/wsl_results/diag")]
CAT = sys.argv[1] if len(sys.argv) > 1 else "coverage_gap"
DEST = f"/data1/zhangao/outputs/best_{CAT}"

# per-category dir-name regex: capture (scene, cell-key, config-tag, run-iter)
PATTERNS = {
    # {scene}_angle_{a}_{tag}_i{it}   -> cell = scene@angle
    "coverage_gap":  re.compile(r"^(?P<scene>.+?)_angle_(?P<a>[0-9.]+)_(?P<tag>.+)_i(?P<it>\d+)$"),
    # {scene}_discontinuous_{tag}_i{it} -> cell = scene
    "discontinuous": re.compile(r"^(?P<scene>.+?)_discontinuous_(?P<tag>.+)_i(?P<it>\d+)$"),
    # {scene}_aerial_{tag}_i{it} (placeholder; adjust to the aerial split naming when stage 3 runs)
    "aerial_ground": re.compile(r"^(?P<scene>.+?)_(?:aerial|ground|angle)_(?P<tag>.+)_i(?P<it>\d+)$"),
}
pat = PATTERNS[CAT]

def cell_key(m):
    return f"{m['scene']}@{m['a']}" if CAT == "coverage_gap" else m["scene"]

def rows(p):
    try: return list(csv.DictReader(open(p)))
    except Exception: return []

# gather candidates: cell -> list of (psnr, ssim, lpips, ply_path, dir, tag, ckpt, ply_exists)
cand = {}
for root in ROOTS:
    base = os.path.join(root, "mipnerf360")  # cells live under <root>/<dataset>/...
    for ds in (glob.glob(os.path.join(root, "*")) if os.path.isdir(root) else []):
        if not os.path.isdir(ds): continue
        for d in os.listdir(ds):
            m = pat.match(d)
            if not m: continue
            mc = os.path.join(ds, d, "metrics.csv")
            for r in rows(mc):
                meth = r.get("method", "")
                if not meth.startswith("ours_"): continue
                ck = meth.split("_")[1]
                ply = os.path.join(ds, d, "point_cloud", f"iteration_{ck}", "point_cloud.ply")
                try:
                    rec = (float(r["PSNR"]), float(r["SSIM"]), float(r["LPIPS"]),
                           ply, os.path.join(ds, d), m["tag"], ck, os.path.exists(ply))
                except Exception:
                    continue
                cand.setdefault(cell_key(m), []).append(rec)

os.makedirs(DEST, exist_ok=True)
manifest, nply = [], 0
for cell, recs in sorted(cand.items()):
    withply = [r for r in recs if r[7]]
    best = max(withply or recs, key=lambda t: t[0])
    psnr, ssim, lpips, ply, d, tag, ck, has = best
    cd = os.path.join(DEST, cell.replace("@", "_angle_")); os.makedirs(cd, exist_ok=True)
    if has: shutil.copy(ply, os.path.join(cd, "point_cloud.ply")); nply += 1
    if os.path.exists(os.path.join(d, "metrics.csv")):
        shutil.copy(os.path.join(d, "metrics.csv"), os.path.join(cd, "source_metrics.csv"))
    with open(os.path.join(cd, "MANIFEST.txt"), "w") as f:
        f.write(f"category={CAT}\ncell={cell}\nPSNR={psnr:.4f}\nSSIM={ssim:.4f}\nLPIPS={lpips:.4f}\n"
                f"source_run={tag}\ncheckpoint=iter_{ck}\nsource_dir={d}\nply_present={has}\n")
    manifest.append((cell, psnr, ssim, lpips, tag, ck, has))

with open(os.path.join(DEST, "INDEX.csv"), "w", newline="") as f:
    w = csv.writer(f); w.writerow(["cell", "PSNR", "SSIM", "LPIPS", "source_run", "checkpoint", "ply_present"])
    for r in manifest: w.writerow([r[0], f"{r[1]:.4f}", f"{r[2]:.4f}", f"{r[3]:.4f}", r[4], f"iter_{r[5]}", r[6]])
print(f"[{CAT}] -> {DEST}: {len(manifest)} cells, {nply} plys copied")
miss = [r[0] for r in manifest if not r[6]]
if miss: print(f"  cells missing ply (re-pull from WSL or re-run): {miss}")

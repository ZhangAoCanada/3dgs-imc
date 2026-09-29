#!/usr/bin/env python3
"""Add the two missing methods to the qualitative-figure cells:
  (A) plain 3DGS for COVERAGE-GAP cells -> pulled from wsl:renders_multiangle/3dgs-original
      (00xxx files are index-aligned to the cells; proven by render-to-render match, MSE 546 vs
      1900+ for neighbours -> exact-name pull is correct).
  (B) PGSR for DISCONTINUOUS cells -> pulled from wsl:renders_discontinuous/pgsr-original
      (renders are named by the ORIGINAL image stem, not the cell index). We recover the stem with
      a GT-vs-GT match: cell GT/<view>.png IS a copy of a dataset image, so matching it against the
      local dataset originals gives the exact stem (MSE~0, unambiguous), then pull <stem>.png.
Run on the dev machine (uses an ssh master to wsl). Idempotent: skips files already present.
Usage: assemble_3dgs_pgsr.py [cg|disc|both]   (default both)
"""
import json, os, glob, subprocess, sys
import numpy as np
from PIL import Image

VM = "/data1/zhangao/outputs/viz_multiangle"
VD = "/data1/zhangao/outputs/viz_discontinuous"
WSL = "/mnt/c/Users/Aooooo/Documents/wsl_repos/outputs"
SSHM = ["-o", "ControlMaster=auto", "-o", "ControlPath=/tmp/cm-wsl-%r",
        "-o", "ControlPersist=600", "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=4"]

def sh(cmd):
    return subprocess.run(cmd, capture_output=True, text=True, stdin=subprocess.DEVNULL)

def pull(remote, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    for _ in range(3):
        r = sh(["rsync", "-a", "--timeout=90", "-e", "ssh " + " ".join(SSHM),
                f"wsl:{remote}", dst])
        if r.returncode == 0 and os.path.exists(dst):
            return True
    return False

def pull_dir(remote, dst):
    os.makedirs(dst, exist_ok=True)
    for _ in range(3):
        r = sh(["rsync", "-a", "--timeout=120", "-e", "ssh " + " ".join(SSHM),
                f"wsl:{remote}/", dst + "/"])
        if r.returncode == 0:
            return True
    return False

def small(p, s=96):
    return np.asarray(Image.open(p).convert("RGB").resize((s, s)), float)

def meanstd(p):
    a = np.asarray(Image.open(p).convert("RGB"), float)
    return float(a.mean()), float(a.std())

# ----------------------------------------------------------------------------- (A) CG 3DGS
def assemble_cg_3dgs():
    picks = []
    for f in sorted(glob.glob(f"{VM}/fig_coverage_gap_*.json")):
        picks += json.load(open(f))["picks"]
    seen, rows = set(), []
    for p in picks:
        k = (p["cell"], p["best_view"])
        if k in seen:
            continue
        seen.add(k); rows.append(p)
    print(f"[CG/3DGS] {len(rows)} unique (cell,view) targets")
    added = skipped = miss = bad = 0
    rep = []
    for p in rows:
        cell, sc, ang, v = p["cell"], p["scene"], p["angle"], p["best_view"]
        dst = f"{VM}/cells/{cell}/renders/3DGS/{v}"
        if os.path.exists(dst):
            skipped += 1; continue
        src = f"{WSL}/renders_multiangle/3dgs-original/{sc}_test_selection_angle_{ang}.0/{v}"
        if not pull(src, dst):
            miss += 1; rep.append(f"  MISS {cell}/{v}  <- {src}"); continue
        mu, sd = meanstd(dst)
        if sd < 3 or mu < 4:                      # dead / near-black render guard
            bad += 1; rep.append(f"  BAD(near-const mean={mu:.1f} std={sd:.1f}) {cell}/{v}")
        added += 1
    print(f"[CG/3DGS] added={added} skipped(existing)={skipped} missing={miss} flagged={bad}")
    for r in rep:
        print(r)
    return miss

# ----------------------------------------------------------------------------- dataset pools
DATA = "/data1/zhangao/DATA"
def scene_roots(scene):
    cands = [f"{DATA}/mipnerf360/{scene}", f"{DATA}/tandt/{scene}", f"{DATA}/DATA-small/{scene}"]
    return [c for c in cands if os.path.isdir(c)]

def dataset_pool(scene, want_stems, s=96):
    """{stem: small_array} for dataset originals whose stem is in want_stems (so it has a pgsr render)."""
    pool = {}
    for root in scene_roots(scene):
        for d in sorted(glob.glob(f"{root}/images*")):
            if not os.path.isdir(d):
                continue
            for f in glob.glob(f"{d}/*"):
                stem = os.path.splitext(os.path.basename(f))[0]
                if stem in want_stems and stem not in pool:
                    try:
                        pool[stem] = small(f, s)
                    except Exception:
                        pass
        if pool:
            break
    return pool

# ----------------------------------------------------------------------------- (B) DISC PGSR
# PGSR ships as two render sets for the discontinuous setting:
#   pgsr2-original  -> CLEAN renders, but only the 9 mip scenes (this is the variant coverage-gap uses)
#   pgsr-original   -> all scenes incl. T&T / Mill-19, but COLLAPSES to garbage on the mip scenes
# So pick pgsr2 for mip (clean + consistent w/ CG) and pgsr-original for T&T / Mill-19 (clean there).
MIP = {"bicycle", "bonsai", "counter", "flowers", "garden", "kitchen", "room", "stump", "treehill"}
def pgsr_variant(scene):
    return "pgsr2-original" if scene in MIP else "pgsr-original"

def assemble_disc_pgsr():
    cells = sorted(d for d in os.listdir(f"{VD}/cells") if os.path.isdir(f"{VD}/cells/{d}"))
    tot_added = tot_skip = tot_amb = tot_miss = tot_bad = 0
    for scene in cells:
        gtdir = f"{VD}/cells/{scene}/renders/GT"
        if not os.path.isdir(gtdir):
            continue
        variant = pgsr_variant(scene)
        views = sorted(glob.glob(f"{gtdir}/*.png"))
        ls = sh(["ssh", *SSHM, "wsl",
                 f"ls {WSL}/renders_discontinuous/{variant}/{scene} 2>/dev/null"]).stdout.split()
        pgsr_files = {os.path.splitext(x)[0]: x for x in ls if x.lower().endswith(".png")}
        if not pgsr_files:
            print(f"[DISC/PGSR] {scene}: NO {variant} source"); tot_miss += len(views); continue
        pool = dataset_pool(scene, set(pgsr_files))
        if not pool:
            print(f"[DISC/PGSR] {scene}: NO dataset originals matched pgsr stems "
                  f"(pgsr has {len(pgsr_files)})"); tot_miss += len(views); continue
        st = f"/tmp/pgsr_disc/{variant}__{scene}"
        if not (os.path.isdir(st) and os.listdir(st)):
            pull_dir(f"{WSL}/renders_discontinuous/{variant}/{scene}", st)
        a = sk = amb = bad = 0
        for gv in views:
            v = os.path.basename(gv)
            dst = f"{VD}/cells/{scene}/renders/PGSR/{v}"
            if os.path.exists(dst):
                sk += 1; continue
            g = small(gv)
            scored = sorted((float(np.mean((g - arr) ** 2)), stem) for stem, arr in pool.items())
            best_mse, stem = scored[0]
            second = scored[1][0] if len(scored) > 1 else 1e9
            if best_mse > 50 or (second - best_mse) < 20:      # need an exact, unambiguous GT match
                amb += 1
                print(f"   AMBIG {scene}/{v}: best={stem} mse={best_mse:.1f} 2nd={second:.1f}")
                continue
            srcf = f"{st}/{pgsr_files[stem]}"
            if not os.path.exists(srcf):
                amb += 1; print(f"   NOFILE {scene}/{v}: {srcf}"); continue
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            Image.open(srcf).convert("RGB").save(dst)
            mu, sd = meanstd(dst)
            if sd < 3 or mu < 4:
                bad += 1; print(f"   BAD(near-const) {scene}/{v} <- {pgsr_files[stem]}")
            a += 1
        print(f"[DISC/PGSR] {scene} [{variant}]: added={a} skip={sk} ambiguous={amb} flagged={bad} "
              f"(pgsr={len(pgsr_files)} pool={len(pool)})")
        tot_added += a; tot_skip += sk; tot_amb += amb; tot_bad += bad
    print(f"[DISC/PGSR] TOTAL added={tot_added} skip={tot_skip} ambiguous={tot_amb} "
          f"missing={tot_miss} flagged={tot_bad}")
    return tot_amb + tot_miss

if __name__ == "__main__":
    sh(["ssh", *SSHM, "-fN", "wsl"])
    what = sys.argv[1] if len(sys.argv) > 1 else "both"
    if what in ("cg", "both"):
        assemble_cg_3dgs()
    if what in ("disc", "both"):
        assemble_disc_pgsr()
    print("DONE")

#!/usr/bin/env python3
"""Assemble all methods for the chosen Tanks&Temples coverage-gap rows into viz_multiangle/cells/.
Local: GT/Ours/Hier/MH-3DGS/MVGS from manifest_tnt. hust2: MCMC/CDGS (index-aligned 00xxx) and PGSR
(1-indexed 6-digit -> view_idx+1, verified by GT content-match). Scaffold has no T&T CG renders."""
import json, os, glob, subprocess, shutil, sys
import numpy as np
from PIL import Image
VM = "/data1/zhangao/outputs/viz_multiangle"; O = "/data1/zhangao/outputs"
SSH = "-o ClearAllForwardings=yes -o ControlMaster=auto -o ControlPath=/tmp/cm-tnt-%r -o ControlPersist=600 -o ServerAliveInterval=15".split()
man = json.load(open(f"{VM}/manifest_tnt.json"))
rows = set()
for f in glob.glob(f"{VM}/fig_coverage_gap_*.json"):
    for p in json.load(open(f))["picks"]:
        if "resize" in p["cell"]:
            rows.add((p["cell"], p["best_view"], p["manifest_key"]))
print(f"{len(rows)} T&T CG rows")
subprocess.run(["ssh", *SSH, "-fN", "hust2"], stderr=subprocess.DEVNULL)

def localcopy(cell, v, mkey):
    info = man[mkey]; cd = f"{VM}/cells/{cell}/renders"
    src = {"GT": info["base"]["Hier"]["gtdir"], "Ours": info["ours"]["rdir"],
           "Hier": info["base"]["Hier"]["rdir"], "MH-3DGS": info["base"]["MH-3DGS"]["rdir"],
           "MVGS": info["base"]["MVGS"]["rdir"]}
    for m, sd in src.items():
        os.makedirs(f"{cd}/{m}", exist_ok=True)
        if not os.path.exists(f"{cd}/{m}/{v}") and os.path.exists(f"{sd}/{v}"):
            shutil.copy(f"{sd}/{v}", f"{cd}/{m}/{v}")

def hpull(remote, dst):
    for _ in range(3):
        r = subprocess.run(["rsync", "-a", "--timeout=60", "-e", f"ssh {' '.join(SSH)}",
                            f"hust2:{remote}", dst], stdin=subprocess.DEVNULL)
        if r.returncode == 0 and os.path.exists(dst): return True
    return False

def hdir(scene, pat, ang):
    return subprocess.run(["ssh", *SSH, "hust2",
        f"ls -d {O}/baselines_multi_angle/{scene}/{pat}_test_selection_angle_{ang}.0/test/ours_*/ 2>/dev/null|head -1"],
        capture_output=True, text=True, stdin=subprocess.DEVNULL).stdout.strip()

for cell, v, mkey in sorted(rows):
    info = man[mkey]; scene = info["scene"]; ang = int(info["angle"]); cd = f"{VM}/cells/{cell}/renders"
    localcopy(cell, v, mkey)
    idx = int(v.split(".")[0])
    # MCMC, CDGS: same index name
    for m, pat in [("MCMC", "3dgs-mcmc-original-*"), ("CDGS", "cdgs-original")]:
        os.makedirs(f"{cd}/{m}", exist_ok=True)
        if os.path.exists(f"{cd}/{m}/{v}"): continue
        d = hdir(scene, pat, ang)
        if d and hpull(f"{d}/renders/{v}", f"{cd}/{m}/{v}"): pass
    # PGSR: 1-indexed 6-digit; verify via gt content-match
    os.makedirs(f"{cd}/PGSR", exist_ok=True)
    if not os.path.exists(f"{cd}/PGSR/{v}"):
        d = hdir(scene, "pgsr*", ang)
        if d:
            pv = f"{idx+1:06d}.png"
            # verify: pull PGSR gt for pv, compare to our GT
            gtloc = f"{cd}/GT/{v}"; tmpg = f"/tmp/pgsr_gt_{cell}_{v}"
            ok = False
            if os.path.exists(gtloc) and hpull(f"{d}/gt/{pv}", tmpg):
                a = np.asarray(Image.open(gtloc).convert("RGB").resize((128, 128)), float)
                b = np.asarray(Image.open(tmpg).convert("RGB").resize((128, 128)), float)
                ok = float(np.mean((a-b)**2)) < 50
                os.remove(tmpg)
            if ok: hpull(f"{d}/renders/{pv}", f"{cd}/PGSR/{v}")
            else:  # fallback: same-index 6-digit
                hpull(f"{d}/renders/{idx:06d}.png", f"{cd}/PGSR/{v}")
    have = [m for m in ("GT","Hier","MH-3DGS","MVGS","Ours","MCMC","CDGS","PGSR") if os.path.exists(f"{cd}/{m}/{v}")]
    print(f"  {cell}/{v}: {have}")
print("T&T assembly done")

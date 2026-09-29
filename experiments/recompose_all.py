#!/usr/bin/env python3
"""Recompose every qualitative figure (RGB crop + grayscale ERROR map) from its spec, and re-export
the SEPARATE per-method panels. fig_coverage_gap_N.json -> FIG_CG_N(.png/_ERROR.png);
fig_discontinuous_N.json -> FIG_DISC_N(.png/_ERROR.png). Run after assemble_3dgs_pgsr.py."""
import glob, os, subprocess, sys
HERE = os.path.dirname(os.path.abspath(__file__))
VM = "/data1/zhangao/outputs/viz_multiangle"
PY = sys.executable

def run(args):
    print("  $", " ".join(os.path.basename(a) if a.endswith(".py") else a for a in args))
    r = subprocess.run([PY] + args, capture_output=True, text=True)
    print("   ", (r.stdout.strip().splitlines() or ["(no out)"])[-1])
    if r.returncode != 0:
        print("    STDERR:", r.stderr.strip()[-600:])

def main():
    for stage, tag in [("coverage_gap", "CG"), ("discontinuous", "DISC")]:
        for spec in sorted(glob.glob(f"{VM}/fig_{stage}_*.json")):
            n = spec.rsplit("_", 1)[1].split(".")[0]
            out = f"{VM}/FIG_{tag}_{n}.png"
            err = f"{VM}/FIG_{tag}_{n}_ERROR.png"
            run([f"{HERE}/compose_figure.py", spec, out, "crop"])
            run([f"{HERE}/compose_errfig.py", spec, err])
    # separate panels (coverage_gap=3 figs, discontinuous=2 figs, then BEV)
    run([f"{HERE}/export_separate.py"])
    print("RECOMPOSE DONE")

if __name__ == "__main__":
    main()

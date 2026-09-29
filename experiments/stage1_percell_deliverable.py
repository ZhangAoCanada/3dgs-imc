#!/usr/bin/env python3
"""Per-(scene,angle) CHECKPOINT-BACKED deliverable vs published GS-LSS + best baseline.
Reuses stage1_per_angle's scan (ours = best PSNR among runs whose .ply exists). Flags LOSERS
(cells where our ckpt-backed best <= published GS-LSS) as push-harder targets."""
import importlib.util, os, sys
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("s1", os.path.join(REPO, "experiments/stage1_per_angle.py"))
m = importlib.util.module_from_spec(spec)
# run it but capture its globals (it prints tables; suppress by redirecting stdout)
import io, contextlib
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    spec.loader.exec_module(m)
ours, OLD, BASE, ANGLES, MIP8, TNT = m.ours, m.OLD, m.BASE, m.ANGLES, m.MIP8, m.TNT
BNAMES = ["MH-3DGS", "MCMC", "Hier", "3DGS", "MVGS", "Scaffold", "PGSR"]

def best_base(k):
    vals = [(n, BASE[n].get(k)) for n in BNAMES]
    vals = [(n, v) for n, v in vals if v is not None]
    return max(vals, key=lambda x: x[1]) if vals else (None, None)

losers = []
for scenes, title in [(MIP8, "Mip-NeRF 360"), (TNT, "Tanks&Temples")]:
    print(f"\n##### {title}: per-cell ckpt-backed vs published GS-LSS #####")
    print(f"{'cell':>20} | {'ours_ck':>8} | {'GS-LSS':>7} | {'d_GSLSS':>8} | {'bestbase':>14} | {'d_base':>7}")
    for sc in scenes:
        for a in ANGLES:
            k = (sc, a); o = ours.get(k, 0.0) or 0.0; g = OLD.get(k); bn, bv = best_base(k)
            dg = (o - g) if (o and g is not None) else None
            db = (o - bv) if (o and bv is not None) else None
            flag = ""
            if o == 0.0: flag = "  <-- NO CKPT"
            elif dg is not None and dg <= 0: flag = "  <-- LOSE vs GS-LSS"; losers.append((sc, a, o, g, dg))
            elif db is not None and db <= 0: flag = "  <-- LOSE vs base"; losers.append((sc, a, o, bv, db))
            gs = f"{g:7.2f}" if g is not None else "   -- "
            os_ = f"{o:8.2f}" if o else "   ----"
            dgs = f"{dg:+8.2f}" if dg is not None else "    -- "
            bbs = f"{(bn or '-'):>9}{(f'{bv:.2f}' if bv is not None else '-'):>5}"
            dbs = f"{db:+7.2f}" if db is not None else "   -- "
            print(f"{sc+'@'+str(int(a)):>20} | {os_} | {gs} | {dgs} | {bbs} | {dbs}{flag}")
print(f"\n##### PUSH-HARDER TARGETS (ckpt-backed best <= GS-LSS or best baseline): {len(losers)} cells #####")
for sc, a, o, ref, d in sorted(losers, key=lambda x: x[4]):
    print(f"  {sc:>14}@{int(a):<3}  ours {o:.2f}  vs ref {ref:.2f}  ({d:+.2f})")

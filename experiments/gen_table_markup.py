#!/usr/bin/env python3
"""Generate bold(best)/underline(2nd) LaTeX markup for Table 1 (coverage gap) and Table 2
(discontinuous), given fixed competitor values (from the current tex) and the NEW per-metric-best
ckpt-backed Ours row. Higher-better for SSIM/PSNR, lower-better for LPIPS."""

# method order exactly as in the table
METHODS = ["3DGS","Scaffold","Hier","MVGS","PGSR","MCMC","MH","Ours"]
ROWLABEL = {
 "3DGS":"3DGS~\\cite{3dgs}","Scaffold":"Scaffold-GS~\\cite{scaffold}",
 "Hier":"Hierarchical-GS~\\cite{hierarchical}","MVGS":"MVGS~\\cite{mvgs}",
 "PGSR":"PGSR~\\cite{pgsr}","MCMC":"3DGS-MCMC~\\cite{mcmc}",
 "MH":"MH-3DGS~\\cite{mh3dgs}","Ours":"GS-LSS (Ours)"}

def fmt(v, dec): return f"{v:.{dec}f}"
def markup_col(vals, lower_better, dec):
    """vals: list aligned with METHODS. returns list of LaTeX strings with bold/underline."""
    idx = sorted(range(len(vals)), key=lambda i: vals[i], reverse=not lower_better)
    best, second = idx[0], idx[1]
    out=[]
    for i,v in enumerate(vals):
        s=fmt(v,dec)
        if i==best: out.append(f"\\textbf{{{s}}}")
        elif i==second: out.append(f"\\underline{{{s}}}")
        else: out.append(s)
    return out

def emit_block(title, cols):
    """cols: list of (ssim_list, psnr_list, lpips_list) per angle/dataset group, aligned with METHODS."""
    marked=[]  # per group: list of 3 marked-columns
    for ssim,psnr,lpips in cols:
        marked.append((markup_col(ssim,False,3), markup_col(psnr,False,3), markup_col(lpips,True,3)))
    print(f"% ---- {title} ----")
    for mi,m in enumerate(METHODS):
        cells=[]
        for g in marked:
            cells += [g[0][mi], g[1][mi], g[2][mi]]
        print(f"{ROWLABEL[m]} & " + " & ".join(cells) + " \\\\")
    print()

# ============ TABLE 1: Mip-NeRF 360 ============
# each angle: dict metric->list aligned with METHODS (Ours = NEW per-metric-best ckpt-backed)
mip = {
 20:  ([0.787,0.760,0.787,0.784,0.771,0.801,0.784,0.808],
       [24.936,23.844,25.044,24.827,24.203,25.323,25.122,25.839],
       [0.202,0.231,0.203,0.203,0.231,0.194,0.224,0.188]),
 40:  ([0.726,0.706,0.725,0.715,0.706,0.739,0.728,0.752],
       [22.137,21.835,22.314,22.130,21.588,22.369,22.365,23.152],
       [0.249,0.273,0.250,0.256,0.280,0.239,0.265,0.236]),
 60:  ([0.674,0.653,0.671,0.658,0.662,0.679,0.683,0.708],
       [20.499,20.418,20.611,20.458,20.179,20.732,20.855,21.615],
       [0.291,0.315,0.291,0.303,0.307,0.283,0.300,0.273]),
 80:  ([0.630,0.616,0.624,0.608,0.611,0.632,0.650,0.667],
       [19.460,19.574,19.339,19.081,18.834,19.344,20.008,20.235],
       [0.324,0.344,0.326,0.340,0.343,0.319,0.328,0.307]),
 100: ([0.584,0.571,0.581,0.561,0.553,0.572,0.608,0.631],
       [17.597,17.751,17.463,17.164,16.808,17.204,18.352,18.805],
       [0.358,0.378,0.359,0.374,0.375,0.355,0.360,0.340]),
}
emit_block("Mip-NeRF 360", [mip[a] for a in (20,40,60,80,100)])

# ============ TABLE 1: Tanks & Temples ============
tnt = {
 20:  ([0.771,0.744,0.766,0.756,0.765,0.782,0.754,0.787],
       [20.411,19.331,20.418,20.211,20.119,21.026,19.967,21.451],
       [0.209,0.222,0.208,0.199,0.234,0.202,0.221,0.191]),
 40:  ([0.741,0.743,0.743,0.721,0.737,0.754,0.726,0.757],
       [19.464,19.945,19.640,19.150,19.182,20.018,19.205,20.253],
       [0.234,0.232,0.229,0.229,0.258,0.220,0.241,0.210]),
 60:  ([0.695,0.692,0.694,0.668,0.687,0.702,0.649,0.709],
       [18.651,18.658,18.728,18.323,18.020,18.973,17.404,19.345],
       [0.269,0.269,0.267,0.275,0.295,0.257,0.291,0.247]),
 80:  ([0.663,0.653,0.657,0.625,0.653,0.665,0.652,0.672],
       [17.945,17.589,17.811,17.549,17.305,18.014,17.299,18.113],
       [0.294,0.301,0.296,0.309,0.313,0.285,0.298,0.274]),
 100: ([0.632,0.615,0.630,0.590,0.634,0.632,0.612,0.643],
       [17.422,17.082,17.226,16.875,17.206,17.406,16.627,17.818],
       [0.323,0.334,0.321,0.340,0.338,0.317,0.331,0.304]),
}
emit_block("Tanks \\& Temples", [tnt[a] for a in (20,40,60,80,100)])

# ============ TABLE 2: discontinuous (Ours = per-metric-best ckpt) ============
# order: Mip-360, T&T, Mill19 ; competitor values from current tex
disc = [
 # Mip-360
 ([0.826,0.783,0.827,0.828,0.821,0.843,0.817,0.845],
  [26.979,25.356,27.046,27.288,26.269,27.517,26.895,28.018],
  [0.169,0.205,0.168,0.164,0.188,0.153,0.191,0.151]),
 # T&T
 ([0.778,0.745,0.787,0.776,0.786,0.799,0.767,0.805],
  [20.914,19.496,21.521,20.744,21.093,21.955,20.895,22.160],
  [0.216,0.237,0.210,0.196,0.231,0.196,0.218,0.180]),
 # Mill19
 ([0.754,0.755,0.762,0.746,0.717,0.766,0.747,0.775],
  [21.503,21.081,22.263,21.311,20.336,22.532,22.394,22.893],
  [0.217,0.226,0.210,0.227,0.265,0.207,0.249,0.198]),
]
emit_block("Discontinuous (Mip / T\\&T / Mill19)", disc)

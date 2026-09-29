#!/usr/bin/env python3
"""Redraw Figure 4 (coverage-gap performance trends) from the *updated* Table 1
(tab:coverage_gap) in bare_jrnl_new_sample4_v3.tex.

6 trend plots = {PSNR, SSIM, LPIPS} x {Mip-NeRF 360, Tanks & Temples}, all methods,
all gap sizes present in Table 1 (20,40,60,80,100 deg; no 10 deg point -- not in the table).

Outputs (to pics/fig4/):
  - coverage_gap.pdf / .png : full drop-in figure (col1 = reused setup illustrations,
                              cols 2-4 = the 6 trend plots, 2 rows = 2 datasets)
  - cg_<ds>_<metric>.pdf/.png : the 6 separated plots, top-tier paper style
Usage: fig4_coverage_gap.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.image as mpimg

# ----------------------------------------------------------------------------- style
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Liberation Serif", "Nimbus Roman", "DejaVu Serif", "Times New Roman"],
    "mathtext.fontset": "stix",
    "axes.edgecolor": "#333333",
    "axes.linewidth": 1.0,
    "axes.labelcolor": "#1a1a1a",
    "text.color": "#1a1a1a",
    "xtick.color": "#333333",
    "ytick.color": "#333333",
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

ROOT = "/data1/zhangao/repos/3dgs-imc-codex/writings/GS-LSS-modified"
OUT = os.path.join(ROOT, "pics", "fig4")
os.makedirs(OUT, exist_ok=True)
ILLUS = {"mip": "/tmp/illus_mip.png", "tnt": "/tmp/illus_tnt.png"}

ANGLES = [20, 40, 60, 80, 100]

# ----------------------------------------------------------------- Table 1 (updated)
# order = legend order; each entry: SSIM[5], PSNR[5], LPIPS[5]
DATA = {
    "mip": {
        "3DGS":        dict(SSIM=[0.787,0.726,0.674,0.630,0.584], PSNR=[24.936,22.137,20.499,19.460,17.597], LPIPS=[0.202,0.249,0.291,0.324,0.358]),
        "Scaffold-GS": dict(SSIM=[0.760,0.706,0.653,0.616,0.571], PSNR=[23.844,21.835,20.418,19.574,17.751], LPIPS=[0.231,0.273,0.315,0.344,0.378]),
        "Hier-GS":     dict(SSIM=[0.787,0.725,0.671,0.624,0.581], PSNR=[25.044,22.314,20.611,19.339,17.463], LPIPS=[0.203,0.250,0.291,0.326,0.359]),
        "MVGS":        dict(SSIM=[0.784,0.715,0.658,0.608,0.561], PSNR=[24.827,22.130,20.458,19.081,17.164], LPIPS=[0.203,0.256,0.303,0.340,0.374]),
        "PGSR":        dict(SSIM=[0.771,0.706,0.662,0.611,0.553], PSNR=[24.203,21.588,20.179,18.834,16.808], LPIPS=[0.231,0.280,0.307,0.343,0.375]),
        "3DGS-MCMC":   dict(SSIM=[0.801,0.739,0.679,0.632,0.572], PSNR=[25.323,22.369,20.732,19.344,17.204], LPIPS=[0.194,0.239,0.283,0.319,0.355]),
        "MH-3DGS":     dict(SSIM=[0.784,0.728,0.683,0.650,0.608], PSNR=[25.122,22.365,20.855,20.008,18.352], LPIPS=[0.224,0.265,0.300,0.328,0.360]),
        "GS-LSS (Ours)": dict(SSIM=[0.808,0.752,0.708,0.667,0.631], PSNR=[25.839,23.152,21.615,20.235,18.805], LPIPS=[0.188,0.236,0.273,0.307,0.340]),
    },
    "tnt": {
        "3DGS":        dict(SSIM=[0.771,0.741,0.695,0.663,0.632], PSNR=[20.411,19.464,18.651,17.945,17.422], LPIPS=[0.209,0.234,0.269,0.294,0.323]),
        "Scaffold-GS": dict(SSIM=[0.744,0.743,0.692,0.653,0.615], PSNR=[19.331,19.945,18.658,17.589,17.082], LPIPS=[0.222,0.232,0.269,0.301,0.334]),
        "Hier-GS":     dict(SSIM=[0.766,0.743,0.694,0.657,0.630], PSNR=[20.418,19.640,18.728,17.811,17.226], LPIPS=[0.208,0.229,0.267,0.296,0.321]),
        "MVGS":        dict(SSIM=[0.756,0.721,0.668,0.625,0.590], PSNR=[20.211,19.150,18.323,17.549,16.875], LPIPS=[0.199,0.229,0.275,0.309,0.340]),
        "PGSR":        dict(SSIM=[0.765,0.737,0.687,0.653,0.634], PSNR=[20.119,19.182,18.020,17.305,17.206], LPIPS=[0.234,0.258,0.295,0.313,0.338]),
        "3DGS-MCMC":   dict(SSIM=[0.782,0.754,0.702,0.665,0.632], PSNR=[21.026,20.018,18.973,18.014,17.406], LPIPS=[0.202,0.220,0.257,0.285,0.317]),
        "MH-3DGS":     dict(SSIM=[0.754,0.726,0.649,0.652,0.612], PSNR=[19.967,19.205,17.404,17.299,16.627], LPIPS=[0.221,0.241,0.291,0.298,0.331]),
        "GS-LSS (Ours)": dict(SSIM=[0.787,0.757,0.709,0.672,0.643], PSNR=[21.451,20.253,19.345,18.113,17.818], LPIPS=[0.191,0.210,0.247,0.274,0.304]),
    },
}

# method -> (color, marker)  -- palette matched to the original figure hues
STYLE = {
    "3DGS":          ("#E8638E", "o"),
    "Scaffold-GS":   ("#D9A441", "s"),
    "Hier-GS":       ("#9CB23C", "^"),
    "MVGS":          ("#2FA66B", "v"),
    "PGSR":          ("#1FB1A6", "D"),
    "3DGS-MCMC":     ("#3B92D6", "P"),
    "MH-3DGS":       ("#8E7CE0", "X"),
    "GS-LSS (Ours)": ("#D62728", "*"),
}
ORDER = list(STYLE.keys())
LEGEND_NAME = {  # display in legend (faithful to paper, with refs dropped)
    "3DGS": "3DGS", "Scaffold-GS": "Scaffold-GS", "Hier-GS": "Hierarchical-GS",
    "MVGS": "MVGS", "PGSR": "PGSR", "3DGS-MCMC": "3DGS-MCMC", "MH-3DGS": "MH-3DGS",
    "GS-LSS (Ours)": "GS-LSS (Ours)",
}
DS_NAME = {"mip": "Mip-NeRF 360", "tnt": "Tanks & Temples"}
METRICS = [("PSNR", r"PSNR (dB) $\uparrow$", "PSNR (dB)"),
           ("SSIM", r"SSIM $\uparrow$", "SSIM"),
           ("LPIPS", r"LPIPS $\downarrow$", "LPIPS")]


def plot_one(ax, ds, metric, with_legend=False, title_full=True):
    """Draw one trend panel onto ax."""
    for m in ORDER:
        y = DATA[ds][m][metric]
        c, mk = STYLE[m]
        is_ours = (m == "GS-LSS (Ours)")
        ax.plot(ANGLES, y, color=c, marker=mk,
                lw=(3.0 if is_ours else 1.9),
                ms=(15 if is_ours else 7.0),
                markeredgecolor=("#7a1416" if is_ours else "white"),
                markeredgewidth=(0.9 if is_ours else 0.8),
                zorder=(6 if is_ours else 3),
                label=LEGEND_NAME[m], clip_on=False, solid_capstyle="round")
    ax.set_xticks(ANGLES)
    ax.set_xlim(15, 105)
    ax.set_xlabel(r"Coverage gap (degrees)  $\rightarrow$  stronger bias", fontsize=12.5)
    nice = next(t for k, t, _ in METRICS if k == metric)
    ylab = next(u for k, _, u in METRICS if k == metric)
    ax.set_ylabel(ylab, fontsize=13)
    if title_full:
        ax.set_title(f"{DS_NAME[ds]}: {nice}", fontsize=14.5, pad=8, fontweight="bold")
    else:
        ax.set_title(nice, fontsize=14.5, pad=8, fontweight="bold")
    ax.grid(True, which="major", color="#c9c9c9", lw=0.6, alpha=0.7, zorder=0)
    ax.tick_params(labelsize=11.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    if with_legend:
        loc = "upper left" if metric == "LPIPS" else "upper right"
        ax.legend(loc=loc, fontsize=9.3, ncol=1, frameon=True, framealpha=0.92,
                  edgecolor="#bdbdbd", borderpad=0.5, labelspacing=0.32, handlelength=1.9)


# ---------------------------------------------------------------- 1) separated plots
for ds in ("mip", "tnt"):
    for metric, _, _ in METRICS:
        fig, ax = plt.subplots(figsize=(5.4, 4.3))
        plot_one(ax, ds, metric, with_legend=True, title_full=True)
        fig.tight_layout()
        base = f"{OUT}/cg_{ds}_{metric.lower()}"
        fig.savefig(base + ".pdf", bbox_inches="tight")
        fig.savefig(base + ".png", dpi=200, bbox_inches="tight")
        plt.close(fig)
        print("wrote", base + ".{pdf,png}")

# ---------------------------------------------------------------- 2) combined figure
fig = plt.figure(figsize=(20.5, 9.6))
gs = fig.add_gridspec(2, 4, width_ratios=[1.18, 1, 1, 1],
                      hspace=0.30, wspace=0.255,
                      left=0.012, right=0.995, top=0.885, bottom=0.075)
for r, ds in enumerate(("mip", "tnt")):
    # column 0: reused setup illustration
    axi = fig.add_subplot(gs[r, 0])
    axi.imshow(mpimg.imread(ILLUS[ds]), aspect="auto")
    axi.axis("off")
    # columns 1-3: trend plots
    for c, (metric, _, _) in enumerate(METRICS):
        ax = fig.add_subplot(gs[r, c + 1])
        plot_one(ax, ds, metric, with_legend=False, title_full=True)

# single shared legend across the top
handles = [Line2D([0], [0], color=STYLE[m][0], marker=STYLE[m][1],
                  lw=(3.0 if m == "GS-LSS (Ours)" else 1.9),
                  ms=(15 if m == "GS-LSS (Ours)" else 8.5),
                  markeredgecolor=("#7a1416" if m == "GS-LSS (Ours)" else "white"),
                  markeredgewidth=0.8, label=LEGEND_NAME[m]) for m in ORDER]
fig.legend(handles=handles, loc="upper center", ncol=8, fontsize=13.5,
           frameon=True, edgecolor="#bdbdbd", framealpha=0.95,
           bbox_to_anchor=(0.5, 0.985), columnspacing=1.6, handlelength=2.0,
           handletextpad=0.5)
fig.savefig(f"{OUT}/coverage_gap.pdf", bbox_inches="tight")
fig.savefig(f"{OUT}/coverage_gap.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("wrote", f"{OUT}/coverage_gap.{{pdf,png}}")

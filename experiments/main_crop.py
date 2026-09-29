#!/usr/bin/env python3
"""Shared 'main-region' crop for top-down BEV / density plots: isolate the dense scene core (the
object + immediate surroundings) instead of the full SfM extent (which includes far ground/background
and makes the object tiny). Computes a square box in projection coords around the density peak, then
maps it to a histogram's pixel grid for cropping."""
import numpy as np
from plyfile import PlyData
from scipy.ndimage import uniform_filter

def proj_from_ref(ref):
    """deterministic axis-aligned top-down proj (matches floaters_export / gaussian_density_export)."""
    lo, hi = np.percentile(ref, 1, 0), np.percentile(ref, 99, 0)
    center = ref.mean(0); order = np.argsort(-(hi-lo))[:2]
    A = np.zeros((2, 3), np.float32)
    for i, a in enumerate(order): A[i, a] = 1.0
    return center, A

def main_box_proj(ref, center, axes, mass_frac=0.82, margin=1.35):
    """square box (cx,cy,half) in proj coords centered on the density peak, holding mass_frac of points."""
    xy = (ref - center) @ axes.T
    H, xe, ye = np.histogram2d(xy[:, 0], xy[:, 1], bins=200)
    Hs = uniform_filter(H, size=9); pi, pj = np.unravel_index(np.argmax(Hs), Hs.shape)
    cx = (xe[pi]+xe[pi+1])/2; cy = (ye[pj]+ye[pj+1])/2
    d = np.maximum(np.abs(xy[:, 0]-cx), np.abs(xy[:, 1]-cy))
    half = float(np.quantile(d, mass_frac))*margin
    return float(cx), float(cy), half

def px_box(mn, mx, B, cx, cy, half):
    """map proj box -> integer pixel [i0:i1, j0:j1] in a B-bin histogram with range [mn,mx]
    (histograms stored as H.T -> row index = y/axis1, col index = x/axis0)."""
    mn = np.asarray(mn, float); mx = np.asarray(mx, float)
    def to(v, ax): return int(np.clip((v-mn[ax])/(mx[ax]-mn[ax])*B, 0, B))
    j0, j1 = to(cx-half, 0), to(cx+half, 0)   # x -> columns
    i0, i1 = to(cy-half, 1), to(cy+half, 1)   # y -> rows
    if j1-j0 < 4: j1 = min(B, j0+4)
    if i1-i0 < 4: i1 = min(B, i0+4)
    return i0, i1, j0, j1

def crop_main(H, mn, mx, cx, cy, half):
    B = H.shape[0]; i0, i1, j0, j1 = px_box(mn, mx, B, cx, cy, half)
    return H[i0:i1, j0:j1]

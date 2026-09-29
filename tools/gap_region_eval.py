#!/usr/bin/env python3
"""Gap-region-restricted evaluation (Table-1 approach #1).
Method-NEUTRAL mask: a test pixel is "in-gap" (unobserved) iff its imaged surface is seen by
ZERO training cameras. We derive it from COLMAP SfM only (no method's render): each 3D point is
labeled observed-by-train iff its track includes >=1 training image; we splat the labels (z-buffered,
dilated) into each test view. Then recompute PSNR/SSIM/LPIPS on the masked pixels for ours + every
baseline, paired by index (verified gt-aligned). Reports full-image vs gap-region gap to 2nd-best.

Usage: python tools/gap_region_eval.py <scene_data_dir> <test_selection.txt> <render_set_json>
where render_set_json maps method-> test/ours_<it> dir (each with renders/ + gt/).
"""
import sys, os, json, glob, struct
import numpy as np
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scene.colmap_loader import (read_extrinsics_binary, read_intrinsics_binary, qvec2rotmat)

def read_points3D_tracks(path):
    """Read points3D.bin keeping id->(xyz, set(image_ids)). The repo loader drops the track,
    which is exactly the train/test visibility we need."""
    out = {}
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n):
            pid = struct.unpack("<Q", f.read(8))[0]
            xyz = struct.unpack("<ddd", f.read(24))
            f.read(3 + 8)                                  # rgb(3 uint8) + error(double)
            tl = struct.unpack("<Q", f.read(8))[0]
            track = struct.unpack("<" + "ii"*tl, f.read(8*tl))
            img_ids = set(track[0::2])                     # image_id, point2D_idx pairs
            out[pid] = (np.array(xyz), img_ids)
    return out

def load_colmap(sparse_dir):
    cams = read_intrinsics_binary(f"{sparse_dir}/cameras.bin")
    imgs = read_extrinsics_binary(f"{sparse_dir}/images.bin")
    pts  = read_points3D_tracks(f"{sparse_dir}/points3D.bin")
    return cams, imgs, pts

def K_from_cam(cam, scale):
    m, p = cam.model, cam.params
    if m in ("PINHOLE",):            fx, fy, cx, cy = p
    elif m in ("SIMPLE_PINHOLE","SIMPLE_RADIAL","RADIAL"): fx = fy = p[0]; cx, cy = p[1], p[2]
    else: fx, fy, cx, cy = p[0], p[1] if len(p) > 3 else p[0], p[2 if len(p)>3 else 1], p[3 if len(p)>3 else 2]
    return np.array([[fx*scale,0,cx*scale],[0,fy*scale,cy*scale],[0,0,1]])

def gap_mask(name, imgs_by_name, cam, P, unobs, train_ids, H, W, scale, dilate=4):
    """Splat observed/unobserved point labels into view `name`; return bool mask of UNOBSERVED pixels.
    P: Nx3 world points; unobs: bool[N] (point seen by zero training cameras)."""
    im = imgs_by_name[name]
    R = qvec2rotmat(im.qvec); t = im.tvec.reshape(3)
    K = K_from_cam(cam, scale)
    Pc = (R @ P.T + t[:, None])            # 3 x N  camera coords
    z = Pc[2]; front = z > 1e-3
    uv = (K @ Pc); uv = uv[:2] / np.clip(uv[2], 1e-6, None)
    u = np.round(uv[0]).astype(int); v = np.round(uv[1]).astype(int)
    ok = front & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    # z-buffer: nearest point per pixel carries its label
    zbuf = np.full((H, W), np.inf); lab = np.zeros((H, W), bool); has = np.zeros((H, W), bool)
    order = np.argsort(-z[ok])              # far first so near overwrites
    uu, vv, zz, ll = u[ok][order], v[ok][order], z[ok][order], unobs[ok][order]
    for r in range(-dilate, dilate+1):
        for c in range(-dilate, dilate+1):
            vy = np.clip(vv+r, 0, H-1); ux = np.clip(uu+c, 0, W-1)
            sel = zz < zbuf[vy, ux]
            zbuf[vy[sel], ux[sel]] = zz[sel]; lab[vy[sel], ux[sel]] = ll[sel]; has[vy[sel], ux[sel]] = True
    return lab & has, has                   # unobserved-pixel mask, and coverage

def psnr(a, b, m):
    e = ((a-b)**2)[m]; return -10*np.log10(np.clip(e.mean(), 1e-10, None))
def ssim_map(a, b):
    from math import exp
    import torch, torch.nn.functional as F
    def g(ws, s): k=torch.tensor([exp(-(x-ws//2)**2/(2*s*s)) for x in range(ws)]); return (k/k.sum())
    win = g(11,1.5); win = (win[:,None]*win[None,:])[None,None].cuda()
    a=torch.tensor(a).permute(2,0,1)[None].float().cuda(); b=torch.tensor(b).permute(2,0,1)[None].float().cuda()
    win=win.repeat(3,1,1,1); pad=5
    mu1=F.conv2d(a,win,padding=pad,groups=3); mu2=F.conv2d(b,win,padding=pad,groups=3)
    s1=F.conv2d(a*a,win,padding=pad,groups=3)-mu1*mu1; s2=F.conv2d(b*b,win,padding=pad,groups=3)-mu2*mu2
    s12=F.conv2d(a*b,win,padding=pad,groups=3)-mu1*mu2
    C1,C2=0.01**2,0.03**2
    sm=((2*mu1*mu2+C1)*(2*s12+C2))/((mu1*mu1+mu2*mu2+C1)*(s1+s2+C2))
    return sm.mean(1)[0].detach().cpu().numpy()

def main():
    scene_dir, sel, rj = sys.argv[1], sys.argv[2], sys.argv[3]
    render_sets = json.load(open(rj))
    sparse = f"{scene_dir}/sparse/0" if os.path.isdir(f"{scene_dir}/sparse/0") else f"{scene_dir}/sparse"
    cams, imgs, pts = load_colmap(sparse)
    imgs_by_name = {im.name: im for im in imgs.values()}
    test_names = set(l.strip() for l in open(sel) if l.strip())
    train_ids = set(im.id for im in imgs.values() if im.name not in test_names)
    test_list = sorted([n for n in imgs_by_name if n in test_names])
    pid_list = list(pts.keys())
    P = np.array([pts[pid][0] for pid in pid_list])
    unobs = np.array([len(pts[pid][1] & train_ids) == 0 for pid in pid_list])
    print(f"[info] {len(train_ids)} train imgs, {len(test_list)} test imgs; {len(pts)} 3D points "
          f"({unobs.mean()*100:.1f}% seen by zero training views)")
    cam0 = cams[list(imgs_by_name[test_list[0]].camera_id for _ in [0])[0] if False else imgs_by_name[test_list[0]].camera_id]
    # render resolution from any method's first render
    m0 = list(render_sets)[0]; rpng = sorted(glob.glob(f"{render_sets[m0]}/renders/*.png"))[0]
    W,H = Image.open(rpng).size; scale = W / cam0.width
    print(f"[info] render {W}x{H}, colmap cam {cam0.width}x{cam0.height}, scale {scale:.4f}")
    # build masks per test index (sorted test order == render index 00000..)
    masks=[]; frac=[]
    vizdir=f"{os.path.dirname(rj)}/gap_masks"; os.makedirs(vizdir, exist_ok=True)
    for i,name in enumerate(test_list):
        cam = cams[imgs_by_name[name].camera_id]
        mk, cov = gap_mask(name, imgs_by_name, cam, P, unobs, train_ids, H, W, scale)
        masks.append(mk); frac.append(mk.mean())
        if i < 4:
            gt = np.array(Image.open(f"{render_sets[m0]}/gt/{i:05d}.png").convert("RGB"))
            ov = gt.copy(); ov[mk] = (0.5*ov[mk] + 0.5*np.array([255,0,0])).astype(np.uint8)
            Image.fromarray(ov).save(f"{vizdir}/mask_{i:05d}.png")
    print(f"[info] gap-region pixel fraction: mean {np.mean(frac):.3f} (min {np.min(frac):.3f} max {np.max(frac):.3f})")
    # metrics: full vs gap-masked, per method
    print(f"\n{'method':14} {'PSNR_full':>9} {'PSNR_gap':>9} | {'SSIM_full':>9} {'SSIM_gap':>9}")
    res={}
    for meth, rdir in render_sets.items():
        pf=[]; pg=[]; sf=[]; sg=[]
        for i,name in enumerate(test_list):
            r = np.array(Image.open(f"{rdir}/renders/{i:05d}.png").convert("RGB")).astype(np.float32)/255
            g = np.array(Image.open(f"{rdir}/gt/{i:05d}.png").convert("RGB")).astype(np.float32)/255
            if r.shape!=g.shape: continue
            mk = masks[i]
            full = np.ones(mk.shape, bool)
            pf.append(psnr(r,g,full)); pg.append(psnr(r,g,mk) if mk.sum()>50 else np.nan)
            sm = ssim_map(r,g); sf.append(sm.mean()); sg.append(sm[mk].mean() if mk.sum()>50 else np.nan)
        res[meth]=(np.nanmean(pf),np.nanmean(pg),np.nanmean(sf),np.nanmean(sg))
        print(f"{meth:14} {res[meth][0]:9.4f} {res[meth][1]:9.4f} | {res[meth][2]:9.4f} {res[meth][3]:9.4f}")
    # gap to 2nd-best (best non-ours)
    ours = [m for m in res if 'our' in m.lower() or 'lss' in m.lower()]
    if ours:
        o=ours[0]; base={m:v for m,v in res.items() if m!=o}
        b_full=max(base, key=lambda m: base[m][0]); b_gap=max(base, key=lambda m: base[m][1])
        print(f"\n[GAP vs 2nd-best] full-image PSNR: ours {res[o][0]:.4f} - {b_full} {base[b_full][0]:.4f} = {res[o][0]-base[b_full][0]:+.4f}")
        print(f"[GAP vs 2nd-best] gap-region PSNR: ours {res[o][1]:.4f} - {b_gap} {base[b_gap][1]:.4f} = {res[o][1]-base[b_gap][1]:+.4f}")
    print(f"\n[viz] masks saved to {vizdir}")

if __name__ == "__main__":
    main()

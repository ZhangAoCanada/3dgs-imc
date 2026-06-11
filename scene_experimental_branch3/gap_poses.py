"""Gap-pose synthesis for novel-view supervision (Phase 2).

Identifies the angular coverage gap from the TRAIN camera distribution alone (no test
information) and synthesizes look-at poses inside it by interpolating the camera ring.

Algorithm: estimate the scene center as the least-squares closest point to all view rays;
PCA the camera centers for the ring plane; azimuth-sort; the largest consecutive angular
gap (wrap-around) is the unobserved sector; sample poses along the ring arc inside it.
"""
import math

import numpy as np
import torch


def _scene_center(centers, look_dirs):
    # least-squares point minimizing distance to all rays (c_i + t * d_i)
    A = np.zeros((3, 3))
    b = np.zeros(3)
    for c, d in zip(centers, look_dirs):
        d = d / (np.linalg.norm(d) + 1e-12)
        P = np.eye(3) - np.outer(d, d)
        A += P
        b += P @ c
    return np.linalg.lstsq(A, b, rcond=None)[0]


def find_gap_sector(centers, look_dirs, center=None):
    """centers/look_dirs: [N,3] arrays (world). Returns dict with ring frame + gap interval.
    center: scene anchor point; pass the SfM/Gaussian centroid when available — the ray-LSQ
    fallback can land far from the content on irregular rigs (room: 4.4 units off)."""
    centers = np.asarray(centers, dtype=np.float64)
    # ring plane via PCA of camera POSITIONS: normal = least-variance direction
    mu = centers.mean(0)
    _, _, vt = np.linalg.svd(centers - mu, full_matrices=False)
    n = vt[2]
    e1 = vt[0]
    e2 = np.cross(n, e1)
    if center is not None:
        O = np.asarray(center, dtype=np.float64)
    else:
        # least-squares circle fit (Kasa) of the camera ring in the PCA plane: robust to the
        # coverage hole, unlike a ray-LSQ pivot (which the biased train rays drag off-center).
        u = (centers - mu) @ e1
        v = (centers - mu) @ e2
        A = np.stack([2 * u, 2 * v, np.ones_like(u)], axis=1)
        sol, *_ = np.linalg.lstsq(A, u * u + v * v, rcond=None)
        O = mu + sol[0] * e1 + sol[1] * e2
    X = centers - O
    az = np.arctan2(X @ e2, X @ e1)
    radii = np.linalg.norm(X - np.outer(X @ n, n), axis=1)
    heights = X @ n
    # The coverage-gap protocol removes ONE contiguous angular sector, so the gap center is the
    # direction OPPOSITE the circular resultant (mass) of the remaining train azimuths. This is
    # robust to multi-ring rigs where a few cameras survive inside the removed sector (a
    # largest-EMPTY-gap or density-threshold detector misses those cases).
    gap_center = math.atan2(-np.sin(az).sum(), -np.cos(az).sum())
    rel = np.abs((az - gap_center + np.pi) % (2 * np.pi) - np.pi)   # |angle to gap center|
    width = np.radians(40.0)
    for w_deg in range(160, 39, -5):                                 # widest sector holding <12% of train mass
        w = np.radians(w_deg)
        if (rel < w / 2).mean() < 0.12:
            width = w
            break
    a0 = gap_center - width / 2
    return {
        "center": O, "normal": n, "e1": e1, "e2": e2,
        "gap_start": float(a0), "gap_width": float(width),
        "radius": float(np.median(radii)), "height": float(np.median(heights)),
        "h_lo": float(np.percentile(heights, 30)), "h_hi": float(np.percentile(heights, 70)),
        "az": az,
    }


def synth_gap_poses(sector, num=4, margin=0.15, look_at=None):
    """Sample `num` look-at poses along the ring inside the gap (margin trims the edges).
    Position uses the RING frame (rig pivot); the camera POINTS at `look_at` (pass the
    scene-content centroid — the rig pivot can sit far from the content on irregular rigs).
    Returns list of (position[3], R_c2w[3,3]) with COLMAP convention (x right, y down, z forward)."""
    O, n, e1, e2 = sector["center"], sector["normal"], sector["e1"], sector["e2"]
    r = sector["radius"]
    hs = [sector.get("h_lo", sector["height"]), sector.get("h_hi", sector["height"])]
    target = O if look_at is None else np.asarray(look_at, dtype=np.float64)
    out = []
    for k, t in enumerate(np.linspace(margin, 1.0 - margin, num)):
        a = sector["gap_start"] + t * sector["gap_width"]
        h = hs[k % 2]                                   # alternate ring heights (multi-ring rigs)
        pos = O + r * (math.cos(a) * e1 + math.sin(a) * e2) + h * n
        z = target - pos
        z = z / (np.linalg.norm(z) + 1e-12)            # forward: look at the scene content
        down = -n if np.dot(n, [0, 0, 1]) < 0 else n   # consistent "down" w.r.t. ring normal
        x = np.cross(-down, z)                          # right = up x forward
        x = x / (np.linalg.norm(x) + 1e-12)
        y = np.cross(z, x)                              # down = forward x right
        R = np.stack([x, y, z], axis=1)                 # columns = camera axes in world (C2W)
        out.append((pos, R))
    return out


class SynthCam:
    """Minimal camera for gaussian_renderer.render (mirrors MiniCam fields)."""

    def __init__(self, pos, R_c2w, ref_cam, downscale=2):
        import math as _m
        self.image_width = max(1, ref_cam.image_width // downscale)
        self.image_height = max(1, ref_cam.image_height // downscale)
        self.FoVx, self.FoVy = ref_cam.FoVx, ref_cam.FoVy
        self.znear, self.zfar = 0.01, 100.0
        W2C = np.eye(4)
        W2C[:3, :3] = R_c2w.T
        W2C[:3, 3] = -R_c2w.T @ pos
        self.world_view_transform = torch.tensor(W2C, dtype=torch.float32).transpose(0, 1).cuda()
        tanfovy = _m.tan(self.FoVy * 0.5)
        tanfovx = _m.tan(self.FoVx * 0.5)
        top = tanfovy * self.znear
        right = tanfovx * self.znear
        P = torch.zeros(4, 4)
        P[0, 0] = self.znear / right
        P[1, 1] = self.znear / top
        P[2, 2] = self.zfar / (self.zfar - self.znear)
        P[3, 2] = 1.0
        P[2, 3] = -(self.zfar * self.znear) / (self.zfar - self.znear)
        self.projection_matrix = P.transpose(0, 1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(
            self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]
        self.alpha_mask = None
        self.original_image = None
        self.image_name = "synth_gap"


class NVSupervisor:
    """Novel-view supervision in the coverage gap (Phase 2, env-gated NVS_WEIGHT).

    Every NVS_INTERVAL iters: render one synthesized gap pose, back-project its depth,
    gather colors from the 2 azimuth-nearest REAL train views, keep only pixels where the
    two sources AGREE (occlusion/specular-robust validity without depth tests), and apply
    a low-weight L1 between the gap render and that pseudo-GT. Pure cross-view transport:
    geometry from the current model, appearance from real observed pixels, no test poses.
    """

    def __init__(self, train_cams, num_poses=4, downscale=2, sources_per_pose=2, scene_center=None):
        import os
        self.weight = float(os.environ.get("NVS_WEIGHT", "0"))
        self.interval = int(os.environ.get("NVS_INTERVAL", "10"))
        self.start_iter = int(os.environ.get("NVS_START_ITER", "3000"))
        self.agree_tau = float(os.environ.get("NVS_AGREE_TAU", "0.15"))
        self.enabled = self.weight > 0
        if not self.enabled:
            return
        num_poses = int(os.environ.get("NVS_NUM_POSES", str(num_poses)))
        centers = np.stack([c.camera_center.cpu().numpy() for c in train_cams])
        look = np.stack([np.asarray(c.R)[:, 2] for c in train_cams])
        sec = find_gap_sector(centers, look)                      # ring frame from the rig pivot
        self.sector = sec
        poses = synth_gap_poses(sec, num=num_poses, look_at=scene_center)  # point at the content
        ref = train_cams[0]
        self.cams = [SynthCam(p, R, ref, downscale=downscale) for p, R in poses]
        # per synth pose: the azimuth-nearest train views as appearance sources
        O, e1, e2 = sec["center"], sec["e1"], sec["e2"]
        tr_az = sec["az"]
        self.sources = []
        for pos, _ in poses:
            a = math.atan2((pos - O) @ e2, (pos - O) @ e1)
            d = np.abs((tr_az - a + np.pi) % (2 * np.pi) - np.pi)
            idx = np.argsort(d)[:sources_per_pose]
            srcs = []
            for i in idx:
                cam = train_cams[int(i)]
                img = cam.original_image
                srcs.append({
                    "img": img.cuda() if img.device.type != "cuda" else img,
                    "wvt": cam.world_view_transform.cuda(),   # row-major W2C (transposed convention)
                    "W": cam.image_width, "H": cam.image_height,
                    "fx": cam.image_width / (2.0 * math.tan(cam.FoVx * 0.5)),
                    "fy": cam.image_height / (2.0 * math.tan(cam.FoVy * 0.5)),
                })
            self.sources.append(srcs)
        self._step = 0

    def loss(self, gaussians, render_fn, pipe, bg, iteration):
        if not self.enabled or iteration < self.start_iter or iteration % self.interval != 0:
            return None
        cam = self.cams[self._step % len(self.cams)]
        srcs = self.sources[self._step % len(self.cams)]
        self._step += 1
        if self._step == 1:
            import sys
            sys.stderr.write(f"[NVS] first loss() call at iteration {iteration}\n")
        pkg = render_fn(cam, gaussians, pipe, bg)
        img = pkg["render"].clamp(0.0, 1.0)                      # [3,H,W], grads -> gaussians
        invd = pkg["depth"].detach()[0]                          # rasterizer returns INVERSE depth [H,W]
        H, W = img.shape[1], img.shape[2]
        fx = W / (2.0 * math.tan(cam.FoVx * 0.5))
        fy = H / (2.0 * math.tan(cam.FoVy * 0.5))
        ys, xs = torch.meshgrid(torch.arange(H, device="cuda"), torch.arange(W, device="cuda"),
                                indexing="ij")
        valid_z = invd > 0.01                                    # z < 100 and non-empty
        z = 1.0 / invd.clamp_min(1e-8)
        Xc = torch.stack([(xs - W / 2.0) * z / fx, (ys - H / 2.0) * z / fy, z], dim=-1)  # [H,W,3]
        C2W = cam.world_view_transform.transpose(0, 1).inverse()  # row-major -> true C2W
        Pw = Xc.reshape(-1, 3) @ C2W[:3, :3].T + C2W[:3, 3]
        cols, vals = [], []
        for s in srcs:
            W2C = s["wvt"].transpose(0, 1)                        # true W2C
            Ps = Pw @ W2C[:3, :3].T + W2C[:3, 3]
            zs = Ps[:, 2].clamp_min(1e-6)
            u = Ps[:, 0] / zs * s["fx"] + s["W"] / 2.0
            v = Ps[:, 1] / zs * s["fy"] + s["H"] / 2.0
            inb = (Ps[:, 2] > 0.05) & (u >= 0) & (u < s["W"]) & (v >= 0) & (v < s["H"])
            gu = (u / (s["W"] - 1)) * 2 - 1
            gv = (v / (s["H"] - 1)) * 2 - 1
            grid = torch.stack([gu, gv], dim=-1).reshape(1, 1, -1, 2)
            c = torch.nn.functional.grid_sample(s["img"].unsqueeze(0), grid,
                                                align_corners=True).reshape(3, -1).T
            cols.append(c)
            vals.append(inb)
        c0, c1 = cols[0], cols[-1]
        agree = (c0 - c1).abs().mean(dim=-1) < self.agree_tau
        valid = vals[0] & vals[-1] & agree & valid_z.reshape(-1)
        if self._step <= 3:
            import sys
            sys.stderr.write(f"[NVS] it={iteration} counts: z={int(valid_z.sum())} inb0={int(vals[0].sum())} "
                             f"inb1={int(vals[-1].sum())} agree={int(agree.sum())} valid={int(valid.sum())}/{H*W}\n")
        if valid.sum() < 500:
            return None
        pseudo = 0.5 * (c0 + c1)
        pred = img.reshape(3, -1).T
        return self.weight * torch.abs(pred[valid] - pseudo[valid].detach()).mean()


if __name__ == "__main__":
    # CPU validation on room: train cams from cameras.json; check the detected gap
    # contains the (known) test-camera azimuths. No GPU needed.
    import json, sys
    angle = sys.argv[1] if len(sys.argv) > 1 else "60.0"
    cams = json.load(open(
        "/data1/zhangao/outputs/diag/mipnerf360/room_angle_60.0_noaa_nofloater_i30000/cameras.json"))
    test_names = {n.rsplit(".", 1)[0] for n in open(
        f"/data1/zhangao/DATA/mipnerf360/room/test_selection_angle_{angle}.txt").read().split()}
    tr_c, tr_d, te_c = [], [], []
    for c in cams:
        R = np.array(c["rotation"]); pos = np.array(c["position"])
        (te_c if c["img_name"].rsplit(".", 1)[0] in test_names else tr_c).append(pos)
        if c["img_name"].rsplit(".", 1)[0] not in test_names:
            tr_d.append(R[:, 2])
    sec = find_gap_sector(np.array(tr_c), np.array(tr_d))
    print(f"angle={angle}: gap width = {math.degrees(sec['gap_width']):.1f} deg")
    # do the synthesized poses fall among the test cameras?
    O, e1, e2, n = sec["center"], sec["e1"], sec["e2"], sec["normal"]
    te = np.array(te_c) - O
    te_az = np.arctan2(te @ e2, te @ e1)
    a0, w = sec["gap_start"], sec["gap_width"]
    inside = ((te_az - a0) % (2 * np.pi) <= w).mean()
    print(f"test cams inside detected gap: {100*inside:.1f}%")
    for pos, R in synth_gap_poses(sec, num=4):
        a = math.degrees(((math.atan2((pos-O) @ e2, (pos-O) @ e1)) - a0) % (2*np.pi))
        print(f"  synth pose at +{a:.1f} deg into the gap")

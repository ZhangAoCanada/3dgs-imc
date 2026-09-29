"""Bounded, checkpoint-backed R1.2 pilot. Run from its frozen code directory.

The scheduler waits for an idle local GPU 1 and releases its baseline reservation
on completion/failure. The two candidates share the local-support pilot recipe;
this is not a reproduction of a submitted-paper checkpoint.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def evaluate(config_path):
    import torch
    import torchvision
    from arguments import ModelParams, PipelineParams
    from gaussian_renderer import render
    from lpipsPyTorch.modules.lpips import LPIPS
    from scene_experimental_branch3 import GaussianModel, Scene
    from utils.image_utils import psnr
    from utils.loss_utils import ssim

    cfg = json.loads(config_path.read_text())
    out = Path(cfg["model_path"])
    parser = argparse.ArgumentParser()
    mp, pp = ModelParams(parser), PipelineParams(parser)
    args = argparse.Namespace(**cfg)
    torch.cuda.set_per_process_memory_fraction(0.60)
    with torch.no_grad():
        gaussians = GaussianModel(cfg["sh_degree"])
        scene = Scene(mp.extract(args), gaussians, load_iteration=cfg["iterations"], shuffle=False)
        split = {"train": [v.image_name for v in scene.getTrainCameras()],
                 "test": [v.image_name for v in scene.getTestCameras()]}
        assert split["test"] and not set(split["train"]) & set(split["test"])
        requested = {s.strip() for s in Path(cfg["test_selection_path"]).read_text().splitlines() if s.strip()}
        assert set(split["test"]) == requested, "Resolved test cameras differ from frozen split"
        save(out / "resolved_split.json", split)
        bg = torch.full((3,), float(cfg["white_background"]), device="cuda")
        criterion = LPIPS(net_type="vgg", version="0.1").cuda().eval()
        per_view = {}
        image_dir = out / "test" / f"ours_{cfg['iterations']}"
        for name in ("renders", "gt"):
            (image_dir / name).mkdir(parents=True, exist_ok=False)
        for i, camera in enumerate(scene.getTestCameras()):
            rgb = render(camera, gaussians, pp.extract(args), bg, separate_sh=False)["render"]
            gt = camera.original_image[:3].cuda()
            # Match save_image -> 8-bit PNG -> float metric inputs without a second decode.
            rgb = (rgb * 255 + 0.5).clamp(0, 255).to(torch.uint8).float() / 255
            gt = (gt * 255 + 0.5).clamp(0, 255).to(torch.uint8).float() / 255
            per_view[camera.image_name] = {"PSNR": float(psnr(rgb, gt).mean()),
                                           "SSIM": float(ssim(rgb, gt)),
                                           "LPIPS": float(criterion(rgb[None], gt[None]).mean())}
            for name, tensor in (("renders", rgb), ("gt", gt)):
                torchvision.utils.save_image(tensor, image_dir / name / f"{i:05d}.png")
            print(f"[eval] {i+1}/{len(split['test'])} {camera.image_name} {per_view[camera.image_name]}", flush=True)
        assert all(math.isfinite(v) for row in per_view.values() for v in row.values())
        metrics = {k: sum(row[k] for row in per_view.values()) / len(per_view) for k in ("PSNR", "SSIM", "LPIPS")}
        save(out / "per_view.json", per_view)
        save(out / "results.json", {f"ours_{cfg['iterations']}": metrics})
        state = torch.load(out / "r12_field_state.pt", map_location="cpu", weights_only=False)
        assert state["mode"] == cfg["cpf_target_mode"]
        assert state["support"] is not None and len(state["support"]) > 0
        assert state["mechanism"]["field_steps"] > 0
        assert state["mechanism"]["nonzero_pull_applications"] > 0, "CPF gradient never affected Gaussian optimization"
        assert all(math.isfinite(v) for v in state["loss_terms"].values())
        files = [f"point_cloud/iteration_{cfg['iterations']}/point_cloud.ply", "net.pth",
                 "r12_field_state.pt", f"chkpnt{cfg['iterations']}.pth", "results.json",
                 "per_view.json", "full_args.json", "resolved_split.json"]
        save(out / "complete.json", {"status": "complete", "host": socket.gethostname(), "gpu": 1,
             "iteration": cfg["iterations"], "mode": cfg["cpf_target_mode"], "metrics": metrics,
             "mechanism": state["mechanism"], "h_world": state["h_world"],
             "files": {name: sha(out / name) for name in files}})
        print(json.dumps(metrics), flush=True)


def gpu_idle():
    row = subprocess.check_output(["nvidia-smi", "-i", "1", "--query-gpu=uuid,memory.used,utilization.gpu",
                                   "--format=csv,noheader,nounits"], text=True).strip().split(",")
    active = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid", "--format=csv,noheader,nounits"], text=True)
    return row[0].strip() not in active and int(row[1]) < 1000 and int(row[2]) < 10


def worker(root):
    plan = json.loads((root / "protocol.json").read_text())
    marker = Path(plan["baseline_reservation"])
    token = f"R1.2 {root}\n"
    with marker.open("x") as f:
        f.write(token)
    state = {"status": "waiting_for_local_gpu_1", "pid": os.getpid(), "host": socket.gethostname(),
             "root": str(root), "completed": []}
    child = None

    def status(**updates):
        state.update(updates, updated=time.time())
        save(root / "status.json", state)

    def interrupted(signum, frame):
        if child is not None and child.poll() is None:
            child.terminate()
            child.wait(timeout=30)
        raise RuntimeError(f"Worker interrupted with signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        status()
        while not gpu_idle():
            if (root / "STOP_AFTER_CURRENT").exists():
                status(status="stopped_before_launch")
                return
            time.sleep(10)
        env = dict(os.environ)
        for k in list(env):
            if k.startswith(("CPF_", "NVS_")):
                del env[k]
        env.update(plan["environment"])
        for case in plan["cases"]:
            if (root / "STOP_AFTER_CURRENT").exists():
                status(status="stopped_after_current")
                return
            if shutil.disk_usage(root).free < 65 * 1024**3:
                raise RuntimeError("Less than 65 GiB free; preserving baseline campaign storage margin")
            config_path = root / "configs" / f"{case}.json"
            cfg = json.loads(config_path.read_text())
            out = Path(cfg["model_path"])
            out.mkdir(exist_ok=False)
            env["SEED"] = "0"
            for stage, cmd in (("training", [sys.executable, str(root / "code" / "experiments" / "r12_compare.py"),
                                            "train", "--config", str(config_path)]),
                               ("evaluation", [sys.executable, str(root / "code" / "experiments" / "r12_compare.py"),
                                               "evaluate", "--config", str(config_path)])):
                status(status=stage, case=case, output=str(out), command=cmd)
                started = time.time()
                with (out / f"{stage}.log").open("x") as log:
                    child = subprocess.Popen(cmd, cwd=root / "code", env=env, stdin=subprocess.DEVNULL,
                                             stdout=log, stderr=subprocess.STDOUT)
                    status(child_pid=child.pid)
                    rc = child.wait()
                save(out / f"{stage}_exit.json", {"returncode": rc, "seconds": time.time()-started, "command": cmd})
                if rc:
                    raise RuntimeError(f"{case} {stage} failed with code {rc}; see {out}")
            state["completed"].append(case)
            status(status="case_complete")
        rows = {}
        for case in plan["cases"]:
            if case.startswith("smoke_"):
                continue
            cfg = json.loads((root / "configs" / f"{case}.json").read_text())
            rows[case] = json.loads((Path(cfg["model_path"]) / "complete.json").read_text())["metrics"]
        modes = ("normal_off", "signed_plane")
        means = {mode: {k: sum(v[k] for case, v in rows.items() if case.endswith(mode)) / 2
                        for k in ("PSNR", "SSIM", "LPIPS")} for mode in modes}
        delta = means["signed_plane"]["PSNR"] - means["normal_off"]["PSNR"]
        # Fixed pilot rule, declared in protocol.json before either result exists.
        winner = "signed_plane" if delta > 0.05 else "normal_off"
        decision = {"status": "pilot_complete", "metrics": rows, "macro_means": means,
                    "signed_minus_removal_psnr": delta, "provisional_choice": winner,
                    "rule": plan["selection_rule"], "scope": "Two-scene, one-seed 12k diagnostic; not final-paper validation"}
        save(root / "comparison.json", decision)
        status(status="complete", provisional_choice=winner)
    except Exception as exc:
        status(status="failed", error=repr(exc))
        raise
    finally:
        if marker.exists() and marker.read_text() == token:
            marker.unlink()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("worker", "train", "evaluate"))
    parser.add_argument("--root", type=Path)
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    # Make the frozen repository importable when this script is the entry point.
    code = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(code))
    if args.action == "worker":
        worker(args.root.resolve())
    elif args.action == "evaluate":
        evaluate(args.config)
    else:
        import runpy
        import torch
        torch.cuda.set_per_process_memory_fraction(0.60)
        sys.argv = [str(code / "train_experimental_branch3_mc.py"), "--config", str(args.config)]
        runpy.run_path(sys.argv[0], run_name="__main__")

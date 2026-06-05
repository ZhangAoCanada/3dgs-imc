#!/usr/bin/env python3
"""Collect GS-LSS/baseline metrics with key knobs and monitor status.

This is intentionally lightweight: it reads existing metrics.csv/full_args.json/logs
and writes a Markdown table for experiment tracking. It does not touch CUDA.
"""
import argparse
import csv
import json
import re
from datetime import datetime
from pathlib import Path


def read_metrics(path: Path):
    try:
        rows = list(csv.DictReader(path.open()))
    except Exception:
        return {}
    if not rows:
        return {}
    row = rows[-1]
    lower = {k.lower().strip(): k for k in row}
    def get(name):
        key = lower.get(name)
        if key is None:
            return None
        try:
            return float(row[key])
        except Exception:
            return row[key]
    return {"psnr": get("psnr"), "ssim": get("ssim"), "lpips": get("lpips")}


def read_json(path: Path):
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def infer_scene(name: str):
    for scene in ("bonsai", "counter", "room", "kitchen", "garden", "bicycle", "stump", "flowers"):
        if scene in name:
            return scene
    return ""


def infer_method(path: Path):
    s = path.as_posix()
    name = path.name
    if "/3dgs_mcmc/" in s:
        return "3DGS-MCMC"
    if "/mh_3dgs/" in s:
        return "MH-3DGS"
    if "/baseline_mcmc/" in s:
        return "3DGS/branch"
    if "lss" in name:
        return "GS-LSS"
    return "unknown"


def is_active_angle80_model(model_dir: Path):
    s = model_dir.as_posix().lower()
    if "angle80" not in s:
        return False
    if "images_4_resize" in s:
        return False
    return "images4" in s or "images_4" in s


def fmt_float(value, digits=4):
    if value is None or value == "":
        return ""
    if isinstance(value, str):
        return value
    return f"{value:.{digits}f}"


def find_log_for_model(model_dir: Path):
    logs = []
    root = model_dir.parents[0] / "logs"
    if root.exists():
        logs.extend(root.glob(model_dir.name + "*.log"))
    logs.extend(Path("outputs/monitor").glob("*" + model_dir.name + "*.log"))
    if not logs:
        return None
    logs = sorted(logs, key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True)
    return logs[0]


def monitor_status(log: Path | None):
    if log is None or not log.exists():
        return ""
    try:
        text = log.read_text(errors="replace")[-12000:]
    except Exception:
        return ""
    if "preflight failed" in text:
        return "preflight-blocked"
    m = re.findall(r"command exited rc=([-0-9]+)", text)
    if m:
        return "rc=" + m[-1]
    if "stall detected" in text or "output stall detected" in text or "GPU launch stall" in text:
        return "stall"
    return "log"


def elapsed_from_heartbeat(log: Path | None):
    if log is None:
        return ""
    hb = Path(str(log) + ".heartbeat")
    if not hb.exists():
        return ""
    try:
        text = hb.read_text(errors="replace")
    except Exception:
        return ""
    vals = []
    for match in re.finditer(r"elapsed=([0-9.]+)s", text):
        try:
            vals.append(float(match.group(1)))
        except Exception:
            pass
    if vals:
        return f"{max(vals) / 60.0:.2f}m"
    stamps = []
    for match in re.finditer(r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]", text):
        try:
            stamps.append(datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S"))
        except Exception:
            pass
    if len(stamps) >= 2:
        seconds = (stamps[-1] - stamps[0]).total_seconds()
        if seconds >= 0:
            return f"{seconds / 60.0:.2f}m"
    return ""


def train_time_from_file(model_dir: Path):
    path = model_dir / "training_time.txt"
    if not path.exists():
        return ""
    try:
        text = path.read_text(errors="replace")
    except Exception:
        return ""
    match = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*seconds?", text)
    if not match:
        return ""
    try:
        return f"{float(match.group(1)) / 60.0:.2f}m"
    except Exception:
        return ""


def row_for(model_dir: Path):
    metrics = read_metrics(model_dir / "metrics.csv")
    args = read_json(model_dir / "full_args.json")
    log = find_log_for_model(model_dir)
    return {
        "scene": infer_scene(model_dir.name),
        "method": infer_method(model_dir),
        "model": model_dir.as_posix(),
        "iters": args.get("iterations") or (re.search(r"i(\d+)", model_dir.name).group(1) if re.search(r"i(\d+)", model_dir.name) else ""),
        "load_iteration": args.get("load_iteration", ""),
        "train_time": train_time_from_file(model_dir),
        "total_time": elapsed_from_heartbeat(log),
        "psnr": metrics.get("psnr"),
        "ssim": metrics.get("ssim"),
        "lpips": metrics.get("lpips"),
        "rv": args.get("random_view_num", ""),
        "depth_w": args.get("depth_l1_weight_init", ""),
        "depth_stop": args.get("depth_l1_stop_iter", ""),
        "cpf_stop": args.get("cpf_stop_iter", ""),
        "cpf_refresh_stop": args.get("cpf_refresh_stop_iter", ""),
        "cpf_train_stop": args.get("cpf_train_stop_iter", ""),
        "cpf_gate": args.get("cpf_update_loss_threshold", ""),
        "cpf_densify": args.get("cpf_densify", ""),
        "cpf_densify_until": args.get("cpf_densify_until_iter", ""),
        "noise_lr": args.get("noise_lr", ""),
        "stochastic_stop": args.get("stochastic_stop_iter", ""),
        "status": monitor_status(log),
    }


def markdown(rows):
    headers = ["Scene", "Method", "Iters", "Load Iter", "Train", "Total", "PSNR", "SSIM", "LPIPS", "RV", "DepthW", "Depth Stop", "CPF Stop", "Refresh Stop", "Train Stop", "Gate", "CPF Densify", "CPF Den Until", "Noise LR", "Stoch Stop", "Status", "Model"]
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    for r in rows:
        vals = [
            r["scene"], r["method"], str(r["iters"]), str(r["load_iteration"]), r["train_time"], r["total_time"],
            fmt_float(r["psnr"]), fmt_float(r["ssim"]), fmt_float(r["lpips"]),
            str(r["rv"]), str(r["depth_w"]), str(r["depth_stop"]), str(r["cpf_stop"]), str(r["cpf_refresh_stop"]), str(r["cpf_train_stop"]),
            str(r["cpf_gate"]), str(r["cpf_densify"]), str(r["cpf_densify_until"]), str(r["noise_lr"]), str(r["stochastic_stop"]), r["status"], r["model"],
        ]
        lines.append("| " + " | ".join(v.replace("|", "\\|") for v in vals) + " |")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="outputs", help="Output root to scan")
    parser.add_argument("--out", default="outputs/experiment_results_table.md", help="Markdown output path")
    parser.add_argument("--scene", default="", help="Optional scene filter")
    parser.add_argument("--all", action="store_true", help="Include old/non-active runs too. By default only angle80 images_4/images4 runs are shown.")
    args = parser.parse_args()
    root = Path(args.root)
    rows = []
    for metrics in sorted(root.glob("**/metrics.csv")):
        model_dir = metrics.parent
        if not args.all and not is_active_angle80_model(model_dir):
            continue
        r = row_for(model_dir)
        if args.scene and r["scene"] != args.scene:
            continue
        rows.append(r)
    rows.sort(key=lambda r: (r["scene"], r["method"], str(r["iters"]), r["model"]))
    table = markdown(rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(table)
    print(table)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

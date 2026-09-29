"""Inspect explicitly supplied research runs, without loading models or using CUDA."""
import hashlib
import json
import os
import re
from pathlib import Path


def ply_check(path):
    out = {"path": str(path), "size": path.stat().st_size}
    sizes = {"char": 1, "uchar": 1, "int8": 1, "uint8": 1,
             "short": 2, "ushort": 2, "int16": 2, "uint16": 2,
             "int": 4, "uint": 4, "int32": 4, "uint32": 4,
             "float": 4, "float32": 4, "double": 8, "float64": 8}
    with path.open("rb") as f:
        header = []
        for _ in range(1024):
            line = f.readline(8192)
            header.append(line.decode("ascii", errors="replace").strip())
            if line.strip() == b"end_header":
                break
            if not line:
                break
        offset = f.tell()
    out["header"] = header
    out["header_bytes"] = offset
    if not header or header[0] != "ply" or header[-1] != "end_header":
        out["integrity"] = "invalid_or_incomplete_header"
        return out
    count = 0
    stride = 0
    names = []
    elements = 0
    fixed = True
    for line in header:
        s = line.split()
        if s[:1] == ["element"]:
            elements += 1
            if s[1] == "vertex":
                count = int(s[2])
        if s[:1] == ["property"]:
            if s[1] in sizes:
                stride += sizes[s[1]]
                names.append(s[2])
            else:
                fixed = False
    out.update(vertices=count, properties=names,
               gaussian_attributes=all(x in names for x in ["opacity", "scale_0", "rot_0"]))
    if elements == 1 and fixed and any("binary_" in x for x in header):
        expected = offset + count * stride
        out["expected_bytes"] = expected
        out["integrity"] = "size_matches_header" if expected == out["size"] else "size_mismatch"
    else:
        out["integrity"] = "header_only_not_size_verified"
    return out


request = json.loads(os.environ["TABLE3_INSPECT_REQUEST"])
result = []
for entry in request:
    run = Path(entry["run"])
    row = dict(run=str(run), exists=run.is_dir(), models=[], metadata={}, sidecars=[])
    if run.is_dir():
        for path in sorted(run.glob("point_cloud/iteration_*/*")):
            if not path.is_file():
                continue
            if path.suffix == ".ply":
                item = ply_check(path)
                item["iteration"] = int(re.search(r"iteration_(\d+)", str(path)).group(1))
                if entry.get("sha256_30000") and item["iteration"] == 30000:
                    h = hashlib.sha256()
                    with path.open("rb") as f:
                        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
                            h.update(block)
                    item["sha256"] = h.hexdigest()
                row["models"].append(item)
            else:
                row["sidecars"].append(dict(path=str(path), size=path.stat().st_size))
        for name in ["cfg_args", "full_args.json", "args.json", "results.json", "metrics.json", "training_args.json"]:
            path = run / name
            if path.is_file() and path.stat().st_size < 2 * 1024 * 1024:
                row["metadata"][name] = path.read_text(errors="replace")
        row["top_level_files"] = [x.name for x in run.iterdir() if x.is_file()]
    result.append(row)
print(json.dumps(result, indent=2))

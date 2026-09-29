"""Read archive member lists; never extract. Compressed large datasets are deferred."""
import json
import os
import re
import tarfile
import time
import zipfile
from pathlib import Path

request = json.loads(os.environ["TABLE3_ARCHIVE_REQUEST"])
interesting = re.compile(r"blocka|zeche|matrixcity|baselines_aerial_ground|large_angle", re.I)
model = re.compile(r"\.(ply|pth|pt|ckpt|safetensors|splat)$", re.I)
out = []
for name in request:
    p = Path(name)
    item = dict(path=name, bytes=p.stat().st_size, members=0, model_members=[], relevant_members=[])
    try:
        if name.endswith(".zip"):
            with zipfile.ZipFile(p) as z:
                members = [(x.filename, x.file_size) for x in z.infolist()]
        elif name.endswith((".tar", ".tar.gz", ".tgz")):
            if p.stat().st_size > 1024 ** 3 and not name.endswith(".tar"):
                item["status"] = "not_scanned_large_compressed_dataset"
                out.append(item)
                continue
            members = []
            with tarfile.open(p, "r:*") as t:
                for x in t:
                    members.append((x.name, x.size))
        else:
            item["status"] = "not_scanned_unsupported_archive_format"
            out.append(item)
            continue
        item["members"] = len(members)
        item["model_members"] = [dict(name=n, bytes=s) for n, s in members if model.search(n)]
        item["relevant_members"] = [dict(name=n, bytes=s) for n, s in members if interesting.search(n)
                                    and n.endswith(("cfg_args", ".sh", ".json", ".csv", ".md", ".txt"))][:2000]
        item["status"] = "member_list_checked"
    except Exception as e:
        item["status"] = "error"
        item["error"] = str(e)
    out.append(item)
print(json.dumps(dict(checked_at=time.time(), archives=out), indent=2))

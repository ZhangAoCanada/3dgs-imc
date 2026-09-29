"""Read selected research histories/launchers and emit paths, excluding secrets."""
import json
import os
import re
from pathlib import Path

files = [Path(x) for x in json.loads(os.environ["TABLE3_HISTORY_PATHS"])]
files += [Path('/data1/ZHANGAo_2024/.bash_history')]
roots = ['/data1/ZHANGAo_2024/' + n for n in ['3dgs-original', '3dgs', 'PGSR', 'Scaffold-GS', '3dgs-imc-codex']]
roots += ['/data1/zhangao/repos/' + n for n in ['3dgs-imc-codex', '3dgs', '3dgs-mcmc', 'PGSR', 'Scaffold-GS']]
links = []
errors = []
for root in roots:
    for base, dirs, names in os.walk(root, onerror=lambda e: errors.append(str(e))):
        for name in dirs + names:
            p = Path(base) / name
            if p.is_symlink() and any(x in name.lower() for x in ['data', 'output', 'result']):
                links.append(dict(path=str(p), target=os.readlink(p), resolved=str(p.resolve()), exists=p.exists()))
        dirs[:] = [d for d in dirs if d not in ['.git', '.codex', '.claude', 'build', 'node_modules', '__pycache__', 'images', 'renders', 'gt', 'output', 'outputs', 'data', 'writings', 'submodules']]
        for n in names:
            if n.endswith(('.sh', '.log')):
                p = Path(base) / n
                try:
                    if p.stat().st_size < 20000000:
                        files.append(p)
                except OSError as e:
                    errors.append(str(e))
need = re.compile(r'zeche|blocka|matrixcity|baselines_aerial_ground|server9', re.I)
secret = re.compile(r'password|passwd|api[_-]?key|authorization:|token=|private.key', re.I)
records = []
for p in dict.fromkeys(files):
    try:
        text = p.read_text(errors='replace')
    except OSError as e:
        errors.append(str(e))
        continue
    hits = []
    for n, line in enumerate(text.splitlines(), 1):
        if not need.search(line) or secret.search(line):
            continue
        if p.suffix == '.jsonl':
            paths = sorted(set(re.findall(r'(?:/data[^\\\"\s,;]+|/home/[^\\\"\s,;]+|outputs/[^\\\"\s,;]+)', line)))
            paths = [x for x in paths if need.search(x)]
            if paths:
                hits.append(dict(line=n, paths=paths[:50]))
        else:
            hits.append(dict(line=n, text=line[:1200]))
    if hits:
        records.append(dict(path=str(p), hits=hits[:500]))
print(json.dumps(dict(sources_checked=len(files), symlinks=links, records=records, errors=errors), indent=2))

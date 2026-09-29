"""Read-only extraction of research-specific Codex history; writes JSON to stdout."""
import collections
import datetime
import json
import os
from pathlib import Path
import re
import sqlite3
import sys

include_message_text='--include-message-text' in sys.argv

home = Path.home()
db = sqlite3.connect(f'file:{home}/.codex/state_5.sqlite?mode=ro', uri=True)
db.row_factory = sqlite3.Row
rows = [dict(r) for r in db.execute("SELECT id,cwd,title,rollout_path,created_at,updated_at,archived FROM threads WHERE cwd LIKE '%3dgs-imc%' OR title LIKE '%GS-LSS%' ORDER BY created_at")]
known = {r['id'] for r in rows}
for base in (home / '.codex/sessions', home / '.codex/archived_sessions'):
    for path in base.rglob('*.jsonl'):
        try:
            first = json.loads(path.open().readline())
            meta = first.get('payload', {})
            if first.get('type') == 'session_meta' and '3dgs-imc' in meta.get('cwd', '') and meta.get('id') not in known:
                rows.append(dict(id=meta['id'], cwd=meta['cwd'], title='Unindexed session', rollout_path=str(path), created_at=0, updated_at=0, archived='archived_sessions' in str(path)))
                known.add(meta['id'])
        except (OSError, ValueError):
            pass
out = []
for row in rows:
    if row['id'] == '01a0de1d-6475-73e3-9ab2-1ed6e7ea9538':
        continue
    row['approval_mirror'] = row['title'].startswith('The following is the Codex agent history')
    row['title'] = row['title'][:300]
    path = Path(row['rollout_path'])
    row['exists'] = path.exists()
    row['size_bytes'] = path.stat().st_size if path.exists() else None
    counts = collections.Counter()
    messages, compacted, refs = [], [], set()
    if path.exists():
        for ln, line in enumerate(path.open(errors='replace'), 1):
            try:
                event = json.loads(line)
            except ValueError:
                counts['invalid_json'] += 1
                continue
            kind, p = event.get('type'), event.get('payload', {})
            counts[str(kind)] += 1
            if row['approval_mirror']:
                continue
            if kind == 'response_item' and p.get('type') == 'message' and p.get('role') in ('user', 'assistant'):
                txt = '\n'.join(c.get('text', '') for c in p.get('content', []) if isinstance(c, dict))
                if txt and not txt.startswith(('<environment_context>', '# AGENTS.md instructions', '<permissions instructions>')):
                    messages.append(dict(line=ln, timestamp=event.get('timestamp'), role=p.get('role'), channel=p.get('phase', p.get('channel')), **({'text':txt} if include_message_text else {})))
            if kind == 'compacted':
                compacted.append(dict(line=ln, timestamp=event.get('timestamp'), text=p.get('message', '')))
            for match in re.finditer(r'/data[12]/(?:zhangao|ZHANGAo_2024)/(?:outputs|repos/3dgs-imc-codex|DATA)[^\s\"\'<>`\\,;\)\]]*', line):
                refs.add(match.group(0))
    row['event_counts'] = dict(counts)
    row['messages'] = messages
    row['compactions'] = compacted
    row['path_references'] = sorted(refs)
    out.append(row)
print(json.dumps(out, ensure_ascii=False))

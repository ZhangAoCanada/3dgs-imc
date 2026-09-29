#!/usr/bin/env python3
"""Integrity-verify every best-checkpoint .ply in the KEEP manifest WITHOUT a slow full load:
parse the ply header (element vertex N + property list), compute expected binary data bytes, and
confirm the file is not truncated (actual size >= header + expected, within slack). Also confirm all
50 paper Table-1 cells and 14 Table-2 cells are backed. Prints a PASS/FAIL report. DELETES NOTHING."""
import json, os, struct
DEST = "/data1/zhangao/outputs/_cleanup"
M = json.load(open(f"{DEST}/keep_manifest.json"))["manifest"]

TYPE_BYTES = {"char":1,"uchar":1,"int8":1,"uint8":1,"short":2,"ushort":2,"int16":2,"uint16":2,
              "int":4,"uint":4,"int32":4,"uint32":4,"float":4,"float32":4,"double":8,"float64":8}

def header_check(path):
    """returns (ok, msg, nverts). ok=False if header unreadable / file truncated."""
    try:
        with open(path, "rb") as f:
            raw = b""
            while b"end_header" not in raw:
                chunk = f.read(4096)
                if not chunk: return (False, "no end_header (truncated header)", 0)
                raw += chunk
                if len(raw) > 1_000_000: return (False, "header too large", 0)
            hdr_end = raw.index(b"end_header") + len(b"end_header\n")
            header = raw[:hdr_end].decode("latin-1")
    except Exception as e:
        return (False, f"open/read error: {e}", 0)
    if "format binary_little_endian" not in header:
        # ascii ply: just confirm vertex>0 declared and file non-empty
        fmt_ascii = "format ascii" in header
    else:
        fmt_ascii = False
    nv = 0; bytes_per_v = 0; in_vertex = False
    for line in header.splitlines():
        t = line.split()
        if len(t) >= 3 and t[0] == "element":
            in_vertex = (t[1] == "vertex")
            if in_vertex: nv = int(t[2])
        elif len(t) >= 3 and t[0] == "property" and in_vertex:
            if t[1] == "list":  # variable; can't size-check precisely
                bytes_per_v = -1
            elif bytes_per_v != -1:
                bytes_per_v += TYPE_BYTES.get(t[1], 4)
    if nv <= 0: return (False, f"vertex count={nv}", nv)
    try: fsz = os.path.getsize(path)
    except OSError as e: return (False, f"stat error: {e}", nv)
    if fmt_ascii:
        return (fsz > hdr_end, "ascii ok" if fsz > hdr_end else "ascii empty body", nv)
    if bytes_per_v <= 0:
        return (fsz > hdr_end, "binary (list props; size-loose) ok", nv)
    expected = hdr_end + nv * bytes_per_v
    if fsz + 64 < expected:
        return (False, f"TRUNCATED: size {fsz} < expected {expected} (nv={nv}, bpv={bytes_per_v})", nv)
    return (True, f"ok nv={nv} bpv={bytes_per_v} size={fsz}", nv)

# verify unique plys
seen = {}
for m in M:
    seen.setdefault(m["ply"], m)
fails = []
for ply, m in seen.items():
    ok, msg, nv = header_check(ply)
    if not ok: fails.append((ply, msg, m))
print(f"unique KEEP plys checked: {len(seen)}   PASS: {len(seen)-len(fails)}   FAIL: {len(fails)}")
for ply, msg, m in fails:
    print(f"  !! FAIL [{m['category']} {m['cell']} {m['backs_metric']}]  {msg}\n     {ply}")

# coverage of paper cells
MIP8=["bicycle","bonsai","counter","garden","kitchen","room","stump","treehill"]
TNT=["truck_resize","train_resize"]; MILL=["building-small","residence-small","rubble-small","sci-art-small"]
ANG=[20.0,40.0,60.0,80.0,100.0]
t1_need = {f"{s}@{a}" for s in MIP8+TNT for a in ANG}
t1_have = {m["cell"] for m in M if m["category"]=="cg"}
t2_need = {f"{s}" for s in MIP8+TNT+MILL}
t2_have = {m["cell"] for m in M if m["category"]=="dc"}
print(f"\nTable-1 paper cells backed: {len(t1_need & t1_have)}/50   missing: {sorted(t1_need - t1_have)}")
print(f"Table-2 paper cells backed: {len(t2_need & t2_have)}/14   missing: {sorted(t2_need - t2_have)}")
print(f"Table-3 cells backed: {sorted({m['cell'] for m in M if m['category']=='t3'})}")
print("\nVERDICT:", "ALL KEEP PLYS INTACT + ALL PAPER CELLS BACKED" if not fails and not (t1_need-t1_have) and not (t2_need-t2_have)
      else "*** ISSUES ABOVE — DO NOT DELETE ***")

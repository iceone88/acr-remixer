"""Serialisers for the three asset kinds a generated stage rewrites:
pacenote DataTable rows, BakedDataContainer meshes and DMSplineComponent curves.

Each writer re-parses the original export to find the byte ranges it
replaces, so anything it does not understand is carried over verbatim."""
from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np

from ..extract import bakeddata as B
from ..extract.unversioned import Reader
from ..extract.zen import Package


def _hdr(n_values: int, skip: int = 0, last: bool = True) -> bytes:
    return struct.pack("<H", (skip & 0x7F) | (int(last) << 8) | (n_values << 9))


def _fname(names: list[str], s: str) -> bytes:
    return struct.pack("<II", names.index(s), 0)


def _fstring(s: str) -> bytes:
    if not s:
        return struct.pack("<i", 0)
    b = s.encode("latin-1") + b"\0"
    return struct.pack("<i", len(b)) + b


# --------------------------------------------------------------- pacenotes
@dataclass
class NoteOut:
    distance: float
    length: float
    tokens: list[str]
    flag: int = 0
    line_id: str = ""


def pacenote_payload(pk: Package, notes: list[NoteOut]) -> tuple[bytes, list[str]]:
    """Returns (export payload, full name list incl. any appended names)."""
    e = pk.exports[0]
    d = pk.d
    r = Reader(d, e.data_offset)
    for _ in r.header():
        r.i32()
    r.i32()
    count_pos = r.p
    n = r.i32()
    # skip original rows to find the suffix
    for _ in range(n):
        r.fname(pk.names)
        for f in r.header():
            if f in (0, 1):
                r.f32()
            elif f == 2:
                for _g in r.header():
                    for _k in range(r.u32()):
                        r.fname(pk.names)
            elif f == 3:
                r.u8()
            elif f == 4:
                r.fstring()
            elif f == 5:
                r.i32()
    prefix = d[e.data_offset: count_pos]
    suffix = d[r.p: e.data_offset + e.serial_size]

    names = list(pk.names)
    def need(s):
        if s not in names:
            names.append(s)
    rows = bytearray(struct.pack("<i", len(notes)))
    for k, note in enumerate(notes):
        rn = f"R{k:04d}"
        need(rn)
        for t in note.tokens:
            need(t)
    for k, note in enumerate(notes):
        rows += _fname(names, f"R{k:04d}")
        rows += _hdr(5)
        rows += struct.pack("<ff", note.distance, note.length)
        rows += _hdr(1) + struct.pack("<I", len(note.tokens))
        for t in note.tokens:
            rows += _fname(names, t)
        rows += struct.pack("<B", note.flag)
        rows += _fstring(note.line_id)
    return bytes(prefix) + bytes(rows) + bytes(suffix), names


# --------------------------------------------------------------- baked data
@dataclass
class MeshOut:
    vertices: np.ndarray       # (N,3) float, baked frame (x, up, y)
    indices: np.ndarray        # (M,) uint32
    material_ref: int          # raw package index (e.g. -4)
    soft_pkg: tuple[int, int]  # raw FName pairs of the soft object path
    soft_asset: tuple[int, int]
    soft_sub: bytes            # raw length-prefixed string bytes
    extra: np.ndarray          # the 4 trailing floats


def read_baked_raw(pk: Package):
    """Original meshes with raw references + byte ranges for rewriting."""
    e = pk.exports[0]
    d = pk.d
    r = Reader(d, e.data_offset)
    top = r.header()
    assert top[0] == 0
    start_meshes = r.p
    n = r.u32()
    meshes = []
    transform_bytes = None
    for _ in range(n):
        verts = np.zeros((0, 3), np.float32); extra = np.zeros(0, np.float32)
        idx = np.zeros(0, np.uint32); mat = 0; sp = (0, 0); sa = (0, 0); sub = struct.pack("<i", 0)
        for f in r.header():
            if f == 0:
                t0 = r.p
                rot, tr, sc = B._transform(r)
                if not (np.allclose(rot, (0, 0, 0, 1)) and np.allclose(tr, 0) and np.allclose(sc, 1)):
                    raise NotImplementedError("non-identity baked mesh transform")
                transform_bytes = d[t0: r.p]
            elif f == 1:
                verts = B._array(r, np.float32, 3).astype(np.float64)
            elif f == 2:
                extra = B._array(r, np.float32).copy()
            elif f == 3:
                idx = B._array(r, np.uint32).copy()
            elif f == 4:
                mat = r.i32()
            elif f == 5:
                sp = struct.unpack_from("<II", d, r.p); r.p += 8
                sa = struct.unpack_from("<II", d, r.p); r.p += 8
                n_sub = r.i32()
                sub = d[r.p - 4: r.p + n_sub]          # raw length-prefixed bytes (no terminator)
                r.p += n_sub
        meshes.append(MeshOut(verts, idx, mat, sp, sa, sub, extra))
    tail = d[r.p: e.data_offset + e.serial_size]
    k = tail.find(struct.pack("<I", n))
    tail_prefix = tail[:k]
    table_len = 4 + 7 + 10 * (n - 1) if n else 4
    tail_suffix = tail[k + table_len:]
    head = d[e.data_offset: start_meshes]
    return meshes, head, transform_bytes, tail_prefix, tail_suffix


def baked_payload(head: bytes, transform_bytes: bytes, meshes: list[MeshOut],
                  tail_prefix: bytes, tail_suffix: bytes) -> bytes:
    out = bytearray(head)
    out += struct.pack("<I", len(meshes))
    for m in meshes:
        out += _hdr(6)
        out += transform_bytes
        v = np.ascontiguousarray(m.vertices, dtype=np.float32).ravel()
        out += struct.pack("<I", len(v)) + v.tobytes()
        ex = np.ascontiguousarray(m.extra, dtype=np.float32)
        out += struct.pack("<I", len(ex)) + ex.tobytes()
        ix = np.ascontiguousarray(m.indices, dtype=np.uint32).ravel()
        out += struct.pack("<I", ix.size) + ix.tobytes()
        out += struct.pack("<i", m.material_ref)
        out += struct.pack("<II", *m.soft_pkg) + struct.pack("<II", *m.soft_asset) + m.soft_sub
    out += tail_prefix
    out += struct.pack("<I", len(meshes))
    for i in range(len(meshes)):
        out += (b"\x80\x05\x01" + struct.pack("<i", 1)) if i == 0 else (b"\x00\x05" + struct.pack("<ii", i, 1))
    out += tail_suffix
    return bytes(out)


# --------------------------------------------------------------- splines
def _skip_curve(r: Reader, vw: int, dbl: bool) -> tuple[list[int], bytes]:
    """Skip one FInterpCurve; return (its header indices, bytes of the fields
    after Points: bIsLooped u8 / LoopKeyOffset f32)."""
    h = r.header()
    if not h:
        return h, b""
    if h[0] == 0:
        n = r.u32()
        for _ in range(n):
            for f in r.header():
                if f == 0:
                    r.f32()
                elif f in (1, 2, 3):
                    r.raw(vw * (8 if dbl else 4))
                elif f == 4:
                    r.u8()
    start = r.p
    for f in h:
        if f == 1:
            r.u8()
        elif f == 2:
            r.f32()
        elif f != 0:
            raise ValueError(f"unexpected FInterpCurve field {f}")
    return h, r.d[start:r.p]


def _hdr_for(indices: list[int]) -> bytes:
    """Encode an unversioned header for exactly these (sorted) indices."""
    frags = []
    prev = 0
    i = 0
    idx = sorted(indices)
    while i < len(idx):
        skip = idx[i] - prev
        while skip > 127:
            frags.append([127, 0]); skip -= 127
        run = 1
        while i + run < len(idx) and idx[i + run] == idx[i] + run and run < 127:
            run += 1
        frags.append([skip, run])
        prev = idx[i] + run
        i += run
    if not frags:
        frags = [[0, 0]]
    out = b""
    for k, (skip, run) in enumerate(frags):
        out += struct.pack("<H", skip | (int(k == len(frags) - 1) << 8) | (run << 9))
    return out


def _quat_from_dir(t: np.ndarray) -> np.ndarray:
    """UE quaternion (x,y,z,w) rotating +X onto direction t (yaw + pitch, no roll)."""
    yaw = np.arctan2(t[1], t[0])
    pitch = np.arctan2(t[2], np.hypot(t[0], t[1]))
    cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
    cp, sp = np.cos(-pitch / 2), np.sin(-pitch / 2)   # UE pitch: rotation about Y, nose-up positive
    # q = qYaw * qPitch
    return np.array([-sy * sp, cy * sp, sy * cp, cy * cp])


def spline_payload(pk: Package, export_name: str, pts_cm: np.ndarray) -> bytes:
    """Replace the SplineCurves of a DMSplineComponent with control points pts_cm (N,3)."""
    e = next(x for x in pk.exports if x.name == export_name)
    d = pk.d
    r = Reader(d, e.data_offset)
    top = r.header()
    if top and top[0] == 0:
        r.i32()
    curves_start = r.p
    cv = r.header()
    assert cv[:3] == [0, 1, 2] and set(cv) <= {0, 1, 2, 3}, cv
    has_reparam = 3 in cv
    curve_meta = [_skip_curve(r, 3, True), _skip_curve(r, 4, True), _skip_curve(r, 3, True)]
    if has_reparam:
        curve_meta.append(_skip_curve(r, 1, False))
    prefix = d[e.data_offset: curves_start]
    suffix = d[r.p: e.data_offset + e.serial_size]

    P = np.asarray(pts_cm, float)
    n = len(P)
    T = np.zeros_like(P)
    T[1:-1] = (P[2:] - P[:-2]) / 2
    T[0] = P[1] - P[0]
    T[-1] = P[-1] - P[-2]

    out = bytearray(prefix)
    out += _hdr(4 if has_reparam else 3)            # FSplineCurves: Position, Rotation, Scale[, Reparam]
    def curve_head(k):
        h, _ = curve_meta[k]
        return _hdr_for(sorted(set(h) | {0}))

    def curve_tail(k):
        return curve_meta[k][1]

    # Position
    out += curve_head(0) + struct.pack("<I", n)
    for i in range(n):
        out += _hdr(5) + struct.pack("<f", float(i))
        out += struct.pack("<3d", *P[i]) + struct.pack("<3d", *T[i]) + struct.pack("<3d", *T[i]) + b"\x03"
    out += curve_tail(0)
    # Rotation
    out += curve_head(1) + struct.pack("<I", n)
    for i in range(n):
        q = _quat_from_dir(T[i])
        out += _hdr(5) + struct.pack("<f", float(i))
        out += struct.pack("<4d", *q) * 3 + b"\x01"
    out += curve_tail(1)
    # Scale
    out += curve_head(2) + struct.pack("<I", n)
    for i in range(n):
        out += _hdr(5) + struct.pack("<f", float(i))
        out += struct.pack("<3d", 1, 1, 1) + struct.pack("<3d", 0, 0, 0) * 2 + b"\x01"
    out += curve_tail(2)
    if not has_reparam:
        return bytes(out) + suffix
    # ReparamTable: 10 steps per segment, (distance -> input key)
    steps = 10
    entries = []
    dist = 0.0
    for i in range(n - 1):
        p0, p1, m0, m1 = P[i], P[i + 1], T[i], T[i + 1]
        prev = p0
        for k in range(steps):
            t = k / steps
            if k:
                h00 = 2 * t**3 - 3 * t**2 + 1; h10 = t**3 - 2 * t**2 + t
                h01 = -2 * t**3 + 3 * t**2; h11 = t**3 - t**2
                cur = h00 * p0 + h10 * m0 + h01 * p1 + h11 * m1
                dist += float(np.linalg.norm(cur - prev)); prev = cur
            entries.append((dist, i + t))
        dist += float(np.linalg.norm(p1 - prev))
    entries.append((dist, float(n - 1)))
    out += curve_head(3) + struct.pack("<I", len(entries))
    for dval, key in entries:
        # InVal, OutVal only; tangents/mode zero -> fragment: 2 values then skip
        out += struct.pack("<H", (2 << 9) | (1 << 8)) + struct.pack("<ff", dval, key)
    out += curve_tail(3)
    out += suffix
    return bytes(out)

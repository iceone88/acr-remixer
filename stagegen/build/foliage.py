"""Instanced (foliage / PCG grass) components: hide instances in place.

Instanced static mesh components serialise their instances as a bulk array of
FMatrix (16 doubles, 128 bytes each): `u32 128, i32 count, count*128 bytes`.
Hiding = zero scale (rows 0-2 zeroed, translation kept), so the count, the
cluster tree and all bounds stay valid.
"""
from __future__ import annotations

import struct

import numpy as np


def find_instances(b: bytes):
    """(offset of the first matrix, count) or None."""
    k = 0
    pat = struct.pack("<I", 128)
    while True:
        k = b.find(pat, k)
        if k < 0 or k + 8 > len(b):
            return None
        n = struct.unpack_from("<i", b, k + 4)[0]
        if 0 < n < 5_000_000 and k + 8 + 128 * n <= len(b):
            M = np.frombuffer(b, "<f8", 16 * n, k + 8).reshape(n, 4, 4)
            if np.allclose(M[:, 3, 3], 1.0) and np.allclose(M[:, :3, 3], 0.0):
                return k + 8, n
        k += 1


def instance_positions(b: bytes):
    hit = find_instances(b)
    if hit is None:
        return None
    off, n = hit
    return np.frombuffer(b, "<f8", 16 * n, off).reshape(n, 4, 4)[:, 3, :3].copy()


def hide_instances(b: bytes, mask=None) -> tuple[bytes, int]:
    """Zero-scale the instances where mask is True (all if None). Returns (payload, hidden)."""
    hit = find_instances(b)
    if hit is None:
        return b, 0
    off, n = hit
    M = np.frombuffer(b, "<f8", 16 * n, off).reshape(n, 4, 4).copy()
    m = np.ones(n, bool) if mask is None else np.asarray(mask, bool)
    M[m, :3, :] = 0.0
    out = bytearray(b)
    out[off: off + 128 * n] = M.astype("<f8").tobytes()
    return bytes(out), int(m.sum())

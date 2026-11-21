"""Minimal read-only IoStore (.utoc/.ucas) reader for UE 5.x containers (TOC v8).

Only what acr-stagegen needs: list files via the directory index and read a
chunk's bytes (Oodle-compressed blocks are decoded with a locally installed
oo2core DLL through ctypes). Never writes to the game folder.
"""
from __future__ import annotations

import ctypes
import glob
import os
import struct
from dataclasses import dataclass
from functools import lru_cache

MAGIC = b"-==--==--==--==-"

# Oodle 2.9 is needed for UE 5.6 Kraken streams (2.4 cannot decode them).
# Only DLLs already on this machine are used; set ACR_OODLE_DLL to override.
OODLE_CANDIDATES = [
    os.environ.get("ACR_OODLE_DLL", ""),
    r"E:\Sando-Rally-Stage-Export-Tool\bin\retoc\oo2core_9_win64.dll",
    r"E:\_old_exports\bin\retoc\oo2core_9_win64.dll",
    r"D:\ME\Mass Effect Legendary Edition\Game\ME3\Binaries\Win64\oo2core_8_win64.dll",
]


@lru_cache(maxsize=1)
def _oodle():
    for p in OODLE_CANDIDATES:
        if p and os.path.isfile(p):
            dll = ctypes.WinDLL(p)
            f = dll.OodleLZ_Decompress
            f.restype = ctypes.c_int64
            f.argtypes = [ctypes.c_void_p, ctypes.c_int64, ctypes.c_void_p, ctypes.c_int64,
                          ctypes.c_int, ctypes.c_int, ctypes.c_int,
                          ctypes.c_void_p, ctypes.c_int64, ctypes.c_void_p, ctypes.c_void_p,
                          ctypes.c_void_p, ctypes.c_int64, ctypes.c_int]
            return f
    raise RuntimeError("No oo2core_*_win64.dll found; set ACR_OODLE_DLL")


def oodle_decompress(src: bytes, raw_size: int) -> bytes:
    out = ctypes.create_string_buffer(raw_size)
    n = _oodle()(src, len(src), out, raw_size, 1, 0, 0, None, 0, None, None, None, 0, 3)
    if n != raw_size:
        raise RuntimeError(f"Oodle decompress failed ({n} != {raw_size})")
    return out.raw


def _u40be(b: bytes) -> int:
    return int.from_bytes(b, "big")


@dataclass
class Block:
    offset: int
    csize: int
    usize: int
    method: int


class Container:
    def __init__(self, utoc_path: str):
        self.utoc = utoc_path
        self.ucas = utoc_path[:-5] + ".ucas"
        with open(utoc_path, "rb") as f:
            d = f.read()
        if d[:16] != MAGIC:
            raise ValueError("not a utoc")
        (self.version,) = struct.unpack_from("<B", d, 16)
        (hsize, n_entries, n_blocks, block_entry_size, n_methods, method_len,
         self.block_size, dir_size, self.partition_count) = struct.unpack_from("<9I", d, 20)
        self.container_id, = struct.unpack_from("<Q", d, 56)
        self.flags = d[80]
        n_seeds, = struct.unpack_from("<I", d, 84)
        self.partition_size, = struct.unpack_from("<Q", d, 88)
        n_noperfect, = struct.unpack_from("<I", d, 96)
        p = hsize
        self.chunk_ids = [d[p + 12 * i: p + 12 * i + 12] for i in range(n_entries)]
        p += 12 * n_entries
        self.offlen = []
        for i in range(n_entries):
            e = d[p + 10 * i: p + 10 * i + 10]
            self.offlen.append((_u40be(e[:5]), _u40be(e[5:])))
        p += 10 * n_entries
        p += 4 * n_seeds + 4 * n_noperfect
        self.blocks = []
        for i in range(n_blocks):
            e = d[p + 12 * i: p + 12 * i + 12]
            off = int.from_bytes(e[0:5], "little")
            cs = int.from_bytes(e[5:8], "little")
            us = int.from_bytes(e[8:11], "little")
            self.blocks.append(Block(off, cs, us, e[11]))
        p += block_entry_size * n_blocks
        self.methods = ["None"] + [d[p + method_len * i: p + method_len * (i + 1)].split(b"\0")[0].decode()
                                   for i in range(n_methods)]
        p += method_len * n_methods
        if self.flags & 4:  # signed
            hs, = struct.unpack_from("<i", d, p)
            p += 4 + 2 * hs + 20 * n_blocks
        self.files: dict[str, int] = {}
        if self.flags & 8 and dir_size:
            self._read_dir(d[p: p + dir_size])
        self.chunk_index = {cid: i for i, cid in enumerate(self.chunk_ids)}

    def _read_dir(self, d: bytes):
        p = 0

        def fstring():
            nonlocal p
            n, = struct.unpack_from("<i", d, p); p += 4
            if n >= 0:
                s = d[p: p + n].rstrip(b"\0").decode("latin-1"); p += n
            else:
                s = d[p: p - 2 * n].decode("utf-16-le").rstrip("\0"); p += -2 * n
            return s

        mount = fstring()
        nd, = struct.unpack_from("<I", d, p); p += 4
        dirs = [struct.unpack_from("<4I", d, p + 16 * i) for i in range(nd)]; p += 16 * nd
        nf, = struct.unpack_from("<I", d, p); p += 4
        files = [struct.unpack_from("<3I", d, p + 12 * i) for i in range(nf)]; p += 12 * nf
        ns, = struct.unpack_from("<I", d, p); p += 4
        strings = [fstring() for _ in range(ns)]
        NONE = 0xFFFFFFFF
        stack = [(0, mount.lstrip("/").replace("../../../", ""))]
        while stack:
            di, path = stack.pop()
            name, child, sib, ffile = dirs[di]
            here = path if name == NONE else f"{path}{strings[name]}/"
            fi = ffile
            while fi != NONE:
                fn, nxt, user = files[fi]
                self.files[here + strings[fn]] = user
                fi = nxt
            if child != NONE:
                stack.append((child, here))
            if sib != NONE:
                stack.append((sib, path))

    def read_entry(self, idx: int) -> bytes:
        off, length = self.offlen[idx]
        bs = self.block_size
        first, last = off // bs, (off + length - 1) // bs
        parts = []
        with open(self.ucas, "rb") as f:
            for bi in range(first, last + 1):
                b = self.blocks[bi]
                f.seek(b.offset)
                raw = f.read((b.csize + 15) & ~15)[: b.csize]
                m = self.methods[b.method]
                if m == "None":
                    parts.append(raw[: b.usize])
                elif m.lower() == "oodle":
                    parts.append(oodle_decompress(raw, b.usize))
                else:
                    raise RuntimeError(f"unsupported compression {m}")
        data = b"".join(parts)
        start = off - first * bs
        return data[start: start + length]

    def read(self, path: str) -> bytes:
        return self.read_entry(self.files[path])


class GameFS:
    """All containers of an install, base game first, mods last (mods override)."""

    def __init__(self, paks_dir: str, include_mods: bool = False):
        paths = sorted(glob.glob(os.path.join(paks_dir, "*.utoc")))
        if include_mods:
            # never read our own generated containers back as source data
            paths += sorted(p for p in glob.glob(os.path.join(paks_dir, "~mods", "*.utoc"))
                            if "_Stagegen_" not in os.path.basename(p))
        self.containers = [Container(p) for p in paths]
        self.index: dict[str, tuple[Container, int]] = {}
        self.by_chunk: dict[bytes, tuple[Container, int]] = {}
        for c in self.containers:
            for fpath, i in c.files.items():
                self.index[fpath] = (c, i)
            for cid, i in c.chunk_index.items():
                self.by_chunk[cid] = (c, i)

    def find(self, substr: str) -> list[str]:
        s = substr.lower()
        return sorted(p for p in self.index if s in p.lower())

    def read(self, path: str) -> bytes:
        c, i = self.index[path]
        return c.read_entry(i)

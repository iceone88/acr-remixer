"""Write a small uncompressed IoStore container (.utoc/.ucas) + companion .pak.

Layout mirrors what UE 5.6 mod containers use (TOC v8, flags = Indexed,
no compression, no perfect hash, 64 KiB blocks, one partition). Chunk hashes
are left zero (only checked for signed containers).
"""
from __future__ import annotations

import struct
from dataclasses import dataclass

MAGIC = b"-==--==--==--==-"
BLOCK = 65536


@dataclass
class ChunkIn:
    chunk_id: bytes        # 12 bytes
    data: bytes
    path: str | None = None   # mount-relative path for the directory index


def _fstring(s: str) -> bytes:
    b = s.encode("latin-1") + b"\0"
    return struct.pack("<i", len(b)) + b


def _dir_index(mount: str, files: list[tuple[str, int]]) -> bytes:
    strings: list[str] = []
    sidx: dict[str, int] = {}

    def S(s):
        if s not in sidx:
            sidx[s] = len(strings); strings.append(s)
        return sidx[s]

    NONE = 0xFFFFFFFF
    dirs = [[NONE, NONE, NONE, NONE]]          # name, first_child, next_sibling, first_file
    fentries: list[list[int]] = []              # name, next_file, user
    children: dict[tuple[int, str], int] = {}
    for path, user in files:
        parts = path.split("/")
        d = 0
        for name in parts[:-1]:
            key = (d, name)
            if key not in children:
                nd = len(dirs)
                dirs.append([S(name), NONE, dirs[d][1], NONE])
                dirs[d][1] = nd
                children[key] = nd
            d = children[key]
        fi = len(fentries)
        fentries.append([S(parts[-1]), dirs[d][3], user])
        dirs[d][3] = fi
    out = bytearray(_fstring(mount))
    out += struct.pack("<I", len(dirs))
    for e in dirs:
        out += struct.pack("<4I", *e)
    out += struct.pack("<I", len(fentries))
    for e in fentries:
        out += struct.pack("<3I", *e)
    out += struct.pack("<I", len(strings))
    for s in strings:
        out += _fstring(s)
    return bytes(out)


def write_container(base_path: str, container_id: int, chunks: list[ChunkIn],
                    pak_template: bytes, mount: str = "../../../") -> None:
    """Writes base_path + .utoc/.ucas/.pak."""
    offlen = []
    blocks = []                      # (ucas_offset, size)
    ucas_pos = 0
    uoff = 0
    with open(base_path + ".ucas", "wb") as ucas:
        for c in chunks:
            offlen.append((uoff, len(c.data)))
            for p in range(0, max(len(c.data), 1), BLOCK):
                piece = c.data[p: p + BLOCK]
                blocks.append((ucas_pos, len(piece)))
                ucas.write(piece)
                ucas_pos += len(piece)
                # keep ucas blocks 16-byte aligned like UE does
                pad = (-ucas_pos) % 16
                if pad:
                    ucas.write(b"\0" * pad); ucas_pos += pad
            uoff += ((len(c.data) + BLOCK - 1) // BLOCK) * BLOCK or BLOCK

    files = [(c.path, i) for i, c in enumerate(chunks) if c.path]
    dindex = _dir_index(mount, files)

    hdr = bytearray(144)
    hdr[0:16] = MAGIC
    hdr[16] = 8                                       # version
    struct.pack_into("<I", hdr, 20, 144)              # header size
    struct.pack_into("<I", hdr, 24, len(chunks))
    struct.pack_into("<I", hdr, 28, len(blocks))
    struct.pack_into("<I", hdr, 32, 12)
    struct.pack_into("<I", hdr, 36, 0)                # compression method count
    struct.pack_into("<I", hdr, 40, 32)
    struct.pack_into("<I", hdr, 44, BLOCK)
    struct.pack_into("<I", hdr, 48, len(dindex))
    struct.pack_into("<I", hdr, 52, 1)                # partition count
    struct.pack_into("<Q", hdr, 56, container_id)
    hdr[80] = 0x08                                    # Indexed
    struct.pack_into("<I", hdr, 84, 0)                # perfect hash seeds
    struct.pack_into("<Q", hdr, 88, 0xFFFFFFFFFFFFFFFF)
    struct.pack_into("<I", hdr, 96, 0)

    out = bytearray(hdr)
    for c in chunks:
        assert len(c.chunk_id) == 12
        out += c.chunk_id
    for off, ln in offlen:
        out += off.to_bytes(5, "big") + ln.to_bytes(5, "big")
    for off, size in blocks:
        out += off.to_bytes(5, "little") + size.to_bytes(3, "little") + size.to_bytes(3, "little") + b"\0"
    out += dindex
    out += b"\0" * (24 * len(chunks))                 # chunk metas (hash + flags)
    with open(base_path + ".utoc", "wb") as f:
        f.write(out)
    with open(base_path + ".pak", "wb") as f:
        f.write(pak_template)


def container_header_chunk_id(container_id: int) -> bytes:
    return struct.pack("<Q", container_id) + b"\0\0\0" + bytes([6])


def retarget_container_header(data: bytes, container_id: int) -> bytes:
    if data[:4] != b"nCoI":
        raise ValueError("not a container header")
    return data[:8] + struct.pack("<Q", container_id) + data[16:]

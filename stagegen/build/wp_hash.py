"""World Partition runtime spatial hash (UE 5.6, cooked, unversioned).

    WorldPartitionRuntimeSpatialHash
      ...                   (2 small values)
      TArray<FSpatialHashStreamingGrid> StreamingGrids     (count at payload offset 4)
        [0] GridName FName  [1] Origin FVector  [2] CellSize i32  [3] LoadingRange f32
        [4] bBlockOnSlowStreaming  [5] DebugColor FLinearColor
        [6] GridLevels TArray<FSpatialHashStreamingGridLevel>
              [0] LayerCells TArray<{[0] GridCells TArray<cell ref>}>
              [1] LayerCellsMapping TMap<int64 coord, int32 layer cell index>
        [7..] GridBounds etc. (copied verbatim)

Cells are found at runtime by grid coordinate around the streaming source,
i.e. by where their actors *used* to be. `all_cells_in_top_level` registers
every cell in the highest occupied level's cell, the one that already holds
the variant route cells and is loaded wherever the stage is driven.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass

from ..extract.unversioned import Reader
from .writers import _hdr_for

GRID_FIELD_SIZES = {0: 8, 1: 24, 2: 4, 3: 4, 4: 1, 5: 16}


@dataclass
class Level:
    layer_cells: list          # [[ref, ...], ...]
    mapping: list              # [(coord, index), ...]


def _parse_levels(r: Reader):
    levels = []
    for _ in range(r.i32()):
        h = r.header()
        cells, mapping = [], []
        for f in h:
            if f == 0:
                for _ in range(r.i32()):
                    hh = r.header()
                    refs = []
                    for g in hh:
                        if g != 0:
                            raise ValueError(f"unknown layer cell field {g}")
                        refs = [r.i32() for _ in range(r.i32())]
                    cells.append(refs)
            elif f == 1:
                if r.i32() != 0:
                    raise ValueError("TMap with removed keys")
                for _ in range(r.i32()):
                    k, = struct.unpack_from("<q", r.d, r.p); r.p += 8
                    mapping.append((k, r.i32()))
            else:
                raise ValueError(f"unknown grid level field {f}")
        levels.append(Level(cells, mapping))
    return levels


def _write_levels(levels) -> bytes:
    out = bytearray(struct.pack("<i", len(levels)))
    for lv in levels:
        out += _hdr_for([0, 1])
        out += struct.pack("<i", len(lv.layer_cells))
        for refs in lv.layer_cells:
            out += _hdr_for([0]) + struct.pack("<i", len(refs)) + b"".join(struct.pack("<i", x) for x in refs)
        out += struct.pack("<ii", 0, len(lv.mapping))
        for k, v in lv.mapping:
            out += struct.pack("<qi", k, v)
    return bytes(out)


def read_grid(payload: bytes):
    """-> (levels, start of GridLevels, end of GridLevels, loading range offset)."""
    if struct.unpack_from("<i", payload, 4)[0] != 1:
        raise ValueError("expected exactly one streaming grid")
    r = Reader(payload, 8)
    h = r.header()
    lr_off = None
    for f in h:
        if f == 6:
            start = r.p
            levels = _parse_levels(r)
            return levels, start, r.p, lr_off
        if f not in GRID_FIELD_SIZES:
            raise ValueError(f"unknown grid field {f} before GridLevels")
        if f == 3:
            lr_off = r.p
        r.p += GRID_FIELD_SIZES[f]
    raise ValueError("grid has no GridLevels")


def all_cells_in_top_level(payload: bytes) -> tuple[bytes, dict]:
    levels, a, b, _ = read_grid(payload)
    refs = []
    for lv in levels:
        for cell in lv.layer_cells:
            refs += [x for x in cell if x not in refs]
    top = max(i for i, lv in enumerate(levels) if lv.layer_cells)
    if len(levels[top].layer_cells) != 1 or len(levels[top].mapping) != 1:
        raise ValueError("top grid level has more than one cell")
    coord = levels[top].mapping[0][0]
    new = [Level([], []) for _ in levels]
    new[top] = Level([refs], [(coord, 0)])
    stats = {"cells": len(refs), "level": top, "before": [sum(map(len, lv.layer_cells)) for lv in levels]}
    return payload[:a] + _write_levels(new) + payload[b:], stats

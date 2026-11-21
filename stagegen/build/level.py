"""Exact, per-class decoding and patching of root scene-component transforms
in a cooked level (UE 5.6, unversioned properties).

Unversioned property indices count the most-derived class first, so the
SceneComponent block (RelativeLocation L, RelativeRotation L+1,
RelativeScale3D L+2, AttachParent L-5) sits at a class-specific index. The
rules below were derived from the WRC Generations stage levels and are only
applied to components whose decoded values pass sanity checks.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np

from ..extract.unversioned import Reader
from ..extract.zen import Package
from .writers import _hdr_for


@dataclass(frozen=True)
class Rule:
    L: int
    trailer: int
    extras: dict          # index -> byte size, for indices > L+2 seen in practice


# keyed by a representative export name of the class
RULES = {
    "StaticMeshComponent": Rule(163, 20, {171: 1, 177: 1}),
    "NewDecalComponent": Rule(15, 8, {}),
    "SceneComponent": Rule(5, 8, {33: 1, 41: 1}),
    "BoxComponent": Rule(125, 8, {}),
    "VirtualTextureComponent": Rule(26, 8, {}),
}
_REPRESENTATIVE = {
    "StaticMeshComponent0": "StaticMeshComponent",
    "NewDecalComponent": "NewDecalComponent",
    "SceneComp": "SceneComponent",
    "DefaultSceneRoot": "SceneComponent",
    "Trigger": "BoxComponent",
    "VirtualTextureComponent": "VirtualTextureComponent",
}


@dataclass
class Comp:
    export: int
    owner: int
    rule: Rule
    present: list
    hdr_end: int
    block_start: int
    block_end: int
    loc: np.ndarray | None
    rot: np.ndarray | None
    scale: np.ndarray | None

    @property
    def root(self) -> bool:
        return (self.rule.L - 5) not in self.present


def _class_rules(pk: Package, votes: dict | None = None) -> dict[int, Rule]:
    votes = {} if votes is None else votes
    for e in pk.exports:
        key = _REPRESENTATIVE.get(e.name)
        if key:
            votes.setdefault(e.class_index, {}).setdefault(key, 0)
            votes[e.class_index][key] += 1
    return {cls: RULES[max(v, key=v.get)] for cls, v in votes.items()}


def learn_rules(pkgs) -> dict[int, Rule]:
    """Class rules voted over several packages. Script class import indices are
    global hashes, so a rule learned in one World Partition cell applies to all."""
    votes: dict = {}
    for pk in pkgs:
        _class_rules(pk, votes)
    return {cls: RULES[max(v, key=v.get)] for cls, v in votes.items()}


def _sane(loc, rot, scale) -> bool:
    if loc is not None and not (np.all(np.isfinite(loc)) and np.all(np.abs(loc) < 2e6)):
        return False
    if rot is not None and not (np.all(np.isfinite(rot)) and np.all(np.abs(rot) <= 720)):
        return False
    if scale is not None and not (np.all(np.isfinite(scale)) and np.all(np.abs(scale) < 1e7)):
        return False
    return True


def decode_components(pk: Package, rules: dict | None = None) -> list[Comp]:
    rules = _class_rules(pk) if rules is None else rules
    d = pk.d
    out = []
    for i, e in enumerate(pk.exports):
        rule = rules.get(e.class_index)
        if rule is None:
            continue
        r = Reader(d, e.data_offset)
        try:
            h = r.header()
        except Exception:     # noqa: BLE001
            continue
        L = rule.L
        if not ({L, L + 1, L + 2} & set(h)):
            continue
        after = [x for x in h if x > L + 2]
        if any(x not in rule.extras for x in after):
            continue
        end = e.data_offset + e.serial_size
        block_end = end - rule.trailer - sum(rule.extras[x] for x in after)
        n_trip = sum(x in h for x in (L, L + 1, L + 2))
        block_start = block_end - 24 * n_trip
        if block_start < r.p:
            continue
        p = block_start
        loc = rot = sc = None
        if L in h:
            loc = np.array(struct.unpack_from("<3d", d, p)); p += 24
        if L + 1 in h:
            rot = np.array(struct.unpack_from("<3d", d, p)); p += 24
        if L + 2 in h:
            sc = np.array(struct.unpack_from("<3d", d, p)); p += 24
        if not _sane(loc, rot, sc):
            continue
        out.append(Comp(i, e.outer, rule, h, r.p, block_start, block_end, loc, rot, sc))
    return out


def patched_payload(pk: Package, c: Comp, loc: np.ndarray, rot: np.ndarray,
                    scale: np.ndarray | None = None) -> bytes:
    """Payload of component c with RelativeLocation/Rotation (and optionally
    Scale) set; adds the properties to the header if they were default."""
    e = pk.exports[c.export]
    d = pk.d
    L = c.rule.L
    present = set(c.present) | {L, L + 1}
    keep_scale = (L + 2 in c.present) or scale is not None
    if keep_scale:
        present.add(L + 2)
    sc = scale if scale is not None else (c.scale if c.scale is not None else np.ones(3))
    block = struct.pack("<3d", *loc) + struct.pack("<3d", *rot)
    if keep_scale:
        block += struct.pack("<3d", *sc)
    head = _hdr_for(sorted(present))
    return head + d[c.hdr_end: c.block_start] + block + d[c.block_end: e.data_offset + e.serial_size]

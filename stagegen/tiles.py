"""Tile-aligned recombination for base-game (World Partition) stages.

The base game ships every stage's road + terrain pre-split on a square grid
(`SM_<Level>Track<YY><XX>`, 350 m cells). Pieces are cut exactly where the
route crosses a grid line and joined so the two grid lines coincide: road and
terrain meshes then meet edge to edge, with at most a small kink. Each tile
belongs to at most one piece; pieces that are not neighbours keep enough
distance that no piece's (clipped) terrain reaches another piece's road.
"""
from __future__ import annotations

import collections
import random
import re
from dataclasses import dataclass, field

import numpy as np

from .geometry import headings, rot_z, wrap_angle
from .model import Piece, Segment, SourceStage

TILE_RE = re.compile(rb"Track(\d\d)(\d\d)")


@dataclass
class TileGrid:
    cell: float
    x0: float
    y0: float
    tiles: set            # {(iy, ix)}

    @staticmethod
    def from_baked(meshes) -> "TileGrid":
        bounds: dict[tuple, list] = collections.defaultdict(list)
        for m in meshes:
            mm = TILE_RE.search(m.soft_sub)
            if mm and len(m.vertices):
                v = m.vertices[:, [0, 2, 1]]
                bounds[(int(mm.group(1)), int(mm.group(2)))].append((v.min(0), v.max(0)))
        lo = {k: np.min([b[0] for b in v], 0) for k, v in bounds.items()}
        hi = {k: np.max([b[1] for b in v], 0) for k, v in bounds.items()}
        ext = max(float(max(hi[k][0] - lo[k][0], hi[k][1] - lo[k][1])) for k in lo)
        cell = round(ext / 10) * 10
        x0 = collections.Counter(round(lo[k][0] - cell * k[1]) for k in lo).most_common(1)[0][0]
        y0 = collections.Counter(round(lo[k][1] - cell * k[0]) for k in lo).most_common(1)[0][0]
        g = TileGrid(float(cell), float(x0), float(y0), set(lo))
        bad = [k for k in lo if not (np.all(lo[k][:2] >= g.lo(k) - 2) and np.all(hi[k][:2] <= g.lo(k) + cell + 2))]
        if len(bad) > len(lo) // 4:
            raise ValueError(f"tile grid does not fit ({len(bad)}/{len(lo)} tiles outside their cell)")
        return g

    def lo(self, k) -> np.ndarray:
        return np.array([self.x0 + self.cell * k[1], self.y0 + self.cell * k[0]])

    def of(self, xy: np.ndarray) -> np.ndarray:
        """(N, 2) -> (N, 2) int array of (iy, ix)."""
        xy = np.atleast_2d(xy)
        return np.stack([np.floor((xy[:, 1] - self.y0) / self.cell),
                         np.floor((xy[:, 0] - self.x0) / self.cell)], 1).astype(int)


@dataclass
class Cut:
    s: float               # distance along the source route (m)
    p: np.ndarray          # exact crossing point (3,)
    line: float            # source angle of the grid line through p (0 = along x, pi/2 = along y)
    heading: float         # route heading at p (forwards)
    grade: float
    width: float
    kind: str = "cross"    # "cross" | "start" | "finish"


@dataclass
class Range:
    id: str
    a: int                 # index into cuts
    b: int
    tiles: frozenset
    seg: Segment


def find_cuts(stage: SourceStage, grid: TileGrid, step: float, min_gap: float = 30.0) -> list[Cut]:
    P = stage.points
    tid = grid.of(P[:, :2])
    hd = headings(P)
    g = np.gradient(P[:, 2]) / step
    raw = []
    for i in np.where(np.any(tid[1:] != tid[:-1], axis=1))[0]:
        a, b = tid[i], tid[i + 1]
        if (a != b).sum() != 1:
            continue                                     # through a corner: no single grid line
        if tuple(a) not in grid.tiles or tuple(b) not in grid.tiles:
            continue
        if a[1] != b[1]:                                 # ix changed: line x = const (runs along y)
            X = grid.x0 + grid.cell * max(a[1], b[1])
            t = (X - P[i, 0]) / (P[i + 1, 0] - P[i, 0])
            line = np.pi / 2
        else:
            Y = grid.y0 + grid.cell * max(a[0], b[0])
            t = (Y - P[i, 1]) / (P[i + 1, 1] - P[i, 1])
            line = 0.0
        p = P[i] + t * (P[i + 1] - P[i])
        raw.append(Cut(float((i + t) * step), p, line, float(hd[i]), float(np.mean(g[max(0, i - 8): i + 9])),
                       float(stage.width[i])))
    cuts = [c for j, c in enumerate(raw)
            if (j == 0 or c.s - raw[j - 1].s > min_gap) and (j == len(raw) - 1 or raw[j + 1].s - c.s > min_gap)]
    L = (len(P) - 1) * step
    start = Cut(0.0, P[0], 0.0, float(hd[0]), float(g[:9].mean()), float(stage.width[0]), "start")
    fin = Cut(L, P[-1], 0.0, float(hd[-1]), float(g[-9:].mean()), float(stage.width[-1]), "finish")
    return [start] + [c for c in cuts if 40 < c.s < L - 40] + [fin]


def make_ranges(stage: SourceStage, grid: TileGrid, cuts: list[Cut], step: float,
                min_len: float, max_len: float) -> list[Range]:
    P = stage.points
    tid = grid.of(P[:, :2])
    out = []
    for a in range(len(cuts)):
        for b in range(a + 1, len(cuts)):
            L = cuts[b].s - cuts[a].s
            if L > max_len:
                break
            if L < min_len and not (cuts[a].kind == "start" or cuts[b].kind == "finish"):
                continue
            i0 = int(np.ceil(cuts[a].s / step))
            i1 = min(int(np.floor(cuts[b].s / step)), len(P) - 1)
            if i1 - i0 < 5:
                continue
            tiles = frozenset(map(tuple, tid[i0: i1 + 1]))
            rid = f"{stage.name}#{a}-{b}"
            out.append(Range(rid, a, b, tiles, Segment(rid, stage, i0, i1, step)))
    return out


@dataclass
class TileSearchConfig:
    target_length: float = 8000.0
    length_tolerance: float = 0.12
    min_len: float = 250.0
    max_len: float = 1400.0
    max_kink_deg: float = 15.0
    max_grade_mismatch: float = 0.06
    max_width_mismatch: float = 2.0
    clearance: float = 95.0        # min distance between roads of non-adjacent pieces (m)
    beam_width: int = 128
    branch_per_state: int = 24
    allow_reversed: bool = True
    novelty: float = 30.0          # penalty for continuing exactly like the source
    variety: float = 25.0
    seed: int = 0
    step: float = 2.0
    coarse_step: float = 10.0


@dataclass
class TState:
    parent: "TState | None"
    piece: Piece
    rng: Range
    entry: int                     # cut index we entered through
    exit: int                      # cut index we leave through
    exit_p: np.ndarray             # world exit point
    exit_line: float               # world angle of the exit grid line
    exit_heading: float            # world heading (direction of travel) at the exit
    exit_grade: float
    exit_width: float
    length: float
    tiles: frozenset
    coarse: np.ndarray
    penalty: float
    kinks: list = field(default_factory=list)

    def chain(self):
        s = self
        while s is not None:
            yield s
            s = s.parent

    def pieces(self) -> list[Piece]:
        return [s.piece for s in reversed(list(self.chain()))]

    all_pieces = pieces          # layout.build_layout() interface


def _coarse(pts, cfg):
    return pts[:: max(1, int(cfg.coarse_step / cfg.step)), :2]


def _too_close(st: TState, new_xy: np.ndarray, cfg: TileSearchConfig) -> bool:
    skip = int(np.ceil(cfg.clearance * 1.3 / cfg.coarse_step))
    test = new_xy[skip:]
    if len(test) == 0:
        return False
    c2 = cfg.clearance ** 2
    lo, hi = test.min(0) - cfg.clearance, test.max(0) + cfg.clearance
    for i, s in enumerate(st.chain()):
        pts = s.coarse[:-skip] if i == 0 else s.coarse
        if len(pts) == 0:
            continue
        m = np.all((pts >= lo) & (pts <= hi), axis=1)
        if m.any() and (((test[:, None, :] - pts[m][None]) ** 2).sum(-1) < c2).any():
            return True
    # the joint region of the new piece against everything except its predecessor
    head = new_xy[:skip]
    for s in list(st.chain())[1:]:
        pts = s.coarse
        if len(pts) and len(head) and (((head[:, None, :] - pts[None]) ** 2).sum(-1) < c2).any():
            return True
    return False


def _place(rng: Range, rev: bool, cuts: list[Cut], st: TState, cfg: TileSearchConfig):
    ent = cuts[rng.b] if rev else cuts[rng.a]
    ext = cuts[rng.a] if rev else cuts[rng.b]
    h_in = ent.heading + (np.pi if rev else 0.0)
    best = None
    for k in (0, 1):
        yaw = st.exit_line - ent.line + k * np.pi
        kink = wrap_angle(yaw + h_in - st.exit_heading)
        if best is None or abs(kink) < abs(best[1]):
            best = (yaw, kink)
    yaw, kink = best
    if abs(np.degrees(kink)) > cfg.max_kink_deg:
        return None
    R = rot_z(yaw)
    trans = st.exit_p - R @ ent.p
    pts = rng.seg.points(rev) @ R.T + trans
    g_in = -ent.grade if rev else ent.grade
    return pts, yaw, trans, kink, g_in, ent, ext


def tile_search(stage: SourceStage, ranges: list[Range], cuts: list[Cut], cfg: TileSearchConfig) -> TState | None:
    rnd = random.Random(cfg.seed)
    bias = {r.id: rnd.uniform(0, cfg.variety) for r in ranges}
    lo_len = cfg.target_length * (1 - cfg.length_tolerance)
    hi_len = cfg.target_length * (1 + cfg.length_tolerance)
    last = len(cuts) - 1
    fin_lens = [cuts[-1].s - cuts[r.a].s for r in ranges if r.b == last]
    min_fin = min(fin_lens) if fin_lens else 0.0
    beam = []
    for r in ranges:
        if r.a != 0:
            continue
        pts = r.seg.points(False)
        ext = cuts[r.b]
        if ext.kind == "finish":
            continue
        pc = Piece("segment", pts, r.seg.widths(False), r.id, False, 0.0, np.zeros(3), r.seg.pacenotes(False) or [])
        beam.append(TState(None, pc, r, r.a, r.b, ext.p.copy(), ext.line, ext.heading, ext.grade, ext.width,
                           ext.s, r.tiles, _coarse(pts, cfg), bias[r.id]))
    finished, near = [], []
    for _ in range(60):
        nxt = []
        for st in beam:
            kids = []
            for r in ranges:
                if r.tiles & st.tiles or r.a == 0:
                    continue
                is_fin = r.b == last
                for rev in ((False, True) if cfg.allow_reversed and not is_fin else (False,)):
                    if rev and (cuts[r.b].kind != "cross" or cuts[r.a].kind != "cross"):
                        continue
                    if not rev and cuts[r.a].kind != "cross":
                        continue
                    if not is_fin and not rev and cuts[r.b].kind != "cross":
                        continue
                    placed = _place(r, rev, cuts, st, cfg)
                    if placed is None:
                        continue
                    pts, yaw, trans, kink, g_in, ent, ext = placed
                    if abs(g_in - st.exit_grade) > cfg.max_grade_mismatch:
                        continue
                    if abs(ent.width - st.exit_width) > cfg.max_width_mismatch:
                        continue
                    L = st.length + r.seg.length
                    if not is_fin and L + min_fin > hi_len:
                        continue
                    in_range = lo_len <= L <= hi_len
                    cxy = _coarse(pts, cfg)
                    if _too_close(st, cxy, cfg):
                        continue
                    R = rot_z(yaw)
                    follows = (rev == st.piece.reversed) and (r.b if rev else r.a) == st.exit
                    pen = (st.penalty + abs(np.degrees(kink)) * 0.8
                           + 30 * abs(g_in - st.exit_grade) / cfg.max_grade_mismatch
                           + (cfg.novelty if follows else 0) + bias[r.id])
                    h_out = ext.heading + (np.pi if rev else 0.0) + yaw
                    pc = Piece("segment", pts, r.seg.widths(rev), r.id, rev, yaw, trans,
                               r.seg.pacenotes(rev) or [])
                    kid = TState(st, pc, r, r.b if rev else r.a, r.a if rev else r.b, R @ ext.p + trans,
                                 ext.line + yaw, h_out, -ext.grade if rev else ext.grade, ext.width, L,
                                 st.tiles | r.tiles, cxy, pen, st.kinks + [float(np.degrees(kink))])
                    if not is_fin:
                        kids.append(kid)
                    elif in_range:
                        finished.append(kid)
                    else:
                        near.append(kid)          # complete stage, wrong length: fallback
            rnd.shuffle(kids)
            kids.sort(key=lambda s: s.penalty / max(s.length / 1000, 0.5))
            nxt += kids[: cfg.branch_per_state]
        seen, uniq = set(), []
        for s in sorted(nxt, key=lambda s: s.penalty / max(s.length / 1000, 0.5)):
            key = (s.rng.id, s.piece.reversed, round(s.exit_p[0] / 40), round(s.exit_p[1] / 40))
            if key not in seen:
                seen.add(key)
                uniq.append(s)
        beam = uniq[: cfg.beam_width]
        if not beam or len(finished) >= cfg.beam_width:
            break
    if not finished:
        # requested length not achievable: the complete stage closest to it
        if not near:
            return None
        return min(near, key=lambda s: (round(abs(s.length - cfg.target_length) / 250),
                                        s.penalty / max(s.length / 1000, 0.5)))
    return min(finished, key=lambda s: s.penalty / max(s.length / 1000, 0.5))

"""Beam search that chains segments (via connectors) into a new stage layout."""
from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np

from .connector import connector, max_abs_grade
from .geometry import rot_z
from .model import EndState, Piece, Segment
from .segments import end_states


@dataclass
class SearchConfig:
    target_length: float = 10_000.0
    length_tolerance: float = 0.08      # finish within ±8 %
    beam_width: int = 48
    branch_per_state: int = 24          # best expansions kept per parent
    yaw_offsets_deg: tuple = (-25, -12, 0, 12, 25)
    allow_reversed: bool = True
    corridor_half_width: float = 35.0   # scanned corridor each side of the road
    max_grade_mismatch: float = 0.06
    max_width_mismatch: float = 1.5
    max_connector_grade: float = 0.14
    max_elevation_range: float = 450.0
    same_surface_only: bool = True
    seed: int = 0
    variety: float = 12.0               # random per-segment preference; bigger = more seed-to-seed variation
    step: float = 2.0
    coarse_step: float = 10.0            # spacing used for overlap tests


@dataclass
class State:
    parent: "State | None"
    pieces: list[Piece]                 # pieces added by this step (connector, segment)
    end: EndState
    length: float
    used: frozenset
    coarse: np.ndarray                  # (M, 2) xy points added by this step
    zmin: float
    zmax: float
    penalty: float
    synthetic: float                    # total connector length so far
    last_stage: str
    surface: str
    children_tail_skip: int = 0         # coarse points at our tail to skip in overlap test

    def chain(self):
        s = self
        while s is not None:
            yield s
            s = s.parent

    def all_pieces(self) -> list[Piece]:
        out: list[Piece] = []
        for s in reversed(list(self.chain())):
            out.extend(s.pieces)
        return out


def _coarse(points: np.ndarray, step: float, coarse_step: float) -> np.ndarray:
    k = max(1, int(coarse_step / step))
    return points[::k, :2]


def _overlaps(parent: State, new_xy: np.ndarray, clearance: float, joint_skip: int) -> bool:
    """True if new_xy (minus its first joint_skip points, which sit next to the
    joint) comes within `clearance` of anything already laid down. The parent's
    own tail near the joint is skipped too."""
    test = new_xy[joint_skip:]
    if len(test) == 0:
        return False
    lo, hi = test.min(0) - clearance, test.max(0) + clearance
    c2 = clearance * clearance
    for i, s in enumerate(parent.chain()):
        pts = s.coarse[:-joint_skip] if i == 0 and joint_skip else s.coarse
        if len(pts) == 0:
            continue
        m = np.all((pts >= lo) & (pts <= hi), axis=1)
        if not m.any():
            continue
        p = pts[m]
        d2 = ((test[:, None, :] - p[None, :, :]) ** 2).sum(-1)
        if (d2 < c2).any():
            return True
    return False


def _place(seg: Segment, rev: bool, start_src: EndState, at: EndState, cfg: SearchConfig):
    """Rigidly place a segment so its start state lands on `at` (yaw + translation only)."""
    yaw = at.yaw - start_src.yaw
    R = rot_z(yaw)
    pts = (seg.points(rev) - start_src.pos) @ R.T + at.pos
    return pts, yaw, at.pos - R @ start_src.pos


def _initial_states(segments: list[Segment], cfg: SearchConfig) -> list[State]:
    out = []
    origin = EndState(np.zeros(3), 0.0, 0.0, 0.0)
    for seg in segments:
        for rev in ((False, True) if cfg.allow_reversed else (False,)):
            s0, e0 = end_states(seg, rev, cfg.step)
            pts, yaw, trans = _place(seg, rev, s0, EndState(np.zeros(3), 0.0, s0.grade, s0.width), cfg)
            _, e = end_states(seg, rev, cfg.step)
            R = rot_z(yaw)
            end = EndState(R @ e.pos + trans, e.yaw + yaw, e.grade, e.width)
            piece = Piece("segment", pts, seg.widths(rev), seg.id, rev, yaw, trans,
                          seg.pacenotes(rev) or [])
            out.append(State(None, [piece], end, seg.length, frozenset([seg.id]),
                             _coarse(pts, cfg.step, cfg.coarse_step),
                             float(pts[:, 2].min()), float(pts[:, 2].max()),
                             0.0, 0.0, seg.stage.name, seg.stage.surface))
    return out


def _expand(st: State, segments: list[Segment], cfg: SearchConfig, rng: random.Random,
            bias: dict[str, float]) -> list[State]:
    clearance = 2 * cfg.corridor_half_width
    joint_skip = int(np.ceil(clearance * 1.2 / cfg.coarse_step))
    kids: list[State] = []
    for seg in segments:
        if seg.id in st.used:
            continue
        if cfg.same_surface_only and seg.stage.surface != st.surface:
            continue
        for rev in ((False, True) if cfg.allow_reversed else (False,)):
            s0, e0 = end_states(seg, rev, cfg.step)
            dg = abs(s0.grade - st.end.grade)
            dw = abs(s0.width - st.end.width)
            if dg > cfg.max_grade_mismatch or dw > cfg.max_width_mismatch:
                continue
            for off in cfg.yaw_offsets_deg:
                dyaw = np.radians(off)
                L = float(np.clip(40 + 900 * dg + 160 * abs(dyaw) + 10 * dw, 40, 160))
                cpts, cw, cend = connector(st.end, st.end.yaw + dyaw, L, s0.grade, s0.width, cfg.step)
                if max_abs_grade(cpts) > cfg.max_connector_grade:
                    continue
                spts, yaw, trans = _place(seg, rev, s0, cend, cfg)
                zmin = min(st.zmin, cpts[:, 2].min(), spts[:, 2].min())
                zmax = max(st.zmax, cpts[:, 2].max(), spts[:, 2].max())
                if zmax - zmin > cfg.max_elevation_range:
                    continue
                new_xy = np.vstack([_coarse(cpts, cfg.step, cfg.coarse_step),
                                    _coarse(spts, cfg.step, cfg.coarse_step)])
                if _overlaps(st, new_xy, clearance, joint_skip):
                    continue
                R = rot_z(yaw)
                end = EndState(R @ e0.pos + trans, e0.yaw + yaw, e0.grade, e0.width)
                penalty = (st.penalty
                           + 40 * dg / cfg.max_grade_mismatch
                           + 6 * abs(off) / 25
                           + 4 * dw
                           + (3 if seg.stage.name == st.last_stage else 0)
                           + bias[seg.id])
                kids.append(State(
                    st,
                    [Piece("connector", cpts, cw),
                     Piece("segment", spts, seg.widths(rev), seg.id, rev, yaw, trans,
                           seg.pacenotes(rev) or [])],
                    end, st.length + L + seg.length, st.used | {seg.id}, new_xy,
                    float(zmin), float(zmax), penalty, st.synthetic + L,
                    seg.stage.name, st.surface))
    rng.shuffle(kids)  # tie-break randomly so different seeds give different stages
    kids.sort(key=_score)
    return kids[: cfg.branch_per_state]


def _score(s: State) -> float:
    """Lower is better. Penalty is normalised per km so long and short partial
    layouts compare fairly; synthetic (connector) share is punished hard."""
    km = max(s.length / 1000, 0.3)
    return s.penalty / km + 120 * s.synthetic / max(s.length, 1)


def search(segments: list[Segment], cfg: SearchConfig) -> State | None:
    rng = random.Random(cfg.seed)
    lo = cfg.target_length * (1 - cfg.length_tolerance)
    hi = cfg.target_length * (1 + cfg.length_tolerance)
    bias = {seg.id: rng.uniform(0, cfg.variety) for seg in segments}
    beam = _initial_states(segments, cfg)
    for st in beam:
        st.penalty = bias[st.pieces[0].segment_id]
    rng.shuffle(beam)
    beam = sorted(beam, key=lambda st: st.penalty)[: cfg.beam_width]
    finished: list[State] = []
    for _ in range(200):
        if not beam:
            break
        nxt: list[State] = []
        for st in beam:
            for kid in _expand(st, segments, cfg, rng, bias):
                if lo <= kid.length <= hi:
                    finished.append(kid)
                elif kid.length < lo:
                    nxt.append(kid)
        # Diversity: dedupe states that just reached the same place by the same last segment.
        seen, uniq = set(), []
        for s in sorted(nxt, key=_score):
            key = (s.pieces[-1].segment_id, round(s.end.pos[0] / 50), round(s.end.pos[1] / 50))
            if key not in seen:
                seen.add(key)
                uniq.append(s)
        beam = uniq[: cfg.beam_width]
        if len(finished) >= cfg.beam_width:
            break
    return min(finished, key=_score) if finished else None

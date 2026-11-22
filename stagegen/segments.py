# SPDX-FileCopyrightText: 2026 hiro98 (vale981) and the ACR Remixer contributors
# SPDX-License-Identifier: MIT

"""Cut source stages into reusable segments at 'quiet' points of the road.

A good cut point is somewhere a seam will be least noticeable and easiest to
blend: low horizontal curvature, roughly constant grade, and steady width.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import curvature, grade
from .model import EndState, Segment, SourceStage


@dataclass
class CutConfig:
    step: float = 2.0
    max_curvature: float = 1 / 250   # straight-ish: radius > 250 m around the cut
    max_grade_change: float = 0.015  # grade must be steady across the window
    window_m: float = 30.0           # half-window that has to be quiet
    min_len: float = 250.0
    max_len: float = 1200.0
    anchor_ends: bool = False        # first segment starts at the stage start, last ends at the finish


def cut_candidates(stage: SourceStage, cfg: CutConfig) -> np.ndarray:
    k = np.abs(curvature(stage.points, cfg.step, window=7))
    g = grade(stage.points, cfg.step, window=7)
    w = int(cfg.window_m / cfg.step)
    n = len(stage.points)
    ok = np.zeros(n, dtype=bool)
    for i in range(w, n - w):
        win = slice(i - w, i + w + 1)
        ok[i] = (k[win].max() < cfg.max_curvature
                 and np.ptp(g[win]) < cfg.max_grade_change
                 and np.ptp(stage.width[win]) < 0.6)
    # Always allow the original stage start/end as cut points.
    ok[0] = ok[-1] = True
    return np.flatnonzero(ok)


def cut_stage(stage: SourceStage, cfg: CutConfig) -> list[Segment]:
    """Greedy: from each cut, jump to the furthest candidate within max_len
    (but at least min_len away). Produces non-overlapping segments covering
    as much of the stage as the quiet points allow."""
    cand = cut_candidates(stage, cfg)
    lo, hi = int(cfg.min_len / cfg.step), int(cfg.max_len / cfg.step)
    segs: list[Segment] = []
    i = cand[0]
    if cfg.anchor_ends:
        # The opening segment must contain the real start (gates, start area),
        # even if the first quiet point is further than max_len away.
        first = cand[(cand >= lo) & (cand <= hi)]
        if len(first) == 0:
            first = cand[cand >= lo][:1]
        if len(first):
            j = int(first[-1])
            segs.append(Segment(f"{stage.name}#0", stage, 0, j, cfg.step))
            i = j
    while True:
        nxt = cand[(cand >= i + lo) & (cand <= i + hi)]
        if len(nxt) == 0:
            later = cand[cand > i + hi]
            if len(later) == 0:
                break
            i = later[0]          # gap with no usable cut: skip ahead
            continue
        j = int(nxt[-1])
        segs.append(Segment(f"{stage.name}#{len(segs)}", stage, int(i), j, cfg.step))
        i = j
    n = len(stage.points) - 1
    if cfg.anchor_ends and segs and segs[-1].i1 != n:
        last = segs[-1]
        if (n - last.i1) * cfg.step < cfg.min_len or segs[-1].i1 != i:
            segs[-1] = Segment(last.id, stage, last.i0, n, cfg.step)      # stretch to the finish
        else:
            segs.append(Segment(f"{stage.name}#{len(segs)}", stage, last.i1, n, cfg.step))
    return segs


def end_states(seg: Segment, reversed_: bool, step: float) -> tuple[EndState, EndState]:
    """(start, end) state of a segment in its *source* coordinates."""
    p = seg.points(reversed_)
    w = seg.widths(reversed_)
    n = min(10, len(p) - 1)  # 20 m baseline for heading/grade

    def state(a: int, b: int, at: int) -> EndState:
        d = p[b] - p[a]
        horiz = np.linalg.norm(d[:2])
        return EndState(p[at].copy(), float(np.arctan2(d[1], d[0])),
                        float(d[2] / horiz), float(w[at]))

    return state(0, n, 0), state(-1 - n, -1, -1)

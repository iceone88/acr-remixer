# SPDX-FileCopyrightText: 2026 hiro98 (vale981) and the ACR Remixer contributors
# SPDX-License-Identifier: MIT

"""Core data types: source stages, cut segments, end states, placed pieces."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Pacenote:
    s: float          # distance along the piece/stage where the call is triggered (m)
    text: str         # e.g. "4 left long", "crest", "into 2 right"
    origin: str = "original"   # "original" (carried over) or "generated"
    length: float = 30.0       # extent of the call (m), as stored in DT_Pacenote tables


@dataclass
class SourceStage:
    """A stage as extracted from the game: centreline + metadata."""
    name: str
    location: str      # e.g. "wales"; only segments from one location are mixed
    surface: str       # "gravel" / "tarmac" / "snow"
    points: np.ndarray # (N, 3) centreline, resampled to `step`
    width: np.ndarray  # (N,) road width (m)
    pacenotes: list[Pacenote] = field(default_factory=list)


@dataclass(frozen=True)
class EndState:
    pos: np.ndarray    # (3,)
    yaw: float         # radians
    grade: float       # dz/ds
    width: float       # m


@dataclass
class Segment:
    """A contiguous slice [i0, i1] of a source stage, usable forwards or reversed."""
    id: str
    stage: SourceStage
    i0: int
    i1: int
    step: float

    def points(self, reversed_: bool = False) -> np.ndarray:
        p = self.stage.points[self.i0 : self.i1 + 1]
        return p[::-1].copy() if reversed_ else p.copy()

    def widths(self, reversed_: bool = False) -> np.ndarray:
        w = self.stage.width[self.i0 : self.i1 + 1]
        return w[::-1].copy() if reversed_ else w.copy()

    @property
    def length(self) -> float:
        return (self.i1 - self.i0) * self.step

    @property
    def s0(self) -> float:
        return self.i0 * self.step

    def pacenotes(self, reversed_: bool = False) -> list[Pacenote] | None:
        """Original notes re-keyed to segment-local distance. None when reversed:
        calls are direction-dependent (left/right, cue placement before corners),
        so reversed segments get freshly generated notes instead."""
        if reversed_:
            return None
        s0, s1 = self.s0, self.s0 + self.length
        return [Pacenote(n.s - s0, n.text, "original", n.length)
                for n in self.stage.pacenotes if s0 <= n.s < s1]


@dataclass
class Piece:
    """One element of a generated layout: either a placed segment or a connector."""
    kind: str                      # "segment" | "connector"
    points: np.ndarray             # world-space centreline (N, 3)
    widths: np.ndarray
    segment_id: str | None = None
    reversed: bool = False
    yaw: float = 0.0               # rigid transform applied to the source geometry
    translation: np.ndarray | None = None
    pacenotes: list[Pacenote] = field(default_factory=list)

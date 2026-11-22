# SPDX-FileCopyrightText: 2026 hiro98 (vale981) and the ACR Remixer contributors
# SPDX-License-Identifier: MIT

"""Smooth the joins between pieces (kinks in direction and gradient).

Pieces meet on a grid line, edge to edge. Rotating a piece would open the
edge, so instead each side of the line is *sheared along the line*: a point at
distance d from the line moves by A * g(d) along it (and by Az * g(d) up),
with g(0) = 0, g'(0) = 1, g(D) = g'(D) = 0. Points on the line stay put (the
meshes stay seamless), the road's crossing angle (and slope) at the line
changes by the amount needed to meet the other side, and beyond D nothing
moves. A kink becomes a smooth bend about 2 D long.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _g(d: np.ndarray, D: float) -> np.ndarray:
    u = np.clip(d / D, 0.0, 1.0)
    return d * (1.0 - u) ** 2 * (d < D)


def _dg(d: np.ndarray, D: float) -> np.ndarray:
    u = np.clip(d / D, 0.0, 1.0)
    return ((1.0 - u) ** 2 - 2.0 * u * (1.0 - u)) * (d < D)


@dataclass
class Shear:
    p0: np.ndarray        # (2,) a point on the join line (layout frame, m)
    e: np.ndarray         # (2,) unit vector along the line
    n: np.ndarray         # (2,) unit normal pointing into this side
    A: float              # along-line shear at the line (dx_e / dd)
    Az: float             # vertical shear at the line (dz / dd)
    D: float              # reach (m)

    def shifted(self, off: np.ndarray) -> "Shear":
        return Shear(self.p0 + off[:2], self.e, self.n, self.A, self.Az, self.D)

    def _d(self, xy):
        return (np.atleast_2d(xy)[:, :2] - self.p0) @ self.n

    def apply(self, P: np.ndarray) -> np.ndarray:
        """(N,2|3) -> same shape: points on this side within D moved."""
        P = np.array(P, np.float64, copy=True)
        d = self._d(P)
        side = d > 0
        if not side.any():
            return P
        g = np.where(side, _g(np.maximum(d, 0), self.D), 0.0)
        P[:, :2] += g[:, None] * self.A * self.e
        if P.shape[1] > 2:
            P[:, 2] += g * self.Az
        return P

    def yaw(self, xy: np.ndarray) -> np.ndarray:
        """Local rotation (rad) of the map at xy (for objects' headings): the
        shear turns directions along n by atan(A g'(d))."""
        d = self._d(xy)
        dg = np.where(d > 0, _dg(np.maximum(d, 0), self.D), 0.0)
        # a direction along n becomes n + A g' e: turned towards e (z of n x e gives the sense)
        sense = np.sign(self.n[0] * self.e[1] - self.n[1] * self.e[0])
        return np.arctan(self.A * dg) * sense


def _cross_angle(u: np.ndarray, e: np.ndarray, n: np.ndarray) -> float:
    return float(np.arctan2(u @ e, u @ n))


def join_shears(p0, line_angle: float, h_prev: float, h_next: float, g_prev: float, g_next: float,
                D_prev: float, D_next: float, max_kink_deg: float = 60.0):
    """Shears for both sides of one join. p0: join point (layout frame);
    line_angle: world angle of the grid line; h_prev/h_next: road headings
    (rad, direction of travel) arriving/leaving; g_prev/g_next: gradients.
    Returns (shear for the previous piece's side, shear for the next one's)."""
    e = np.array([np.cos(line_angle), np.sin(line_angle)])
    n = np.array([-e[1], e[0]])
    u_out = np.array([np.cos(h_next), np.sin(h_next)])
    if u_out @ n < 0:
        n = -n                                     # n points along travel, into the next piece
    u_in = np.array([np.cos(h_prev), np.sin(h_prev)])
    a_in, a_out = _cross_angle(u_in, e, n), _cross_angle(u_out, e, n)
    if abs(np.degrees(a_out - a_in)) > max_kink_deg:
        return None, None
    a_mid = (a_in + a_out) / 2
    # direction (cos a, sin a) in (n, e); a shear x_e += A d (d along the side's
    # normal) maps it to (cos a, sin a + A cos a [n-dir sign]) -> tan a' = tan a + A s
    # next side: normal n, d grows along travel
    A_next = np.tan(a_mid) - np.tan(a_out)
    # previous side: normal -n, d grows against travel: x_e += A d, d = -(x . n)
    A_prev = -(np.tan(a_mid) - np.tan(a_in))
    g_mid = (g_prev + g_next) / 2
    # slope along travel; d per metre of travel = cos a (next) / -cos a (prev side)
    Az_next = (g_mid - g_next) / max(np.cos(a_out), 0.2)
    Az_prev = -(g_mid - g_prev) / max(np.cos(a_in), 0.2)
    p0 = np.asarray(p0, np.float64)[:2]
    return (Shear(p0, e, -n, A_prev, Az_prev, D_prev), Shear(p0, e, n, A_next, Az_next, D_next))

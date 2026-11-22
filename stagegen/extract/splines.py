# SPDX-FileCopyrightText: 2026 hiro98 (vale981) and the ACR Remixer contributors
# SPDX-License-Identifier: MIT

"""Read the route splines from a stage level (SplinesActor -> DMSplineComponent).

Each DMSplineComponent (CenterSpline, IdealSpline, Left/RightBorderSpline,
ContourSpline, ...) serialises SplineCurves first:

    FSplineCurves [first value]
      [0] Position: FInterpCurveVector
            [0] Points: TArray<FInterpCurvePointVector>
                  [0] InVal float  [1] OutVal FVector(d)  [2] ArriveTangent
                  [3] LeaveTangent [4] InterpMode u8

Coordinates are UE world space (cm, Z up). Component transforms of the
SplinesActor are identity in every level checked so far.
"""
from __future__ import annotations

import numpy as np

from .unversioned import Reader
from .zen import Package


def _vec(r: Reader):
    return (r.f64(), r.f64(), r.f64())


def read_spline_points(pk: Package, export_name: str):
    """Return (inval[N], pos[N,3], arrive[N,3], leave[N,3]) in UE cm."""
    e = next(x for x in pk.exports if x.name == export_name)
    r = Reader(pk.d, e.data_offset)
    top = r.header()
    if top and top[0] == 0:      # 4-byte value before SplineCurves (IdealSpline)
        r.i32()
        top = top[1:]
    if not top or top[0] != 2:
        raise ValueError(f"no SplineCurves in {export_name}: {top}")
    curves = r.header()          # FSplineCurves
    if not curves or curves[0] != 0:
        raise ValueError(f"no Position curve in {export_name}: {curves}")
    pos_curve = r.header()       # FInterpCurveVector
    if not pos_curve or pos_curve[0] != 0:
        return np.zeros(0), np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0, 3))
    n = r.u32()
    inval = np.zeros(n)
    P = np.zeros((n, 3)); A = np.zeros((n, 3)); L = np.zeros((n, 3))
    for k in range(n):
        for f in r.header():
            if f == 0:
                inval[k] = r.f32()
            elif f == 1:
                P[k] = _vec(r)
            elif f == 2:
                A[k] = _vec(r)
            elif f == 3:
                L[k] = _vec(r)
            elif f == 4:
                r.u8()
            else:
                raise ValueError(f"unexpected curve point field {f}")
    return inval, P, A, L


def sample_hermite(P, A, L, step_cm: float = 50.0) -> np.ndarray:
    """Densely sample a UE CurveUser/CurveAuto spline (cubic Hermite per segment)."""
    out = []
    for i in range(len(P) - 1):
        p0, p1, m0, m1 = P[i], P[i + 1], L[i], A[i + 1]
        seg = np.linalg.norm(p1 - p0)
        n = max(2, int(seg / step_cm) + 1)
        t = np.linspace(0, 1, n, endpoint=False)[:, None]
        h00 = 2 * t**3 - 3 * t**2 + 1
        h10 = t**3 - 2 * t**2 + t
        h01 = -2 * t**3 + 3 * t**2
        h11 = t**3 - t**2
        out.append(h00 * p0 + h10 * m0 + h01 * p1 + h11 * m1)
    out.append(P[-1:])
    return np.vstack(out)


def level_splines(pk: Package) -> dict[str, np.ndarray]:
    """All non-empty DMSplineComponents of the level, sampled densely, in metres
    (x, y horizontal, z up; UE axes kept)."""
    res = {}
    for name in ("CenterSpline", "IdealSpline", "LeftBorderSpline", "RightBorderSpline"):
        try:
            _, P, A, L = read_spline_points(pk, name)
        except StopIteration:
            continue
        if len(P) > 1:
            res[name] = sample_hermite(P, A, L) / 100.0
    return res

"""Polyline helpers. Coordinates are metres, x/y horizontal, z up."""
from __future__ import annotations

import numpy as np


def resample(points: np.ndarray, step: float) -> np.ndarray:
    """Resample a 3D polyline to (roughly) uniform spacing along its 2D length."""
    d = np.linalg.norm(np.diff(points[:, :2], axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(d)])
    n = max(2, int(round(s[-1] / step)) + 1)
    t = np.linspace(0.0, s[-1], n)
    return np.stack([np.interp(t, s, points[:, i]) for i in range(3)], axis=1)


def arclength(points: np.ndarray) -> np.ndarray:
    d = np.linalg.norm(np.diff(points[:, :2], axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(d)])


def headings(points: np.ndarray) -> np.ndarray:
    """Yaw (radians) at each point, unwrapped."""
    v = np.gradient(points[:, :2], axis=0)
    return np.unwrap(np.arctan2(v[:, 1], v[:, 0]))


def smooth(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return x
    k = np.ones(window) / window
    pad = window // 2
    xp = np.pad(x, pad, mode="edge")
    return np.convolve(xp, k, mode="valid")[: len(x)]


def curvature(points: np.ndarray, step: float, window: int = 5) -> np.ndarray:
    """Signed horizontal curvature (1/m); positive = turning left."""
    h = smooth(headings(points), window)
    return np.gradient(h) / step


def grade(points: np.ndarray, step: float, window: int = 5) -> np.ndarray:
    """dz/ds along the road."""
    return np.gradient(smooth(points[:, 2], window)) / step


def rot_z(yaw: float) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def wrap_angle(a: float) -> float:
    return (a + np.pi) % (2 * np.pi) - np.pi

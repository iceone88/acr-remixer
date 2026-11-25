# SPDX-FileCopyrightText: 2026 hiro98 (vale981) and the ACR Remixer contributors
# SPDX-License-Identifier: MIT
"""Create tiny road/terrain PLY fixtures for the real Blender bridge CI test."""
from __future__ import annotations

import argparse
from pathlib import Path

import open3d as o3d


def plane(z: float, size: float = 20.0) -> o3d.geometry.TriangleMesh:
    half = size / 2.0
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector([
        (-half, -half, z), (half, -half, z),
        (half, half, z), (-half, half, z),
    ])
    mesh.triangles = o3d.utility.Vector3iVector([(0, 1, 2), (0, 2, 3)])
    mesh.compute_vertex_normals()
    return mesh


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    exchange = args.out / "exchange"
    exchange.mkdir(parents=True, exist_ok=True)
    for name, mesh in (("road_collision.ply", plane(2.0)), ("terrain_collision.ply", plane(0.0, 30.0))):
        path = exchange / name
        if not o3d.io.write_triangle_mesh(str(path), mesh, write_vertex_normals=True):
            raise RuntimeError(f"Could not write test fixture: {path}")
    print(f"Created deterministic bridge fixtures in {exchange}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

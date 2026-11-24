# SPDX-FileCopyrightText: 2026 ACR Remixer contributors
# SPDX-License-Identifier: MIT
"""Verify Open3D CPU triangle ray casting in a clean, headless environment."""
from __future__ import annotations

import sys

import numpy as np
import open3d as o3d


def main() -> int:
    vertices = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    triangles = np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int32)

    legacy_mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(vertices),
        o3d.utility.Vector3iVector(triangles),
    )
    tensor_mesh = o3d.t.geometry.TriangleMesh.from_legacy(legacy_mesh)
    scene = o3d.t.geometry.RaycastingScene()
    geometry_id = scene.add_triangles(tensor_mesh)

    rays = o3d.core.Tensor(
        [[0.25, 0.25, 1.0, 0.0, 0.0, -1.0]],
        dtype=o3d.core.Dtype.Float32,
    )
    hit = scene.cast_rays(rays)
    t_hit = float(hit["t_hit"].numpy()[0])
    if not np.isfinite(t_hit) or abs(t_hit - 1.0) > 1e-5:
        print(
            f"Open3D smoke test FAILED: expected t_hit=1.0, received {t_hit}",
            file=sys.stderr,
        )
        return 1

    print(
        "Open3D smoke test PASS: "
        f"version={o3d.__version__}, geometry_id={geometry_id}, t_hit={t_hit:.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

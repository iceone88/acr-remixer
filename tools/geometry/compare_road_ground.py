# SPDX-FileCopyrightText: 2026 ACR Remixer contributors
# SPDX-License-Identifier: MIT
"""Read-only vertical road/ground proximity diagnostic using Open3D ray casting.

The road OBJ must represent the road's upper driving surface (not a closed
collision hull with underside faces). The terrain OBJ must represent the terrain
surface. Both meshes must already share coordinates, units, and a Z-up frame.
This script never moves or edits vertices.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import open3d as o3d


def read_triangle_mesh(path: Path) -> o3d.geometry.TriangleMesh:
    mesh = o3d.io.read_triangle_mesh(str(path), enable_post_processing=False)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    triangles = np.asarray(mesh.triangles, dtype=np.int64)
    if len(vertices) == 0 or len(triangles) == 0:
        raise ValueError(f"Input has no triangle mesh geometry: {path}")
    if not np.isfinite(vertices).all():
        raise ValueError(f"Input contains non-finite vertices: {path}")
    if triangles.min() < 0 or triangles.max() >= len(vertices):
        raise ValueError(f"Input contains invalid triangle indices: {path}")
    return mesh


def compare_surfaces(
    road_path: Path,
    terrain_path: Path,
    sample_count: int = 20_000,
    max_search_distance: float = 100.0,
    seed: int = 7,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if sample_count < 1:
        raise ValueError("sample_count must be at least 1")
    if not math.isfinite(max_search_distance) or max_search_distance <= 0:
        raise ValueError("max_search_distance must be finite and greater than zero")

    road = read_triangle_mesh(road_path)
    terrain = read_triangle_mesh(terrain_path)
    o3d.utility.random.seed(int(seed))
    sample_cloud = road.sample_points_uniformly(number_of_points=int(sample_count))
    points = np.asarray(sample_cloud.points, dtype=np.float64)
    if len(points) == 0 or not np.isfinite(points).all():
        raise ValueError("Could not generate finite sample points from the road mesh")

    terrain_tensor = o3d.t.geometry.TriangleMesh.from_legacy(terrain)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(terrain_tensor)

    # RayCastingScene uses float32 rays. The small origin offset avoids treating
    # an exactly coincident source point as a zero-distance self-hit.
    epsilon = max(1e-4, max_search_distance * 1e-7)
    point32 = points.astype(np.float32)
    down_rays = np.empty((len(points), 6), dtype=np.float32)
    down_rays[:, :3] = point32
    down_rays[:, 2] += epsilon
    down_rays[:, 3:] = (0.0, 0.0, -1.0)

    up_rays = np.empty((len(points), 6), dtype=np.float32)
    up_rays[:, :3] = point32
    up_rays[:, 2] -= epsilon
    up_rays[:, 3:] = (0.0, 0.0, 1.0)

    down_t = scene.cast_rays(o3d.core.Tensor(down_rays))["t_hit"].numpy().astype(np.float64)
    up_t = scene.cast_rays(o3d.core.Tensor(up_rays))["t_hit"].numpy().astype(np.float64)

    down_distance = down_t - epsilon
    up_distance = up_t - epsilon
    down_valid = np.isfinite(down_t) & (down_t <= max_search_distance + epsilon) & (down_distance >= -epsilon)
    up_valid = np.isfinite(up_t) & (up_t <= max_search_distance + epsilon) & (up_distance >= -epsilon)

    down_abs = np.where(down_valid, np.maximum(down_distance, 0.0), np.inf)
    up_abs = np.where(up_valid, np.maximum(up_distance, 0.0), np.inf)
    use_down = down_abs <= up_abs
    matched = np.isfinite(np.minimum(down_abs, up_abs))

    signed_gap = np.full(len(points), np.nan, dtype=np.float64)
    signed_gap[matched & use_down] = -down_abs[matched & use_down]
    signed_gap[matched & ~use_down] = up_abs[matched & ~use_down]
    matched_count = int(matched.sum())

    if matched_count:
        values = signed_gap[matched]
        quantiles = np.quantile(values, [0.0, 0.05, 0.50, 0.95, 1.0])
        stats = {
            "min": float(quantiles[0]),
            "p05": float(quantiles[1]),
            "median": float(quantiles[2]),
            "p95": float(quantiles[3]),
            "max": float(quantiles[4]),
            "terrain_below_road_samples": int(np.count_nonzero(values < -epsilon)),
            "terrain_above_road_samples": int(np.count_nonzero(values > epsilon)),
            "near_coincident_samples": int(np.count_nonzero(np.abs(values) <= epsilon)),
        }
    else:
        stats = {"min": None, "p05": None, "median": None, "p95": None, "max": None,
                 "terrain_below_road_samples": 0, "terrain_above_road_samples": 0,
                 "near_coincident_samples": 0}

    report: dict[str, Any] = {
        "schema_version": 1,
        "tool": "ACR Remixer road/ground vertical proximity diagnostic",
        "road_file": str(road_path),
        "terrain_file": str(terrain_path),
        "road_vertices": int(len(road.vertices)),
        "road_triangles": int(len(road.triangles)),
        "terrain_vertices": int(len(terrain.vertices)),
        "terrain_triangles": int(len(terrain.triangles)),
        "requested_samples": int(sample_count),
        "matched_samples": matched_count,
        "unmatched_samples": int(len(points) - matched_count),
        "coverage_percent": float(100.0 * matched_count / len(points)),
        "max_search_distance_input_units": float(max_search_distance),
        "signed_vertical_gap_input_units": stats,
        "interpretation": "Negative means the nearest vertical terrain hit is below the sampled road surface; positive means above. This is a read-only proximity diagnostic, not a drivability verdict or an automatic correction.",
        "coordinate_assumptions": "Both meshes must already share the same coordinate frame and units, with Z as vertical. No registration, axis conversion, or unit conversion is performed.",
        "sampling_note": "Samples are drawn uniformly over the road mesh surface; if the road OBJ includes underside/side collision faces, results may not represent only the driving surface.",
    }

    sample_rows: list[dict[str, Any]] = []
    for index, point in enumerate(points):
        gap = float(signed_gap[index]) if matched[index] else None
        terrain_z = float(point[2] + gap) if gap is not None else None
        sample_rows.append({
            "sample_id": index,
            "road_x": float(point[0]),
            "road_y": float(point[1]),
            "road_z": float(point[2]),
            "terrain_z_hit": terrain_z,
            "signed_vertical_gap": gap,
        })
    return report, sample_rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--road", required=True, type=Path, help="Road upper-surface OBJ")
    parser.add_argument("--terrain", required=True, type=Path, help="Terrain surface OBJ")
    parser.add_argument("--samples", type=int, default=20_000)
    parser.add_argument("--max-search", type=float, default=100.0,
                        help="Maximum vertical ray distance in the meshes' existing units (default: 100)")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--json", type=Path, default=Path("artifacts/road_ground_report.json"))
    parser.add_argument("--csv", type=Path, default=Path("artifacts/road_ground_samples.csv"))
    args = parser.parse_args()

    report, rows = compare_surfaces(args.road, args.terrain, args.samples, args.max_search, args.seed)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    with args.csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Road/ground samples matched: {report['matched_samples']}/{report['requested_samples']} ({report['coverage_percent']:.2f}%)")
    print(f"Signed vertical gap: {json.dumps(report['signed_vertical_gap_input_units'])}")
    print(f"JSON: {args.json}\nCSV:  {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

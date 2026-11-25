# SPDX-FileCopyrightText: 2026 hiro98 (vale981) and the ACR Remixer contributors
# SPDX-License-Identifier: MIT

"""Create and process an editable mesh workspace from an ACR stock stage.

This module deliberately works on exported collision meshes (OBJ/PLY). It does
not claim to import or rewrite Unreal .uasset/.umap/IoStore packages from Blender.
The StageGen package reader performs the game-specific extraction; Open3D and
Blender operate on the neutral mesh files produced by that extraction.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np

from . import toolchain


class GeometryWorkspaceError(RuntimeError):
    """A stock-stage mesh workspace could not be created or processed."""


def _open3d():
    try:
        import open3d as o3d
    except Exception as exc:  # pragma: no cover - depends on the frozen runtime
        raise GeometryWorkspaceError(
            f"Open3D is not available in this backend: {type(exc).__name__}: {exc}"
        ) from exc
    return o3d


def _finite_mesh(mesh, label: str) -> tuple[np.ndarray, np.ndarray]:
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    triangles = np.asarray(mesh.triangles, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) == 0:
        raise GeometryWorkspaceError(f"{label}: mesh has no valid vertices")
    if triangles.ndim != 2 or triangles.shape[1] != 3 or len(triangles) == 0:
        raise GeometryWorkspaceError(f"{label}: mesh has no triangles")
    if not np.isfinite(vertices).all():
        raise GeometryWorkspaceError(f"{label}: mesh contains NaN or infinite vertex coordinates")
    if triangles.min() < 0 or triangles.max() >= len(vertices):
        raise GeometryWorkspaceError(f"{label}: mesh contains out-of-range triangle indices")
    return vertices, triangles


def _mesh_stats(mesh, label: str) -> dict[str, Any]:
    vertices, triangles = _finite_mesh(mesh, label)
    tri = vertices[triangles]
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    twice_area = np.linalg.norm(cross, axis=1)
    degenerate = twice_area <= 1e-10
    nz_fraction = np.abs(cross[:, 2]) / np.maximum(twice_area, 1e-15)
    try:
        area = float(mesh.get_surface_area())
    except Exception:
        area = float(np.sum(twice_area) * 0.5)
    bounds_min = vertices.min(axis=0)
    bounds_max = vertices.max(axis=0)
    return {
        "vertices": int(len(vertices)),
        "triangles": int(len(triangles)),
        "degenerate_triangles": int(np.count_nonzero(degenerate)),
        "surface_area_m2": area,
        "bounds_min_xyz_m": bounds_min.tolist(),
        "bounds_max_xyz_m": bounds_max.tolist(),
        "bounds_size_xyz_m": (bounds_max - bounds_min).tolist(),
        "upward_or_downward_facing_fraction_abs_nz_ge_0_3": float(np.mean(nz_fraction >= 0.3)),
        "coordinate_units": "metres (StageGen world coordinates)",
    }


def _read_open3d_mesh(path: Path, label: str):
    o3d = _open3d()
    if not path.is_file():
        raise GeometryWorkspaceError(f"{label}: file does not exist: {path}")
    try:
        mesh = o3d.io.read_triangle_mesh(str(path), enable_post_processing=False)
    except Exception as exc:
        raise GeometryWorkspaceError(f"{label}: Open3D could not read {path.name}: {exc}") from exc
    _finite_mesh(mesh, label)
    return mesh


def _write_ply(mesh, path: Path, label: str) -> None:
    o3d = _open3d()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        mesh.compute_vertex_normals()
        ok = o3d.io.write_triangle_mesh(
            str(path), mesh, write_ascii=False, compressed=False,
            write_vertex_normals=True, write_vertex_colors=True,
            write_triangle_uvs=False, print_progress=False,
        )
    except Exception as exc:
        raise GeometryWorkspaceError(f"{label}: could not write PLY {path}: {exc}") from exc
    if not ok or not path.is_file() or path.stat().st_size == 0:
        raise GeometryWorkspaceError(f"{label}: Open3D failed to write {path}")


def _raycast_road_to_terrain(road_mesh, terrain_mesh, sample_count: int,
                             max_down_distance_m: float = 100.0,
                             max_up_distance_m: float = 100.0,
                             epsilon_m: float = 0.02) -> dict[str, Any]:
    """Sample upward-facing road triangles and cast separate up/down terrain rays.

    The downward ray starts just below the sampled road face. This prevents an
    elevated terrain/bridge surface above the road from being misidentified as
    the supporting ground. An upward ray separately reports overhead surfaces.
    It is still a geometric mesh diagnostic, not Unreal's runtime physics query.
    """
    o3d = _open3d()
    road_vertices, road_triangles = _finite_mesh(road_mesh, "road collision")
    terrain_vertices, terrain_triangles = _finite_mesh(terrain_mesh, "terrain collision")

    tri_xyz = road_vertices[road_triangles]
    cross = np.cross(tri_xyz[:, 1] - tri_xyz[:, 0], tri_xyz[:, 2] - tri_xyz[:, 0])
    norm = np.linalg.norm(cross, axis=1)
    # Winding in cooked collision may vary, so use |normal.z|. Side walls and
    # near-vertical faces are excluded; banked road decks remain eligible.
    upwardish = (norm > 1e-12) & (np.abs(cross[:, 2]) / np.maximum(norm, 1e-15) >= 0.30)
    road_surface = o3d.geometry.TriangleMesh()
    road_surface.vertices = o3d.utility.Vector3dVector(road_vertices)
    road_surface.triangles = o3d.utility.Vector3iVector(road_triangles[upwardish].astype(np.int32))
    if len(road_surface.triangles) == 0:
        raise GeometryWorkspaceError("road collision has no non-degenerate, upward-facing triangles to sample")

    # Open3D's sampling is area-weighted. Use a fixed seed for reproducible
    # reports. The count is clamped to protect the GUI from pathological jobs.
    sample_count = max(100, min(int(sample_count), 100_000))
    try:
        points_cloud = road_surface.sample_points_uniformly(
            number_of_points=sample_count, use_triangle_normal=True, seed=17
        )
    except TypeError:
        # Compatibility fallback for Open3D builds that predate the seed arg.
        points_cloud = road_surface.sample_points_uniformly(
            number_of_points=sample_count, use_triangle_normal=True
        )
    points = np.asarray(points_cloud.points, dtype=np.float64)
    if len(points) == 0:
        raise GeometryWorkspaceError("Open3D did not produce road surface samples")

    terrain_for_raycast = o3d.t.geometry.TriangleMesh.from_legacy(terrain_mesh)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(terrain_for_raycast)

    down_origins = points.copy()
    down_origins[:, 2] -= epsilon_m
    down_rays = np.column_stack((down_origins, np.tile([0.0, 0.0, -1.0], (len(points), 1))))
    up_origins = points.copy()
    up_origins[:, 2] += epsilon_m
    up_rays = np.column_stack((up_origins, np.tile([0.0, 0.0, 1.0], (len(points), 1))))

    down_hit = scene.cast_rays(o3d.core.Tensor(down_rays.astype(np.float32)))
    up_hit = scene.cast_rays(o3d.core.Tensor(up_rays.astype(np.float32)))
    down_t = down_hit["t_hit"].numpy().astype(np.float64)
    up_t = up_hit["t_hit"].numpy().astype(np.float64)

    down_found = np.isfinite(down_t) & (down_t <= max_down_distance_m)
    up_found = np.isfinite(up_t) & (up_t <= max_up_distance_m)
    down_gap = epsilon_m + down_t[down_found]
    overhead_clearance = epsilon_m + up_t[up_found]

    def distribution(values: np.ndarray) -> dict[str, float | None]:
        if len(values) == 0:
            return {"min": None, "p50": None, "p95": None, "max": None}
        return {
            "min": float(np.min(values)),
            "p50": float(np.percentile(values, 50)),
            "p95": float(np.percentile(values, 95)),
            "max": float(np.max(values)),
        }

    return {
        "method": "Open3D RaycastingScene; area-weighted road samples; separate vertical rays",
        "sample_count": int(len(points)),
        "road_surface_triangle_count": int(len(road_surface.triangles)),
        "road_source_triangle_count": int(len(road_triangles)),
        "downward_ground_hits": int(np.count_nonzero(down_found)),
        "downward_ground_coverage_fraction": float(np.mean(down_found)),
        "downward_ground_gap_m": distribution(down_gap),
        "ground_within_1m_fraction": float(np.mean(down_found & (down_t + epsilon_m <= 1.0))),
        "ground_within_3m_fraction": float(np.mean(down_found & (down_t + epsilon_m <= 3.0))),
        "ground_within_6m_fraction": float(np.mean(down_found & (down_t + epsilon_m <= 6.0))),
        "no_ground_within_search_distance_count": int(np.count_nonzero(~down_found)),
        "upward_surface_hits": int(np.count_nonzero(up_found)),
        "upward_surface_clearance_m": distribution(overhead_clearance),
        "overhead_surface_within_1m_count": int(np.count_nonzero(up_found & (up_t + epsilon_m <= 1.0))),
        "max_down_search_m": float(max_down_distance_m),
        "max_up_search_m": float(max_up_distance_m),
        "warning": "Geometric mesh comparison only; not an Unreal runtime physics/collision query.",
    }


def analyse_stage(level: str, paks: str, out_dir: str, corridor_m: float = 30.0,
                  sample_count: int = 20_000) -> dict[str, Any]:
    """Extract a stock stage's baked collision mesh exports and analyse with Open3D."""
    from .extract.iostore import GameFS
    from .inspect_collision import inspect_one

    out = Path(out_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    print("@progress 0.03 Opening stock stage packages", flush=True)
    fs = GameFS(os.path.normpath(paks), include_mods=False)
    print("@progress 0.12 Extracting the stock road/terrain collision meshes", flush=True)
    source_report = inspect_one(fs, level, str(out), corridor_m=corridor_m, write_obj=True)

    stem = re_safe(level)
    road_obj = out / f"{stem}_road_collision.obj"
    terrain_obj = out / f"{stem}_terrain_collision.obj"
    exchange = out / "exchange"
    source_meshes: dict[str, Path] = {}
    stats: dict[str, Any] = {}

    for label, source in (("road", road_obj), ("terrain", terrain_obj)):
        if not source.is_file():
            stats[label] = {"available": False, "reason": "No collision OBJ was exported for this class."}
            continue
        print(f"@progress 0.20 Reading {label} collision mesh with Open3D", flush=True)
        mesh = _read_open3d_mesh(source, f"{label} collision")
        stats[label] = {"available": True, **_mesh_stats(mesh, f"{label} collision")}
        ply = exchange / f"{label}_collision.ply"
        _write_ply(mesh, ply, f"{label} collision")
        source_meshes[label] = ply

    road_terrain: dict[str, Any]
    if "road" in source_meshes and "terrain" in source_meshes:
        print("@progress 0.42 Running Open3D road-to-ground ray analysis", flush=True)
        road_mesh = _read_open3d_mesh(source_meshes["road"], "road collision PLY")
        terrain_mesh = _read_open3d_mesh(source_meshes["terrain"], "terrain collision PLY")
        road_terrain = _raycast_road_to_terrain(road_mesh, terrain_mesh, sample_count)
    else:
        road_terrain = {
            "available": False,
            "reason": "Both road and terrain collision exports are required for road-to-ground ray analysis.",
        }

    report = {
        "schema_version": 1,
        "stage": level,
        "source_kind": "original stock stage (GameFS opened with include_mods=False)",
        "workspace": str(out),
        "source_collision_report": f"{stem}_collision.json",
        "source_meshes": {
            "road_obj": str(road_obj) if road_obj.is_file() else None,
            "terrain_obj": str(terrain_obj) if terrain_obj.is_file() else None,
            "road_ply": str(source_meshes["road"]) if "road" in source_meshes else None,
            "terrain_ply": str(source_meshes["terrain"]) if "terrain" in source_meshes else None,
        },
        "open3d_mesh_statistics": stats,
        "road_to_terrain_analysis": road_terrain,
        "limitations": [
            "These exports are baked collision meshes, not every visible render mesh in the level.",
            "OBJ/PLY files are interchange geometry, not cooked Unreal .uasset/.umap packages.",
            "This command does not edit or install the original game stage.",
            "Ray tests are geometric diagnostics, not Unreal runtime physics queries.",
        ],
        "source_report_summary": {
            "collision_mesh_count": source_report.get("collision_mesh_count"),
            "collision_triangle_count": source_report.get("collision_triangle_count"),
            "source_asset_inventory": source_report.get("source_asset_inventory"),
        },
    }
    report_path = out / "open3d_stage_analysis.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "WORKSPACE_README.txt").write_text(
        "ACR Remixer geometry workspace\n"
        f"Stage: {level}\n\n"
        "Original collision OBJ files are in this folder.\n"
        "Open3D interchange PLY files are in exchange/.\n"
        "Blender-processed PLY copies and native .blend projects are written to processed/.\n\n"
        "IMPORTANT: processed OBJ/PLY files are not automatically repacked into ACR cooked packages.\n"
        "Do not copy them into the game's Paks folder.\n",
        encoding="utf-8",
    )
    print(f"@progress 1.0 Open3D analysis complete: {report_path}", flush=True)
    print(f"geometry workspace: {out}")
    print(f"Open3D report: {report_path}")
    return report


def re_safe(value: str) -> str:
    import re
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "stage"


def _stats_equivalent(before: dict[str, Any], after: dict[str, Any], operation: str) -> dict[str, Any]:
    """Assess whether Blender altered the geometric surface during the selected operation."""
    checks = {
        "vertex_count_same": before["vertices"] == after["vertices"],
        "triangle_count_same": before["triangles"] == after["triangles"],
        "bounds_min_same_within_5e-3m": bool(np.allclose(
            before["bounds_min_xyz_m"], after["bounds_min_xyz_m"], atol=5e-3, rtol=0.0)),
        "bounds_max_same_within_5e-3m": bool(np.allclose(
            before["bounds_max_xyz_m"], after["bounds_max_xyz_m"], atol=5e-3, rtol=0.0)),
        "surface_area_same_within_1e-4_relative": bool(math.isclose(
            before["surface_area_m2"], after["surface_area_m2"], rel_tol=1e-4, abs_tol=1e-6)),
    }
    return {
        "operation": operation,
        "checks": checks,
        "geometry_unchanged": all(checks.values()) if operation == "recalculate_normals" else None,
        "note": "The normals-only operation is expected to preserve geometry; a 5 mm bounds tolerance accommodates Blender mesh-coordinate float precision.",
    }


def process_workspace_with_blender(workspace: str, operation: str = "recalculate_normals") -> dict[str, Any]:
    """Run Blender headlessly on copied PLY exports; never overwrite source meshes."""
    root = Path(workspace).expanduser().resolve()
    exchange = root / "exchange"
    processed = root / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    candidates = [p for p in (exchange / "road_collision.ply", exchange / "terrain_collision.ply") if p.is_file()]
    if not candidates:
        raise GeometryWorkspaceError(
            f"No exchange/*.ply meshes found in {root}. Run 'Analyze selected stock stage' first."
        )

    blender = toolchain.find_blender()
    if blender is None:
        print("@progress 0.02 Blender is missing; downloading the managed Blender runtime", flush=True)
        toolchain.install_blender(force=False)
        blender = toolchain.find_blender()
    if blender is None:
        raise GeometryWorkspaceError("Blender installation completed but blender.exe could not be found")

    results: list[dict[str, Any]] = []
    total = len(candidates)
    for index, source in enumerate(candidates):
        label = source.stem.replace("_collision", "")
        target = processed / f"{label}_collision_processed.ply"
        blend_project = processed / f"{label}_collision_processed.blend"
        print(f"@progress {0.08 + 0.70 * index / total:.3f} Blender processing {label} mesh", flush=True)
        before_mesh = _read_open3d_mesh(source, f"input {label}")
        before_stats = _mesh_stats(before_mesh, f"input {label}")
        blender_result = toolchain.blender_process_mesh(
            source, target, operation=operation, blend_output_path=blend_project
        )
        after_mesh = _read_open3d_mesh(target, f"Blender output {label}")
        after_stats = _mesh_stats(after_mesh, f"Blender output {label}")
        comparison = _stats_equivalent(before_stats, after_stats, operation)
        if operation == "recalculate_normals" and not comparison["geometry_unchanged"]:
            target.unlink(missing_ok=True)
            raise GeometryWorkspaceError(
                f"Blender normals-only pass unexpectedly changed {label} geometry; processed copy was discarded. "
                f"Checks: {comparison['checks']}"
            )
        results.append({
            "mesh": label,
            "input": str(source),
            "output": str(target),
            "blend_project": str(blend_project),
            "operation": operation,
            "blender": blender_result,
            "before": before_stats,
            "after": after_stats,
            "comparison": comparison,
        })

    report = {
        "schema_version": 1,
        "workspace": str(root),
        "blender_executable": str(blender),
        "operation": operation,
        "processed_meshes": results,
        "safety": {
            "source_exchange_meshes_overwritten": False,
            "game_packages_modified": False,
            "game_stage_installed": False,
        },
        "limitations": [
            "Blender processed exported collision PLY meshes only and saved editable .blend project files.",
            "This result is not a cooked Unreal StaticMesh or BakedData package.",
            "Package reintegration is a separate future implementation phase.",
        ],
    }
    report_path = root / "blender_processing_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"@progress 1.0 Blender mesh processing complete: {report_path}", flush=True)
    print(f"processed mesh workspace: {processed}")
    print(f"Blender report: {report_path}")
    return report

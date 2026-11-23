# SPDX-FileCopyrightText: 2026 hiro98 (vale981) and the ACR Remixer contributors
# SPDX-License-Identifier: MIT

"""Read-only stage geometry/collision inventory for UE 5.6 World Partition stages.

The inspector deliberately does not modify assets. It exports collision mesh
statistics, approximate road-to-collision correspondence, and the primitive
shape inventory from DA_<Level>BakedData.
"""
from __future__ import annotations

import csv
import json
import os
import re
import struct
from collections import Counter

import numpy as np
from scipy.spatial import cKDTree

from .build import writers as W
from .extract.real import load_wp_source
from .extract.zen import Package


def _fstring_from_raw(data: bytes) -> str:
    if len(data) < 4:
        return ""
    n, = struct.unpack_from("<i", data, 0)
    if n == 0:
        return ""
    if n > 0 and 4 + n <= len(data):
        return data[4:4 + n].rstrip(b"\0").decode("latin-1", errors="replace")
    if n < 0 and 4 + (-2 * n) <= len(data):
        return data[4:4 + (-2 * n)].decode("utf-16-le", errors="replace").rstrip("\0")
    return "<invalid-fstring>"


def _actor_path(pk: Package, mesh) -> str:
    pkg = pk.mapped(*mesh.soft_pkg)
    asset = pk.mapped(*mesh.soft_asset)
    sub = _fstring_from_raw(mesh.soft_sub)
    return sub or f"{pkg}.{asset}"


def _material_path(pk: Package, ref: int) -> str:
    if ref < 0:
        i = -ref - 1
        try:
            return pk.imported_package_names()[i]
        except (IndexError, ValueError):
            return f"import[{i}]"
    if ref > 0:
        i = ref - 1
        return pk.exports[i].name if i < len(pk.exports) else f"export[{i}]"
    return "none"


def _kind(actor: str, material: str) -> str:
    text = f"{actor} {material}".lower()
    if "originquad" in text:
        return "origin"
    if "road" in text:
        return "road"
    if "terrain" in text or "landscape" in text:
        return "terrain"
    if "pcggrass" in text or "grass" in text:
        return "vegetation"
    return "other"


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "stage"



def _closest_points_on_triangles(points: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    """Closest points on triangles for broadcast-compatible point/triangle arrays.

    points: (..., 3), triangles: (..., 3, 3), with matching leading dimensions.
    Uses plane projection when the projection lies inside the face; otherwise
    selects the nearest of the three edge segments. Degenerate triangles fall
    back to their edges.
    """
    p = np.asarray(points, dtype=np.float64)
    tri = np.asarray(triangles, dtype=np.float64)
    a, b, c = tri[..., 0, :], tri[..., 1, :], tri[..., 2, :]

    def on_segment(x, u, v):
        uv = v - u
        den = np.sum(uv * uv, axis=-1)
        t = np.divide(np.sum((x - u) * uv, axis=-1), den,
                      out=np.zeros_like(den), where=den > 1e-20)
        t = np.clip(t, 0.0, 1.0)
        return u + t[..., None] * uv

    q_ab = on_segment(p, a, b)
    q_bc = on_segment(p, b, c)
    q_ca = on_segment(p, c, a)
    d_ab = np.sum((p - q_ab) ** 2, axis=-1)
    d_bc = np.sum((p - q_bc) ** 2, axis=-1)
    d_ca = np.sum((p - q_ca) ** 2, axis=-1)
    edge_q = np.stack((q_ab, q_bc, q_ca), axis=-2)
    edge_d = np.stack((d_ab, d_bc, d_ca), axis=-1)
    best_edge = np.argmin(edge_d, axis=-1)
    edge_best = np.take_along_axis(edge_q, best_edge[..., None, None], axis=-2)[..., 0, :]

    ab, ac = b - a, c - a
    normal = np.cross(ab, ac)
    n2 = np.sum(normal * normal, axis=-1)
    signed_num = np.sum((p - a) * normal, axis=-1)
    proj = p - np.divide(signed_num, n2, out=np.zeros_like(signed_num), where=n2 > 1e-20)[..., None] * normal
    ap = proj - a
    d00 = np.sum(ab * ab, axis=-1)
    d01 = np.sum(ab * ac, axis=-1)
    d11 = np.sum(ac * ac, axis=-1)
    d20 = np.sum(ap * ab, axis=-1)
    d21 = np.sum(ap * ac, axis=-1)
    denom = d00 * d11 - d01 * d01
    v = np.divide(d11 * d20 - d01 * d21, denom,
                  out=np.zeros_like(denom), where=np.abs(denom) > 1e-20)
    w = np.divide(d00 * d21 - d01 * d20, denom,
                  out=np.zeros_like(denom), where=np.abs(denom) > 1e-20)
    inside = (n2 > 1e-20) & (v >= -1e-10) & (w >= -1e-10) & (v + w <= 1.0 + 1e-10)
    return np.where(inside[..., None], proj, edge_best)


def _resample_route(points: np.ndarray, widths: np.ndarray, step_m: float = 10.0):
    """Resample route and width arrays at an approximately uniform arc-length step."""
    p = np.asarray(points, dtype=np.float64)
    w = np.asarray(widths, dtype=np.float64)
    if len(p) < 2:
        return p, w
    seg = np.linalg.norm(np.diff(p[:, :3], axis=0), axis=1)
    cumulative = np.r_[0.0, np.cumsum(seg)]
    total = float(cumulative[-1])
    if total <= 0:
        return p[:1], w[:1]
    stations = np.arange(0.0, total, max(1.0, step_m))
    if len(stations) == 0 or stations[-1] < total:
        stations = np.r_[stations, total]
    result = np.column_stack([np.interp(stations, cumulative, p[:, axis]) for axis in range(3)])
    out_width = np.interp(stations, cumulative, w)
    return result, out_width

def inspect_one(fs, level: str, out_dir: str, corridor_m: float = 30.0,
                write_obj: bool = False) -> dict:
    """Inspect one source level and write JSON/CSV plus optional OBJ exports."""
    ws = load_wp_source(fs, level)
    baked_pk = Package(fs.read(ws.baked))
    baked = W.read_baked_full(baked_pk)
    shape_info = W.inspect_baked_shapes(baked_pk, baked)

    route = np.asarray(ws.stage.points, dtype=np.float64)
    if len(route) < 2:
        raise ValueError(f"{level}: route spline has fewer than two points")
    route_tree = cKDTree(route[:, :2])
    route_step = float(np.median(np.linalg.norm(np.diff(route[:, :2], axis=0), axis=1)))
    mesh_rows = []
    kind_counts = Counter()
    group_counts = Counter()
    total_tris = 0
    road_triangles = []
    obj_buffers = {"road": [], "terrain": []}

    for mesh_index, mesh in enumerate(baked.meshes):
        actor = _actor_path(baked_pk, mesh)
        material = _material_path(baked_pk, mesh.material_ref)
        kind = _kind(actor, material)
        kind_counts[kind] += 1
        group_counts[str(mesh.group)] += 1

        # BakedData stores (X, up, Y); convert to UE world axes (X, Y, Z-up).
        vertices = np.asarray(mesh.vertices, dtype=np.float64)[:, [0, 2, 1]]
        raw_indices = np.asarray(mesh.indices, dtype=np.uint32).reshape(-1)
        if raw_indices.size % 3:
            raise ValueError(f"{actor}: index count {raw_indices.size} is not divisible by 3")
        tris_all = raw_indices.reshape(-1, 3)
        valid = np.all(tris_all < len(vertices), axis=1)
        invalid_count = int((~valid).sum())
        tris = tris_all[valid].astype(np.int64, copy=False)
        tri_count = len(tris)
        total_tris += tri_count

        row = {
            "mesh_index": mesh_index, "kind": kind, "actor": actor,
            "material_ref": int(mesh.material_ref), "material": material,
            "collision_group": int(mesh.group), "vertices": int(len(vertices)),
            "triangles": int(tri_count), "invalid_index_triangles": invalid_count,
            "degenerate_triangles": 0, "zero_area_triangles": 0,
            "min_route_distance_m": None, "p50_route_distance_m": None,
            "p95_route_distance_m": None, "max_route_distance_m": None,
            "within_corridor_triangles": 0, "route_s_min_m": None, "route_s_max_m": None,
            "bounds_min_x": None, "bounds_min_y": None, "bounds_min_z": None,
            "bounds_max_x": None, "bounds_max_y": None, "bounds_max_z": None,
            "median_triangle_area_m2": None, "upward_normal_fraction": None,
        }
        if len(vertices):
            lo, hi = vertices.min(0), vertices.max(0)
            row.update({
                "bounds_min_x": float(lo[0]), "bounds_min_y": float(lo[1]), "bounds_min_z": float(lo[2]),
                "bounds_max_x": float(hi[0]), "bounds_max_y": float(hi[1]), "bounds_max_z": float(hi[2]),
            })
        if tri_count:
            p = vertices[tris]
            repeated = ((tris[:, 0] == tris[:, 1]) | (tris[:, 1] == tris[:, 2]) |
                        (tris[:, 0] == tris[:, 2]))
            cross = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
            area = np.linalg.norm(cross, axis=1) * 0.5
            zero_area = area <= 1e-10
            row["degenerate_triangles"] = int(repeated.sum())
            row["zero_area_triangles"] = int(zero_area.sum())
            row["median_triangle_area_m2"] = float(np.median(area))
            nz = np.abs(cross[:, 2]) / np.maximum(np.linalg.norm(cross, axis=1), 1e-12)
            row["upward_normal_fraction"] = float(np.mean(nz > 0.5))

            # Triangle-centre proximity is an inventory metric, not a collision
            # raycast. It helps rank which source meshes lie along the route.
            centres = p.mean(axis=1)
            distances, nearest = route_tree.query(centres[:, :2], k=1)
            nearest_s = nearest.astype(np.float64) * route_step
            row.update({
                "min_route_distance_m": float(np.min(distances)),
                "p50_route_distance_m": float(np.median(distances)),
                "p95_route_distance_m": float(np.percentile(distances, 95)),
                "max_route_distance_m": float(np.max(distances)),
                "within_corridor_triangles": int(np.count_nonzero(distances <= corridor_m)),
                "route_s_min_m": float(np.min(nearest_s[distances <= corridor_m])) if np.any(distances <= corridor_m) else None,
                "route_s_max_m": float(np.max(nearest_s[distances <= corridor_m])) if np.any(distances <= corridor_m) else None,
            })
            if kind == "road":
                road_triangles.append(p)
            if write_obj and kind in obj_buffers and len(tris):
                obj_buffers[kind].append((vertices, tris))
        mesh_rows.append(row)

    # More precise route-to-collision correspondence: resample the route at 10 m,
    # find the nearest 64 road-triangle centroids in XY, then calculate exact
    # point-to-triangle distances in 3D for those candidates. This is still a
    # bounded-neighbour approximation (not a full Unreal raycast), but unlike
    # centroid-only matching it measures distance to the actual triangle faces.
    route_check, width_check = _resample_route(route, np.asarray(ws.stage.width, dtype=np.float64), 10.0)
    coverage = {"available": False, "method": "nearest 64 XY triangle centroids + exact 3D point-to-triangle distance",
                "sample_spacing_m": 10.0, "sample_count": int(len(route_check)), "covered_samples": 0,
                "coverage_fraction": None, "p50_3d_distance_m": None,
                "p95_3d_distance_m": None, "max_3d_distance_m": None,
                "p95_xy_distance_m": None, "median_height_delta_m": None,
                "height_tolerance_m": [-3.0, 1.5]}
    if road_triangles:
        RT = np.concatenate(road_triangles, axis=0)
        RT_centres = RT.mean(axis=1)
        rt = cKDTree(RT_centres[:, :2])
        k = min(64, len(RT_centres))
        _centroid_dist, candidate_ix = rt.query(route_check[:, :2], k=k)
        if k == 1:
            candidate_ix = candidate_ix[:, None]
        candidates = RT[candidate_ix]
        query_points = np.broadcast_to(route_check[:, None, :], (len(route_check), k, 3))
        closest = _closest_points_on_triangles(query_points, candidates)
        delta = closest - query_points
        d3 = np.linalg.norm(delta, axis=2)
        best = np.argmin(d3, axis=1)
        rows = np.arange(len(route_check))
        best_d3 = d3[rows, best]
        best_closest = closest[rows, best]
        best_xy = np.linalg.norm(best_closest[:, :2] - route_check[:, :2], axis=1)
        dz = best_closest[:, 2] - route_check[:, 2]
        half_width = width_check / 2.0
        xy_ok = best_xy <= np.maximum(half_width + 2.0, 4.0)
        z_ok = (dz >= -3.0) & (dz <= 1.5)
        good = xy_ok & z_ok
        coverage = {
            "available": True,
            "method": "nearest 64 XY triangle centroids + exact 3D point-to-triangle distance",
            "sample_spacing_m": 10.0, "sample_count": int(len(route_check)),
            "covered_samples": int(good.sum()),
            "coverage_fraction": float(good.mean()),
            "p50_3d_distance_m": float(np.percentile(best_d3, 50)),
            "p95_3d_distance_m": float(np.percentile(best_d3, 95)),
            "max_3d_distance_m": float(np.max(best_d3)),
            "p95_xy_distance_m": float(np.percentile(best_xy, 95)),
            "median_height_delta_m": float(np.median(dz)),
            "height_tolerance_m": [-3.0, 1.5],
            "xy_tolerance_per_sample_m": "max(road_half_width + 2 m, 4 m)",
            "candidate_triangle_count": int(len(RT)),
            "candidate_centroids_per_sample": int(k),
        }

    env_prefix = f"acr/Content/Environments/{level}/"
    source_assets = {
        "road_mesh_chunks": [], "terrain_mesh_chunks": [],
        "physical_material_masks": [], "terrain_masks": [], "terrain_meshes": [],
    }
    for path in sorted(fs.index):
        if not path.startswith(env_prefix):
            continue
        lower = path.lower()
        if "/generatedsplits/" in lower and "_road_" in lower and path.endswith(".uasset"):
            source_assets["road_mesh_chunks"].append(path)
        elif "/generatedsplits/" in lower and "_terrain_" in lower and path.endswith(".uasset"):
            source_assets["terrain_mesh_chunks"].append(path)
        elif "/physmatmasks/" in lower:
            source_assets["physical_material_masks"].append(path)
        elif "/terrainmasks/" in lower:
            source_assets["terrain_masks"].append(path)
        elif "/meshesterrain/" in lower:
            source_assets["terrain_meshes"].append(path)

    report = {
        "schema_version": 1,
        "stage": level,
        "source_map": ws.main,
        "baked_data_asset": ws.baked,
        "route_length_m": float(np.linalg.norm(np.diff(route, axis=0), axis=1).sum()),
        "route_sample_step_m": route_step,
        "route_sample_count": int(len(route)),
        "route_centreline_m": np.round(route, 3).tolist(),
        "route_width_m": np.round(np.asarray(ws.stage.width, dtype=np.float64), 3).tolist(),
        "source_asset_inventory": {k: {"count": len(v), "paths": v} for k, v in source_assets.items()},
        "collision_mesh_count": int(len(mesh_rows)),
        "collision_triangle_count": int(total_tris),
        "mesh_kind_counts": dict(kind_counts),
        "collision_group_mesh_counts": dict(group_counts),
        "road_collision_centreline_coverage": coverage,
        "primitive_collision_shapes": shape_info,
        "notes": [
            "Road-to-collision coverage uses exact point-to-triangle distance for the 64 closest triangle centroids in XY at 10 m route intervals; it is a bounded candidate search, not an Unreal physics raycast.",
            "Primitive-shape field payloads are inventoried but not decoded semantically.",
            "A clean report does not prove Unreal's runtime physics/replay systems will accept the stage."
        ],
        "meshes": mesh_rows,
    }

    os.makedirs(out_dir, exist_ok=True)
    stem = _safe_name(level)
    json_path = os.path.join(out_dir, stem + "_collision.json")
    csv_path = os.path.join(out_dir, stem + "_collision_meshes.csv")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    if mesh_rows:
        with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(mesh_rows[0]))
            writer.writeheader(); writer.writerows(mesh_rows)
    if write_obj:
        for kind, items in obj_buffers.items():
            if not items:
                continue
            path = os.path.join(out_dir, f"{stem}_{kind}_collision.obj")
            with open(path, "w", encoding="ascii", newline="\n") as f:
                f.write(f"# ACR Remixer collision export: {level} / {kind}\n")
                offset = 1
                for vertices, tris in items:
                    for v in vertices:
                        f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
                    for t in tris:
                        f.write(f"f {int(t[0])+offset} {int(t[1])+offset} {int(t[2])+offset}\n")
                    offset += len(vertices)
        route_path = os.path.join(out_dir, f"{stem}_route_centreline.obj")
        with open(route_path, "w", encoding="ascii", newline="\n") as f:
            f.write(f"# ACR Remixer source route centreline: {level}\n")
            for p in route:
                f.write(f"v {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
            if len(route) >= 2:
                f.write("l " + " ".join(str(i) for i in range(1, len(route) + 1)) + "\n")
    print(f"  collision inspector: {len(mesh_rows)} meshes, {total_tris:,} triangles; "
          f"road-centreline coverage {coverage['coverage_fraction'] if coverage['available'] else 'n/a'}")
    print(f"  wrote {json_path}")
    print(f"  wrote {csv_path}")
    if write_obj:
        print(f"  wrote {os.path.join(out_dir, stem + '_route_centreline.obj')}")
    return report


def inspect_levels(fs, levels: list[str], out_dir: str,
                   corridor_m: float = 30.0, write_obj: bool = False) -> dict:
    all_reports, errors = [], []
    for level in levels:
        try:
            all_reports.append(inspect_one(fs, level, out_dir, corridor_m, write_obj))
        except Exception as exc:  # report per-stage failure and continue through remaining stages
            errors.append({"stage": level, "error": f"{type(exc).__name__}: {exc}"})
            print(f"  skip {level}: {type(exc).__name__}: {exc}")
    index = {"schema_version": 1, "inspected": [r["stage"] for r in all_reports], "errors": errors,
             "report_count": len(all_reports)}
    with open(os.path.join(out_dir, "collision_inspection_index.json"), "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)
    return index

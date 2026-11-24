# SPDX-FileCopyrightText: 2026 ACR Remixer contributors
# SPDX-License-Identifier: MIT
"""Read-only inventory of OBJ triangle meshes using Open3D.

No transforms, repair, coordinate conversion, or asset modification is performed.
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


def discover_obj_files(inputs: list[Path]) -> list[Path]:
    found: set[Path] = set()
    for item in inputs:
        if item.is_file() and item.suffix.lower() == ".obj":
            found.add(item.resolve())
        elif item.is_dir():
            found.update(p.resolve() for p in item.rglob("*") if p.is_file() and p.suffix.lower() == ".obj")
        else:
            raise FileNotFoundError(f"Input does not exist or is not an OBJ/directory: {item}")
    return sorted(found, key=lambda p: str(p).lower())


def inspect_mesh(path: Path) -> dict[str, Any]:
    mesh = o3d.io.read_triangle_mesh(str(path), enable_post_processing=False)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    triangles = np.asarray(mesh.triangles, dtype=np.int64)

    if vertices.size == 0 or triangles.size == 0:
        raise ValueError(f"No triangle mesh geometry could be read from {path}")
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(f"Unexpected vertex array shape {vertices.shape} in {path}")
    if not np.isfinite(vertices).all():
        raise ValueError(f"Mesh contains non-finite vertex coordinates: {path}")
    if triangles.ndim != 2 or triangles.shape[1] != 3:
        raise ValueError(f"Unexpected triangle array shape {triangles.shape} in {path}")
    if int(triangles.min()) < 0 or int(triangles.max()) >= len(vertices):
        raise ValueError(f"Mesh contains out-of-range triangle indices: {path}")

    minimum = vertices.min(axis=0)
    maximum = vertices.max(axis=0)
    dimensions = maximum - minimum

    return {
        "file": path.name,
        "relative_or_absolute_path": str(path),
        "vertices": int(len(vertices)),
        "triangles": int(len(triangles)),
        "finite_vertices": True,
        "bounds_min_xyz": [float(v) for v in minimum],
        "bounds_max_xyz": [float(v) for v in maximum],
        "dimensions_xyz": [float(v) for v in dimensions],
        "surface_area": float(mesh.get_surface_area()),
        "watertight": bool(mesh.is_watertight()),
        "edge_manifold_allowing_boundary": bool(mesh.is_edge_manifold(allow_boundary_edges=True)),
        "coordinate_note": "Coordinates are reported as imported. Units and axis orientation are not inferred; no transform is applied.",
    }


def write_reports(rows: list[dict[str, Any]], json_path: Path, csv_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "tool": "ACR Remixer Open3D mesh inventory",
        "open3d_version": o3d.__version__,
        "mesh_count": len(rows),
        "coordinate_note": "No unit inference, axis conversion, alignment, repair, or geometry modification is performed.",
        "meshes": rows,
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    columns = [
        "file", "vertices", "triangles", "surface_area", "watertight",
        "edge_manifold_allowing_boundary", "bounds_min_xyz", "bounds_max_xyz", "dimensions_xyz",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(row[key]) if isinstance(row[key], list) else row[key] for key in columns})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="OBJ file(s) and/or directories to scan recursively")
    parser.add_argument("--json", dest="json_path", type=Path, default=Path("artifacts/mesh_inventory.json"))
    parser.add_argument("--csv", dest="csv_path", type=Path, default=Path("artifacts/mesh_inventory.csv"))
    args = parser.parse_args()

    files = discover_obj_files(args.inputs)
    if not files:
        parser.error("No OBJ files were found in the supplied inputs")

    rows = [inspect_mesh(path) for path in files]
    write_reports(rows, args.json_path, args.csv_path)
    print(f"Mesh inventory: {len(rows)} OBJ file(s); Open3D {o3d.__version__}")
    for row in rows:
        print(f"  {row['file']}: {row['vertices']} vertices, {row['triangles']} triangles")
    print(f"JSON: {args.json_path}\nCSV:  {args.csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# SPDX-FileCopyrightText: 2026 ACR Remixer contributors
# SPDX-License-Identifier: MIT
"""Headless Blender OBJ importer and mesh inventory reporter.

Run with Blender, not the system Python:
  blender --background --factory-startup --python-exit-code 1 \
    --python tools/blender/inspect_mesh_exports.py -- \
    --input-dir tests/fixtures/geometry --json artifacts/blender_inventory.json

The script is read-only with respect to the OBJ inputs. It reports imported
object counts and world-space bounds; no geometry is transformed or repaired.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import bpy


def parse_args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--json", required=True, type=Path)
    parser.add_argument("--save-blend", type=Path, default=None)
    return parser.parse_args(argv)


def discover_obj_files(directory: Path) -> list[Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"OBJ input directory not found: {directory}")
    return sorted((p.resolve() for p in directory.rglob("*") if p.is_file() and p.suffix.lower() == ".obj"),
                  key=lambda p: str(p).lower())


def mesh_object_summary(obj: bpy.types.Object, source_file: str) -> dict:
    mesh = obj.data
    mesh.calc_loop_triangles()
    world_vertices = [obj.matrix_world @ vertex.co for vertex in mesh.vertices]
    finite_vertices = [v for v in world_vertices if all(math.isfinite(float(c)) for c in v)]
    if not finite_vertices:
        raise ValueError(f"Imported object has no finite vertices: {obj.name}")
    minimum = [min(float(v[i]) for v in finite_vertices) for i in range(3)]
    maximum = [max(float(v[i]) for v in finite_vertices) for i in range(3)]
    return {
        "source_file": source_file,
        "object_name": obj.name,
        "vertices": len(mesh.vertices),
        "polygons": len(mesh.polygons),
        "triangles": len(mesh.loop_triangles),
        "finite_vertices": len(finite_vertices),
        "world_bounds_min_xyz": minimum,
        "world_bounds_max_xyz": maximum,
        "world_dimensions_xyz": [maximum[i] - minimum[i] for i in range(3)],
        "materials": [slot.material.name if slot.material else None for slot in obj.material_slots],
    }


def main() -> int:
    args = parse_args()
    inputs = discover_obj_files(args.input_dir)
    if not inputs:
        raise RuntimeError(f"No OBJ files found in {args.input_dir}")
    if not hasattr(bpy.ops.wm, "obj_import"):
        raise RuntimeError("This Blender build does not expose bpy.ops.wm.obj_import")

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)

    report = {
        "schema_version": 1,
        "tool": "ACR Remixer Blender mesh inventory",
        "blender_version": bpy.app.version_string,
        "coordinate_note": "Blender reports imported OBJ coordinates. No unit inference, axis conversion, repair, or geometry modification is performed.",
        "files": [],
        "objects": [],
    }

    for path in inputs:
        before = {obj.as_pointer() for obj in bpy.context.scene.objects}
        # StageGen diagnostic OBJs use Z-up coordinates. Explicitly match that
        # orientation so Blender's importer does not apply its default axis swap.
        bpy.ops.wm.obj_import(
            filepath=str(path),
            forward_axis="Y",
            up_axis="Z",
            global_scale=1.0,
        )
        imported = [obj for obj in bpy.context.scene.objects
                    if obj.as_pointer() not in before and obj.type == "MESH"]
        report["files"].append({"file": path.name, "mesh_objects_imported": len(imported)})
        for obj in imported:
            report["objects"].append(mesh_object_summary(obj, path.name))

    report["totals"] = {
        "files": len(report["files"]),
        "mesh_objects": len(report["objects"]),
        "vertices": sum(item["vertices"] for item in report["objects"]),
        "triangles": sum(item["triangles"] for item in report["objects"]),
    }
    if not report["objects"]:
        raise RuntimeError("OBJ files were found, but Blender imported no mesh objects")

    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["totals"], indent=2))
    print(f"Wrote Blender inventory: {args.json}")

    if args.save_blend:
        args.save_blend.parent.mkdir(parents=True, exist_ok=True)
        bpy.ops.wm.save_as_mainfile(filepath=str(args.save_blend))
        print(f"Saved inspection scene: {args.save_blend}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# SPDX-FileCopyrightText: 2026 hiro98 (vale981) and the ACR Remixer contributors
# SPDX-License-Identifier: MIT

"""Controlled Blender background worker for ACR Remixer mesh workspaces.

Invoked only by stagegen.toolchain.blender_process_mesh. It deliberately accepts
an allow-listed operation rather than executing caller-supplied Python code.
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path


def _args() -> argparse.Namespace:
    if "--" not in sys.argv:
        raise RuntimeError("Blender worker arguments must follow '--'")
    parser = argparse.ArgumentParser(prog="acr-remixer-blender-worker")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--blend-output", required=True, help="native Blender project saved after processing")
    parser.add_argument("--operation", choices=("recalculate_normals", "triangulate"), default="recalculate_normals")
    return parser.parse_args(sys.argv[sys.argv.index("--") + 1:])


def _mesh_objects():
    import bpy
    return [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]


def _apply_operation(objects, operation: str) -> dict:
    import bmesh
    counts = {"objects": len(objects), "vertices_before": 0, "vertices_after": 0,
              "faces_before": 0, "faces_after": 0}
    for obj in objects:
        mesh = obj.data
        counts["vertices_before"] += len(mesh.vertices)
        counts["faces_before"] += len(mesh.polygons)
        bm = bmesh.new()
        try:
            bm.from_mesh(mesh)
            if operation == "recalculate_normals":
                bmesh.ops.recalc_face_normals(bm, faces=list(bm.faces))
            elif operation == "triangulate":
                bmesh.ops.triangulate(bm, faces=list(bm.faces))
            bm.to_mesh(mesh)
            mesh.update()
        finally:
            bm.free()
        counts["vertices_after"] += len(mesh.vertices)
        counts["faces_after"] += len(mesh.polygons)
    return counts


def main() -> int:
    args = _args()
    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()
    report_path = Path(args.report).resolve()
    blend_output_path = Path(args.blend_output).resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Input PLY does not exist: {input_path}")
    if input_path.suffix.lower() != ".ply" or output_path.suffix.lower() != ".ply":
        raise ValueError("Blender bridge accepts PLY input and output to keep the exchange coordinate convention explicit")
    if input_path == output_path:
        raise ValueError("Input and output paths must be different; source mesh is never overwritten")
    if blend_output_path in (input_path, output_path, report_path):
        raise ValueError("Blender project path must be separate from input, mesh output and report")

    import bpy
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.wm.ply_import(
        filepath=str(input_path), global_scale=1.0, use_scene_unit=False,
        forward_axis="Y", up_axis="Z", merge_verts=False,
    )
    objects = _mesh_objects()
    if not objects:
        raise RuntimeError("Blender imported no mesh objects from the PLY file")
    # Give the project useful defaults for downstream mesh work. The PLY exchange
    # coordinates are preserved; scene units are labelled metric without scaling.
    bpy.context.scene.unit_settings.system = "METRIC"
    bpy.context.scene.unit_settings.scale_length = 1.0
    for index, obj in enumerate(objects, start=1):
        obj.name = f"{input_path.stem}_{index:02d}"
        obj.data.name = f"{input_path.stem}_{index:02d}_Mesh"
    counts = _apply_operation(objects, args.operation)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Blender 5.2's PLY exporter does not expose ``use_scene_unit`` (the
    # importer does). Keep the exchange in unscaled StageGen world coordinates
    # by using global_scale=1.0 and omitting unit conversion on export.
    result = bpy.ops.wm.ply_export(
        filepath=str(output_path), global_scale=1.0,
        forward_axis="Y", up_axis="Z", export_selected_objects=False,
        export_normals=True, export_triangulated_mesh=False, ascii_format=False,
    )
    if "FINISHED" not in result or not output_path.is_file() or output_path.stat().st_size == 0:
        raise RuntimeError(f"Blender PLY export did not finish: {result}")

    blend_output_path.parent.mkdir(parents=True, exist_ok=True)
    save_result = bpy.ops.wm.save_as_mainfile(
        filepath=str(blend_output_path), check_existing=False, compress=True
    )
    if "FINISHED" not in save_result or not blend_output_path.is_file() or blend_output_path.stat().st_size == 0:
        raise RuntimeError(f"Blender project save did not finish: {save_result}")

    report = {
        "success": True,
        "blender_version": bpy.app.version_string,
        "operation": args.operation,
        "input": str(input_path),
        "output": str(output_path),
        "blend_project": str(blend_output_path),
        "counts": counts,
        "coordinate_space": "PLY world coordinates; no explicit unit or axis conversion applied by worker",
        "scene_units": "METRIC with scale_length=1.0; coordinate values are not rescaled",
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("ACR_REMIXER_BLENDER_JOB=" + json.dumps(report, separators=(",", ":")), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ACR Remixer Blender worker failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        traceback.print_exc()
        raise SystemExit(2)

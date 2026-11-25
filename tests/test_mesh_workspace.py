# SPDX-FileCopyrightText: 2026 hiro98 (vale981) and the ACR Remixer contributors
# SPDX-License-Identifier: MIT

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from stagegen import mesh_workspace, toolchain


def _plane(o3d, z: float, size: float = 10.0):
    half = size / 2.0
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector([
        (-half, -half, z), (half, -half, z), (half, half, z), (-half, half, z)
    ])
    mesh.triangles = o3d.utility.Vector3iVector([(0, 1, 2), (0, 2, 3)])
    mesh.compute_triangle_normals()
    mesh.compute_vertex_normals()
    return mesh


def test_blender_worker_is_valid_python_and_has_allowlisted_operations():
    worker = Path(__file__).parents[1] / "stagegen" / "blender_worker.py"
    source = worker.read_text(encoding="utf-8")
    ast.parse(source)
    assert 'choices=("recalculate_normals", "triangulate")' in source
    assert 'input_path == output_path' in source
    assert '--blend-output' in source
    assert 'bpy.ops.wm.save_as_mainfile' in source


def test_blender_operation_allowlist_rejects_arbitrary_scripts():
    assert toolchain.BLENDER_MESH_OPERATIONS == ("recalculate_normals", "triangulate")


def test_open3d_road_ground_raycast_uses_below_surface_and_reports_overhead_separately():
    o3d = pytest.importorskip("open3d")
    road = _plane(o3d, 2.0)
    lower_terrain = _plane(o3d, 0.0)
    upper_terrain = _plane(o3d, 5.0)
    terrain = lower_terrain + upper_terrain

    report = mesh_workspace._raycast_road_to_terrain(road, terrain, sample_count=500)

    assert report["sample_count"] == 500
    assert report["downward_ground_coverage_fraction"] == pytest.approx(1.0)
    assert report["downward_ground_gap_m"]["p50"] == pytest.approx(2.0, abs=1e-3)
    assert report["upward_surface_hits"] == 500
    assert report["upward_surface_clearance_m"]["p50"] == pytest.approx(3.0, abs=1e-3)


def test_open3d_mesh_stats_rejects_non_finite_geometry():
    o3d = pytest.importorskip("open3d")
    mesh = _plane(o3d, 0.0)
    vertices = np.asarray(mesh.vertices).copy()
    vertices[0, 0] = np.nan
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    with pytest.raises(mesh_workspace.GeometryWorkspaceError, match="NaN or infinite"):
        mesh_workspace._mesh_stats(mesh, "test mesh")


def test_blender_mesh_bridge_rejects_same_input_and_output_path(tmp_path: Path, monkeypatch):
    source = tmp_path / "source.ply"
    source.write_text("fixture", encoding="ascii")
    monkeypatch.setattr(toolchain, "ensure_blender", lambda: tmp_path / "blender.exe")
    with pytest.raises(toolchain.ToolchainError, match="different"):
        toolchain.blender_process_mesh(source, source)


def test_blender_mesh_bridge_rejects_non_ply_inputs(tmp_path: Path):
    source = tmp_path / "source.obj"
    source.write_text("fixture", encoding="ascii")
    with pytest.raises(toolchain.ToolchainError, match="PLY"):
        toolchain.blender_process_mesh(source, tmp_path / "out.ply")


def test_blender_runner_requests_ply_and_native_blend_outputs(tmp_path: Path, monkeypatch):
    import json
    import subprocess

    source = tmp_path / "road.ply"
    target = tmp_path / "processed" / "road_processed.ply"
    blend = tmp_path / "processed" / "road_processed.blend"
    source.write_bytes(b"test PLY input")
    monkeypatch.setattr(toolchain, "ensure_blender", lambda: tmp_path / "blender.exe")
    monkeypatch.setattr(toolchain, "blender_worker_path", lambda: tmp_path / "worker.py")
    monkeypatch.setattr(toolchain, "_tool_root", lambda: (_ for _ in ()).throw(toolchain.ToolchainError("test")))
    captured = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"processed PLY")
        blend.write_bytes(b"native blend project")
        report_path = target.with_suffix(target.suffix + ".blender.json")
        report_path.write_text(json.dumps({
            "success": True, "operation": "recalculate_normals",
            "blend_project": str(blend),
        }), encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, "ok", "")

    monkeypatch.setattr(toolchain.subprocess, "run", fake_run)
    report = toolchain.blender_process_mesh(source, target, blend_output_path=blend)
    command = captured["command"]
    assert "--background" in command
    assert "--blend-output" in command
    assert str(blend) in command
    assert target.is_file() and blend.is_file()
    assert report["blend_project"] == str(blend)

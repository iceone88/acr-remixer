# SPDX-FileCopyrightText: 2026 ACR Remixer contributors
# SPDX-License-Identifier: MIT
from __future__ import annotations

from pathlib import Path

import pytest

from tools.geometry.mesh_inventory import inspect_mesh
from tools.geometry.compare_road_ground import compare_surfaces

FIXTURES = Path(__file__).parent / "fixtures" / "geometry"
ROAD = FIXTURES / "road_surface.obj"
TERRAIN = FIXTURES / "terrain_surface.obj"


def test_open3d_reads_synthetic_road_mesh():
    result = inspect_mesh(ROAD)
    assert result["vertices"] == 4
    assert result["triangles"] == 2
    assert result["finite_vertices"] is True
    assert result["bounds_min_xyz"] == [0.0, -1.0, 2.0]
    assert result["bounds_max_xyz"] == [10.0, 1.0, 2.0]


def test_vertical_surface_comparison_finds_two_metre_gap():
    report, rows = compare_surfaces(ROAD, TERRAIN, sample_count=128, max_search_distance=10.0, seed=7)
    assert report["matched_samples"] == report["requested_samples"]
    assert report["coverage_percent"] == pytest.approx(100.0)
    assert report["signed_vertical_gap_input_units"]["median"] == pytest.approx(-2.0, abs=1e-4)
    assert len(rows) == 128
    assert all(row["signed_vertical_gap"] is not None for row in rows)


def test_search_distance_limit_reports_unmatched_samples():
    report, _ = compare_surfaces(ROAD, TERRAIN, sample_count=64, max_search_distance=1.0, seed=7)
    assert report["matched_samples"] == 0
    assert report["unmatched_samples"] == report["requested_samples"]
    assert report["coverage_percent"] == 0.0


def test_nonpositive_search_distance_is_rejected():
    with pytest.raises(ValueError, match="greater than zero"):
        compare_surfaces(ROAD, TERRAIN, sample_count=10, max_search_distance=0.0)

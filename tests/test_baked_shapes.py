import struct
from types import SimpleNamespace

from stagegen.build.writers import (
    BakedFull, _hdr, inspect_baked_shapes, without_baked_shapes,
)


def _shape_mid_one_box():
    # One shape entry with one default-layout box primitive, no sphere/capsule,
    # no physical-material ref and an empty soft path. Top-level field 2 empty.
    box_fields = _hdr(10) + bytes(range(75))
    entry = (
        _hdr(5) + struct.pack("<I", 1) + box_fields +
        struct.pack("<II", 0, 0) + struct.pack("<i", 0) +
        struct.pack("<IIIII", 0, 0, 0, 0, 0)
    )
    return struct.pack("<I", 1) + entry + struct.pack("<I", 0)


def _shape_group_rest():
    # List 1 empty. List 2: collision key 12 refers to shape entry range [0,1).
    list1 = struct.pack("<i", 0)
    list2 = (
        struct.pack("<iii", 1, 12, 0) + b"\x00\x03" +
        struct.pack("<ii", 0, 1) + _hdr(2) + struct.pack("<ii", 0, 1)
    )
    return list1 + list2 + bytes(4)


def _bf(mid, rest=None):
    # A BakedFull carrying no meshes; the test focuses on the shape arrays/lists.
    return BakedFull(b"top", b"transform", [], mid, [], rest or struct.pack("<III", 0, 0, 0), set())


def test_inspect_primitive_shape_inventory_and_samples():
    pk = SimpleNamespace(names=["None"], imported_package_names=lambda: [])
    info = inspect_baked_shapes(pk, _bf(_shape_mid_one_box(), _shape_group_rest()))
    assert info["entry_count"] == 1
    assert info["primitive_counts"] == {"box": 1, "sphere": 0, "capsule": 0}
    assert info["entries"][0]["material"] == "none"
    assert info["entries"][0]["groups"] == [12]
    assert info["primitive_raw_samples_hex"]["box"][0]["0"] == bytes(range(24)).hex()


def test_strip_shapes_preserves_triangle_groups_and_clears_shape_groups():
    original = _bf(_shape_mid_one_box(), _shape_group_rest())
    stripped, count = without_baked_shapes(original)
    assert count == 1
    assert stripped.mid == struct.pack("<II", 0, 0)
    # Empty list 1 + empty list 2 + required trailer.
    assert stripped.rest == struct.pack("<III", 0, 0, 0)
    assert stripped.meshes == original.meshes
    assert stripped.list0_hdr == original.list0_hdr


def test_point_to_triangle_distance_projects_to_face_and_edges():
    import numpy as np
    from stagegen.inspect_collision import _closest_points_on_triangles

    tri = np.array([[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]]])
    # Point above the face projects vertically onto its interior.
    p_face = np.array([[0.5, 0.5, 3.0]])
    q_face = _closest_points_on_triangles(p_face, tri)
    np.testing.assert_allclose(q_face, [[0.5, 0.5, 0.0]], atol=1e-8)

    # Point outside the triangle uses the nearest edge segment.
    p_edge = np.array([[2.0, 2.0, 0.0]])
    q_edge = _closest_points_on_triangles(p_edge, tri)
    np.testing.assert_allclose(q_edge, [[1.0, 1.0, 0.0]], atol=1e-8)


def test_point_to_degenerate_triangle_falls_back_to_edges():
    import numpy as np
    from stagegen.inspect_collision import _closest_points_on_triangles

    tri = np.array([[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [4.0, 0.0, 0.0]]])
    point = np.array([[1.0, 1.0, 0.0]])
    closest = _closest_points_on_triangles(point, tri)
    np.testing.assert_allclose(closest, [[1.0, 0.0, 0.0]], atol=1e-8)

# Upload instructions — cloud-first 3D toolchain

This is an incremental overlay for the existing ACR Remixer repository. Extract
the ZIP and merge its repository files into the repository root; do not upload
the ZIP as one file. `UPLOAD_INSTRUCTIONS.md` itself is just this convenience
guide and is not required by the workflow.

## Files that replace earlier optional-toolkit files

If you already uploaded the earlier kit, overwrite these files with the copies
from this package. If you have not uploaded the earlier kit, simply add them:

- `requirements-geometry.txt`
- `.github/workflows/geometry-tools-smoke.yml`
- `tools/geometry/open3d_smoke_test.py`
- `tools/blender/inspect_mesh_exports.py`
- `tools/unreal/README.md`
- `docs/THREE_D_TOOLCHAIN_SETUP.md`

The workflow has the same filename as the previous Open3D-only workflow, so it
replaces it rather than creating a duplicate workflow.

## Add new files

- `tools/__init__.py`
- `tools/geometry/__init__.py`
- `tools/geometry/README.md`
- `tools/geometry/mesh_inventory.py`
- `tools/geometry/compare_road_ground.py`
- `tools/blender/__init__.py`
- `tools/blender/README.md`
- `tests/test_geometry_tools.py`
- `tests/fixtures/geometry/README.md`
- `tests/fixtures/geometry/road_surface.obj`
- `tests/fixtures/geometry/terrain_surface.obj`
- `diagnostics/private-input/README.md`

Do not replace `requirements.txt`, `requirements-stagegen-build.txt`,
`packaging/backend.spec`, or `.github/workflows/nightly.yml`.

Suggested branch: `cloud-geometry-toolchain`.

After pushing, open **Actions → Cloud 3D geometry diagnostics → Run workflow**
and leave `use_private_diagnostics` set to `false`. This uses synthetic meshes
only and requires no local software installation.

For actual ACR-derived OBJ inputs, use a separate private diagnostics
repository, copy this kit there, place `road_surface.obj` and `terrain_surface.obj`
in `diagnostics/private-input/`, and run with `use_private_diagnostics=true`.
Never commit game-derived meshes to a public repository.

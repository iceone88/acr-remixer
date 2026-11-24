# Headless Blender tools

`inspect_mesh_exports.py` imports OBJ files in Blender background mode and
writes a JSON inventory of imported mesh objects, triangle counts and world
bounds. It does not change source files or repair geometry.

The GitHub Actions workflow downloads the pinned Blender portable archive,
verifies its SHA-256 checksum against Blender's official checksum file, then
runs this script with `--background`.

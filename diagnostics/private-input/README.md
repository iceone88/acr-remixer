# Private ACR diagnostic inputs

Do not add ACR-derived OBJ exports to a public repository.

For real-game diagnostics, use a **separate private GitHub repository** (or make
this repository private before adding any game-derived exports). Put the two
matching exports in this directory:

- `road_surface.obj` — the road's upper/driving surface only, not a closed hull
  containing underside and side faces.
- `terrain_surface.obj` — the terrain surface that occupies the same coordinate
  frame and area.

Optional additional OBJ exports may also be placed here; they are included in
mesh inventory and Blender inspection.

The `Cloud 3D geometry diagnostics` workflow refuses private-input mode when
its repository is public. It uploads reports and sampled CSV measurements as
workflow artifacts, not copies of the input OBJ files. The repository itself
must remain private because its Git history contains the source exports.

Both files must use the same coordinate system, units, and Z-up convention.
The diagnostics do not perform registration, unit conversion, axis conversion,
or automatic correction.

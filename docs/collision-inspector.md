# Collision Inspector and crash-isolation notes

## Why this is being added

The current ACR Remixer rewrites the baked triangle meshes in
`DA_<Location>BakedData`: it clips and relocates the FBakedMesh triangle
collision with the same tile ownership rules used for the road/terrain mesh.
The current writer preserves the baked primitive-shape array (box, sphere,
capsule) and its shape-index group list byte-for-byte. Those primitive shapes
are not transformed by the current build path. This is a concrete compatibility
risk when source actors are relocated, but it is not yet proven to be the cause
of the user's post-race crash.

## Read-only inspection

Run:

```powershell
.\stagegen-backend.exe inspect-collision --level WalesS4HafrenSouth `
  --paks "D:\SteamLibrary\steamapps\common\Assetto Corsa Rally\acr\Content\Paks" `
  --out "$env:USERPROFILE\Desktop\ACR-Collision-Inspection"
```

Or use `--all` instead of `--level ...` for every discoverable original stage.
To inspect installed custom outputs, add `--include-stagegen`; this opt-in lets
the inspector read `~mods/*_Stagegen_*` containers, which normal source loading
intentionally excludes:

```powershell
.\stagegen-backend.exe inspect-collision --level SGQ2195E225HafrenSouth --include-stagegen `
  --paks "D:\SteamLibrary\steamapps\common\Assetto Corsa Rally\acr\Content\Paks" `
  --out "$env:USERPROFILE\Desktop\ACR-Collision-Inspection"
```

The tool writes:

- `<Stage>_collision.json`: route, baked-mesh, primitive-shape and group summary.
- `<Stage>_collision_meshes.csv`: one row per baked collision mesh, with actor,
  material reference, collision group, vertex/triangle counts, bounds, nearest
  route-centreline distances, and triangle validity statistics.
- `collision_inspection_index.json`: inspected stages and any per-stage errors.
- Optional `<Stage>_road_collision.obj` and `<Stage>_terrain_collision.obj`
  when `--obj` is supplied.

The road-to-collision metric resamples the centreline at 10 m intervals,
selects the 64 nearest road-triangle centroids in XY, then computes exact 3D
point-to-triangle distances for those candidate faces. This is more precise
than centroid-only distance, but remains a bounded candidate search rather than
an Unreal physics raycast. A high coverage score does not certify runtime
physics behaviour.
Primitive shape component fields are only inventoried (shape counts, material,
owner path, collision-group mapping); their field payloads have not yet been
reverse-engineered semantically.

## Diagnostic build: remove primitive shapes

The GUI contains an **Diagnostic (experimental) → Remove primitive collision
shapes** option. It removes the baked box/sphere/capsule shape array and clears
only its corresponding shape-index group list. It preserves triangle collision
meshes and their mesh-group index list. This intentionally removes some prop
collisions, so it is a test build only, not a normal stage setting.

Use it as a controlled test:

1. Rebuild the same manual sequence with all vegetation disabled, but leave
   primitive shapes enabled. Test entering, driving, finishing and exiting.
2. If it still crashes, rebuild the same sequence with all vegetation still
   disabled and **Remove primitive collision shapes** enabled. Test again.
3. If step 1 crashes and step 2 does not, primitive shape data becomes the
   leading suspect. If both crash, investigate triangle collision, terrain
   height blending, actor transforms, and route/replay data next.
4. If step 1 works, re-enable one vegetation category at a time to isolate
   forest, grass/rocks, or foliage relocation.

The test mode should not be distributed as a playable production stage because
some props will have no primitive collision.

## Validation currently performed

The baked-data validator checks that:

- the BakedDataContainer export parses to its exact end;
- mesh index groups cover all collision meshes;
- every mesh's index count is triangular;
- mesh vertices are finite and triangle indices are in range;
- primitive shape group indices do not point beyond the shape array.

The generation pipeline also retains its existing route drivability, road-step,
seam-height and terrain checks. A passing report is a data-integrity check, not
a substitute for in-game testing.

## Still required to implement full shape relocation

The shape entry structs contain box/sphere/capsule fields with opaque byte
schemas in `stagegen/build/writers.py`. Before transforming those shapes, each
field must be decoded into its actual transform, dimensions, material and
flags, then re-serialized and the shape-group index list rebuilt for each
repeated/moved source actor. Guessing field meanings risks producing invalid
Unreal physics data; the inspector therefore preserves raw field layouts until
we can compare it against representative source assets from multiple stages.

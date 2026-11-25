# ACR Remixer: Open3D + Blender mesh-workspace integration

## Purpose of this phase

This update integrates Open3D and Blender into the existing `acr-remixer.exe` workflow. It is the tool-integration and mesh-exchange phase, **not** the automatic terrain/road correction phase.

The app can read a selected original stock level through StageGen's existing UE/IoStore reader, export the baked road and terrain collision meshes, analyse them with Open3D, and run controlled Blender background operations on copies. Blender saves both processed PLY files and native `.blend` projects. No Blender window or local Python installation is required.

**Important boundary:** this version exports baked collision geometry, not every visible StaticMesh, Landscape, material, texture, tree, bush, foliage instance, or complete visual scene. It can edit the exported collision-mesh copies, but it does not yet repack those edits into cooked ACR `.uasset/.umap/.ucas/.utoc` packages or install a corrected playable stage. Package reintegration is a later engineering phase.

## Runtime architecture

- **Open3D 0.20.0** is bundled in the frozen Python backend by PyInstaller (`collect_all("open3d")`). It reads/writes mesh interchange files, measures geometry, and performs triangle ray queries.
- **Blender 5.2.5 LTS** is downloaded on demand from the official Blender release server. Its official SHA-256 manifest is checked before extraction. The EXE launches it in background mode with a bundled, allow-listed Python worker.
- **Rust/egui GUI** starts backend jobs asynchronously, shows progress/log output, and opens the generated workspace folder.
- **StageGen** remains the game-specific reader/writer. Blender/Open3D do not parse Unreal IoStore packages directly.

Blender uses the documented `bpy.ops.wm.ply_import`, `bpy.ops.wm.ply_export`, and `bpy.ops.wm.save_as_mainfile` APIs. The worker explicitly keeps scale and axes at 1.0 / Y-forward / Z-up and labels the scene as metric without rescaling coordinates. See the Blender 5.2 API documentation: https://docs.blender.org/api/5.2/bpy.ops.wm.html.

## In-app workflow

1. Select an original stock stage in ACR Remixer.
2. Open **Setup → 3D geometry tools (Open3D / Blender)**.
3. Click **Analyze stock stage with Open3D**.
   - StageGen opens stock PAK/IoStore data with `include_mods=False`.
   - It exports the baked road/terrain collision classes and route centreline.
   - Open3D checks mesh integrity, bounds, area and vertical road-to-terrain relationships.
4. Click **Edit collision mesh copies with Blender**. The EXE runs Blender in background mode; you do not need to open Blender yourself.
   - If the managed Blender copy is missing, the EXE downloads and verifies it automatically.
   - Blender imports each copied PLY in background mode.
   - It runs the selected allow-listed operation, exports a processed PLY, and saves a native `.blend` project.
5. Click **Open workspace folder** to inspect reports, processed PLY files and the saved `.blend` projects.

The original PAK files and source OBJ/PLY exports are never overwritten. The GUI's current default Blender operation is `recalculate_normals`; `triangulate` is also available through the CLI.

## Workspace output

```text
out/geometry-workspaces/<Level>/
  <Level>_collision.json
  <Level>_collision_meshes.csv
  <Level>_road_collision.obj
  <Level>_terrain_collision.obj
  <Level>_route_centreline.obj
  exchange/
    road_collision.ply
    terrain_collision.ply
  open3d_stage_analysis.json
  processed/
    road_collision_processed.ply
    road_collision_processed.blend
    terrain_collision_processed.ply
    terrain_collision_processed.blend
    road_collision_processed.ply.blender.json
    terrain_collision_processed.ply.blender.json
  blender_processing_report.json
```

Only classes successfully exported by StageGen are present. A missing road/terrain OBJ is reported; it is not fabricated.

## CLI equivalents

```powershell
# Read/extract a stock stage and analyse its baked road/terrain collision meshes.
acr-stagegen.exe geometry analyze-stage --level WalesS4HafrenSouth --paks "C:\...\acr\Content\Paks" --out ".\out\geometry-workspaces\WalesS4HafrenSouth"

# Run Blender headlessly on copied PLY files and save .blend projects.
acr-stagegen.exe geometry process-workspace --workspace ".\out\geometry-workspaces\WalesS4HafrenSouth" --operation recalculate_normals
```

## Managed Blender installation

Blender 5.2.5 portable x64 is downloaded only when a Blender mesh job is requested. The EXE stores it under:

```text
%LOCALAPPDATA%\ACRRemixer\Tools\Blender\5.2.5
```

The app needs an internet connection for the first download and several GB of free disk space. Later runs reuse the verified installation. `ACR_REMIXER_BLENDER_EXE` can point to an existing compatible Blender 5.2.5 executable for development/testing.

## Safety and limitations

- All edits are performed on copies in `exchange/` and `processed/`; the original game containers are read-only during this workflow.
- The Blender worker accepts only an allow-list of operations. It does not execute arbitrary scripts passed from the GUI.
- `recalculate_normals` is expected to preserve geometry; StageGen compares counts, bounds (5 mm tolerance for float coordinates) and surface area before accepting the result.
- The worker saves a `.blend` project so later correction scripts can run in the same background environment.
- PLY is an interchange format; it does not preserve Unreal material slots, LODs, StaticMesh build settings, collision metadata, actor transforms, HLOD relationships, Data Layers, or IoStore package identity.
- Open3D ray checks are geometric queries, not Unreal runtime physics queries. They do not by themselves prove vehicle support in-game.
- This phase does not automatically repair terrain, modify visible road meshes, move trees/bushes, regenerate collision, change road direction, or install a corrected stage.
- To make corrected geometry playable, a later schema-aware reintegration layer must associate edited mesh results with their original Unreal asset/actor identities and rebuild the relevant cooked package data without discarding materials, UVs, LODs, collision and World Partition references.

## GitHub and tests

The existing `nightly` workflow builds the frozen Python backend and Rust GUI, then publishes the EXE. It also runs Python tests and the Open3D smoke test.

A separate Windows workflow, `.github/workflows/blender-mesh-bridge.yml`, downloads/verifies Blender 5.2.5 and runs a real background PLY import → processing → PLY export + `.blend` save round trip on deterministic synthetic meshes. It uploads the resulting files as an Actions artifact. This validates the Blender runner without putting ACR game assets in a public workflow.

The standard `acr-stagegen` Windows workflow also runs the geometry workspace tests and CLI smoke test.

## Next phase

After the EXE build and Blender bridge workflow pass on Windows, implement the full stock-stage analysis engine and correction algorithms. The first package-reintegration milestone should preserve mesh identity and metadata and initially target collision-only assets. Visible render StaticMeshes, terrain, foliage and World Partition authoring need separate extraction/serialization work. No corrected PAK should be installed until the rebuilt package passes validation and a test-stage round trip.

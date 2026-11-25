# ACR Remixer: Open3D + Blender mesh workspace update

## Before you upload

This is an incremental update for the branch that already contains the previous Open3D/Blender runtime manager and Stock Stage Explorer. It adds the real stage-collision mesh workspace, Blender background worker, saved `.blend` projects, and a Windows Blender bridge CI workflow.

If your GitHub branch does not yet contain the earlier Open3D/Blender runtime integration, use the complete source ZIP instead of this incremental overlay.

## Apply

1. Create a new branch from your latest GUI / Stock Stage Explorer / Open3D-Blender integration branch, e.g. `geometry-mesh-workspace`.
2. Extract this ZIP at the repository root, preserving paths.
3. Replace files with matching paths and add new files/folders.
4. Do not replace `requirements.txt`, `requirements-stagegen-build.txt`, `.github/workflows/nightly.yml` with unrelated local versions, or existing source files not present in this ZIP. The included `nightly.yml` and StageGen workflow are intentional updates and should replace the corresponding files on the expected baseline.
5. Commit and push.

## Workflows to run

1. Actions → **Blender Mesh Bridge Windows** → Run workflow. It installs Open3D, downloads and verifies Blender 5.2.5, creates small synthetic PLY meshes, runs Blender headlessly, and verifies the processed PLY and native `.blend` outputs. Download artifact `acr-blender-mesh-bridge-test` to inspect the result.
2. Actions → **nightly** → Run workflow on the same branch. This freezes the Python backend, runs tests and Open3D smoke tests, compiles the Rust GUI, then publishes the nightly EXE.

## In-app use after the new EXE is built

1. Open ACR Remixer and select a stock stage.
2. Setup → 3D geometry tools → **Analyze stock stage with Open3D**.
3. Setup → **Edit collision mesh copies with Blender**. On first use, ACR Remixer downloads Blender automatically and runs it in background mode. No Blender window or local Python is required.
4. **Open workspace folder** to see the `exchange/`, `processed/`, JSON and CSV reports.

Each available road/terrain collision mesh produces a processed PLY copy, a native `.blend` project, and a Blender worker JSON report.

## Important scope boundary

This phase reads and exports the selected stage's baked road/terrain collision meshes. It does not yet export every visible StaticMesh, Landscape, material, texture, tree, bush, foliage instance or HLOD. Blender edits the exported collision-mesh copies only. It does not rewrite cooked ACR `.uasset/.umap/.ucas/.utoc` packages or install the processed PLY as a playable stage. That package-reintegration layer and automatic terrain/road correction are later phases.

Do not copy processed PLY files into `Paks/~mods`.

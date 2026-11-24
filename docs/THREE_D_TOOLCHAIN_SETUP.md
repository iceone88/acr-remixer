# ACR Remixer — Cloud-first 3D geometry toolchain

This setup runs Open3D and Blender on GitHub-hosted Windows runners. No local
Open3D, Blender, or Python installation is required on the developer's PC.
It does not change the released `acr-remixer.exe` or its runtime dependencies.

## What the workflow installs

| Component | Pinned version | Where it runs | Purpose |
|---|---:|---|---|
| Python | 3.12 x64 | GitHub Actions runner | Runs analysis scripts and tests |
| Open3D | 0.20.0 | Python environment on runner | Triangle mesh loading, ray casting and read-only surface comparisons |
| Blender | 5.2.5 LTS | Portable ZIP extracted on runner | Independent OBJ import and mesh inventory, headless |
| NumPy | Compatible 2.x | Python environment on runner | Array operations |
| Unreal Engine | Not installed in this workflow | Deferred | A full UE installation is not a practical dependency of the standard runner; see below |

Versions are intentionally pinned for reproducibility. Blender 5.2.5 is the
5.2 LTS patch available on 2026-11-25. The workflow downloads the official
Windows x64 portable ZIP and verifies it against Blender's published SHA-256
file before extracting it.

## 1. Add the files to the existing ACR Remixer repository

Extract the cloud-toolchain ZIP into a temporary folder. Merge its contents
into the repository root, preserving paths. Do not upload the ZIP as one file.

Files supplied by this kit (add them if absent, overwrite only the listed
previous-toolkit files when they already exist):

```text
requirements-geometry.txt
.github/workflows/geometry-tools-smoke.yml
tools/__init__.py
tools/geometry/__init__.py
tools/geometry/README.md
tools/geometry/open3d_smoke_test.py
tools/geometry/mesh_inventory.py
tools/geometry/compare_road_ground.py
tools/blender/__init__.py
tools/blender/README.md
tools/blender/inspect_mesh_exports.py
tools/unreal/README.md
tests/test_geometry_tools.py
tests/fixtures/geometry/README.md
tests/fixtures/geometry/road_surface.obj
tests/fixtures/geometry/terrain_surface.obj
diagnostics/private-input/README.md
docs/THREE_D_TOOLCHAIN_SETUP.md
```

`UPLOAD_INSTRUCTIONS.md` at the ZIP root is a convenience guide for applying
the kit; it is not required by the workflow.

Do not replace or modify these existing files as part of this toolchain setup:

- `requirements.txt`
- `requirements-stagegen-build.txt`
- `packaging/backend.spec`
- `.github/workflows/nightly.yml`
- `.github/workflows/build-stagegen-windows.yml`

If you previously uploaded the earlier optional toolchain kit, overwrite its
`.github/workflows/geometry-tools-smoke.yml` with the copy in this kit. The
filename is deliberately unchanged, so you have one workflow with Open3D and
Blender checks rather than two overlapping workflows.

Suggested branch: `cloud-geometry-toolchain`.

## 2. Run the cloud workflow (no private game files)

1. Push the files to GitHub.
2. Open **Actions**.
3. Select **Cloud 3D geometry diagnostics**.
4. Click **Run workflow**.
5. Leave **use_private_diagnostics** set to `false`.
6. Run the workflow.
7. Open the completed run and download the artifact named
   `acr-cloud-geometry-diagnostics-...`.

The workflow uses two original synthetic OBJ fixtures: a flat road at Z=2 and
flat terrain at Z=0. These files are deliberately not extracted from ACR. The
expected road-to-ground median vertical gap is -2 input units. This checks that
Open3D reads triangle meshes and that Blender independently imports the OBJ
files. It is a toolchain test, not a validation of any ACR stage.

## 3. Understand the generated reports

The artifact contains:

- `mesh_inventory.json` and `.csv`: vertex/triangle counts, bounds, surface area,
  watertight status and edge-manifold status for each OBJ.
- `road_ground_report.json`: sample coverage and vertical-gap quantiles.
- `road_ground_samples.csv`: sampled road points and the nearest vertical
  terrain hit where found.
- `blender_mesh_inventory.json`: independent Blender import counts and bounds.

All scripts are read-only. They do not transform, repair, clip or write back to
source meshes. No axis conversion or unit inference is performed.

### Important meaning of the road/ground metric

The road/ground tool samples the surface area of the supplied road OBJ and casts
vertical rays in both Z directions into the terrain OBJ. Negative signed gaps
mean the nearest vertical terrain hit is below the sampled road surface;
positive gaps mean it is above. The result is a proximity measurement, not a
universal “drivable / not drivable” verdict.

The road input must be the **upper driving surface**, not a closed collision
hull that includes underside and side faces. The road and terrain meshes must
already share coordinates, units and a Z-up convention. The tool does not align
or rescale them. Bridges/overpasses need separate interpretation because a
vertical ray can hit terrain below a deck without that terrain being the deck's
support surface.

## 4. Analyse actual ACR exports safely

Do not upload ACR-derived geometry to the public source repository. Git history
and workflow artifacts may be visible to people who can access that repository.

For real-game cloud analysis, create a separate **private** repository, for
example `acr-remixer-private-diagnostics`:

1. Set its visibility to **Private** when creating it.
2. Copy the cloud-toolchain files/folders from this kit into that private repo.
3. Create `diagnostics/private-input/` in the private repo.
4. Put the matching files there:

   ```text
   diagnostics/private-input/
   ├── road_surface.obj
   └── terrain_surface.obj
   ```

5. Do not add `.pak`, `.utoc`, `.ucas`, `.uasset`, `.umap`, or complete game
   packages. Use only the specific OBJ diagnostic exports required for analysis.
6. In the private repo, run the same workflow and set
   `use_private_diagnostics` to `true`.

The workflow checks the repository visibility through the GitHub API and fails
closed if private-input mode is requested from a public repository. It uploads
only reports/CSV samples as artifacts, not copies of the input OBJ files. Keep
the repository private because its Git history will still contain the OBJ
exports.

The current scripts expect a road upper-surface OBJ and a terrain OBJ that
cover the same coordinate region. If your current StageGen inspector only
exports a combined road collision mesh, do not rename that file and assume it
is the road upper surface. We must first confirm its triangles represent the
driving face. The current setup does not yet extract semantic road/terrain
meshes from IoStore automatically.

## 5. Why Unreal Engine is deferred

The repository's ACR inventory identifies an Unreal Engine 5.6 game build. A
full Unreal Editor installation is much larger than Open3D or Blender and is
not included in this standard Windows GitHub runner workflow. Do not upload
Engine binaries or source to the public ACR Remixer repository. Epic source
access and engine use are governed by Epic's account and licence requirements.

For the first terrain-validation phase, the StageGen package reader/writer plus
Open3D and Blender are sufficient for mesh diagnostics. If a later phase
requires Unreal automation (for example, authoring/cooking a new StaticMesh
asset), evaluate a dedicated private cloud VM/self-hosted runner with UE 5.6
installed under the applicable Epic licence. That is a separate, potentially
paid infrastructure decision; it is not required to run this workflow.

## 6. What this setup does not do yet

It does not automatically correct terrain, deform road geometry, reposition
foliage, rebuild collision, author Unreal assets, or prove in-game drivability.
The next code change should be a read-only Phase 0 diagnostic on a known-good
Wales/Greece export and the problematic Saverne export. Only after that
comparison is trustworthy should we implement bounded terrain correction.

## 7. Cost and runner notes

The workflow uses `windows-2025`. GitHub's published standard Windows runner
specification is 4 CPUs / 16 GB RAM / 14 GB SSD for public repositories; private
repository standard runners have different included compute/resource rules.
Blender's portable archive is roughly 406 MB, so the job downloads and extracts
it at run time and does not commit it to the repository.

The workflow does not run on every commit to unrelated source files. It runs
manually and on pushes/pull requests that touch the listed geometry tools,
fixtures, requirements, or workflow.

## Official tool references

- Open3D Python package and supported Python/Windows platforms: https://pypi.org/project/open3d/
- Blender 5.2 LTS releases and downloads: https://www.blender.org/download/lts/
- Blender 5.2.5 Windows x64 release archive and checksum index: https://download.blender.org/release/Blender5.2/
- GitHub-hosted runner specifications: https://docs.github.com/en/actions/reference/runners/github-hosted-runners

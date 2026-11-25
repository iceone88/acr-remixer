# Open3D and Blender runtime integration

## Scope of this phase

This integration makes the EXE able to extract a selected stock level's baked road/terrain collision meshes, analyse them with Open3D, and run a controlled Blender background mesh pass on copied PLY files. It does **not** implement automatic terrain/road/foliage correction, and it does **not** repack Blender-edited PLY files into cooked Unreal packages. Existing stage generation remains unchanged.

## Open3D: bundled into the application

- Runtime dependency: `open3d==0.20.0` in `requirements.txt`.
- The standalone StageGen build also pins Open3D in `requirements-stagegen-build.txt`.
- `packaging/backend.spec` uses PyInstaller `collect_all("open3d")` to include package modules, data, and native binaries. Open3D's MIT notice is also placed inside the frozen backend under `licenses/`.
- The Rust GUI already embeds `packaging/dist/stagegen-backend.exe`, so Open3D becomes part of the released `acr-remixer.exe` through the existing packaging path.
- The backend command `toolchain test-open3d` performs a CPU triangle-raycast smoke test against a plane exactly 2 m below the ray origin.
- Windows build workflows run this test against the frozen executable before building or publishing the GUI.

The Windows CPython 3.14 Open3D wheel is 79.3 MB before freezing. The final `acr-remixer.exe` will grow; its exact size must be measured in the Windows build. Open3D supports Python 3.10 through 3.14 on Windows x64. See https://pypi.org/project/open3d/0.20.0/.

## Blender: managed portable companion

Blender is **not embedded** in the EXE. The official Windows portable ZIP is about 406 MB compressed and would make every app download unnecessarily large. Instead, ACR Remixer downloads it only when the user clicks **Download / repair Blender**.

Pinned tool version: Blender 5.2.5 LTS, Windows x64.

Official URLs:

- Archive: `https://download.blender.org/release/Blender5.2/blender-5.2.5-windows-x64.zip`
- SHA-256 manifest: `https://download.blender.org/release/Blender5.2/blender-5.2.5.sha256`

The runtime manager:

1. Downloads the official SHA-256 manifest over HTTPS.
2. Downloads the portable ZIP over HTTPS.
3. Verifies the archive against the official manifest before extraction.
4. Rejects archive paths that could escape the extraction directory and rejects symlink entries.
5. Checks available disk space before extraction.
6. Runs Blender's background Python API smoke test.
7. Installs into `%LOCALAPPDATA%\ACRRemixer\Tools\Blender\5.2.5`.

The download is user-initiated and happens once per pinned version. A user can override the executable with `ACR_REMIXER_BLENDER_EXE`.

## GUI controls

The Setup sidebar includes **3D geometry tools (Open3D / Blender)** with three actions:

- **Check tools**: runs the Open3D raycast test and checks whether the managed Blender version is installed.
- **Download / repair Blender**: downloads, verifies, extracts, and tests Blender.
- **Test Blender**: runs Blender in background mode and checks its bundled `bpy` API.

All actions run through the embedded StageGen backend and the existing asynchronous job/log/progress path.

## CLI commands

```text
acr-remixer backend: toolchain status
acr-remixer backend: toolchain test-open3d
acr-remixer backend: toolchain install-blender
acr-remixer backend: toolchain test-blender
```

For development, use the equivalent StageGen backend command (for example `python -m stagegen.cli toolchain status`).

## Verification and known limits

- Unit tests cover checksum-manifest parsing, ZIP path traversal rejection, and safe extraction.
- Windows build workflows test the frozen executable by invoking `toolchain test-open3d`.
- Blender is downloaded at runtime; it is not part of the release payload. The EXE launches the bundled worker in background mode. The worker imports copied PLY meshes, performs an allow-listed mesh operation, exports a processed PLY, and saves a native `.blend` project. The app reports and logs download/install errors.
- This Linux working environment cannot execute the Windows frozen binary or compile the Rust GUI. The first GitHub Windows workflow run is the authoritative packaging and GUI-build check.
- No game assets or generated stages are modified by any of these tool-management commands.

## Licensing and packaging

Open3D is MIT-licensed. Its copyright and permission notice is included in `LICENSES/Open3D-MIT.txt`, bundled into the frozen backend, and attached beside release executables as `THIRD-PARTY-NOTICES-Open3D.txt`. Blender is GPL-licensed and is downloaded unmodified from Blender's official distribution server; ACR Remixer does not repackage Blender into its own executable. Users launch Blender only as a companion process. See the upstream projects for their complete license notices.

## Next phase (not included here)

The next implementation phase should export visible road/terrain StaticMeshes and foliage instances, classify the full source stage, then develop bounded terrain/vegetation correction. After that, a schema-aware Unreal package reintegration layer is needed before corrected PLY data can become an installable ACR stage.


## Mesh workspace bridge

The selected stock-stage analysis exports baked road/terrain collision meshes only. The Blender worker produces `*_processed.ply`, `*_processed.blend`, and a JSON report for each available mesh class. These are editable intermediate assets; they are not repacked or installed as ACR stages by this phase. See `OPEN3D_BLENDER_MESH_WORKSPACE_GUIDE.md` for the end-to-end workflow and exact limitations.

# SPDX-FileCopyrightText: 2026 hiro98 (vale981) and the ACR Remixer contributors
# SPDX-License-Identifier: MIT

"""Runtime management and smoke tests for ACR Remixer's optional 3D tools.

Open3D is bundled into the frozen StageGen backend. Blender is downloaded on
explicit user request from Blender's official release server and cached under
%LOCALAPPDATA%\\ACRRemixer\\Tools. No correction or game-asset code lives here.
"""
from __future__ import annotations

import hashlib
import importlib
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

OPEN3D_VERSION = "0.20.0"
BLENDER_VERSION = "5.2.5"
BLENDER_DIRNAME = f"blender-{BLENDER_VERSION}-windows-x64"
BLENDER_ARCHIVE = f"{BLENDER_DIRNAME}.zip"
BLENDER_BASE_URL = "https://download.blender.org/release/Blender5.2"
BLENDER_ARCHIVE_URL = f"{BLENDER_BASE_URL}/{BLENDER_ARCHIVE}"
BLENDER_CHECKSUM_URL = f"{BLENDER_BASE_URL}/blender-{BLENDER_VERSION}.sha256"
DOWNLOAD_CHUNK = 8 * 1024 * 1024


class ToolchainError(RuntimeError):
    """A 3D tool is unavailable or failed validation."""


def _tool_root() -> Path:
    local = os.environ.get("LOCALAPPDATA")
    if not local:
        raise ToolchainError("LOCALAPPDATA is not set; Blender's managed install is Windows-only")
    return Path(local) / "ACRRemixer" / "Tools"


def blender_install_root() -> Path:
    return _tool_root() / "Blender" / BLENDER_VERSION


def find_blender() -> Path | None:
    """Return the managed Blender executable if it exists and reports the pinned version."""
    candidates: list[Path] = []
    override = os.environ.get("ACR_REMIXER_BLENDER_EXE")
    if override:
        candidates.append(Path(override).expanduser())
    try:
        root = blender_install_root()
        candidates.extend(root.glob("**/blender.exe"))
    except ToolchainError:
        pass
    for candidate in candidates:
        if candidate.is_file():
            try:
                result = subprocess.run([str(candidate), "--version"], capture_output=True,
                                        text=True, timeout=20, check=False)
                first = (result.stdout or result.stderr).splitlines()[0]
                if result.returncode == 0 and first.startswith(f"Blender {BLENDER_VERSION}"):
                    return candidate
            except (OSError, subprocess.SubprocessError, IndexError):
                continue
    return None


def open3d_smoke_test() -> str:
    """Import bundled Open3D and raycast a known 2 m plane from above."""
    try:
        o3d = importlib.import_module("open3d")
        legacy = o3d.geometry.TriangleMesh()
        legacy.vertices = o3d.utility.Vector3dVector(
            [(-1.0, -1.0, 0.0), (1.0, -1.0, 0.0), (1.0, 1.0, 0.0), (-1.0, 1.0, 0.0)]
        )
        legacy.triangles = o3d.utility.Vector3iVector([(0, 1, 2), (0, 2, 3)])
        tensor_mesh = o3d.t.geometry.TriangleMesh.from_legacy(legacy)
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(tensor_mesh)
        rays = o3d.core.Tensor([[0.0, 0.0, 2.0, 0.0, 0.0, -1.0]], dtype=o3d.core.Dtype.Float32)
        hit = scene.cast_rays(rays)
        distance = float(hit["t_hit"][0].item())
        if not (abs(distance - 2.0) <= 1e-4):
            raise ToolchainError(f"Open3D raycast expected 2.0 m, got {distance:.6f} m")
        version = str(getattr(o3d, "__version__", "unknown"))
        return f"Open3D {version}: OK (raycast = {distance:.3f} m)"
    except ToolchainError:
        raise
    except Exception as exc:
        raise ToolchainError(f"Open3D import/raycast failed: {type(exc).__name__}: {exc}") from exc


def blender_smoke_test(executable: Path | None = None) -> str:
    exe = executable or find_blender()
    if exe is None:
        raise ToolchainError(f"Blender {BLENDER_VERSION} is not installed. Use 'toolchain install-blender'.")
    expression = (
        "import bpy, sys; "
        "print('ACR_REMIXER_BLENDER_OK=' + bpy.app.version_string); "
        "sys.exit(0 if bpy.app.version_string.startswith('5.2.5') else 9)"
    )
    try:
        result = subprocess.run(
            [str(exe), "--background", "--factory-startup", "--python-expr", expression, "--quit"],
            capture_output=True, text=True, timeout=180, check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ToolchainError(f"Could not start Blender: {exc}") from exc
    output = "\n".join(part for part in (result.stdout, result.stderr) if part)
    match = re.search(r"ACR_REMIXER_BLENDER_OK=([^\r\n]+)", output)
    if result.returncode != 0 or not match:
        tail = output[-2500:] if output else "(no output)"
        raise ToolchainError(f"Blender smoke test failed (exit {result.returncode}):\n{tail}")
    return f"Blender {match.group(1)}: OK (background Python API)"


def _read_expected_sha256(manifest: bytes, filename: str) -> str:
    """Parse Blender's official multi-file SHA256 manifest, failing closed."""
    text = manifest.decode("utf-8", errors="strict")
    for line in text.splitlines():
        if filename not in line:
            continue
        match = re.search(r"\b([0-9a-fA-F]{64})\b", line)
        if match:
            return match.group(1).lower()
    raise ToolchainError(f"Official Blender checksum manifest has no SHA-256 entry for {filename}")


def _download(url: str, destination: Path, label: str, progress_start: float = 0.03,
              progress_end: float = 0.68) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "ACR-Remixer/3D-toolchain"})
    try:
        with urllib.request.urlopen(request, timeout=45) as response, destination.open("wb") as out:
            total = int(response.headers.get("Content-Length") or 0)
            done = 0
            while True:
                chunk = response.read(DOWNLOAD_CHUNK)
                if not chunk:
                    break
                out.write(chunk)
                done += len(chunk)
                if total:
                    fraction = min(1.0, done / total)
                    progress = progress_start + (progress_end - progress_start) * fraction
                    print(f"@progress {progress:.3f} {label}: {done / 1_000_000:.0f}/{total / 1_000_000:.0f} MB",
                          flush=True)
        if not destination.is_file() or destination.stat().st_size == 0:
            raise ToolchainError(f"Download produced an empty file: {url}")
    except ToolchainError:
        raise
    except Exception as exc:
        destination.unlink(missing_ok=True)
        raise ToolchainError(f"Download failed from {url}: {exc}") from exc


def _safe_extract(zip_path: Path, target: Path) -> int:
    """Extract ZIP without allowing absolute paths or path traversal."""
    root = target.resolve()
    total_unpacked = 0
    with zipfile.ZipFile(zip_path) as archive:
        members = archive.infolist()
        for member in members:
            name = member.filename.replace("\\", "/")
            pure = PurePosixPath(name)
            if (pure.is_absolute() or any(part == ".." for part in pure.parts)
                    or (pure.parts and ":" in pure.parts[0])):
                raise ToolchainError(f"Unsafe path in Blender archive: {member.filename!r}")
            # The official Blender ZIP should not need symlinks. Reject them so
            # extraction cannot create links outside the managed tool folder.
            mode = (member.external_attr >> 16) & 0o170000
            if mode == 0o120000:
                raise ToolchainError(f"Symlink entry is not allowed in Blender archive: {member.filename!r}")
            destination = (root / Path(*pure.parts)).resolve()
            if destination != root and root not in destination.parents:
                raise ToolchainError(f"Archive entry escapes extraction directory: {member.filename!r}")
            total_unpacked += member.file_size
        free = shutil.disk_usage(root.parent).free
        required = total_unpacked + 300 * 1024 * 1024
        if free < required:
            raise ToolchainError(
                f"Not enough free space to extract Blender: need about {required / 1_000_000_000:.1f} GB, "
                f"have {free / 1_000_000_000:.1f} GB"
            )
        archive.extractall(root)
    return total_unpacked


def install_blender(force: bool = False) -> str:
    """Download, verify, extract and smoke-test official Blender portable ZIP."""
    if sys.platform != "win32":
        raise ToolchainError("Managed Blender installation is currently supported only on Windows x64.")
    if platform.machine().lower() not in ("amd64", "x86_64"):
        raise ToolchainError(f"Blender package is Windows x64; detected machine {platform.machine()!r}")
    if not force:
        existing = find_blender()
        if existing:
            return blender_smoke_test(existing) + f" (already installed at {existing})"

    root = _tool_root()
    root.mkdir(parents=True, exist_ok=True)
    final = blender_install_root()
    # Ensure the pinned-version parent exists even on a first-time install.
    # The staging and backup directories live under `root`, while `final` is
    # nested one level deeper under `Blender/`.
    final.parent.mkdir(parents=True, exist_ok=True)
    archive = root / BLENDER_ARCHIVE
    stage = root / f".blender-{BLENDER_VERSION}.installing"
    stage_backup = root / f".blender-{BLENDER_VERSION}.previous"
    archive.unlink(missing_ok=True)
    if stage.exists():
        shutil.rmtree(stage, ignore_errors=True)
    stage.mkdir(parents=True, exist_ok=True)

    try:
        print("@progress 0.01 Fetching official Blender checksum manifest", flush=True)
        manifest_request = urllib.request.Request(BLENDER_CHECKSUM_URL,
                                                  headers={"User-Agent": "ACR-Remixer/3D-toolchain"})
        with urllib.request.urlopen(manifest_request, timeout=45) as response:
            manifest = response.read(256 * 1024)
        expected = _read_expected_sha256(manifest, BLENDER_ARCHIVE)

        print("@progress 0.03 Downloading Blender portable archive", flush=True)
        _download(BLENDER_ARCHIVE_URL, archive, "Downloading Blender", 0.03, 0.68)
        digest = hashlib.sha256()
        with archive.open("rb") as f:
            for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        actual = digest.hexdigest()
        if actual != expected:
            raise ToolchainError(f"Blender archive SHA-256 mismatch: expected {expected}, got {actual}")
        print("@progress 0.72 Blender archive checksum verified; extracting", flush=True)
        _safe_extract(archive, stage)
        candidates = list(stage.glob("**/blender.exe"))
        if not candidates:
            raise ToolchainError("Blender archive extracted but blender.exe was not found")
        candidates.sort(key=lambda p: len(p.parts))
        staged_exe = candidates[0]
        version = subprocess.run([str(staged_exe), "--version"], capture_output=True,
                                 text=True, timeout=30, check=False)
        first = (version.stdout or version.stderr).splitlines()[0]
        if version.returncode != 0 or not first.startswith(f"Blender {BLENDER_VERSION}"):
            raise ToolchainError(f"Downloaded Blender reports unexpected version: {first!r}")
        print("@progress 0.90 Running Blender background API test", flush=True)
        blender_smoke_test(staged_exe)

        had_previous = final.exists()
        if had_previous:
            if stage_backup.exists():
                shutil.rmtree(stage_backup, ignore_errors=True)
            final.rename(stage_backup)
        staged_root = next((p for p in stage.iterdir() if p.is_dir()), stage)
        # Move the already-tested portable folder into the versioned cache.
        staged_root.rename(final)
        installed_exe = next(iter(final.glob("**/blender.exe")), None)
        if installed_exe is None:
            shutil.rmtree(final, ignore_errors=True)
            if had_previous and stage_backup.exists():
                stage_backup.rename(final)
            raise ToolchainError("Blender install moved but executable is missing")
        if stage.exists():
            shutil.rmtree(stage, ignore_errors=True)
        shutil.rmtree(stage_backup, ignore_errors=True)
        archive.unlink(missing_ok=True)
        print("@progress 1.0 Blender installed and verified", flush=True)
        return f"Blender {BLENDER_VERSION}: installed and verified at {installed_exe}"
    except Exception:
        archive.unlink(missing_ok=True)
        shutil.rmtree(stage, ignore_errors=True)
        # Restore an old installation if replacement failed before completion.
        if stage_backup.exists() and not final.exists():
            stage_backup.rename(final)
        raise


def status_lines() -> tuple[str, str]:
    try:
        open3d_status = open3d_smoke_test()
    except ToolchainError as exc:
        open3d_status = f"Open3D: NOT READY ({exc})"
    blender = find_blender()
    blender_status = f"Blender {BLENDER_VERSION}: installed at {blender}" if blender else (
        f"Blender {BLENDER_VERSION}: not installed (download on request)"
    )
    return open3d_status, blender_status


BLENDER_MESH_OPERATIONS = ("recalculate_normals", "triangulate")


def blender_worker_path() -> Path:
    """Locate the Blender worker script in source or PyInstaller's data directory."""
    frozen_root = getattr(sys, "_MEIPASS", None)
    if frozen_root:
        candidate = Path(frozen_root) / "stagegen" / "blender_worker.py"
    else:
        candidate = Path(__file__).with_name("blender_worker.py")
    if not candidate.is_file():
        raise ToolchainError(f"Bundled Blender worker script is missing: {candidate}")
    return candidate


def ensure_blender() -> Path:
    """Return the managed Blender executable, installing the pinned build if needed."""
    exe = find_blender()
    if exe is not None:
        return exe
    install_blender(force=False)
    exe = find_blender()
    if exe is None:
        raise ToolchainError("Blender installation finished but the executable is still unavailable")
    return exe


def blender_process_mesh(input_path: str | Path, output_path: str | Path,
                         operation: str = "recalculate_normals", timeout_s: int = 900,
                         blend_output_path: str | Path | None = None) -> dict:
    """Run an allow-listed Blender background operation on a copied PLY mesh.

    Blender is automatically provisioned on first use. The input is never
    overwritten. Output and a JSON worker report are written beside the target.
    """
    if operation not in BLENDER_MESH_OPERATIONS:
        raise ToolchainError(
            f"Unsupported Blender mesh operation {operation!r}; allowed: {', '.join(BLENDER_MESH_OPERATIONS)}"
        )
    source = Path(input_path).expanduser().resolve()
    target = Path(output_path).expanduser().resolve()
    if source == target:
        raise ToolchainError("Blender input and output paths must be different; source meshes are never overwritten")
    if source.suffix.lower() != ".ply" or target.suffix.lower() != ".ply":
        raise ToolchainError("Blender mesh bridge currently requires PLY input and output")
    if not source.is_file() or source.stat().st_size == 0:
        raise ToolchainError(f"Blender input PLY is missing or empty: {source}")

    exe = ensure_blender()
    worker = blender_worker_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    report_path = target.with_suffix(target.suffix + ".blender.json")
    blend_target = (Path(blend_output_path).expanduser().resolve() if blend_output_path
                    else target.with_suffix(".blend"))
    if blend_target in (source, target, report_path):
        raise ToolchainError("Blender project path must be separate from input, mesh output and report")
    report_path.unlink(missing_ok=True)
    blend_target.unlink(missing_ok=True)
    env = os.environ.copy()
    try:
        tool_root = _tool_root()
        config_root = tool_root / "BlenderConfig" / BLENDER_VERSION
        scripts_root = tool_root / "BlenderScripts" / BLENDER_VERSION
        config_root.mkdir(parents=True, exist_ok=True)
        scripts_root.mkdir(parents=True, exist_ok=True)
        env["BLENDER_USER_CONFIG"] = str(config_root)
        env["BLENDER_USER_SCRIPTS"] = str(scripts_root)
    except ToolchainError:
        # Development/testing on non-Windows may supply a Blender override.
        pass

    command = [
        str(exe), "--background", "--factory-startup", "--python-exit-code", "17",
        "--python", str(worker), "--", "--input", str(source), "--output", str(target),
        "--report", str(report_path), "--blend-output", str(blend_target), "--operation", operation,
    ]
    print(f"@progress 0.05 Launching Blender background worker ({operation})", flush=True)
    try:
        result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8",
                                errors="replace", timeout=timeout_s, check=False, env=env)
    except subprocess.TimeoutExpired as exc:
        raise ToolchainError(f"Blender mesh operation timed out after {timeout_s} seconds") from exc
    except OSError as exc:
        raise ToolchainError(f"Could not launch Blender: {exc}") from exc
    output = "\n".join(part for part in (result.stdout, result.stderr) if part)
    if result.returncode != 0:
        raise ToolchainError(
            f"Blender mesh operation failed (exit {result.returncode}):\n{output[-5000:]}"
        )
    if not target.is_file() or target.stat().st_size == 0:
        raise ToolchainError(f"Blender exited successfully but did not create output PLY: {target}\n{output[-2000:]}")
    if not blend_target.is_file() or blend_target.stat().st_size == 0:
        raise ToolchainError(f"Blender exited successfully but did not create its project file: {blend_target}\n{output[-2000:]}")
    if not report_path.is_file():
        raise ToolchainError(f"Blender did not create its worker report: {report_path}\n{output[-2000:]}")
    try:
        report = __import__("json").loads(report_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ToolchainError(f"Blender worker report is invalid: {exc}") from exc
    if not report.get("success") or report.get("operation") != operation:
        raise ToolchainError(f"Blender worker report does not confirm the requested operation: {report}")
    if Path(report.get("blend_project", "")).resolve() != blend_target:
        raise ToolchainError("Blender worker report references a different project path than requested")
    print(f"@progress 0.95 Blender finished {source.name}; project saved: {blend_target.name}", flush=True)
    return report

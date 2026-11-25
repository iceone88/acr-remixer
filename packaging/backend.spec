# SPDX-FileCopyrightText: 2026 hiro98 (vale981) and the ACR Remixer contributors
# SPDX-License-Identifier: MIT

# PyInstaller spec: one-file stagegen backend (the GUI embeds this exe).
#   cd acr-stagegen; python -m PyInstaller packaging/backend.spec --distpath packaging/dist --workpath packaging/build
import os
from PyInstaller.utils.hooks import collect_all

ROOT = os.path.abspath(os.path.join(SPECPATH, ".."))

# Open3D ships native extension modules and runtime data. Collect the complete
# package so its CPU raycasting API is available in the frozen one-file backend.
o3d_datas, o3d_binaries, o3d_hiddenimports = collect_all("open3d")

a = Analysis(
    [os.path.join(SPECPATH, "backend_main.py")],
    pathex=[ROOT],
    binaries=[(os.path.join(ROOT, "oodle_shim", "target", "release", "oo2core_shim.dll"), ".")] + o3d_binaries,
    datas=[
        (os.path.join(ROOT, "stagegen", "build", "empty_mod.pak"), os.path.join("stagegen", "build")),
        (os.path.join(ROOT, "stagegen", "blender_worker.py"), os.path.join("stagegen")),
        (os.path.join(ROOT, "LICENSES", "Open3D-MIT.txt"), os.path.join("licenses")),
    ] + o3d_datas,
    hiddenimports=["scipy.spatial._ckdtree", "scipy.interpolate", "scipy.ndimage"] + o3d_hiddenimports,
    excludes=["matplotlib", "tkinter", "PIL", "IPython", "pandas", "pytest", "vgamepad"],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz, a.scripts, a.binaries, a.datas, [],
    name="stagegen-backend",
    console=True,
    upx=False,
    strip=False,
)

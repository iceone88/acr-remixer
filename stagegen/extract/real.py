"""Build SourceStage objects from the game's own data.

Per level:
  * centreline + road edges  <- SplinesActor (CenterSpline, Left/RightBorderSpline)
  * pacenotes                <- the DT_Pacenote* table the level references
  * surface                  <- dominant surface physical material in BakedData names

Levels are read straight from the IoStore containers; nothing is written.
"""
from __future__ import annotations

import os
import re

import numpy as np
from scipy.spatial import cKDTree

from ..geometry import arclength, resample
from ..model import Pacenote, SourceStage
from .iostore import GameFS
from .pacenote_table import read_pacenote_table
from .splines import level_splines
from .zen import Package

DEFAULT_PAKS = os.path.join(os.path.dirname(__file__), "..", "..", "..", "acr", "Content", "Paks")


REGION_SURFACE = {"wales": "gravel", "greece": "gravel", "sweden": "snow", "livigno": "snow",
                  "alsace": "tarmac", "montecarlo": "tarmac", "wrcges": "tarmac"}


def _surface(pk: Package, level: str = "") -> str:
    for region, surf in REGION_SURFACE.items():
        if level.lower().startswith(region):
            return surf
    names = " ".join(pk.names).lower()
    for key, surf in (("snow", "snow"), ("ice", "snow"), ("gravel", "gravel"), ("tarmac", "tarmac")):
        if f"pm_surface{key}" in names:
            return surf
    return "tarmac"


def _pacenote_tables(pk: Package) -> list[str]:
    return [n for n in pk.names if n.startswith("/Game/Data/Pacenote/Tables/") and "DT_Pacenote" in n]


def _game_path(game_ref: str) -> str:
    return "acr/Content/" + game_ref[len("/Game/"):] + ".uasset"


def load_level(fs: GameFS, level_path: str, step: float = 2.0, name: str | None = None,
               location: str | None = None) -> SourceStage:
    pk = Package(fs.read(level_path))
    sp = level_splines(pk)
    if "CenterSpline" not in sp:
        # World Partition levels keep actors in _Generated_ cell packages.
        cell_dir = level_path[:-5] + "/_Generated_/"
        for cell in sorted(p for p in fs.index if p.startswith(cell_dir)):
            d = fs.read(cell)
            if b"CenterSpline" not in d:
                continue
            try:
                sp = level_splines(Package(d))
            except Exception:      # noqa: BLE001
                continue
            if "CenterSpline" in sp:
                break
    if "CenterSpline" not in sp:
        raise ValueError(f"{level_path}: no CenterSpline")
    centre = resample(sp["CenterSpline"], step)
    width = np.full(len(centre), 7.0)
    if "LeftBorderSpline" in sp and "RightBorderSpline" in sp:
        dl, _ = cKDTree(sp["LeftBorderSpline"][:, :2]).query(centre[:, :2])
        dr, _ = cKDTree(sp["RightBorderSpline"][:, :2]).query(centre[:, :2])
        width = np.clip(dl + dr, 3.0, 20.0)

    lvl = name or os.path.splitext(os.path.basename(level_path))[0]
    notes: list[Pacenote] = []
    # Prefer the table named after this level; mods sometimes leave a stray
    # reference to another stage's table in the name map.
    tables = sorted(_pacenote_tables(pk), key=lambda t: (lvl.lower() not in t.lower(), t))
    if not tables:
        # Base game: one table per variant, named after the stage without the
        # region/stage-number prefix (WalesS3HafrenNorth -> HafrenNorth).
        short = re.sub(r"^[A-Za-z]+?S\d+", "", lvl) or lvl
        tables = ["/Game/" + p[len("acr/Content/"):-len(".uasset")]
                  for p in fs.find(f"DT_Pacenote{short}FullForward") if p.endswith(".uasset")]
    for t in tables:
        path = _game_path(t)
        if path in fs.index:
            for row in read_pacenote_table(Package(fs.read(path))):
                notes.append(Pacenote(float(row.distance), " ".join(row.tokens), "original", float(row.length)))
            break
    # Table distances are measured along the original spline; rescale to
    # our resampled arclength (they agree to <0.5 % but keep it exact).
    L_src = float(arclength(sp["CenterSpline"])[-1])
    L_new = float(arclength(centre)[-1])
    for n in notes:
        n.s *= L_new / L_src
    loc = location or re.sub(r"(SS\d+|S\d+.*)$", "", lvl)
    return SourceStage(lvl, loc, _surface(pk, lvl), centre, width, sorted(notes, key=lambda n: n.s))


def find_levels(fs: GameFS) -> list[str]:
    return [p for p in fs.find("/Content/Levels/") if p.endswith(".umap") and "_Generated_" not in p]


def load_location(location: str, paks_dir: str = DEFAULT_PAKS, step: float = 2.0,
                  include_mods: bool = True) -> list[SourceStage]:
    fs = GameFS(paks_dir, include_mods=include_mods)
    stages = []
    for lv in find_levels(fs):
        base = os.path.splitext(os.path.basename(lv))[0]
        if not base.lower().startswith(location.lower()):
            continue
        try:
            stages.append(load_level(fs, lv, step, location=location))
        except Exception as ex:   # noqa: BLE001 - report and keep going
            print(f"  skip {base}: {ex}")
    return stages

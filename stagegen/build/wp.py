"""Build a recombined base-game (World Partition) stage as a mod container.

What gets rewritten (everything else is still served by the game):
  * main level: streaming LoadingRange raised so every cell stays loaded
    (cells' actors no longer sit where the grid expects them); RVT volume
    moved over the new area.
  * every World Partition cell that holds a moved actor: root scene
    components get their piece's rigid transform. Track tiles move whole.
  * route cells (Forward/Reverse data layers): splines rebuilt along the new
    route, start/finish/split triggers re-placed.
  * the track tile meshes that are used: triangles outside the piece's road
    corridor are made degenerate, so no terrain sticks into other pieces.
  * DA_<Level>BakedData collision: tile triangles clipped with the same rule
    and transformed; props follow their piece.
  * DT_Pacenote<Name>Forward/Reverse.
"""
from __future__ import annotations

import collections
import os
import re
import struct
from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import cKDTree

from ..extract.iostore import Container, GameFS
from ..extract.real import WPSource
from ..extract.splines import level_splines, read_spline_points
from ..extract.zen import Package
from ..geometry import arclength, headings, resample, rot_z
from ..layout import Layout
from ..pacenotes import generate
from ..tiles import Cut, TileGrid, TState
from . import writers as W
from .cityhash import name_hash
from .iostore_write import ChunkIn, container_header_chunk_id, retarget_container_header, write_container
from .level import decode_components, learn_rules, patched_payload
from .mesh_clip import clip_mesh_payload, mesh_export_index
from .zen_write import rewrite_package

SPLINES = ("CenterSpline", "IdealSpline", "LeftBorderSpline", "RightBorderSpline", "ContourSpline")
SKIP_OWNERS = ("InstancedFoliageActor", "PCG", "HLOD", "WorldDataLayers", "WorldSettings", "RespawnPlanes",
               "BC_ReplayCamerasActor", "SplinesActor")
SKIP_WORDS = ("Sky", "Weather", "Fog", "Cloud", "Atmosphere", "PostProcess", "Moon", "Sun")


@dataclass
class WPConfig:
    prop_lateral: float = 60.0     # m: props/decals this close to the route move with their piece
    terrain_clip: float = 80.0     # m: tile terrain kept this far either side of its road
    road_clip: float = 30.0
    range_ext: float = 80.0        # m: along-route slack at piece ends (tiles end at the grid line anyway)
    margin: float = 1500.0         # m: gap between the original map and the new stage
    ctrl_spacing: float = 8.0
    loading_range_cm: float = 3.0e6
    tag: str = "Stagegen"


@dataclass
class WPReport:
    out_files: list = field(default_factory=list)
    moved: collections.Counter = field(default_factory=collections.Counter)
    relocated: list = field(default_factory=list)
    clipped: list = field(default_factory=list)
    baked_meshes: int = 0
    baked_tris: int = 0
    notes: dict = field(default_factory=dict)
    offset: tuple = (0, 0, 0)
    packages: int = 0
    checks: list = field(default_factory=list)


@dataclass
class PieceX:
    sa: float
    sb: float
    R: np.ndarray
    t: np.ndarray
    yaw: float
    tiles: frozenset


class _Assign:
    def __init__(self, src_pts: np.ndarray, step: float, pieces: list[PieceX], ext: float):
        self.tree = cKDTree(src_pts[:, :2])
        self.step = step
        self.p = pieces
        self.ext = ext

    def __call__(self, xy: np.ndarray, ext: float | None = None):
        d, idx = self.tree.query(np.atleast_2d(xy)[:, :2])
        s = idx * self.step
        k = np.full(len(s), -1)
        e = self.ext if ext is None else ext
        for j, pc in enumerate(self.p):           # exact ranges first
            k[(k < 0) & (s >= pc.sa) & (s <= pc.sb)] = j
        for j, pc in enumerate(self.p):
            k[(k < 0) & (s >= pc.sa - e) & (s <= pc.sb + e)] = j
        return k, d, s, idx


def build_wp_stage(layout: Layout, best: TState, cuts: list[Cut], grid: TileGrid, ws: WPSource,
                   paks_dir: str, out_dir: str, cfg: WPConfig | None = None) -> WPReport:
    cfg = cfg or WPConfig()
    rep = WPReport()
    fs = GameFS(paks_dir, include_mods=False)
    src = ws.stage
    step = float(np.median(np.linalg.norm(np.diff(src.points[:, :2], axis=0), axis=1)))
    L_src = float(arclength(src.points)[-1])
    lvl = src.name

    # ---- placement of the new stage
    xs = [grid.x0 + grid.cell * k[1] for k in grid.tiles]
    ys = [grid.y0 + grid.cell * k[0] for k in grid.tiles]
    old_max_x, old_min_y = max(xs) + grid.cell, min(ys)
    cl = layout.centreline
    offset = np.array([old_max_x + cfg.margin - cl[:, 0].min(), old_min_y - cl[:, 1].min(), 0.0])
    rep.offset = tuple(np.round(offset, 1))
    states = list(reversed(list(best.chain())))
    pieces = [PieceX(cuts[s.rng.a].s, cuts[s.rng.b].s, rot_z(s.piece.yaw), np.asarray(s.piece.translation) + offset,
                     float(s.piece.yaw), s.rng.tiles) for s in states]
    assign = _Assign(src.points, step, pieces, cfg.range_ext)
    tile_piece = {t: j for j, pc in enumerate(pieces) for t in pc.tiles}
    new_cl = cl + offset
    new_s = arclength(new_cl)
    new_head = headings(new_cl)
    src_head = headings(src.points)

    def rigid(j, p_m, rot):
        pc = pieces[j]
        return pc.R @ p_m + pc.t, (rot if rot is not None else np.zeros(3)) + np.array([0, np.degrees(pc.yaw), 0])

    def along(p_m, rot, idx):
        """Keep the relative position along the route (fraction of length), facing the same way."""
        s_new = idx * step / L_src * new_s[-1]
        j = min(int(np.searchsorted(new_s, s_new)), len(new_cl) - 1)
        dh = new_head[j] - src_head[idx]
        off = (p_m[:2] - src.points[idx, :2]) @ rot_z(dh)[:2, :2].T
        loc = np.r_[new_cl[j, :2] + off, new_cl[j, 2] + (p_m[2] - src.points[idx, 2])]
        return loc, (rot if rot is not None else np.zeros(3)) + np.array([0, np.degrees(dh), 0]), s_new

    # ---- route stretches no track tile covers (start areas etc.) live on the
    # coarse backdrop terrain (SM_<Level>TerrainYYXX); it follows the piece
    # that owns most of them, clipped to those stretches.
    bpk = Package(fs.read(ws.baked))
    meshes, head, tb, tp, ts = W.read_baked_raw(bpk)
    tile_v = np.vstack([m.vertices[::4, [0, 2, 1]] for m in meshes
                        if re.search(rb"Track\d{4}", m.soft_sub) and len(m.vertices)])
    dcov, _ = cKDTree(tile_v[:, :2]).query(src.points[:, :2])
    uncovered = dcov > 15.0
    grow = int(40 / step)
    unc_ext = np.convolve(uncovered, np.ones(2 * grow + 1), "same") > 0
    s_all = np.arange(len(src.points)) * step
    unc_len = [float(uncovered[(s_all >= pc.sa) & (s_all <= pc.sb)].sum() * step) for pc in pieces]
    bd_piece = int(np.argmax(unc_len)) if max(unc_len) > 20 else None
    if bd_piece is not None:
        rep.checks.append(f"backdrop terrain follows piece {bd_piece} ({unc_len[bd_piece]:.0f} m without track tiles)" +
                          "".join(f"; piece {j}: {u:.0f} m uncovered (no ground)" for j, u in enumerate(unc_len)
                                  if j != bd_piece and u > 20))
    bd_re = re.compile(rf"SM_{lvl}Terrain\d{{4}}$")
    bd_pivot: dict[str, np.ndarray] = {}

    # ---- packages
    pk_paths = [ws.main] + ws.cells
    pkgs = {p: Package(fs.read(p)) for p in pk_paths}
    rules = learn_rules(pkgs.values())
    tile_re = re.compile(rf"SM_{lvl}Track(\d\d)(\d\d)_GEN")
    payloads: dict[str, dict[int, bytes]] = collections.defaultdict(dict)
    tile_pivot: dict[tuple, np.ndarray] = {}

    for path, pk in pkgs.items():
        ex = pk.exports
        route = ws.route_cells.get(path)
        comps = [c for c in decode_components(pk, rules) if c.root and c.loc is not None]
        for c in comps:
            owner = ex[c.owner].name if c.owner < len(ex) else ""
            p_m = c.loc / 100
            if c.rule.L == 26:                     # runtime virtual texture volume
                lo = np.r_[new_cl[:, :2].min(0) - 300, new_cl[:, 2].min() - 150]
                hi = np.r_[new_cl[:, :2].max(0) + 300, new_cl[:, 2].max() + 300]
                payloads[path][c.export] = patched_payload(pk, c, lo * 100, c.rot if c.rot is not None
                                                           else np.zeros(3), (hi - lo) * 100)
                rep.moved["RVT volume"] += 1
                continue
            m = tile_re.match(owner)
            if m:
                key = (int(m.group(1)), int(m.group(2)))
                tile_pivot[key] = c.loc.copy()
                if key in tile_piece:
                    loc, rot = rigid(tile_piece[key], p_m, c.rot)
                    payloads[path][c.export] = patched_payload(pk, c, loc * 100, rot)
                    rep.moved["track tile"] += 1
                continue
            if bd_re.match(owner):
                bd_pivot[owner] = c.loc.copy()
                if bd_piece is not None:
                    loc, rot = rigid(bd_piece, p_m, c.rot)
                    payloads[path][c.export] = patched_payload(pk, c, loc * 100, rot)
                    rep.moved["backdrop terrain"] += 1
                continue
            if owner.startswith(SKIP_OWNERS) or any(w in owner for w in SKIP_WORDS):
                continue
            kind = owner.split("_UAID")[0].rstrip("0123456789_")
            k, d, s, idx = (v[0] for v in assign(p_m[None]))
            if route == "cut":
                continue
            if route:
                # start/finish areas travel with the first/last piece; intermediate
                # splits are re-spaced along the new route so they stay in order
                if kind.startswith("RaceSector") and k not in (0, len(pieces) - 1):
                    k = -1
                if k >= 0 and d < 150:
                    loc, rot = rigid(k, p_m, c.rot)
                    rep.moved[kind] += 1
                else:
                    loc, rot, s_new = along(p_m, c.rot, idx)
                    rep.relocated.append((f"{route}:{kind}", round(s), round(s_new)))
                payloads[path][c.export] = patched_payload(pk, c, loc * 100, rot)
                continue
            if k >= 0 and d <= cfg.prop_lateral:
                loc, rot = rigid(k, p_m, c.rot)
                payloads[path][c.export] = patched_payload(pk, c, loc * 100, rot)
                rep.moved[kind] += 1

    # ---- route splines (per variant direction)
    ctrl = resample(new_cl, cfg.ctrl_spacing)
    w_ctrl = np.interp(arclength(ctrl), new_s, layout.widths)
    for path, kind in ws.route_cells.items():
        if kind == "cut":
            continue
        pk = pkgs[path]
        C, Wd = (ctrl, w_ctrl) if kind == "fwd" else (ctrl[::-1].copy(), w_ctrl[::-1].copy())
        t = np.gradient(C[:, :2], axis=0)
        t /= np.linalg.norm(t, axis=1, keepdims=True)
        n_left = np.c_[t[:, 1], -t[:, 0]]
        _, sc_P, _, _ = read_spline_points(pk, "CenterSpline")
        _, co_P, _, _ = read_spline_points(pk, "ContourSpline")
        dc, _ = cKDTree(sc_P[:, :2]).query(co_P[:, :2])
        margin = max(1.0, float(np.median(dc) / 100 - np.median(src.width) / 2))
        left = np.c_[C[:, :2] + n_left * (Wd / 2)[:, None], C[:, 2]]
        right = np.c_[C[:, :2] - n_left * (Wd / 2)[:, None], C[:, 2]]
        cleft = np.c_[C[:, :2] + n_left * (Wd / 2 + margin)[:, None], C[:, 2]]
        cright = np.c_[C[:, :2] - n_left * (Wd / 2 + margin)[:, None], C[:, 2]]
        pts = {"CenterSpline": C, "IdealSpline": C, "LeftBorderSpline": left, "RightBorderSpline": right,
               "ContourSpline": np.vstack([cleft, cright[::-1], cleft[:1]])}
        for name in SPLINES:
            i = next((i for i, e in enumerate(pk.exports) if e.name == name), None)
            if i is not None:
                payloads[path][i] = W.spline_payload(pk, name, pts[name] * 100)

    # ---- keep every cell loaded
    main = pkgs[ws.main]
    hi_ = next(i for i, e in enumerate(main.exports) if e.name.startswith("WorldPartitionRuntimeSpatialHash"))
    e = main.exports[hi_]
    b = bytearray(payloads[ws.main].get(hi_, main.export_bytes(e)))
    cell_size, rng = struct.unpack_from("<if", b, 20)
    if not (1000 <= cell_size <= 10_000_000 and 1000 <= rng <= 10_000_000):
        raise ValueError(f"unexpected streaming grid layout (cell {cell_size}, range {rng})")
    struct.pack_into("<f", b, 24, cfg.loading_range_cm)
    payloads[ws.main][hi_] = bytes(b)
    rep.checks.append(f"streaming grid: cell {cell_size / 100:.0f} m, loading range {rng / 100:.0f} m -> "
                      f"{cfg.loading_range_cm / 100:.0f} m")

    new_pkgs: dict[str, bytes] = {p: rewrite_package(pkgs[p], pl) for p, pl in payloads.items() if pl}

    # ---- clip the used tiles' meshes
    def keeper(pivot_cm, j, lim):
        pc = pieces[j]

        def keep(cen_cm):
            w = (cen_cm + pivot_cm) / 100
            d, idx = assign.tree.query(w[:, :2])
            s = idx * step
            return (s >= pc.sa - cfg.range_ext) & (s <= pc.sb + cfg.range_ext) & (d <= lim)
        return keep

    gen_dir = f"/{lvl}/GeneratedSplits/"
    for key, j in sorted(tile_piece.items()):
        if key not in tile_pivot:
            rep.checks.append(f"tile {key} has no actor"); continue
        stem = f"SM_{lvl}Track{key[0]:02d}{key[1]:02d}_"
        for p in sorted(x for x in fs.index if gen_dir in x and x.rsplit("/", 1)[-1].startswith(stem)
                        and x.endswith(".uasset")):
            mpk = Package(fs.read(p))
            mi = mesh_export_index(mpk)
            lim = cfg.road_clip if "_Road_" in p else cfg.terrain_clip
            nb, st = clip_mesh_payload(mpk.export_bytes(mpk.exports[mi]), keeper(tile_pivot[key], j, lim))
            new_pkgs[p] = rewrite_package(mpk, {mi: nb})
            rep.clipped.append((p.rsplit("/", 1)[-1], st["lods"], st["dropped"], st["tris"]))

    def bd_keep_world(w_m):
        pc = pieces[bd_piece]
        d, idx = assign.tree.query(w_m[:, :2])
        s = idx * step
        return unc_ext[idx] & (s >= pc.sa - 40) & (s <= pc.sb + 40) & (d <= cfg.terrain_clip)

    if bd_piece is not None:
        for name, pivot in bd_pivot.items():
            p = next((x for x in fs.index if x.endswith(f"/{name}.uasset") and "/MeshesTerrain/" in x), None)
            if p is None:
                rep.checks.append(f"backdrop mesh for {name} not found"); continue
            mpk = Package(fs.read(p))
            mi = mesh_export_index(mpk)
            nb, st = clip_mesh_payload(mpk.export_bytes(mpk.exports[mi]),
                                       lambda cen, pv=pivot: bd_keep_world((cen + pv) / 100))
            new_pkgs[p] = rewrite_package(mpk, {mi: nb})
            rep.clipped.append((p.rsplit("/", 1)[-1], st["lods"], st["dropped"], st["tris"]))

    # ---- baked collision
    out_meshes: list[W.MeshOut] = []
    for m in meshes:
        if b"OriginQuad" in m.soft_sub:
            out_meshes.append(m); continue
        if len(m.indices) < 3:
            continue
        U = m.vertices[:, [0, 2, 1]]
        tri = m.indices.reshape(-1, 3)
        cen = U[tri].mean(1)
        mt = re.search(rb"Track(\d\d)(\d\d)", m.soft_sub)
        if mt:
            key = (int(mt.group(1)), int(mt.group(2)))
            if key not in tile_piece:
                continue
            j = tile_piece[key]
            pc = pieces[j]
            d, idx = assign.tree.query(cen[:, :2])
            s = idx * step
            lim = cfg.road_clip if b"Road" in m.soft_sub else cfg.terrain_clip
            keep = (s >= pc.sa - cfg.range_ext) & (s <= pc.sb + cfg.range_ext) & (d <= lim)
            k = np.where(keep, j, -1)
        elif re.search(rb"Terrain\d{4}", m.soft_sub):
            if bd_piece is None:
                continue
            k = np.where(bd_keep_world(cen), bd_piece, -1)
        elif b"PCGGrass" in m.soft_sub:
            continue                                   # grass/rock instances stay behind (not moved)
        else:
            k, d, _, _ = assign(cen)
            k[d > cfg.prop_lateral] = -1
        for kk in np.unique(k[k >= 0]):
            sel = tri[k == kk]
            used, inv = np.unique(sel, return_inverse=True)
            pc = pieces[int(kk)]
            V = (U[used] @ pc.R.T + pc.t)[:, [0, 2, 1]]
            out_meshes.append(W.MeshOut(V, inv.ravel().astype(np.uint32), m.material_ref, m.soft_pkg,
                                        m.soft_asset, m.soft_sub, m.extra))
            rep.baked_tris += len(sel)
    rep.baked_meshes = len(out_meshes)
    new_pkgs[ws.baked] = rewrite_package(bpk, {0: W.baked_payload(head, tb, out_meshes, tp, ts)})

    # ---- pacenotes
    for key, path in ws.tables.items():
        tpk = Package(fs.read(path))
        if key == "forward":
            notes = [W.NoteOut(float(n.s), float(getattr(n, "length", 30.0)), n.text.split())
                     for n in layout.pacenotes if n.text.strip()]
        else:
            notes = [W.NoteOut(float(n.s), 30.0, n.text.split()) for n in generate(new_cl[::-1].copy(), step)
                     if n.text.strip()]
        pay, names = W.pacenote_payload(tpk, notes)
        new_pkgs[path] = rewrite_package(tpk, {0: pay}, names)
        rep.notes[key] = len(notes)

    # ---- container
    os.makedirs(out_dir, exist_ok=True)
    lvl_container, _ = fs.index[ws.main]
    pak_name = f"pakchunk99-{lvl}_{cfg.tag}_9_P"
    cid = name_hash(pak_name)
    hdr_idx = next(i for i, x in enumerate(lvl_container.chunk_ids) if x[11] == 6)
    chunks = []
    for path, data in new_pkgs.items():
        c, i = fs.index[path]
        chunks.append(ChunkIn(c.chunk_ids[i], data, path))
    chunks.append(ChunkIn(container_header_chunk_id(cid),
                          retarget_container_header(lvl_container.read_entry(hdr_idx), cid)))
    with open(os.path.join(os.path.dirname(__file__), "empty_mod.pak"), "rb") as f:
        pak_template = f.read()
    base = os.path.join(out_dir, pak_name)
    write_container(base, cid, chunks, pak_template)
    rep.out_files = [base + ext for ext in (".utoc", ".ucas", ".pak")]
    rep.packages = len(new_pkgs)
    rep.checks += verify_wp(base + ".utoc", ws, new_pkgs, payloads, rules)
    return rep


def verify_wp(utoc: str, ws: WPSource, new_pkgs: dict, payloads: dict, rules) -> list[str]:
    from ..extract.bakeddata import read_baked
    from ..extract.pacenote_table import read_pacenote_table
    out = []
    c = Container(utoc)
    bad = 0
    n_patched = 0
    for path, pl in payloads.items():
        pk = Package(c.read(path))
        comps = {cc.export for cc in decode_components(pk, rules)}
        want = [i for i in pl if pk.exports[i].name not in SPLINES
                and not pk.exports[i].name.startswith("WorldPartitionRuntimeSpatialHash")]
        n_patched += len(want)
        bad += sum(i not in comps for i in want)
    out.append(f"level: {n_patched} patched components in {len(payloads)} packages, {bad} fail to re-decode")
    for path, kind in ws.route_cells.items():
        if kind == "cut":
            continue
        sp = level_splines(Package(c.read(path)))
        out.append(f"route {kind}: " + ", ".join(f"{k}={float(arclength(v)[-1]):.0f}m" for k, v in sp.items()))
    meshes = read_baked(Package(c.read(ws.baked)))
    out.append(f"baked: {len(meshes)} meshes, indices in range: "
               f"{all((m.indices.max() < len(m.vertices)) if len(m.indices) else True for m in meshes)}")
    for key, path in ws.tables.items():
        out.append(f"pacenotes {key}: {len(read_pacenote_table(Package(c.read(path))))} rows")
    return out

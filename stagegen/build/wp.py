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
from .mesh_clip import clip_mesh_payload, displace_payload, mesh_export_index
from .wp_hash import all_cells_in_top_level, drop_cells, widen_cell_data
from .foliage import hide_instances, move_hism
from .hlod import mesh_world_bounds, transform_mesh
from .zen_write import rewrite_package

SPLINES = ("CenterSpline", "IdealSpline", "LeftBorderSpline", "RightBorderSpline", "ContourSpline")
SKIP_OWNERS = ("InstancedFoliageActor", "PCG", "HLOD", "WorldDataLayers", "WorldSettings", "RespawnPlanes",
               "BC_ReplayCamerasActor", "SplinesActor")
SKIP_WORDS = ("Sky", "Weather", "Fog", "Cloud", "Atmosphere", "PostProcess", "Moon", "Sun")


@dataclass
class WPConfig:
    prop_lateral: float = 60.0     # m: props/decals this close to the route move with their piece
    terrain_clip: float = 2000.0   # m: tile terrain kept this far either side of its road (ownership
                                   # still decides; a narrow corridor looks like floating ribbons from afar)
    road_clip: float = 30.0
    range_ext: float = 80.0        # m: along-route slack at piece ends (tiles end at the grid line anyway)
    end_radius: float = 400.0      # m: around the route ends (start/finish areas) keep everything this far
    margin: float = 1500.0         # m: gap between the original map and the new stage
    ctrl_spacing: float = 8.0
    loading_range_cm: float = 3.0e6
    tag: str = "Stagegen"
    debug_spawn_s: float | None = None   # test builds: spawn the car this far along the new route
    all_cells_loaded: bool = False       # experimental, hides HLODs (see build_wp_stage)
    hlod_visuals: bool = False           # experiment: tile cells never load (also hides the forest)
    hlod_hide: bool = True               # empty all HLOD proxy meshes (cells are always loaded)
    filler: bool = True                  # backdrop terrain becomes ground under the whole new stage
    filler_below: float = 1.5            # m: filler sits this far below the stage terrain
    bg_clear: float = 2000.0            # m: background (satellite map) mesh hidden this close to the new route
    move_forest: bool = True             # PCG forest trees follow the piece owning their spot
    move_grass: bool = True              # ... and the PCG grass/branches/rocks
    hide_foliage: bool = True          # zero-scale the (unmoved) per-cell foliage instances
    hide_unused_tiles: bool = True       # tiles the layout does not use would float at the old location
    seam_width: float = 60.0            # m: terrain height blending either side of piece boundaries (0 = off)
    debug_skip_kinds: tuple = ()         # test builds: actor kinds left unmoved
    debug_no_backdrop: bool = False     # test builds: hide the coarse backdrop terrain entirely
    debug_no_clip: bool = False         # test builds: keep tile/backdrop meshes unclipped
    debug_offset_zero: bool = False      # test builds: build on top of the original map (piece 0 stays put)
    debug_offset: tuple | None = None    # test builds: explicit placement offset (m)


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


def _filler_height_field(surf, pieces, tree_new, tree_new_all, seam_dz, new_cl, mesh_list, bd_re_bytes,
                         bd_piece, bd_keep, bd_xform, below, cell=20.0, pad=2500.0, erode_m=45.0):
    """Height (m) of the filler ground at new-layout xy: the stage's own terrain
    (owned samples, seam-corrected) lowered by `below`; the start/finish areas
    without track tiles at the exact (moved) backdrop height; elsewhere the
    nearest heights, smoothed, so the gaps between stretches get plausible hills."""
    from scipy.interpolate import RegularGridInterpolator
    from scipy.ndimage import distance_transform_edt, gaussian_filter

    pts, zs = [], []
    for j, (tr, z) in surf.items():
        P = np.c_[tr.data, z]
        d_j, _ = tree_new[j].query(P[:, :2])
        d_a, _ = tree_new_all.query(P[:, :2])
        P = P[d_j <= d_a + 2.0]
        P[:, 2] += seam_dz(j, P) - below
        pts.append(P[:, :2]); zs.append(P[:, 2])
    if bd_piece is not None and bd_xform is not None:
        for U, sub in mesh_list:
            if re.search(bd_re_bytes, sub):
                k = bd_keep(U)
                if k.any():
                    V = bd_xform(U[k])
                    pts.append(V[:, :2]); zs.append(V[:, 2])
    P = np.vstack(pts); Z = np.concatenate(zs)
    # the road is a hard ceiling (below+1 m under it), spread over erode_m so
    # coarse filler triangles cannot span a road cut from hillside to hillside
    cl = resample(new_cl, 2.0)
    Rp, Rz = cl[:, :2], cl[:, 2] - below - 1.0
    lo = new_cl[:, :2].min(0) - pad
    hi = new_cl[:, :2].max(0) + pad
    gx = np.arange(lo[0], hi[0] + cell, cell)
    gy = np.arange(lo[1], hi[1] + cell, cell)
    ix = np.clip(((P[:, 0] - lo[0]) / cell).round().astype(int), 0, len(gx) - 1)
    iy = np.clip(((P[:, 1] - lo[1]) / cell).round().astype(int), 0, len(gy) - 1)
    # lowest sample per cell (the filler must stay under the detailed terrain)
    G = np.full((len(gx), len(gy)), np.inf)
    np.minimum.at(G, (ix, iy), Z)
    valid = np.isfinite(G)
    _, (nx, ny) = distance_transform_edt(~valid, return_indices=True)
    F = G[nx, ny]
    from scipy.ndimage import minimum_filter
    # the filler mesh is coarse (adaptive, up to ~100 m triangles): keep it under
    # the lowest stage height within erode_m so it does not form shelves in valleys
    E = minimum_filter(F, size=int(2 * erode_m / cell) + 1)
    # (continuing the edge heights outside the detailed terrain was tried: it
    # forms overhanging shelves over valleys, worse than the occasional gap)
    Ht = np.minimum(gaussian_filter(E, sigma=3), E)
    GR = np.full_like(G, np.inf)
    rx = np.clip(((Rp[:, 0] - lo[0]) / cell).round().astype(int), 0, len(gx) - 1)
    ry = np.clip(((Rp[:, 1] - lo[1]) / cell).round().astype(int), 0, len(gy) - 1)
    np.minimum.at(GR, (rx, ry), Rz)
    GR = minimum_filter(GR, size=int(2 * erode_m / cell) + 1)  # +inf away from the road
    H = np.minimum(Ht, GR)
    f = RegularGridInterpolator((gx, gy), H, bounds_error=False, fill_value=None)
    return lambda xy: f(np.atleast_2d(xy)[:, :2])


def _component_transform_for_mesh(pk: Package, mesh_pkg_path: str):
    """(RelativeLocation cm, RelativeScale3D) of the StaticMeshComponent in pk that
    draws the mesh package mesh_pkg_path (via imported package names -> import map).
    StaticMeshComponent props here: 4 StaticMesh (i32), 42 OverrideMaterials
    (i32 count + i32 each), 163 location, 164 rotation, 165 scale (3 doubles each)."""
    from ..extract.unversioned import Reader
    from ..extract.zen import read_name_batch
    game = "/Game/" + mesh_pkg_path.split("/Content/", 1)[1].rsplit(".", 1)[0]
    names, _ = read_name_batch(pk.d, pk.imported_pkg_names_off)
    if game not in names:
        return None
    k = names.index(game)
    imps = [i for i, v in enumerate(pk.imports) if (v >> 62) == 2 and ((v >> 32) & 0x3FFFFFFF) == k]
    for i in imps:
        for e in pk.exports:
            b = pk.export_bytes(e)
            r = Reader(pk.d, e.data_offset)
            pres, _ = r.header_full()
            q = r.p - e.data_offset
            if not pres or pres[0] != 4 or struct.unpack_from("<i", b, q)[0] != -(i + 1):
                continue
            q += 4
            if 42 in pres:
                q += 4 + 4 * struct.unpack_from("<i", b, q)[0]
            loc = np.array(struct.unpack_from("<3d", b, q)) if 163 in pres else np.zeros(3)
            q += 24 if 163 in pres else 0
            q += 24 if 164 in pres else 0
            scale = np.array(struct.unpack_from("<3d", b, q)) if 165 in pres else np.ones(3)
            return loc, scale
    return None


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
    if cfg.debug_offset_zero:
        offset = np.zeros(3)
    if cfg.debug_offset is not None:
        offset = np.asarray(cfg.debug_offset, float)
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

    def lat_ok(s, d, base):
        """Lateral limit; widened at the route ends, where start/finish areas sit off the spline."""
        end = (np.asarray(s) <= 50) | (np.asarray(s) >= L_src - 50)
        return np.asarray(d) <= np.where(end, max(base, cfg.end_radius), base)

    # Voronoi ownership on the *new* layout: after moving, a triangle/prop of
    # piece j is kept only if j's road is (about) the nearest road to it, so
    # no piece reaches over another piece's road.
    piece_new = []
    for pc in pieces:
        i0, i1 = int(pc.sa / step), min(int(np.ceil(pc.sb / step)), len(src.points) - 1)
        piece_new.append(src.points[i0: i1 + 1, :2] @ pc.R[:2, :2].T + pc.t[:2])
    tree_new_all = cKDTree(np.vstack(piece_new))
    tree_new = [cKDTree(p) for p in piece_new]

    def owns(j, src_pts_m, margin: float = 2.0):
        pc = pieces[j]
        q = np.atleast_2d(src_pts_m)[:, :2] @ pc.R[:2, :2].T + pc.t[:2]
        d_own, _ = tree_new[j].query(q)
        d_all, _ = tree_new_all.query(q)
        return d_own <= d_all + margin

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
    bf = W.read_baked_full(bpk)
    meshes = bf.meshes
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
    # ---- seam blending: neighbouring pieces' terrain meets at different heights
    # beside the road (one source spot is a hillside, the other a valley). Near
    # each ownership boundary, vertices are pulled towards the mean of both
    # surfaces (weight 1 at the boundary, 0 at seam_width into the own piece).
    s_tree_src = cKDTree(src.points[:, :2])
    surf = {}
    for j, pc in enumerate(pieces):
        pts = []
        for m in meshes:
            mt = re.search(rb"Track(\d\d)(\d\d)", m.soft_sub)
            if not (mt and len(m.vertices)) or tile_piece.get((int(mt.group(1)), int(mt.group(2)))) != j:
                continue
            U = m.vertices[:, [0, 2, 1]].astype(np.float64)
            _, ix = s_tree_src.query(U[:, :2])
            s_ = ix * step
            U = U[(s_ >= pc.sa - 150) & (s_ <= pc.sb + 150)]
            pts.append(U @ pc.R.T + pc.t)
        if pts:
            P = np.vstack(pts)
            surf[j] = (cKDTree(P[:, :2]), P[:, 2])

    def surface_z(j, xy):
        """(height, confidence): IDW of piece j's surface; confidence fades from 1
        (sample within 12 m) to 0 (none within 40 m) so corrections never jump."""
        if j not in surf:
            return np.full(len(xy), np.nan), np.zeros(len(xy))
        tr, z = surf[j]
        d, ii = tr.query(xy, k=4, distance_upper_bound=40.0)
        ok = np.isfinite(d)
        ii = np.where(ok, ii, 0)
        wts = np.where(ok, 1.0 / (d + 0.5), 0.0)
        sw = wts.sum(1)
        h = np.where(sw > 0, (wts * z[ii]).sum(1) / np.maximum(sw, 1e-9), np.nan)
        dmin = np.where(ok[:, 0], d[:, 0], np.inf)
        conf = np.clip((40.0 - dmin) / 28.0, 0.0, 1.0)
        return h, conf

    def seam_dz(j, P_new_m):
        """Vertical correction (m) for points of piece j at new-layout positions."""
        P = np.atleast_2d(P_new_m)
        dz = np.zeros(len(P))
        if cfg.seam_width <= 0:
            return dz
        d_j, _ = tree_new[j].query(P[:, :2])
        for b in range(len(pieces)):
            if b == j:
                continue
            d_b, _ = tree_new[b].query(P[:, :2])
            w = np.clip(1.0 - (d_b - d_j) / 2.0 / cfg.seam_width, 0.0, 1.0)
            near = w > 0
            if not near.any():
                continue
            w = w * w * (3 - 2 * w)
            zb, conf = surface_z(b, P[near, :2])
            ok = np.isfinite(zb)
            idx = np.flatnonzero(near)[ok]
            dz[idx] += w[idx] * conf[ok] * (zb[ok] - P[idx, 2]) / 2.0
        return dz

    bd_re = re.compile(rf"SM_{lvl}Terrain\d{{4}}$")
    bd_pivot: dict[str, np.ndarray] = {}

    # ---- packages
    pk_paths = [ws.main] + ws.cells
    pkgs = {p: Package(fs.read(p)) for p in pk_paths}
    rules = learn_rules(pkgs.values())
    tile_re = re.compile(rf"SM_{lvl}Track(\d\d)(\d\d)_GEN")
    # filler ground: the coarse backdrop terrain tiles are re-centred on the new
    # stage and reshaped (below) to its height field, so there is ground between
    # stretches of the route (without it, distant stretches float over the void)
    bd_pivots0 = [c.loc / 100 for c in decode_components(pkgs[ws.main], rules)
                  if c.root and c.loc is not None and c.owner < len(pkgs[ws.main].exports)
                  and bd_re.match(pkgs[ws.main].exports[c.owner].name)]
    filler_shift = np.zeros(2)
    if cfg.filler and bd_pivots0:
        filler_shift = (new_cl[:, :2].min(0) + new_cl[:, :2].max(0)) / 2 - np.mean(bd_pivots0, axis=0)[:2]
        if bd_piece is None:
            rep.checks.append(f"filler ground: {len(bd_pivots0)} backdrop tiles re-centred on the new stage")
    elif cfg.filler:
        cfg.filler = False
        rep.checks.append("filler ground: no backdrop tiles in the main level, off")
    payloads: dict[str, dict[int, bytes]] = collections.defaultdict(dict)
    tile_pivot: dict[tuple, np.ndarray] = {}

    def place_trees(W):
        """world_m (N,3) source -> (keep, new world, yaw) for forest instances."""
        k, d, s, _ = assign(W)
        keep = (k >= 0) & lat_ok(s, d, cfg.terrain_clip)
        NW = W.copy()
        yaw = np.zeros(len(W))
        for kk in np.unique(k[k >= 0]):
            m_ = k == kk
            pc = pieces[int(kk)]
            keep[m_] &= owns(int(kk), W[m_])
            NW[m_] = W[m_] @ pc.R.T + pc.t
            NW[m_, 2] += seam_dz(int(kk), NW[m_])
            yaw[m_] = pc.yaw
            if int(kk) in surf:                            # only where the piece's terrain is
                dd, _ = surf[int(kk)][0].query(NW[m_, :2], distance_upper_bound=15.0)
                keep[np.flatnonzero(m_)] &= np.isfinite(dd)
        if keep.any():                                     # park hidden ones among the kept (tight bounds)
            NW[~keep] = NW[keep].mean(0)
            yaw[~keep] = 0.0
        return keep, NW, yaw

    forest_nobox = [0]
    for path, pk in pkgs.items():
        ex = pk.exports
        route = ws.route_cells.get(path)
        if cfg.hide_foliage:
            # per-cell grass/foliage instances are not moved: at the original
            # location they float in the void left by the moved tiles
            for i, e in enumerate(ex):
                own = ex[e.outer].name if 0 <= e.outer < len(ex) else ""
                if own.startswith("InstancedFoliageActor") or (own.startswith("PCGGrassContainer")
                                                                and not (cfg.move_forest and cfg.move_grass)):
                    nb, nh = hide_instances(pk.export_bytes(e))
                    if nh:
                        payloads[path][i] = nb
                        rep.moved["foliage instances hidden"] += nh
        comps = [c for c in decode_components(pk, rules) if c.root and c.loc is not None]
        pcg_roots: dict[int, np.ndarray] = {}
        for c in comps:
            owner = ex[c.owner].name if c.owner < len(ex) else ""
            p_m = c.loc / 100
            if c.rule.L == 26:                     # runtime virtual texture volume
                lo = np.r_[new_cl[:, :2].min(0) - 300, new_cl[:, 2].min() - 150]
                hi = np.r_[new_cl[:, :2].max(0) + 300, new_cl[:, 2].max() + 300]
                pl = patched_payload(pk, c, lo * 100, c.rot if c.rot is not None else np.zeros(3),
                                     (hi - lo) * 100)
                pl, dropped = _drop_streaming_mips(pl, pk.names)
                payloads[path][c.export] = pl
                rep.moved["RVT volume"] += 1
                rep.checks.append("RVT: " + ("baked streaming low mips disabled (rendered live from the moved "
                                             "geometry)" if dropped else "no streaming mips found"))
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
                if cfg.filler and bd_piece is None:
                    loc = p_m + np.r_[filler_shift, 0.0]
                    payloads[path][c.export] = patched_payload(pk, c, loc * 100,
                                                               c.rot if c.rot is not None else np.zeros(3))
                    rep.moved["filler ground"] += 1
                elif bd_piece is not None:
                    loc, rot = rigid(bd_piece, p_m, c.rot)
                    payloads[path][c.export] = patched_payload(pk, c, loc * 100, rot)
                    rep.moved["backdrop terrain"] += 1
                continue
            if c.root and (owner.startswith("PCGPartitionActor")
                           or (cfg.move_grass and owner.startswith("PCGGrassContainer"))):
                pcg_roots[c.owner] = c.loc.copy()
            if owner.startswith(SKIP_OWNERS) or any(w in owner for w in SKIP_WORDS):
                continue
            kind = owner.split("_UAID")[0].rstrip("0123456789_")
            if kind in cfg.debug_skip_kinds:
                continue
            if "PlayerStart" in owner:
                # where the car is spawned (sits behind the start line, often
                # far from the route spline): always with the opening piece
                loc, rot = rigid(0, p_m, c.rot)
                if cfg.debug_spawn_s is not None:          # test builds: spawn anywhere on the route
                    j = min(int(np.searchsorted(new_s, cfg.debug_spawn_s)), len(new_cl) - 1)
                    loc = new_cl[j] + np.array([0, 0, 1.0])
                    rot = np.array([0.0, np.degrees(new_head[j]), 0.0])
                    rep.checks.append(f"DEBUG: car spawns at {cfg.debug_spawn_s:.0f} m on the new route")
                payloads[path][c.export] = patched_payload(pk, c, loc * 100, rot)
                rep.moved[kind] += 1
                continue
            k, d, s, idx = (v[0] for v in assign(p_m[None]))
            if route == "cut":
                continue
            if route:
                # start/finish areas travel with the first/last piece; intermediate
                # splits are re-spaced along the new route so they stay in order
                if kind.startswith("RaceSector") and k not in (0, len(pieces) - 1):
                    k = -1
                if cfg.debug_spawn_s is not None and route == "fwd" and "StartTargetPoint" in kind:
                    j = min(int(np.searchsorted(new_s, cfg.debug_spawn_s)), len(new_cl) - 1)
                    payloads[path][c.export] = patched_payload(
                        pk, c, (new_cl[j] + np.array([0, 0, 0.5])) * 100,
                        np.array([0.0, np.degrees(new_head[j]), 0.0]))
                    rep.checks.append(f"DEBUG: start target point at {cfg.debug_spawn_s:.0f} m")
                    continue
                if k >= 0 and d < 150:
                    loc, rot = rigid(k, p_m, c.rot)
                    rep.moved[kind] += 1
                else:
                    loc, rot, s_new = along(p_m, c.rot, idx)
                    rep.relocated.append((f"{route}:{kind}", round(s), round(s_new)))
                payloads[path][c.export] = patched_payload(pk, c, loc * 100, rot)
                continue
            if k >= 0 and lat_ok(s, d, cfg.prop_lateral) and owns(k, p_m[None])[0]:
                loc, rot = rigid(k, p_m, c.rot)
                loc[2] += seam_dz(k, loc[None])[0]
                payloads[path][c.export] = patched_payload(pk, c, loc * 100, rot)
                rep.moved[kind] += 1

        # PCG forest (HISM trees on PCGPartitionActors, positions relative to the
        # actor root): each tree moves with the piece owning its spot, if that
        # piece's terrain is under it; the rest are hidden
        if cfg.move_forest:
            for i, e in enumerate(ex):
                if e.outer not in pcg_roots:
                    continue
                if ex[e.outer].name.startswith("PCGPartitionActor") and not e.name.startswith(("HISM_", "ISM_")):
                    continue
                try:
                    nb, st = move_hism(pk.export_bytes(e), pcg_roots[e.outer], place_trees)
                except Exception as ex_:                       # layout not understood: hide instead
                    nb, _ = hide_instances(pk.export_bytes(e))
                    rep.checks.append(f"forest {e.name}: hidden ({ex_})")
                    st = {"n": 0}
                if st.get("n"):
                    payloads[path][i] = nb
                    rep.moved["forest trees"] += st["kept"]
                    rep.moved["forest trees hidden"] += st["n"] - st["kept"]
                    if st.get("boxes", 0) == 0:
                        forest_nobox[0] += 1

    if cfg.move_forest:
        rep.checks.append(f"forest: {rep.moved['forest trees']} trees moved, {rep.moved['forest trees hidden']} "
                          f"hidden; {forest_nobox[0]} components without cached bounds found")

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
    if cfg.hlod_visuals:
        # track tiles do not render once moved; keep their cells unloaded so the
        # (moved) HLOD track meshes in the main level are what is drawn
        tile_cell_names = {p.rsplit("/", 1)[-1][:-5] for p, pk in pkgs.items()
                           if p != ws.main and any(tile_re.match(e.name) for e in pk.exports)}
        refs = {i + 1 for i, e in enumerate(main.exports) if e.name in tile_cell_names}
        nb2, ndrop = drop_cells(payloads[ws.main][hi_], refs)
        payloads[ws.main][hi_] = nb2
        rep.checks.append(f"streaming: {ndrop} track-tile cells never load (HLOD visuals)")
    if cfg.all_cells_loaded:
        # experimental: register every cell in the top-level cell and widen its
        # record. NOTE: this hides the HLODs, which are the visible track in ACR.
        nb, hst = all_cells_in_top_level(bytes(b))
        payloads[ws.main][hi_] = nb
        for i, e in enumerate(main.exports):
            if e.name.startswith("WorldPartitionRuntimeCellDataSpatialHash"):
                nbp = widen_cell_data(main.export_bytes(e), hst["level"])
                if nbp is not None:
                    payloads[ws.main][i] = nbp
        rep.checks.append(f"streaming: all {hst['cells']} cells in grid level {hst['level']}, records widened")

    # ---- HLOD track meshes: the visible road/terrain in ACR. World-space
    # meshes (one per track tile) in the main level; move + clip them in place.
    moved_hlod = 0
    for i, e in enumerate(main.exports):
        if not e.name.startswith("StaticMesh_"):
            continue
        mb = main.export_bytes(e)
        wb = mesh_world_bounds(mb)
        if wb is None:
            continue
        lo, hi = wb
        if np.any((hi - lo)[:2] > grid.cell * 100 * 1.2):
            continue
        key = tuple(grid.of(((lo + hi) / 2 / 100)[None, :2])[0])
        if cfg.hlod_hide:
            # every cell stays loaded, so the full-detail geometry is always drawn;
            # the HLOD proxies (clipped to the corridor) only showed up as slabs
            # floating in the distance. Empty them (same size, all triangles degenerate).
            try:
                nbm, _ = clip_mesh_payload(mb, lambda cen: np.zeros(len(cen), bool))
            except ValueError:
                continue
            payloads[ws.main][i] = nbm
            moved_hlod += 1
            continue
        if key not in tile_piece:
            continue
        j = tile_piece[key]
        pc = pieces[j]

        def keep(cen_cm, pc=pc, j=j):
            d, idx = assign.tree.query(cen_cm[:, :2] / 100)
            s = idx * step
            return ((s >= pc.sa - cfg.range_ext) & (s <= pc.sb + cfg.range_ext) & lat_ok(s, d, cfg.terrain_clip)
                    & owns(j, cen_cm / 100))
        nbm, hs = transform_mesh(mb, pc.R, pc.t * 100, None if cfg.debug_no_clip else keep)
        payloads[ws.main][i] = nbm
        moved_hlod += 1
    rep.moved["HLOD mesh " + ("hidden" if cfg.hlod_hide else "moved")] += moved_hlod
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
            return (s >= pc.sa - cfg.range_ext) & (s <= pc.sb + cfg.range_ext) & lat_ok(s, d, lim) & owns(j, w)
        return keep

    gen_dir = f"/{lvl}/GeneratedSplits/"
    seam_max = 0.0
    for key, j in ([] if cfg.debug_no_clip else sorted(tile_piece.items())):
        if key not in tile_pivot:
            rep.checks.append(f"tile {key} has no actor"); continue
        stem = f"SM_{lvl}Track{key[0]:02d}{key[1]:02d}_"
        for p in sorted(x for x in fs.index if gen_dir in x and x.rsplit("/", 1)[-1].startswith(stem)
                        and x.endswith(".uasset")):
            mpk = Package(fs.read(p))
            mi = mesh_export_index(mpk)
            lim = cfg.road_clip if "_Road_" in p else cfg.terrain_clip
            nb, st = clip_mesh_payload(mpk.export_bytes(mpk.exports[mi]), keeper(tile_pivot[key], j, lim))
            pc_, pv_ = pieces[j], tile_pivot[key]
            nb, mx = displace_payload(
                nb, lambda P, pc_=pc_, pv_=pv_, j=j: 100.0 * seam_dz(j, ((P + pv_) / 100) @ pc_.R.T + pc_.t))
            seam_max = max(seam_max, mx / 100)
            new_pkgs[p] = rewrite_package(mpk, {mi: nb})
            rep.clipped.append((p.rsplit("/", 1)[-1], st["lods"], st["dropped"], st["tris"]))

    # ---- background meshes (satellite-map terrain around the original region,
    # world space): near the new stage they cut through or float above it
    if cfg.bg_clear > 0:
        cl_tree = cKDTree(new_cl[:, :2])
        for p in sorted(x for x in fs.index if f"/Environments/{lvl}/Backgrounds/SM_" in x and x.endswith(".uasset")):
            tf = _component_transform_for_mesh(main, p)
            if tf is None:
                rep.checks.append(f"background {p.rsplit('/', 1)[-1]}: placing component not found"); continue
            loc_cm, scale = tf
            mpk = Package(fs.read(p))
            mi = mesh_export_index(mpk)
            mb = mpk.export_bytes(mpk.exports[mi])
            lo_, hi_ = mesh_world_bounds(mb)
            wb = (lo_ * scale + loc_cm, hi_ * scale + loc_cm)
            try:
                nb, st = clip_mesh_payload(
                    mb, lambda cen, L=loc_cm, S=scale: cl_tree.query((cen * S + L)[:, :2] / 100)[0] > cfg.bg_clear)
            except ValueError as ex_:
                rep.checks.append(f"background {p.rsplit('/', 1)[-1]}: not clipped ({ex_})"); continue
            new_pkgs[p] = rewrite_package(mpk, {mi: nb})
            rep.checks.append(f"background {p.rsplit('/', 1)[-1]}: {st['dropped']}/{st['tris']} triangles within "
                              f"{cfg.bg_clear:.0f} m of the new route hidden; bounds "
                              f"{np.round(wb[0] / 100) if wb else '?'}..{np.round(wb[1] / 100) if wb else '?'} m")

    if cfg.hide_unused_tiles and not cfg.debug_no_clip:
        n_un = 0
        for key in sorted(set(tile_pivot) - set(tile_piece)):
            stem = f"SM_{lvl}Track{key[0]:02d}{key[1]:02d}_"
            for p in sorted(x for x in fs.index if gen_dir in x and x.rsplit("/", 1)[-1].startswith(stem)
                            and x.endswith(".uasset")):
                mpk = Package(fs.read(p))
                mi = mesh_export_index(mpk)
                try:
                    nb, _ = clip_mesh_payload(mpk.export_bytes(mpk.exports[mi]), lambda c: np.zeros(len(c), bool))
                except ValueError:
                    continue
                new_pkgs[p] = rewrite_package(mpk, {mi: nb})
                n_un += 1
        rep.checks.append(f"unused tiles: {n_un} meshes hidden ({len(set(tile_pivot) - set(tile_piece))} tiles)")
    rep.checks.append(f"seam blending: up to {seam_max:.1f} m vertical correction within {cfg.seam_width:.0f} m "
                      f"of piece boundaries")

    def bd_keep_world(w_m):
        if cfg.debug_no_backdrop:
            return np.zeros(len(w_m), bool)
        pc = pieces[bd_piece]
        d, idx = assign.tree.query(w_m[:, :2])
        s = idx * step
        return (unc_ext[idx] & (s >= pc.sa - 40) & (s <= pc.sb + 40) & lat_ok(s, d, cfg.terrain_clip)
                & owns(bd_piece, w_m))

    filler_h = None
    if cfg.filler:
        filler_h = _filler_height_field(
            surf, pieces, tree_new, tree_new_all, seam_dz, new_cl,
            [(m.vertices[:, [0, 2, 1]].astype(np.float64), m.soft_sub) for m in meshes if len(m.vertices)],
            bd_re_bytes=rb"Terrain\d{4}", bd_piece=bd_piece, bd_keep=bd_keep_world,
            bd_xform=(lambda U: U @ pieces[bd_piece].R.T + pieces[bd_piece].t) if bd_piece is not None else None,
            below=cfg.filler_below)
        for name, pivot in bd_pivot.items():
            p = next((x for x in fs.index if x.endswith(f"/{name}.uasset") and "/MeshesTerrain/" in x), None)
            if p is None:
                rep.checks.append(f"backdrop mesh for {name} not found"); continue
            mpk = Package(fs.read(p))
            mi = mesh_export_index(mpk)
            def to_h(P, pv=pivot):
                # world = R (P + pivot) + t (rigid with bd_piece; z unaffected by the yaw),
                # or the plain shift when no piece carries the backdrop
                src = (P + pv) / 100
                if bd_piece is not None:
                    pc0 = pieces[bd_piece]
                    xy = (src @ pc0.R.T + pc0.t)[:, :2]
                    return (filler_h(xy) - pc0.t[2]) * 100 - pv[2] - P[:, 2]
                return filler_h(src[:, :2] + filler_shift) * 100 - pv[2] - P[:, 2]
            nb, mx = displace_payload(mpk.export_bytes(mpk.exports[mi]), to_h)
            new_pkgs[p] = rewrite_package(mpk, {mi: nb})
        rep.checks.append(f"filler ground: {len(bd_pivot)} tiles reshaped to the stage height field "
                          f"({cfg.filler_below:.1f} m below the stage terrain)")
    elif bd_piece is not None and not cfg.debug_no_clip:
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
            keep = (s >= pc.sa - cfg.range_ext) & (s <= pc.sb + cfg.range_ext) & lat_ok(s, d, lim) & owns(j, cen)
            k = np.where(keep, j, -1)
        elif re.search(rb"Terrain\d{4}", m.soft_sub):
            if filler_h is not None:
                V = U.astype(np.float64).copy()
                if bd_piece is not None:
                    V = V @ pieces[bd_piece].R.T + pieces[bd_piece].t
                else:
                    V[:, :2] += filler_shift
                V[:, 2] = filler_h(V[:, :2])
                out_meshes.append(W.MeshOut(V[:, [0, 2, 1]], m.indices.astype(np.uint32), m.material_ref, m.soft_pkg,
                                            m.soft_asset, m.soft_sub, m.extra, m.group))
                rep.baked_tris += len(tri)
                continue
            if bd_piece is None:
                continue
            k = np.where(bd_keep_world(cen), bd_piece, -1)
        elif b"PCGGrass" in m.soft_sub:
            continue                                   # grass/rock instances stay behind (not moved)
        else:
            k, d, s, _ = assign(cen)
            k[~lat_ok(s, d, cfg.prop_lateral)] = -1
            for kk in np.unique(k[k >= 0]):
                m_ = k == kk
                k[np.flatnonzero(m_)[~owns(int(kk), cen[m_])]] = -1
        for kk in np.unique(k[k >= 0]):
            sel = tri[k == kk]
            used, inv = np.unique(sel, return_inverse=True)
            pc = pieces[int(kk)]
            V = U[used] @ pc.R.T + pc.t
            V[:, 2] += seam_dz(int(kk), V)
            V = V[:, [0, 2, 1]]
            out_meshes.append(W.MeshOut(V, inv.ravel().astype(np.uint32), m.material_ref, m.soft_pkg,
                                        m.soft_asset, m.soft_sub, m.extra, m.group))
            rep.baked_tris += len(sel)
    rep.baked_meshes = len(out_meshes)
    # seams: where terrain of two pieces meets at different heights (shows as
    # floating sheets, the lower side is culled from below)
    seam_pts = collections.defaultdict(list)
    for mo in out_meshes:
        if not (re.search(rb"Track\d{4}", mo.soft_sub) and len(mo.indices)):      # not the filler
            continue
        V = mo.vertices[:, [0, 2, 1]]
        cen = V[mo.indices.reshape(-1, 3)].mean(1)
        jj = int(np.argmin(np.array([np.linalg.norm(
            tree_new[j].query(cen[:50, :2])[0]) for j in range(len(pieces))])))
        seam_pts[jj].append(cen)
    rep.seams = []
    P = {j: np.vstack(v) for j, v in seam_pts.items()}
    for a in P:
        ta = cKDTree(P[a][:, :2])
        for b_ in P:
            if b_ <= a:
                continue
            d, ii = ta.query(P[b_][:, :2], distance_upper_bound=6.0)
            ok = np.isfinite(d)
            if not ok.any():
                continue
            dz = P[b_][ok, 2] - P[a][ii[ok], 2]
            big = np.abs(dz) > 3.0
            if big.any():
                q = P[b_][ok][big]
                ds, _ = cKDTree(new_cl[:, :2]).query(q[:, :2])
                rep.checks.append(f"seam pieces {a}/{b_}: {int(big.sum())} spots with height step >3 m "
                                  f"(max {np.abs(dz).max():.0f} m, {np.median(ds):.0f} m from the road)")
    new_pkgs[ws.baked] = rewrite_package(bpk, {0: W.baked_payload_full(bf, out_meshes)})

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


def _drop_streaming_mips(payload: bytes, names) -> tuple[bytes, bool]:
    """RuntimeVirtualTextureComponent: remove StreamingTexture (12) and
    StreamLowMips (13). The streaming texture is baked for the original map;
    without it the whole RVT is rendered at runtime from the moved writers."""
    from ..extract.unversioned import Reader
    from .writers import _hdr_for
    r = Reader(payload, 0)
    h = r.header()
    if not {12, 13} <= set(h) or [x for x in h if x < 12] != [0, 4]:
        return payload, False
    p = r.p
    p += 16                                     # soft object path: two FNames
    n = struct.unpack_from("<i", payload, p)[0]
    p += 4 + n                                  # + sub-path string
    p += 4                                      # [4] VirtualTexture (object ref)
    body = payload[r.p: p] + payload[p + 8:]    # skip [12] and [13] (4 bytes each)
    return _hdr_for([x for x in h if x not in (12, 13)]) + body, True


def verify_wp(utoc: str, ws: WPSource, new_pkgs: dict, payloads: dict, rules) -> list[str]:
    from ..extract.bakeddata import read_baked
    from ..extract.pacenote_table import read_pacenote_table
    out = []
    c = Container(utoc)
    bad = 0
    n_patched = 0
    bad_names = collections.Counter()
    for path, pl in payloads.items():
        pk = Package(c.read(path))
        comps = {cc.export for cc in decode_components(pk, rules)}
        want = [i for i in pl if pk.exports[i].name not in SPLINES
                and not pk.exports[i].name.startswith(("WorldPartitionRuntimeSpatialHash",
                                                       "WorldPartitionRuntimeCellData", "StaticMesh_",
                                                       "HISM_", "ISM_"))      # instance data, no transform
                and not (0 <= pk.exports[i].outer < len(pk.exports) and pk.exports[pk.exports[i].outer].name.startswith(
                    ("InstancedFoliageActor", "PCGGrassContainer", "PCGPartitionActor")))]
        n_patched += len(want)
        bad += sum(i not in comps for i in want)
        bad_names.update(re.sub(r"(_UAID_\w+|_\d+)+$", "", pk.exports[i].name) + " < " +
                         re.sub(r"(_UAID_\w+|_\d+)+$", "", pk.exports[pk.exports[i].outer].name
                                if 0 <= pk.exports[i].outer < len(pk.exports) else "-")
                         for i in want if i not in comps)
    out.append(f"level: {n_patched} patched components in {len(payloads)} packages, {bad} fail to re-decode"
               + (f" ({', '.join(f'{k} x{v}' for k, v in bad_names.most_common(5))})" if bad else ""))
    for path, kind in ws.route_cells.items():
        if kind == "cut":
            continue
        sp = level_splines(Package(c.read(path)))
        out.append(f"route {kind}: " + ", ".join(f"{k}={float(arclength(v)[-1]):.0f}m" for k, v in sp.items()))
    bpk = Package(c.read(ws.baked))
    meshes = read_baked(bpk)
    out.append(f"baked: {W.check_baked(bpk)}; triangle indices in range: "
               f"{all((m.indices.max() < len(m.vertices)) if len(m.indices) else True for m in meshes)}")
    for key, path in ws.tables.items():
        out.append(f"pacenotes {key}: {len(read_pacenote_table(Package(c.read(path))))} rows")
    return out

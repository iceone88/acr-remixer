"""acr-stagegen command line.

    python -m stagegen.cli generate --fake --length 10000 --seed 3 --out out/
    python -m stagegen.cli generate --location WrcgEs --length 8000 --seed 1 --out out/
    python -m stagegen.cli list
"""
from __future__ import annotations

import argparse
import os
import time

from .connector import max_abs_grade
from .fake import fake_location, self_clear
from .layout import build_layout, to_json
from .plot import plot_layout
from .search import SearchConfig, search
from .segments import CutConfig, cut_stage

# Real roads are twistier than the synthetic ones: accept cuts on gentler
# curves (R > 120 m) and slightly less steady grades.
REAL_CUT = CutConfig(max_curvature=1 / 120, max_grade_change=0.04, window_m=20.0)


def _list(a) -> int:
    from .extract.iostore import GameFS
    from .extract.real import DEFAULT_PAKS, find_levels
    fs = GameFS(a.paks or DEFAULT_PAKS, include_mods=True)
    for lv in find_levels(fs):
        c, _ = fs.index[lv]
        comp = "compressed" if any(b.method for b in c.blocks[:1]) else "uncompressed"
        print(f"{os.path.basename(lv)[:-5]:40s} {os.path.basename(c.utoc):34s} {comp}")
    return 0


# Single-level builds: shorter segments so a 6-7 km stage still recombines,
# with the opening/closing segments anchored to the real start and finish.
BUILD_CUT = CutConfig(max_curvature=1 / 90, max_grade_change=0.05, window_m=16.0,
                      min_len=200.0, max_len=700.0, anchor_ends=True)


def _build_wp(a, paks: str) -> int:
    """Base-game World Partition level: tile-aligned recombination."""
    from .build import writers as W
    from .build.wp import build_wp_stage
    from .extract.iostore import GameFS
    from .extract.real import load_wp_source
    from .extract.zen import Package
    from .tiles import TileGrid, TileSearchConfig, find_cuts, make_ranges, tile_search
    fs = GameFS(paks, include_mods=False)
    ws = load_wp_source(fs, a.level)
    st = ws.stage
    meshes, *_ = W.read_baked_raw(Package(fs.read(ws.baked)))
    grid = TileGrid.from_baked(meshes)
    cuts = find_cuts(st, grid, 2.0)
    cfg = TileSearchConfig(target_length=a.length, seed=a.seed)
    ranges = make_ranges(st, grid, cuts, 2.0, cfg.min_len, cfg.max_len)
    print(f"{st.name}: {(len(st.points)-1)*2/1000:.2f} km, {len(grid.tiles)} tiles of {grid.cell:.0f} m, "
          f"{len(cuts)-2} usable tile crossings, {len(ranges)} candidate pieces")
    t = time.time()
    best = tile_search(st, ranges, cuts, cfg)
    if best is None:
        print("no layout found; try another --seed or --length")
        return 1
    layout = build_layout(best, 2.0)
    order = " ".join(p.segment_id.split("#")[1] + ("r" if p.reversed else "") for p in layout.pieces)
    print(f"layout {layout.length/1000:.2f} km in {time.time()-t:.1f}s: pieces {order}, "
          f"kinks {', '.join(f'{k:+.0f}' for k in best.kinks)} deg, {len(layout.pacenotes)} calls")
    out = os.path.join(a.out, f"{a.level}_seed{a.seed}")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "layout.json"), "w") as f:
        f.write(to_json(layout, {"seed": a.seed, "source": "game", "level": a.level}))
    plot_layout(layout, os.path.join(out, "layout.png"),
                f"{a.level} seed {a.seed}: {layout.length/1000:.2f} km ({order})", 40)
    from .build.wp import WPConfig
    rep = build_wp_stage(layout, best, cuts, grid, ws, paks, out, WPConfig(debug_spawn_s=a.spawn_at, debug_no_clip=a.no_clip, debug_offset_zero=a.offset_zero, debug_offset=None if a.offset is None else tuple(float(v) for v in a.offset.split(',')) ))
    from .build.preview import render_collision
    fwd = next(p for p, k in ws.route_cells.items() if k == "fwd")
    render_collision(rep.out_files[0], ws.baked, fwd, os.path.join(out, "collision.png"),
                     f"{a.level} seed {a.seed}: built collision (start o, finish #)")
    print(f"placed at offset {tuple(float(x) for x in rep.offset)} m; {rep.packages} packages rewritten; moved " +
          ", ".join(f"{k} {v}" for k, v in rep.moved.most_common(12)))
    for r in rep.relocated:
        print(f"  relocated {r[0][:40]} {r[1]} m -> {r[2]} m")
    dropped = sum(c[2] for c in rep.clipped); total = sum(c[3] for c in rep.clipped)
    print(f"tile meshes clipped: {len(rep.clipped)} ({dropped}/{total} triangles hidden); "
          f"baked: {rep.baked_meshes} meshes, {rep.baked_tris} triangles; pacenotes {rep.notes}")
    for c in rep.checks:
        print("  check:", c)
    for f_ in rep.out_files:
        print("  wrote", f_)
    return 0


def _stop_game() -> None:
    """ACR keeps the mod container open; close it before overwriting."""
    import subprocess
    out = subprocess.run(["tasklist", "/FI", "IMAGENAME eq acr.exe"], capture_output=True, text=True).stdout
    if "acr.exe" in out.lower():
        print("closing ACR (it holds the installed container open)")
        subprocess.run(["taskkill", "/F", "/IM", "acr.exe"], capture_output=True)
        time.sleep(3)


def _build(a) -> int:
    rc = _build_impl(a)
    if rc == 0 and (a.install or a.launch):
        _stop_game()
        a.build_dir = os.path.join(a.out, f"{a.level}_seed{a.seed}")
        rc = _install(a)
        if rc == 0 and a.launch:
            os.startfile("steam://rungameid/3917090")
            print("launched ACR")
    return rc


def _build_impl(a) -> int:
    from .build.stage import build_stage
    from .extract.iostore import GameFS
    from .extract.real import DEFAULT_PAKS, load_location
    paks = a.paks or DEFAULT_PAKS
    fs = GameFS(paks, include_mods=False)
    if any(p.startswith(f"acr/Content/Levels/{a.level}/_Generated_/") for p in fs.index):
        return _build_wp(a, paks)
    stages = [s for s in load_location(a.level, paks) if s.name == a.level]
    if not stages:
        print(f"level {a.level!r} not found or not readable (try `list`)")
        return 1
    st = stages[0]
    segs = cut_stage(st, BUILD_CUT)
    print(f"{st.name}: {(len(st.points)-1)*2/1000:.2f} km, {len(segs)} segments")
    cfg = SearchConfig(target_length=a.length, seed=a.seed, corridor_half_width=a.corridor,
                       start_segment=segs[0].id, end_segment=segs[-1].id, novelty=25, variety=25)
    t = time.time()
    best = search(segs, cfg)
    if best is None:
        print("no layout found; try another --seed or --length")
        return 1
    layout = build_layout(best, cfg.step)
    order = " ".join(p.segment_id.split("#")[1] + ("r" if p.reversed else "") for p in layout.pieces
                     if p.kind == "segment")
    print(f"layout {layout.length/1000:.2f} km in {time.time()-t:.1f}s: segments {order}, "
          f"{layout.synthetic_share:.1%} connector, {len(layout.pacenotes)} calls")
    out = os.path.join(a.out, f"{a.level}_seed{a.seed}")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "layout.json"), "w") as f:
        f.write(to_json(layout, {"seed": a.seed, "source": "game", "level": a.level}))
    plot_layout(layout, os.path.join(out, "layout.png"),
                f"{a.level} seed {a.seed}: {layout.length/1000:.2f} km ({order})", a.corridor)
    rep = build_stage(layout, st, {s.id: s for s in segs}, a.level, paks, out)
    print(f"placed at offset {tuple(float(x) for x in rep.offset)} m; moved {sum(rep.moved.values())} actors: " +
          ", ".join(f"{k} {v}" for k, v in rep.moved.most_common(10)) +
          f" ... RVT volumes {rep.moved.get('RVT volume', 0)}")
    for r in rep.relocated:
        print(f"  relocated {r[0][:40]} {r[1]} m -> {r[2]} m")
    print(f"baked: {rep.baked_meshes} meshes, {rep.baked_tris} triangles + {rep.connector_tris} connector triangles; "
          f"{rep.notes} pacenotes")
    for c in rep.checks:
        print("  check:", c)
    for f_ in rep.out_files:
        print("  wrote", f_)
    return 0


def _mods_dir(paks: str | None) -> str:
    from .extract.real import DEFAULT_PAKS
    return os.path.normpath(os.path.join(paks or DEFAULT_PAKS, "~mods"))


def _install(a) -> int:
    import glob
    import shutil
    src = glob.glob(os.path.join(a.build_dir, "*_Stagegen_9_P.utoc"))
    if not src:
        print(f"no built container in {a.build_dir}")
        return 1
    mods = _mods_dir(a.paks)
    _uninstall(a, quiet=True)
    base = src[0][:-5]
    for ext in (".utoc", ".ucas", ".pak"):
        shutil.copy2(base + ext, mods)
        print("installed", os.path.join(mods, os.path.basename(base + ext)))
    return 0


def _uninstall(a, quiet: bool = False) -> int:
    import glob
    removed = 0
    for p in glob.glob(os.path.join(_mods_dir(a.paks), "*_Stagegen_9_P.*")):
        os.remove(p)
        removed += 1
        if not quiet:
            print("removed", p)
    if not quiet and not removed:
        print("nothing to remove")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(prog="acr-stagegen")
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate", help="generate a new stage layout")
    src = g.add_mutually_exclusive_group(required=True)
    src.add_argument("--fake", action="store_true", help="use synthetic source stages")
    src.add_argument("--location", help="level name prefix to recombine, e.g. WrcgEs")
    g.add_argument("--paks", help="path to acr/Content/Paks (default: the install this tool lives in)")
    g.add_argument("--length", type=float, default=10_000)
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--no-reverse", action="store_true")
    g.add_argument("--corridor", type=float, default=35.0, help="scanned corridor half-width (m)")
    g.add_argument("--out", default="out")
    ls = sub.add_parser("list", help="list stage levels found in the install")
    ls.add_argument("--paks")
    b = sub.add_parser("build", help="generate a stage from one level and build an installable mod container")
    b.add_argument("--level", default="WrcgEsSS1", help="stage level to recombine and replace")
    b.add_argument("--paks")
    b.add_argument("--length", type=float, default=5500)
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--corridor", type=float, default=35.0)
    b.add_argument("--out", default="out")
    b.add_argument("--spawn-at", type=float, default=None, help="test builds: spawn the car this many metres along the route")
    b.add_argument("--install", action="store_true", help="install into ~mods after building (closes ACR if running)")
    b.add_argument("--launch", action="store_true", help="install and start ACR through Steam")
    b.add_argument("--no-clip", action="store_true", help="test builds: leave track meshes unclipped")
    b.add_argument("--offset-zero", action="store_true", help="test builds: build on top of the original map")
    b.add_argument("--offset", default=None, help="test builds: placement offset x,y,z in metres")
    ins = sub.add_parser("install", help="copy a built stage container into acr/Content/Paks/~mods")
    ins.add_argument("build_dir", help="folder written by `build`, e.g. out/WrcgEsSS1_seed0")
    ins.add_argument("--paks")
    uni = sub.add_parser("uninstall", help="remove installed stagegen containers (restores the original stage)")
    uni.add_argument("--paks")
    a = ap.parse_args(argv)
    if a.cmd == "install":
        return _install(a)
    if a.cmd == "uninstall":
        return _uninstall(a)

    if a.cmd == "list":
        return _list(a)
    if a.cmd == "build":
        return _build(a)

    if a.fake:
        stages, cut, source = fake_location(), CutConfig(), "fake"
    else:
        from .extract.real import DEFAULT_PAKS, load_location
        stages = load_location(a.location, a.paks or DEFAULT_PAKS)
        if not stages:
            ap.error(f"no readable levels start with {a.location!r} (try `list`)")
        cut, source = REAL_CUT, "game"
    segments = [s for st in stages for s in cut_stage(st, cut)]
    print(f"{len(stages)} source stages ({', '.join(s.name for s in stages)}), {len(segments)} segments "
          f"({sum(s.length for s in segments)/1000:.1f} km of reusable road)")

    cfg = SearchConfig(target_length=a.length, seed=a.seed,
                       allow_reversed=not a.no_reverse, corridor_half_width=a.corridor)
    t = time.time()
    best = search(segments, cfg)
    if best is None:
        print("no layout found; try a shorter --length, smaller --corridor, or another --seed")
        return 1
    layout = build_layout(best, cfg.step)
    print(f"found {layout.length/1000:.2f} km layout in {time.time()-t:.1f}s; "
          f"{sum(p.kind=='segment' for p in layout.pieces)} segments, "
          f"{layout.synthetic_share:.1%} generated connector road, "
          f"{len(layout.pacenotes)} calls "
          f"({sum(n.origin=='original' for n in layout.pacenotes)} carried over)")

    clear = self_clear(layout.centreline, 2 * a.corridor * 0.95, cfg.step)
    print(f"check: corridors clear={clear}, max grade={max_abs_grade(layout.centreline[::5]):.1%}")

    os.makedirs(a.out, exist_ok=True)
    tag = "fake" if a.fake else a.location
    base = os.path.join(a.out, f"stage_{tag}_seed{a.seed}")
    with open(base + ".json", "w") as f:
        f.write(to_json(layout, {"seed": a.seed, "source": source, "location": stages[0].location,
                                 "surface": stages[0].surface,
                                 "source_stages": [s.name for s in stages]}))
    plot_layout(layout, base + ".png",
                f"Generated stage ({tag}), seed {a.seed}: {layout.length/1000:.2f} km, "
                f"{layout.synthetic_share:.0%} connector", a.corridor)
    print(f"wrote {base}.json, {base}.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

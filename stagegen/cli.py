"""acr-stagegen command line.

    python -m stagegen.cli generate --fake --length 10000 --seed 3 --out out/
"""
from __future__ import annotations

import argparse
import os
import time

from .fake import fake_location
from .layout import build_layout, to_json
from .plot import plot_layout
from .search import SearchConfig, search
from .segments import CutConfig, cut_stage


def main(argv=None):
    ap = argparse.ArgumentParser(prog="acr-stagegen")
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate", help="generate a new stage layout")
    g.add_argument("--fake", action="store_true", help="use synthetic source stages (no game data yet)")
    g.add_argument("--length", type=float, default=10_000)
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--no-reverse", action="store_true")
    g.add_argument("--corridor", type=float, default=35.0, help="scanned corridor half-width (m)")
    g.add_argument("--out", default="out")
    a = ap.parse_args(argv)

    if not a.fake:
        ap.error("real-game extraction isn't implemented yet; use --fake (see README: roadmap)")

    stages = fake_location()
    cut = CutConfig()
    segments = [s for st in stages for s in cut_stage(st, cut)]
    print(f"{len(stages)} source stages, {len(segments)} segments "
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

    from .fake import self_clear
    from .connector import max_abs_grade
    clear = self_clear(layout.centreline, 2 * a.corridor * 0.95, cfg.step)
    print(f"check: corridors clear={clear}, max grade={max_abs_grade(layout.centreline[::5]):.1%}")

    os.makedirs(a.out, exist_ok=True)
    base = os.path.join(a.out, f"stage_seed{a.seed}")
    with open(base + ".json", "w") as f:
        f.write(to_json(layout, {"seed": a.seed, "source": "fake", "location": stages[0].location}))
    plot_layout(layout, base + ".png",
                f"Generated stage, seed {a.seed}: {layout.length/1000:.2f} km, "
                f"{layout.synthetic_share:.0%} connector", a.corridor)
    print(f"wrote {base}.json, {base}.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

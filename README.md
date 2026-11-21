# acr-stagegen

Generate new Assetto Corsa Rally stages by recombining road segments from the
stages you already own. It runs on your own install and is meant to reference
the game's assets in place, so mods never redistribute scanned geometry.

**Status:** layout generator works on synthetic data. Nothing touches the game yet.

```
python3 -m stagegen.cli generate --fake --length 10000 --seed 3 --out out/
```

Writes `out/stage_seed3.json` (the layout manifest) and `out/stage_seed3.png`
(map + elevation profile).

## How it works

1. **Cut** (`segments.py`): split each source stage at "quiet" points:
   radius > 250 m, steady grade and steady width over ±30 m. Segments are
   250–1200 m long.
2. **Search** (`search.py`): beam search that chains segments, forwards or
   reversed, from one location and surface. For each joint:
   - The segment is placed using **only yaw + translation**. Scans are never
     tilted.
   - A short **connector** (`connector.py`, 40–160 m) bridges the joint. Its
     horizontal Hermite curve allows ±25° of bend, and its vertical Hermite
     matches both grades so there's no slope step. Width blends linearly.
   - **Corridor check:** a new piece may not come within 2 × corridor
     half-width (default 70 m) of anything already laid down, so the scanned
     scenery bands never overlap.
   - Elevation range and connector grade are capped.
   - Scoring prefers small grade/width mismatches, gentle bends, little
     connector road, and variety of source stages. A seeded per-segment bias
     makes each seed give a different stage.
3. **Pacenotes** (`pacenotes.py`): forward segments keep their **original
   calls**, re-keyed to the new distance. Connectors and reversed segments get
   generated calls from curvature (1–6 scale by minimum radius, "long",
   "into", crests).
4. **Manifest** (`layout.py`): for each piece it records the source segment,
   direction, rigid transform and distance range. Connectors also carry their
   centreline. A future build step consumes this.

## Roadmap (needs a real install)

- [ ] **Extract**: find where the route spline, road width, surface, pacenotes
      and split/reset data live in the paks (UE 5.6, probably IoStore
      `.utoc/.ucas`). Write a reader that produces `SourceStage` objects.
- [ ] **Asset granularity**: are stage meshes and collision split into
      reusable chunks, or baked into large World Partition cells / HLODs?
      This decides whether segments can be placed by referencing base-game
      assets.
- [ ] **Build**: create a level that references the base-game assets by
      `/Game/...` path with the manifest's transforms. Loft connector road and
      collision meshes, clip the scenery where corridors meet, place
      gates/resets, write pacenotes. Then cook it to a mod pak.
- [ ] **Install/uninstall**: drop the pak into the mods folder and remove it
      cleanly. Never modify the base game files.
- [ ] Tune cut/joint thresholds on real roads, which are twistier and less
      regular than the fake ones.
- [ ] Speed: the search takes about 30 s in pure Python. Cache end states and
      vectorise the overlap test.

## Layout

```
stagegen/
  geometry.py   polyline maths
  model.py      SourceStage, Segment, EndState, Piece, Pacenote
  segments.py   cut-point detection, segment end states
  connector.py  joint geometry
  search.py     beam search
  pacenotes.py  curvature → calls
  layout.py     flatten result, JSON manifest
  plot.py       map + elevation PNG
  fake.py       synthetic source stages
  cli.py
```

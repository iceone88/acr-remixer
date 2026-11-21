# acr-stagegen

Generate new Assetto Corsa Rally stages by recombining road segments from the
stages you already own. Everything is read in place from your install; the
output is a small mod container that only references game/mod assets.

**Status (2026-11-21):**
- Reads real game data: route splines, road width and original pacenotes for all
  10 base-game stages and mod stages (~150 km of road). Base-game containers are
  Oodle-compressed and need an `oo2core_9_win64.dll` already on the machine
  (auto-detected, or set `ACR_OODLE_DLL`).
- Generates layouts from real segments (any location).
- **Builds an installable stage** by replacing the WRC Generations mod's
  `WrcgEsSS1` level with a recombined version of itself. It is built and
  self-verified by re-reading the output, but **not yet tested in-game**.

```
py -m stagegen.cli list                                   # levels found in the install
py -m stagegen.cli generate --location Wales --length 10000 --seed 2 --out out/
py -m stagegen.cli build --level WrcgEsSS1 --length 5500 --seed 0 --out out
py -m stagegen.cli install out/WrcgEsSS1_seed0           # copies 3 files into acr/Content/Paks/~mods
py -m stagegen.cli uninstall                              # deletes them again -> original SS1 back
```
(Use the full path to Python 3.14: `%LOCALAPPDATA%\Programs\Python\Python314\python.exe`; needs numpy, scipy, matplotlib.)

## How the build works (`stagegen/build/`)

The generated stage takes over an existing stage slot through a higher-priority
IoStore container (`pakchunk99-WrcgEsSS1_Stagegen_9_P`) that holds only three
rewritten packages; every other asset is still served by the original mod.

1. **Layout**: segments of the one level, with the opening segment (start area)
   and closing segment (finish area) fixed, middle segments shuffled/reversed.
2. **Level (.umap)**: root scene components of actors in the used segments are
   moved with their segment's rigid transform to a fresh area next to the old
   map (exact per-class decoding, see `level.py`). Intermediate split triggers
   are re-placed along the new route in order. Route splines (centre, ideal,
   borders, looped contour) are rebuilt. The RVT volumes are moved over the
   new area so the road texture renders.
3. **Physics (`DA_*BakedData`, dmphysics)**: collision triangles are split per
   segment by distance along the source route, transformed, and connector
   road + verges are lofted.
4. **Pacenotes**: original calls carried over for forward segments; generated
   (game token vocabulary) for reversed segments and connectors.
5. **Container**: uncompressed TOC v8 + copied container header + `.pak` stub.

### Known limitations
- Connectors (~5-10 % of the stage) get collision plus a transplanted unused
  terrain tile (with its RVT road-texture writer) aligned to the join; the
  road there is visually approximate, and placements are rejected if the
  tile's terrain would rise above the drivable road elsewhere.
- Terrain tiles are 128 m; tiles near joints overlap or leave gaps.
- Not moved: crowds' internal spline paths, foliage instance actors (HISM), TV
  cameras outside used segments, physics state of cones (if stored separately).
- The track list still shows the original SS1 name/length.
- Base-game levels (World Partition) can be read but not yet rebuilt.

## Reverse-engineered formats (`stagegen/extract/`)
- `iostore.py` IoStore TOC v8 reader (Oodle via ctypes); `zen.py` zen package
  header (5.5+ summary with cell maps); `unversioned.py` property headers.
- `bakeddata.py` dmphysics BakedDataContainer (meshes in metres, axes x,up,y).
- `splines.py` DMSplineComponent curves; `pacenote_table.py` PacenoteSplineEntry rows.
- `build/cityhash.py` CityHash64 (verified against stored name hashes).

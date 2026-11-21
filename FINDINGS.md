# What's in the ACR install (as of the Nov 2026 build, UE 5.6, IoStore)

Source: `Manifest_UFSFiles_Win64.txt` in the game root lists every cooked asset.

## Levels
- One World Partition level per location: `acr/Content/Levels/<Location>.umap`
  plus `_Generated_/` cells. Example: `WalesS3HafrenNorth` holds all of its
  Full/Cut/Reverse variants.
- Locations: AlsaceS2Munster, AlsaceS4Saverne, GreeceS3Elatia, GreeceS4Loutraki,
  MonteCarloS1Bollene, MonteCarloS2Sisteron, SwedenS1Vargasen,
  SwedenS3Fredriksberg, WalesS3HafrenNorth, WalesS4HafrenSouth, plus Livigno.

## Road geometry is pre-chunked
`acr/Content/Environments/<Location>/GeneratedSplits/SM_<Location>Track####_Road_0`
and `..._Terrain_0`. Chunk counts per location:

| Location | Road | Terrain |
|---|---|---|
| WalesS3HafrenNorth | 72 | 100 |
| WalesS4HafrenSouth | 36 | 53 |
| Bollene | 110 | 131 |
| Sisteron | 96 | 122 |
| Elatia | 54 | 122 |
| Loutraki | 82 | 66 |
| Munster | 72 | 61 |
| Saverne | 60 | 74 |
| Vargasen (sic "Sweeden") | 269 | 155 |
| Fredriksberg | 176 | 109 |

These look like the natural unit for `Segment`. The next question is whether
chunk boundaries fall on quiet road (see segments.py) or whether cuts must go
mid-chunk.

## Other assets to extract next
- `Environments/<Location>/DataAssets/DA_<Location>BakedData`: probably the
  route spline, progress/sector data, and maybe baked physics. **Top priority.**
- `Environments/<Location>/PhysMatMasks/`, `TerrainMasks/`, `MeshesTerrain/`:
  surface and physics material masks.
- `Data/Pacenote/Tables/DT_Pacenote<Stage><Variant>`: DataTables, one per
  variant (FullForward/FullReverse/CutN...).
- `Data/Database/Main/TrackSelection/DT_Tracks`, `DT_TracksVariants`,
  `DT_TracksLocations`: stage registration. Compare with the WRC Generations
  mod's `pakchunk99-WrcgTables_P` to see how mods add a stage.
- `Data/SplineSpawner/Blueprints/`: spline-driven prop spawning (barriers,
  tape?).

## Mod loading
- `acr/Content/Paks/~mods/` with `pakchunk99-<Name>_P.{pak,utoc,ucas}`.
  The WRC Gen stage mod ships a separate `WrcgTables` pak (~17 MB) next to the
  level paks.

## Tooling to try (local Windows)
- FModel / CUE4Parse (UE 5.6 profile) to browse paks and export DataTables
  and DataAssets to JSON.
- retoc (IoStore <-> legacy pak) for building mod containers.
- Check first whether the game containers need an AES key. Look at
  container flags in the .utoc header.

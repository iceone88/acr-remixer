# Cloud geometry tools

- `open3d_smoke_test.py`: verifies CPU triangle ray casting.
- `mesh_inventory.py`: inventories OBJ meshes without modifying them.
- `compare_road_ground.py`: samples an upper road surface and casts vertical
  rays against a terrain mesh. It reports signed vertical distances; it does
  not decide whether a road is drivable and does not correct geometry.

See `docs/THREE_D_TOOLCHAIN_SETUP.md` for the cloud workflow and input safety.

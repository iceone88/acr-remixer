# Unreal Engine integration status

Unreal Engine is intentionally not installed by the standard GitHub-hosted
workflow. The ACR build reviewed by this project is UE 5.6, but a complete
editor/asset-authoring environment is too large and licence-sensitive to treat
as a small CI dependency.

The initial cloud Phase 0 pipeline uses StageGen's existing package reader and
OBJ exports, Open3D for triangle queries, and Blender for independent mesh
inspection. Consider a dedicated private cloud VM/self-hosted runner with the
matching UE 5.6 Editor only if later work must author/cook new Unreal assets.
Do not commit Epic engine source, binaries, or installers to this repository.

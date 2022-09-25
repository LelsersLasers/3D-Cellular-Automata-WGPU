# 3D-Cellular-Automata-WGPU
3d Cellular Automata using WGPU in Rust (for the web and using compute shaders)

The branches are very messy... I recommend you get working downloads from itch.io: https://lelserslasers.itch.io/3d-cellular-automata-wgpu-rust (and because it runs on the web, you can try it without downloading it).

Current branches:
- NoComputeShader
  - As the name would suggest, this version does not use compute shaders and thus works on almost all browsers
- ComputeShader:
  - Again the name sort of gives it away, but this version uses compute shaders.
  - Much much faster and smoother but does not run on most browsers by default.

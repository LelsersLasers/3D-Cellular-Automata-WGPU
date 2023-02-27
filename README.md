# 3D-Cellular-Automata-WGPU
3d Cellular Automata using WGPU in Rust (for the web and using compute shaders)

The branches are very messy... I recommend you get latest working downloads from itch.io: https://lelserslasers.itch.io/3d-cellular-automata-wgpu-rust (and because it runs on the web, you can try it without downloading it).

https://user-images.githubusercontent.com/48894200/221646009-ff042dd2-a918-4541-ab0c-a964067a7d6a.mp4

Current branches:
- SetRuleFromHTML:
  - The web version
  - Has an HTML/CSS/JS UI for changing the simulation rule and draw mode
    - Not present in the otherversion
  - Works in almost all browsers
- ComputeShader:
  - Uses compute shaders
    - Much much faster and smoother but is not set up to run on the web
  - Also changing any of the simulation rules requires a recompile

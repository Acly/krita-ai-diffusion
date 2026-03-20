# 2026-03-20 - Fix Flux Nunchaku GPU OOM

The `NunchakuFluxDiTLoader` in `ai_diffusion/comfy_workflow.py` was missing several parameters that other Nunchaku loaders (`Qwen`, `ZImage`) had, most notably `cpu_offload="auto"`. 

This deficiency prevented Flux Nunchaku from falling back to CPU/RAM when GPU memory was insufficient for large tensor allocations (such as a 1 GB buffer appearing in some layers).

## Changes:

- Updated `nunchaku_load_flux_diffusion_model` in `ai_diffusion/comfy_workflow.py` to support `cpu_offload`, `num_blocks_on_gpu`, and `use_pin_memory`.
- Set `cpu_offload="auto"` by default for Flux Nunchaku loads.
- Set `use_pin_memory="disable"` to further reduce VRAM overhead for large models.

This aligns with the latest version of the `ComfyUI-nunchaku` custom nodes as supported by the plugin's `v1.2.0` Nunchaku dependency.

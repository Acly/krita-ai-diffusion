import os
from pathlib import Path

from ai_diffusion.resources import Arch, default_checkpoints

_server_dir_env = os.getenv("AI_DIFFUSION_TEST_SERVER_DIR")


test_dir = Path(__file__).parent
root_dir = test_dir.parent
server_dir = Path(_server_dir_env) if _server_dir_env else test_dir / "server"
data_dir = test_dir / "data"
image_dir = test_dir / "images"
result_dir = test_dir / "results"
reference_dir = test_dir / "references"
benchmark_dir = test_dir / "benchmark"

default_checkpoint = {
    Arch.sd15: default_checkpoints[0].filename,
    Arch.sdxl: "RealVisXL_V5.0_fp16.safetensors",
    Arch.flux: "svdq-int4_r32-flux.1-krea-dev.safetensors",
    Arch.flux_k: "svdq-int4_r32-flux.1-kontext-dev.safetensors",
    Arch.flux2_4b: "flux-2-klein-4b.safetensors",
    Arch.zimage: "z_image_turbo_fp8_e4m3fn.safetensors",
}

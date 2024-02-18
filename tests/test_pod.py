import pytest
import json
import subprocess
import os
import sys
import asyncio
import aiohttp

from ai_diffusion.comfyworkflow import ComfyWorkflow, ImageTransferMode
from ai_diffusion.image import Extent, Image
from ai_diffusion.util import ensure
from .config import root_dir, test_dir, result_dir

pod_main = root_dir / "cloud" / "pod" / "pod.py"
run_dir = test_dir / "pod"


@pytest.fixture(scope="module")
def event_loop():
    loop = asyncio.get_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="module")
def pod_server(event_loop):
    async def serve(process: asyncio.subprocess.Process):
        try:
            async for line in ensure(process.stdout):
                print(line.decode("utf-8"), end="")
        except asyncio.CancelledError:
            process.terminate()
            await process.wait()

    # to run via docker instead:
    # docker run --gpus all -p 8000:25601 aclysia/sd-comfyui-pod:v1.14.0 python3.11 /pod.py --rp_serve_api --rp_api_port 25601 --rp_api_host 0.0.0.0

    async def start():
        env = os.environ.copy()
        env["COMFYUI_DIR"] = "C:\\Dev\\ComfyUI"
        args = ["-u", "-Xutf8", str(pod_main), "--rp_serve_api"]
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            *args,
            cwd=run_dir,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        async for line in ensure(process.stdout):
            text = line.decode("utf-8")
            print(text[:80], end="")
            if "Uvicorn running" in text:
                break

        return asyncio.create_task(serve(process))

    task = event_loop.run_until_complete(start())
    yield "http://localhost:8000"
    task.cancel()
    event_loop.run_until_complete(task)


class PodClient:
    default_local_url = "http://localhost:8000"

    def __init__(self, url):
        self.url = url

    async def run(self, prompt):
        payload = {"input": {"prompt": prompt}}
        async with aiohttp.ClientSession() as session:

            async with session.post(f"{self.url}/runsync", json=payload) as response:
                job = await response.json()
            job_id = job["id"]
            print("started job", job_id, "status", job["status"])

            while job["status"] == "IN_QUEUE" or job["status"] == "IN_PROGRESS":
                async with session.get(f"{self.url}/status/{job_id}") as response:
                    job2 = await response.json()
                    print(response.status, "job", job_id, "status", job2)
                    if response.status == 200:
                        job = job2
                    await asyncio.sleep(1)
            if job["status"] == "COMPLETED":
                return job["output"]
            elif job["status"] == "FAILED":
                raise Exception(job["error"])
            elif job["status"] == "CANCELLED":
                raise Exception("job cancelled")
            elif job["status"] == "TIMED_OUT":
                raise Exception("job timed out")
            else:
                raise Exception("unknown job status")


# def test_simple(event_loop, pod_server):
#     w = ComfyWorkflow(image_transfer=ImageTransferMode.memory)
#     model, clip, vae = w.load_checkpoint("dreamshaper_8.safetensors")
#     latent = w.empty_latent_image(Extent(512, 512))
#     positive = w.clip_text_encode(clip, "fluffy ball")
#     negative = w.clip_text_encode(clip, "bad quality")
#     sampled = w.ksampler(model, positive, negative, latent, seed=768)
#     decoded = w.vae_decode(vae, sampled)
#     w.send_image(decoded)

#     client = PodClient(pod_server)
#     output = event_loop.run_until_complete(client.run(w.root))
#     for imgb64 in output["images"]:
#         img = Image.from_base64(imgb64)
#         img.save(result_dir / "pod_simple.png")

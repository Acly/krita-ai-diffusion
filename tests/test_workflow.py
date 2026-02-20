import itertools
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from ai_diffusion import workflow
from ai_diffusion.api import (
    ConditioningInput,
    ControlInput,
    CustomWorkflowInput,
    FillMode,
    ImageInput,
    InpaintMode,
    InpaintParams,
    LoraInput,
    RegionInput,
    SamplingInput,
    UpscaleInput,
    WorkflowInput,
    WorkflowKind,
)
from ai_diffusion.client import CheckpointInfo, Client, ClientEvent, ClientModels
from ai_diffusion.cloud_client import CloudClient
from ai_diffusion.comfy_client import ComfyClient
from ai_diffusion.comfy_workflow import ComfyWorkflow
from ai_diffusion.files import File, FileCollection, FileLibrary, FileSource
from ai_diffusion.image import Bounds, Extent, Image, ImageCollection, Mask
from ai_diffusion.pose import Pose
from ai_diffusion.resources import ControlMode
from ai_diffusion.settings import PerformanceSettings
from ai_diffusion.style import Arch, Style
from ai_diffusion.util import ensure
from ai_diffusion.workflow import detect_inpaint

from . import config
from .config import default_checkpoint, image_dir, reference_dir, result_dir, root_dir, test_dir
from .conftest import CloudService

service_available = (root_dir / "service" / "web" / ".env.local").exists()
client_params = ["local", "cloud"] if service_available else ["local"]
files = FileLibrary(FileCollection(), FileCollection())


async def connect_local():
    client = await ComfyClient.connect()
    async for _ in client.discover_models(refresh=False):
        pass
    return client


async def connect_cloud(service: CloudService):
    user = await service.create_user("workflow-tester")
    return await CloudClient.connect(service.url, user["token"])


@pytest.fixture(params=client_params)
def client(pytestconfig, request, qtapp, cloud_service: CloudService):
    if pytestconfig.getoption("--ci"):
        pytest.skip("Diffusion is disabled on CI")

    if request.param == "local":
        client = qtapp.run(connect_local())
    else:
        if not cloud_service.enabled:
            pytest.skip("Cloud service not running")
        client = qtapp.run(connect_cloud(cloud_service))
    files.loras.update([File.remote(m) for m in client.models.loras], FileSource.remote)

    yield client

    qtapp.run(client.disconnect())


@pytest.fixture()
def local_client(pytestconfig, qtapp):
    if pytestconfig.getoption("--ci"):
        pytest.skip("Diffusion is disabled on CI")

    yield qtapp.run(connect_local())


default_seed = 1234
default_perf = PerformanceSettings(batch_size=1)


def default_style(client: Client, arch=Arch.sd15):
    version_checkpoints = [
        name for name, cp in client.models.checkpoints.items() if cp.arch is arch
    ]
    checkpoint = default_checkpoint[arch]

    style = Style(Path("default.json"))
    style.checkpoints = [checkpoint] + version_checkpoints
    if not arch.is_sdxl_like:
        style.style_prompt = ""
    if arch.is_flux_like:
        style.sampler = "Flux - Euler simple"
        style.cfg_scale = 3.5
    if arch is Arch.zimage:
        style.sampler = "Flux - Euler simple"
        style.cfg_scale = 1.0
        style.sampler_steps = 8
    if arch.is_flux2:
        style.sampler = "Flux 2 - Euler"
        style.cfg_scale = 1.0
        style.sampler_steps = 5
    return style


def create(kind: WorkflowKind, client: Client, **kwargs):
    kwargs.setdefault("cond", ConditioningInput(""))
    kwargs.setdefault("style", default_style(client))
    kwargs.setdefault("seed", default_seed)
    kwargs.setdefault("perf", default_perf)
    kwargs.setdefault("files", files)

    prompt = workflow.prepare_prompts(
        kwargs["cond"], kwargs["style"], kwargs["seed"], Arch.sd15, kwargs["files"]
    )
    kwargs["cond"] = prompt.conditioning
    return workflow.prepare(kind, models=client.models, loras=prompt.loras, **kwargs)


counter = 0


async def receive_images(client: Client, work: WorkflowInput):
    global counter

    counter += 1

    job_id = None
    messages = client.listen()
    async for msg in messages:
        if not job_id:
            job_id = await client.enqueue(work)
        if msg.event is ClientEvent.finished and msg.job_id == job_id:
            assert msg.images is not None
            return msg.images
        if msg.event is ClientEvent.error and msg.job_id == job_id:
            raise RuntimeError(msg.error)
    assert False, "Connection closed without receiving images"


def run_and_save(
    qtapp,
    client: Client,
    work: WorkflowInput,
    filename: str,
    composition_image: Image | None = None,
    composition_mask: Mask | None = None,
    output_dir: Path = result_dir,
):
    dump_workflow(work, filename, client)

    async def runner():
        return await receive_images(client, work)

    results: ImageCollection = qtapp.run(runner())
    assert len(results) == 1
    client_name = "local" if isinstance(client, ComfyClient) else "cloud"
    if composition_image and composition_mask:
        composition_image.draw_image(results[0], composition_mask.bounds.offset)
        composition_image.save(output_dir / f"{filename}_{client_name}.png")
    else:
        results[0].save(output_dir / f"{filename}_{client_name}.png")
    return results[0]


def dump_workflow(work: WorkflowInput, filename: str, client: Client):
    flow = workflow.create(work, client.models)
    flow.embed_images().dump((result_dir / "workflows" / filename).with_suffix(".json"))


_default_conditioning = ConditioningInput("")


def automatic_inpaint(
    image_extent: Extent,
    bounds: Bounds,
    sd_ver: Arch = Arch.sd15,
    cond: ConditioningInput = _default_conditioning,
):
    mode = workflow.detect_inpaint_mode(image_extent, bounds)
    params = detect_inpaint(mode, bounds, sd_ver, cond, strength=1.0)
    params.grow = max(25, int(bounds.width * 0.1))
    params.feather = max(51, int(bounds.width * 0.2))
    params.blend = 25
    return params


def test_inpaint_params():
    bounds = Bounds(0, 0, 100, 100)
    no_cond = ConditioningInput("")

    a = detect_inpaint(InpaintMode.fill, bounds, Arch.sd15, no_cond, 1.0)
    assert a.fill is FillMode.blur and a.use_inpaint_model and a.use_reference

    b = detect_inpaint(InpaintMode.add_object, bounds, Arch.sd15, no_cond, 1.0)
    assert b.fill is FillMode.neutral and not b.use_condition_mask

    c = detect_inpaint(InpaintMode.replace_background, bounds, Arch.sdxl, no_cond, 1.0)
    assert c.fill is FillMode.replace and c.use_inpaint_model and not c.use_reference

    prompt = ConditioningInput("prompt")
    d = detect_inpaint(InpaintMode.add_object, bounds, Arch.sd15, prompt, 1.0)
    assert d.use_condition_mask

    prompt.control = [ControlInput(ControlMode.line_art, Image.create(Extent(4, 4)))]
    e = detect_inpaint(InpaintMode.add_object, bounds, Arch.sd15, prompt, 1.0)
    assert not e.use_condition_mask

    prompt.edit_reference = True
    f = detect_inpaint(InpaintMode.fill, bounds, Arch.sd15, prompt, 1.0)
    assert f.fill is FillMode.none


def test_prepare_lora():
    models = ClientModels()
    models.checkpoints = {"CP": CheckpointInfo("CP", Arch.sd15)}
    models.loras = [
        "PINK_UNICORNS.safetensors",
        "MOTHER_OF_PEARL.safetensors",
        "x/FRACTAL.safetensors",
    ]

    files = FileLibrary(FileCollection(), FileCollection())
    fractal = files.loras.add(File.remote("x/FRACTAL.safetensors"))
    files.loras.set_meta(fractal, "lora_strength", 0.55)

    mop = files.loras.add(File.remote("MOTHER_OF_PEARL.safetensors"))
    files.loras.set_meta(mop, "lora_triggers", "crab")
    files.loras.update([File.remote(m) for m in models.loras], FileSource.remote)

    style = Style(Path("default.json"))
    style.checkpoints = ["CP"]
    style.loras.append({"name": "MOTHER_OF_PEARL.safetensors", "strength": 0.33})

    cond = ConditioningInput("test <lora:PINK_UNICORNS:0.77> baloon <lora:x/FRACTAL> space")
    result = workflow.prepare_prompts(cond, style, seed=29, arch=Arch.sd15, files=files)
    assert result.conditioning and result.conditioning.positive == "test  baloon  space crab"
    assert result.loras
    assert LoraInput("PINK_UNICORNS.safetensors", 0.77) in result.loras
    assert LoraInput("x/FRACTAL.safetensors", 0.55) in result.loras

    job = workflow.prepare(
        WorkflowKind.generate,
        Extent(512, 512),
        result.conditioning,
        style,
        seed=29,
        perf=default_perf,
        loras=result.loras,
        models=models,
        files=files,
    )
    assert job.models and job.models.loras
    assert LoraInput("MOTHER_OF_PEARL.safetensors", 0.33) in job.models.loras
    assert LoraInput("PINK_UNICORNS.safetensors", 0.77) in result.loras
    assert LoraInput("x/FRACTAL.safetensors", 0.55) in result.loras


def test_prepare_negative():
    files = FileLibrary(FileCollection(), FileCollection())
    style = Style(Path("default.json"))
    style.checkpoints = []
    style.negative_prompt = "neg-beg {prompt} neg-end"
    style.cfg_scale = 5.0
    style.live_cfg_scale = 1.0
    cond = ConditioningInput("positive prompt")
    cond.negative = "piong"

    result = workflow.prepare_prompts(cond, style, seed=1, arch=Arch.sd15, files=files)
    assert result.conditioning.negative == "neg-beg piong neg-end"
    assert result.metadata["negative_prompt"] == "piong"
    assert result.metadata["negative_prompt_final"] == "neg-beg piong neg-end"

    # CFG=1.0 -> empty negative prompt
    live = workflow.prepare_prompts(cond, style, 1, Arch.sd15, files=files, is_live=True)
    assert live.conditioning.negative == ""
    assert live.metadata["negative_prompt"] == "piong"
    assert live.metadata["negative_prompt_final"] == ""


def test_prepare_wildcards():
    files = FileLibrary(FileCollection(), FileCollection())
    mask = Mask.rectangle(Bounds(0, 0, 10, 10), Bounds(0, 0, 10, 10)).to_image()
    style = Style(Path("default.json"))
    style.checkpoints = []
    style.style_prompt = "style-beg {prompt} style-end"
    style.negative_prompt = "neg-beg {prompt} neg-end"
    cond = ConditioningInput("a {x|y|z} b {100|200|300} c")
    cond.negative = "{711|pret} +"
    cond.regions = [
        RegionInput(mask, Bounds(0, 0, 10, 10), "region {alpha|beta}"),
        RegionInput(mask, Bounds(0, 0, 10, 10), "no wildcards"),
    ]

    result = workflow.prepare_prompts(cond, style, seed=1, arch=Arch.sd15, files=files)

    assert result.conditioning is not None
    assert result.conditioning.positive == "a x b 300 c"
    assert result.metadata["prompt"] == "a {x|y|z} b {100|200|300} c"
    assert result.metadata["prompt_eval"] == "a x b 300 c"
    assert result.metadata["prompt_final"] == "style-beg a x b 300 c style-end"

    assert result.conditioning.negative == "neg-beg 711 + neg-end"
    assert result.metadata["negative_prompt"] == "{711|pret} +"
    assert result.metadata["negative_prompt_eval"] == "711 +"
    assert result.metadata["negative_prompt_final"] == "neg-beg 711 + neg-end"

    assert result.conditioning.regions[0].positive == "region alpha"
    assert result.metadata["regions"][0]["prompt"] == "region {alpha|beta}"
    assert result.metadata["regions"][0]["prompt_eval"] == "region alpha"

    assert result.conditioning.regions[1].positive == "no wildcards"
    assert result.metadata["regions"][1]["prompt"] == "no wildcards"
    assert result.metadata["regions"][1].get("prompt_eval") is None


@pytest.mark.parametrize("arch", [Arch.sd15, Arch.qwen_e_p])
def test_prepare_prompt_layers(arch: Arch):
    files = FileLibrary(FileCollection(), FileCollection())
    mask = Mask.rectangle(Bounds(0, 0, 10, 10), Bounds(0, 0, 10, 10)).to_image()
    style = Style(Path("default.json"))
    style.checkpoints = []
    cond = ConditioningInput("prompt <layer:layer1> for <layer:layer2>")
    cond.regions = [
        RegionInput(mask, Bounds(0, 0, 10, 10), "region <layer:layer3>"),
        RegionInput(mask, Bounds(0, 0, 10, 10), "region without layer"),
    ]

    result = workflow.prepare_prompts(cond, style, seed=1, arch=arch, files=files)
    assert result.conditioning is not None
    assert result.metadata["prompt"] == "prompt <layer:layer1> for <layer:layer2>"
    assert result.metadata.get("prompt_eval") is None
    if arch is Arch.sd15:
        assert result.conditioning.positive == "prompt  for"
        assert result.metadata["prompt_final"] == f"prompt  for, {style.style_prompt}"
    else:
        assert result.conditioning.positive == "prompt Picture 2 for Picture 3"
        assert (
            result.metadata["prompt_final"]
            == f"prompt Picture 2 for Picture 3, {style.style_prompt}"
        )


def test_prepare_prompt_instructions():
    files = FileLibrary(FileCollection(), FileCollection())
    style = Style(Path("default.json"))
    style.checkpoints = []
    cond = ConditioningInput("base prompt")
    cond.control = [
        ControlInput(ControlMode.style, Image.create(Extent(4, 4))),
        ControlInput(ControlMode.pose, Image.create(Extent(4, 4))),
    ]
    cond.edit_reference = True

    result = workflow.prepare_prompts(cond, style, seed=1, arch=Arch.flux2_4b, files=files)
    assert result.conditioning is not None
    expected_prompt = "Apply the style from image 2.\nMatch the pose in image 3.\n\nbase prompt"
    assert result.conditioning.positive == expected_prompt


def test_prepare_prompt_inpaint():
    files = FileLibrary(FileCollection(), FileCollection())
    style = Style(Path("default.json"))
    style.checkpoints = []
    cond = ConditioningInput("inpaint prompt")

    result = workflow.prepare_prompts(
        cond, style, 1, Arch.flux2_4b, InpaintMode.remove_object, files=files
    )
    assert result.conditioning is not None
    expected_prompt = "Remove the object.\n\ninpaint prompt"
    assert result.conditioning.positive == expected_prompt
    assert result.conditioning.edit_reference


@pytest.mark.parametrize("extent", [Extent(256, 256), Extent(800, 800), Extent(512, 1024)])
def test_generate(qtapp, client, extent: Extent):
    prompt = ConditioningInput("ship")
    job = create(WorkflowKind.generate, client, canvas=extent, cond=prompt)
    result = run_and_save(qtapp, client, job, f"test_generate_{extent.width}x{extent.height}")
    assert result.extent == extent


def test_inpaint(qtapp, client):
    image = Image.load(image_dir / "beach_768x512.webp")
    mask = Mask.rectangle(Bounds(40, 40, 320, 200), Bounds(0, 80, 400, 280))
    cond = ConditioningInput("beach, the sea, cliffs, palm trees")
    inpaint = detect_inpaint(InpaintMode.fill, mask.bounds, Arch.sd15, cond, 1.0)
    inpaint.feather = 32
    inpaint.grow = 4 + inpaint.feather // 2
    inpaint.blend = 21
    job = create(
        WorkflowKind.inpaint,
        client,
        canvas=image,
        mask=mask,
        style=default_style(client, Arch.sd15),
        cond=cond,
        perf=PerformanceSettings(batch_size=3),  # max 3 images@512x512 -> 2 images@768x512
        inpaint=inpaint,
    )

    async def main():
        results = await receive_images(client, job)
        assert len(results) == 2
        client_name = "local" if isinstance(client, ComfyClient) else "cloud"
        for i, result in enumerate(results):
            image.draw_image(result, mask.bounds.offset)
            image.save(result_dir / f"test_inpaint_{i}_{client_name}.png")
            assert result.extent == mask.bounds.extent

    qtapp.run(main())


@pytest.mark.parametrize("sdver", [Arch.sd15, Arch.sdxl, Arch.zimage, Arch.flux2_4b])
def test_inpaint_upscale(qtapp, client, sdver):
    image = Image.load(image_dir / "beach_1536x1024.webp")
    mask = Mask.rectangle(Bounds(150, 150, 768, 512), Bounds(150, 50, 1068, 812))
    prompt = ConditioningInput("ship")
    inpaint = detect_inpaint(InpaintMode.add_object, mask.bounds, sdver, prompt, 1.0)
    inpaint.feather = 50
    inpaint.grow = 4 + inpaint.feather // 2
    inpaint.blend = 25
    job = create(
        WorkflowKind.inpaint,
        client,
        canvas=image,
        mask=mask,
        style=default_style(client, sdver),
        cond=prompt,
        perf=PerformanceSettings(batch_size=3),
        inpaint=inpaint,
    )

    async def main():
        dump_workflow(job, f"test_inpaint_upscale_{sdver.name}.json", client)
        results = await receive_images(client, job)
        client_name = "local" if isinstance(client, ComfyClient) else "cloud"
        for i, result in enumerate(results):
            image.draw_image(result, mask.bounds.offset)
            image.save(result_dir / f"test_inpaint_upscale_{sdver.name}_{i}_{client_name}.png")
            assert result.extent == mask.bounds.extent

    qtapp.run(main())


def test_inpaint_odd_resolution(qtapp, client):
    image = Image.load(image_dir / "beach_768x512.webp")
    image = Image.scale(image, Extent(612, 513))
    mask = Mask.rectangle(Bounds(0, 0, 200, 513), Bounds(0, 0, 350, 513))
    job = create(
        WorkflowKind.inpaint,
        client,
        canvas=image,
        mask=mask,
        inpaint=automatic_inpaint(image.extent, mask.bounds),
    )
    result = run_and_save(qtapp, client, job, "test_inpaint_odd_resolution", image, mask)
    assert result.extent == mask.bounds.extent


def test_inpaint_area_conditioning(qtapp, client):
    image = Image.load(image_dir / "lake_1536x1024.webp")
    mask = Mask.load(image_dir / "lake_1536x1024_mask_bottom_right.png")
    prompt = ConditioningInput("(crocodile)")
    job = create(
        WorkflowKind.inpaint,
        client,
        canvas=image,
        mask=mask,
        cond=prompt,
        inpaint=detect_inpaint(InpaintMode.add_object, mask.bounds, Arch.sd15, prompt, 1.0),
    )
    run_and_save(qtapp, client, job, "test_inpaint_area_conditioning", image, mask)


def test_inpaint_remove_object(qtapp, client):
    image = Image.load(image_dir / "owls_inpaint.webp")
    mask = Mask.load(image_dir / "owls_mask_remove.webp")
    cond = ConditioningInput("tree branch")
    job = create(
        WorkflowKind.inpaint,
        client,
        canvas=image,
        mask=mask,
        cond=cond,
        inpaint=detect_inpaint(InpaintMode.remove_object, mask.bounds, Arch.sd15, cond, 1.0),
    )
    run_and_save(qtapp, client, job, "test_inpaint_remove_object", image, mask)


@pytest.mark.parametrize("setup", ["sd15", "sdxl", "flux", "flux_k", "flux2"])
def test_refine(qtapp, client, setup):
    if isinstance(client, CloudClient) and setup in ["flux", "flux_k"]:
        pytest.skip("Skipping test for CloudClient with flux models")

    sdver, extent, strength = {
        "sd15": (Arch.sd15, Extent(768, 508), 0.5),
        "sdxl": (Arch.sdxl, Extent(1111, 741), 0.65),
        "flux": (Arch.flux, Extent(1111, 741), 0.65),
        "flux_k": (Arch.flux_k, Extent(1111, 741), 1.0),
        "flux2": (Arch.flux2_4b, Extent(1111, 741), 1.0),
    }[setup]
    image = Image.load(image_dir / "beach_1536x1024.webp")
    image = Image.scale(image, extent)
    job = create(
        WorkflowKind.refine,
        client,
        canvas=image,
        style=default_style(client, sdver),
        cond=ConditioningInput("painting in the style of Vincent van Gogh"),
        strength=strength,
        perf=PerformanceSettings(batch_size=1, max_pixel_count=2),
    )
    if sdver.supports_edit:
        ensure(job.conditioning).edit_reference = True
    result = run_and_save(qtapp, client, job, f"test_refine_{setup}")
    assert result.extent == extent


@pytest.mark.parametrize("setup", ["sd15_0.4", "sd15_0.6", "sdxl_0.7", "flux_0.6"])
def test_refine_region(qtapp, client, setup):
    if isinstance(client, CloudClient) and setup == "flux_0.6":
        pytest.skip("Skipping test for CloudClient with flux models")

    sdver, strength = {
        "sd15_0.4": (Arch.sd15, 0.4),
        "sd15_0.6": (Arch.sd15, 0.6),
        "sdxl_0.7": (Arch.sdxl, 0.7),
        "flux_0.6": (Arch.flux, 0.6),
    }[setup]
    image = Image.load(image_dir / "lake_region.webp")
    mask = Mask.load(image_dir / "lake_region_mask.png")
    prompt = ConditioningInput("waterfall")
    params = detect_inpaint(InpaintMode.fill, mask.bounds, Arch.sd15, prompt, strength)
    job = create(
        WorkflowKind.refine_region,
        client,
        canvas=image,
        mask=mask,
        style=default_style(client, sdver),
        cond=prompt,
        strength=strength,
        inpaint=params,
    )
    result = run_and_save(qtapp, client, job, f"test_refine_region_{setup}", image, mask)
    assert result.extent == mask.bounds.extent


def test_differential_diffusion(qtapp, client):
    image = Image.scale(Image.load(image_dir / "beach_1536x1024.webp"), Extent(768, 512))
    mask = Mask.load(image_dir / "differential_diffusion_mask.webp")
    prompt = ConditioningInput("barren plain, volcanic wasteland, burned trees")
    params = detect_inpaint(InpaintMode.fill, mask.bounds, Arch.sd15, prompt, 0.9)
    job = create(
        WorkflowKind.refine_region,
        client,
        canvas=image,
        mask=mask,
        style=default_style(client, Arch.sd15),
        cond=prompt,
        strength=0.9,
        inpaint=params,
    )
    run_and_save(qtapp, client, job, "test_differential_diffusion", image, mask)


def region_prompt():
    root_text = "a collection of objects on a wooden workbench, evening light, dust motes"
    prompt = ConditioningInput(root_text)
    prompt.regions = [
        RegionInput(
            Mask.load(image_dir / "region_mask_bg.png").to_image(),
            Bounds(0, 0, 1024, 1024),
            "a workbench made of wood, sturdy and well-used",
        ),
        RegionInput(
            Mask.load(image_dir / "region_mask_1.png").to_image(),
            Bounds(320, 220, 350, 580),
            "a chemical bottle, something pink oozing out, flask",
        ),
        RegionInput(
            Mask.load(image_dir / "region_mask_2.png").to_image(),
            Bounds(600, 150, 424, 600),
            "a miniature model of a sailing boat made out of light wood, with red sails",
        ),
        RegionInput(
            Mask.load(image_dir / "region_mask_3.png").to_image(),
            Bounds(0, 250, 355, 700),
            "a gramophone with a large horn, made of brass",
        ),
    ]
    for region in prompt.regions:
        region.positive += " " + root_text
    return prompt


@pytest.mark.parametrize("sdver", [Arch.sd15, Arch.sdxl])
def test_regions(qtapp, client, sdver: Arch):
    style = default_style(client, sdver)
    job = create(
        WorkflowKind.generate, client, canvas=Extent(1024, 1024), cond=region_prompt(), style=style
    )
    run_and_save(qtapp, client, job, f"test_regions_{sdver.name}")


@pytest.mark.parametrize("kind", [WorkflowKind.inpaint, WorkflowKind.refine_region])
def test_regions_inpaint(qtapp, client, kind: WorkflowKind):
    image = Image.load(image_dir / "regions_inpaint.webp")
    mask = Mask.load(image_dir / "region_mask_inpaint.png")
    prompt = region_prompt()
    prompt.regions = prompt.regions[:2] + prompt.regions[3:]
    prompt.regions[0].mask = Mask.load(image_dir / "region_mask_bg2.png").to_image()
    params = detect_inpaint(InpaintMode.fill, mask.bounds, Arch.sd15, prompt, 1.0)
    strength = 0.7 if kind is WorkflowKind.refine_region else 1.0
    job = create(
        kind, client, canvas=image, mask=mask, cond=prompt, inpaint=params, strength=strength
    )
    run_and_save(qtapp, client, job, f"test_regions_{kind.name}", image, mask)


def test_regions_upscale(qtapp, client: Client):
    job = create(
        WorkflowKind.upscale_tiled,
        client,
        canvas=Image.load(image_dir / "regions_base.webp"),
        cond=region_prompt(),
        strength=0.5,
        upscale=UpscaleInput(client.models.default_upscaler),
        upscale_factor=2,
    )
    run_and_save(qtapp, client, job, "test_regions_upscale")


def test_regions_ip_adapter(qtapp, client: Client):
    cat = Image.load(image_dir / "cat.webp")
    flowers = Image.load(image_dir / "flowers.webp")
    prompt = region_prompt()
    prompt.regions[2].positive = ""
    prompt.regions[2].control.append(ControlInput(ControlMode.reference, flowers))
    prompt.regions[3].positive = ""
    prompt.regions[3].control.append(ControlInput(ControlMode.reference, cat))
    job = create(WorkflowKind.generate, client, canvas=Extent(1024, 1024), cond=prompt)
    run_and_save(qtapp, client, job, "test_regions_ip_adapter")


def test_regions_lora(qtapp, client: Client):
    files = FileLibrary.instance()
    files.loras.add(File.local(test_dir / "data" / "LowRA.safetensors"))
    files.loras.add(File.local(test_dir / "data" / "Ink scenery.safetensors"))
    root_text = "snowy landscape, tundra, illustration, truck in the distance"
    lines = Image.load(image_dir / "truck_landscape_lines.webp")
    prompt = ConditioningInput(root_text)
    prompt.regions = [
        RegionInput(
            Mask.load(image_dir / "region_mask_bg.png").to_image(),
            Bounds(0, 0, 1024, 1024),
            "frozen lake . " + root_text,
        ),
        RegionInput(
            Mask.load(image_dir / "region_mask_3.png").to_image(),
            Bounds(600, 150, 424, 600),
            "truck on an abandoned road . <lora:LowRA:1.0>" + root_text,
        ),
        RegionInput(
            Mask.load(image_dir / "region_mask_2.png").to_image(),
            Bounds(0, 250, 355, 700),
            "ink scenery, mountains, trees, <lora:Ink scenery:2.0>" + root_text,
        ),
    ]
    prompt.control = [ControlInput(ControlMode.soft_edge, lines, 0.5)]
    job = create(WorkflowKind.generate, client, canvas=Extent(1024, 1024), cond=prompt, files=files)
    run_and_save(qtapp, client, job, "test_regions_lora")


@pytest.mark.parametrize(
    "op", ["generate", "inpaint", "refine", "refine_region", "inpaint_upscale"]
)
def test_control_scribble(qtapp, client, op):
    scribble_image = Image.load(image_dir / "owls_scribble.webp")
    inpaint_image = Image.load(image_dir / "owls_inpaint.webp")
    mask = Mask.load(image_dir / "owls_mask.png")
    mask.bounds = Bounds(256, 0, 256, 512)
    control = [ControlInput(ControlMode.scribble, scribble_image)]
    prompt = ConditioningInput("owls", control=control)

    args: dict[str, Any]
    if op == "generate":
        args = {"kind": WorkflowKind.generate, "canvas": Extent(512, 512)}
    elif op == "inpaint":
        params = automatic_inpaint(inpaint_image.extent, mask.bounds)
        args = {
            "kind": WorkflowKind.inpaint,
            "canvas": inpaint_image,
            "mask": mask,
            "inpaint": params,
        }
    elif op == "refine":
        args = {"kind": WorkflowKind.refine, "canvas": inpaint_image, "strength": 0.7}
    elif op == "refine_region":
        kind = WorkflowKind.refine_region
        crop_image = Image.crop(inpaint_image, mask.bounds)
        control[0].image = Image.crop(scribble_image, mask.bounds)
        crop_mask = Mask(Bounds(0, 0, 256, 512), mask.image)
        params = automatic_inpaint(crop_image.extent, crop_mask.bounds)
        params.grow = params.feather = 0
        args = {
            "kind": kind,
            "canvas": crop_image,
            "mask": crop_mask,
            "strength": 0.7,
            "inpaint": params,
        }
    else:  # op == "inpaint_upscale":
        control[0].image = Image.scale(scribble_image, Extent(1024, 1024))
        inpaint_image = Image.scale(inpaint_image, Extent(1024, 1024))
        scaled_mask = Image.scale(Image(mask.image), Extent(512, 1024))
        mask = Mask(Bounds(512, 0, 512, 1024), scaled_mask._qimage)
        params = detect_inpaint(InpaintMode.fill, mask.bounds, Arch.sd15, prompt, 1.0)
        args = {
            "kind": WorkflowKind.inpaint,
            "canvas": inpaint_image,
            "mask": mask,
            "inpaint": params,
        }

    job = create(client=client, cond=prompt, **args)
    if op in ["inpaint", "refine_region", "inpaint_upscale"]:
        run_and_save(qtapp, client, job, f"test_control_scribble_{op}", inpaint_image, mask)
    else:
        run_and_save(qtapp, client, job, f"test_control_scribble_{op}")


@pytest.mark.parametrize("arch", [Arch.sdxl, Arch.zimage])
def test_control_lines(qtapp, client, arch: Arch):
    lines_image = Image.load(image_dir / "truck_landscape_lines.webp")
    control = [ControlInput(ControlMode.soft_edge, lines_image, 0.7)]
    prompt = ConditioningInput("truck in a snowy landscape, tundra", control=control)
    job = create(
        WorkflowKind.generate,
        client,
        canvas=Extent(1024, 1024),
        cond=prompt,
        style=default_style(client, arch),
    )
    run_and_save(qtapp, client, job, f"test_control_lines_{arch.name}")


def test_control_canny_downscale(qtapp, client):
    canny_image = Image.load(image_dir / "shrine_canny.webp")
    control = [ControlInput(ControlMode.canny_edge, canny_image, 1.0)]
    prompt = ConditioningInput("shrine", control=control)
    job = create(WorkflowKind.generate, client, canvas=Extent(999, 999), cond=prompt)
    run_and_save(qtapp, client, job, "test_control_canny_downscale")


@pytest.mark.parametrize("mode", [m for m in ControlMode if m.has_preprocessor])
def test_create_control_image(qtapp, client: Client, mode):
    if mode is ControlMode.hands:
        pytest.skip("No longer supported")
    skip_cloud_modes = [ControlMode.normal, ControlMode.segmentation, ControlMode.hands]
    if isinstance(client, CloudClient) and mode in skip_cloud_modes:
        pytest.skip("Control image preproccessor not available")

    image_name = f"test_create_control_image_{mode.name}"
    image = Image.load(image_dir / "adobe_stock.jpg")
    job = workflow.prepare_create_control_image(image, mode, default_perf)

    result = run_and_save(qtapp, client, job, image_name)
    if isinstance(client, ComfyClient):
        reference = Image.load(reference_dir / image_name)
        threshold = 0.015 if mode is ControlMode.pose else 0.005
        assert Image.compare(result, reference) < threshold
        # cloud results are a bit different, maybe due to compression of input?


def test_create_open_pose_vector(qtapp, client: Client):
    image_name = "test_create_open_pose_vector.svg"
    image = Image.load(image_dir / "adobe_stock.jpg")
    job = workflow.prepare_create_control_image(image, ControlMode.pose, default_perf)

    async def main():
        job_id = None
        async for msg in client.listen():
            if not job_id:
                job_id = await client.enqueue(job)
            if msg.event is ClientEvent.finished and msg.job_id == job_id:
                assert isinstance(msg.result, (dict, list))
                result = Pose.from_open_pose_json(msg.result).to_svg()
                (result_dir / image_name).write_text(result)
                return
            if msg.event is ClientEvent.error and msg.job_id == job_id:
                raise RuntimeError(msg.error)
        assert False, "Connection closed without receiving images"

    qtapp.run(main())


@pytest.mark.parametrize("sdver", [Arch.sd15, Arch.sdxl])
def test_ip_adapter(qtapp, client, sdver):
    image = Image.load(image_dir / "cat.webp")
    prompt = ConditioningInput("cat on a rooftop in paris")
    prompt.control = [ControlInput(ControlMode.reference, image, 0.6)]
    extent = Extent(512, 512) if sdver == Arch.sd15 else Extent(1024, 1024)
    style = default_style(client, sdver)
    job = create(WorkflowKind.generate, client, style=style, canvas=extent, cond=prompt)
    run_and_save(qtapp, client, job, f"test_ip_adapter_{sdver.name}")


def test_ip_adapter_region(qtapp, client):
    image = Image.load(image_dir / "flowers.webp")
    mask = Mask.load(image_dir / "flowers_mask.png")
    control_img = Image.load(image_dir / "pegonia.webp")
    prompt = ConditioningInput("potted flowers")
    prompt.control = [ControlInput(ControlMode.reference, control_img, 0.7)]
    inpaint = automatic_inpaint(image.extent, mask.bounds, Arch.sd15, prompt)
    job = create(
        WorkflowKind.refine_region,
        client,
        canvas=image,
        mask=mask,
        inpaint=inpaint,
        cond=prompt,
        strength=0.6,
    )
    run_and_save(qtapp, client, job, "test_ip_adapter_region", image, mask)


def test_ip_adapter_batch(qtapp, client):
    image1 = Image.load(image_dir / "cat.webp")
    image2 = Image.load(image_dir / "pegonia.webp")
    control = [
        ControlInput(ControlMode.reference, image1, 1.0),
        ControlInput(ControlMode.reference, image2, 1.0),
    ]
    cond = ConditioningInput("", control=control)
    job = create(WorkflowKind.generate, client, canvas=Extent(512, 512), cond=cond)
    run_and_save(qtapp, client, job, "test_ip_adapter_batch")


def test_style_composition_sdxl(qtapp, client):
    style_image = Image.load(image_dir / "watercolor.webp")
    composition_image = Image.load(image_dir / "flowers.webp")
    control = [
        ControlInput(ControlMode.style, style_image, 1.0),
        ControlInput(ControlMode.composition, composition_image, 0.8),
    ]
    job = create(
        WorkflowKind.generate,
        client,
        canvas=Extent(1024, 1024),
        cond=ConditioningInput("", control=control),
        style=default_style(client, Arch.sdxl),
    )
    run_and_save(qtapp, client, job, "test_style_composition")


@pytest.mark.parametrize("sdver", [Arch.sd15, Arch.sdxl])
def test_ip_adapter_face(qtapp, client, sdver):
    extent = Extent(650, 650) if sdver == Arch.sd15 else Extent(1024, 1024)
    image = Image.load(image_dir / "face.webp")
    cond = ConditioningInput("portrait photo of a woman at a garden party")
    cond.control = [ControlInput(ControlMode.face, image, 0.9)]
    job = create(WorkflowKind.generate, client, canvas=extent, cond=cond)
    run_and_save(qtapp, client, job, f"test_ip_adapter_face_{sdver.name}")


def test_upscale_simple(qtapp, client: Client):
    image = Image.load(image_dir / "beach_768x512.webp")
    job = workflow.prepare_upscale_simple(image, client.models.default_upscaler, 2.0)
    run_and_save(qtapp, client, job, "test_upscale_simple")


@pytest.mark.parametrize("sdver", [Arch.sd15, Arch.sdxl])
def test_upscale_tiled(qtapp, client: Client, sdver):
    image = Image.load(image_dir / "beach_768x512.webp")
    job = create(
        WorkflowKind.upscale_tiled,
        client,
        canvas=image,
        upscale=UpscaleInput(client.models.default_upscaler),
        upscale_factor=2.0,
        style=default_style(client, sdver),
        cond=ConditioningInput("4k uhd", control=[ControlInput(ControlMode.blur, None)]),
        strength=0.5,
    )
    run_and_save(qtapp, client, job, f"test_upscale_tiled_{sdver.name}")


def test_generate_live(qtapp, client):
    scribble = Image.load(image_dir / "owls_scribble.webp")
    job = create(
        WorkflowKind.generate,
        client,
        canvas=Extent(512, 512),
        cond=ConditioningInput("owls", control=[ControlInput(ControlMode.scribble, scribble)]),
        is_live=True,
    )
    run_and_save(qtapp, client, job, "test_generate_live")


@pytest.mark.parametrize("sdver", [Arch.sd15, Arch.sdxl])
def test_refine_live(qtapp, client, sdver):
    image = Image.load(image_dir / "pegonia.webp")
    if sdver is Arch.sdxl:
        image = Image.scale(image, Extent(1024, 1024))  # result will be a bit blurry
    job = create(
        WorkflowKind.refine,
        client,
        style=default_style(client, sdver),
        canvas=image,
        cond=ConditioningInput(""),
        strength=0.4,
        is_live=True,
    )
    run_and_save(qtapp, client, job, f"test_refine_live_{sdver.name}")


@pytest.mark.parametrize("arch", [Arch.flux_k, Arch.flux2_4b])
def test_edit(qtapp, client, arch):
    if isinstance(client, CloudClient) and arch is Arch.flux_k:
        pytest.skip("Flux Kontext not supported on Cloud")
    image = Image.load(image_dir / "flowers.webp")
    style = default_style(client, arch)
    cond = ConditioningInput("turn the image into a minimalistic vector illustration")
    cond.edit_reference = True
    job = create(WorkflowKind.refine, client, style=style, canvas=image, cond=cond)
    run_and_save(qtapp, client, job, f"test_edit_{arch.name}")


@pytest.mark.parametrize("arch", [Arch.flux_k, Arch.flux2_4b])
def test_edit_selection(qtapp, client, arch):
    if isinstance(client, CloudClient) and arch is Arch.flux_k:
        pytest.skip("Flux Kontext not supported on Cloud")
    image = Image.load(image_dir / "flowers.webp")
    mask = Mask.load(image_dir / "flowers_mask.png")
    cond = ConditioningInput("make all flowers have yellow blossoms")
    cond.edit_reference = True
    job = create(
        WorkflowKind.refine_region,
        client,
        style=default_style(client, arch),
        canvas=image,
        cond=cond,
        mask=mask,
        inpaint=automatic_inpaint(image.extent, mask.bounds, arch, cond),
    )
    run_and_save(qtapp, client, job, f"test_edit_selection_{arch.name}")


def test_edit_reference(qtapp, client):
    image = Image.load(image_dir / "flowers.webp")
    ref_image = Image.load(image_dir / "cat.webp")
    style = default_style(client, Arch.flux2_4b)
    cond = ConditioningInput("put the cat in the flower pot")
    cond.control = [ControlInput(ControlMode.reference, ref_image, 1.0)]
    cond.edit_reference = True
    job = create(WorkflowKind.refine, client, style=style, canvas=image, cond=cond)
    run_and_save(qtapp, client, job, "test_edit_reference")


def test_flux2_outpaint_lora(qtapp, client):
    image = Image.load(image_dir / "beach_1536x1024.webp")
    mask = Mask.rectangle(Bounds(0, 0, 400, 1024), Bounds(0, 0, 880, 1024))
    style = default_style(client, Arch.flux2_4b)
    cond = ConditioningInput("")
    cond, _, _md = workflow.prepare_prompts(cond, style, 1, Arch.flux2_4b, InpaintMode.expand)
    inpaint = workflow.detect_inpaint(InpaintMode.expand, mask.bounds, Arch.flux2_4b, cond, 1.0)
    inpaint.grow = 40
    inpaint.feather = 37
    inpaint.blend = 17
    job = create(
        WorkflowKind.inpaint,
        client,
        canvas=image,
        mask=mask,
        cond=cond,
        inpaint=inpaint,
        style=style,
    )
    run_and_save(qtapp, client, job, "test_flux2_outpaint_lora", image, mask)


def test_refine_max_pixels(qtapp, client):
    perf_settings = PerformanceSettings(max_pixel_count=1)  # million pixels
    image = Image.load(image_dir / "lake_1536x1024.webp")
    cond = ConditioningInput("watercolor painting on structured paper, aquarelle, stylized")
    job = create(
        WorkflowKind.refine, client, canvas=image, cond=cond, strength=0.6, perf=perf_settings
    )
    run_and_save(qtapp, client, job, "test_refine_max_pixels")


def test_fill_control_max_pixels(qtapp, client):
    perf_settings = PerformanceSettings(max_pixel_count=2)  # million pixels
    image = Image.load(image_dir / "beach_1536x1024.webp")
    image = Image.scale(image, Extent(2304, 1536))
    mask = Mask.load(image_dir / "beach_mask_2304x1536.webp")
    mask.bounds = Bounds(700, 0, 2304 - 700, 1536)
    depth = Image.load(image_dir / "beach_depth_2304x1536.webp")
    prompt = ConditioningInput("beach, the sea, cliffs, palm trees")
    prompt.control = [ControlInput(ControlMode.depth, depth)]
    inpaint = detect_inpaint(InpaintMode.fill, mask.bounds, Arch.sd15, prompt, 1.0)
    job = create(
        WorkflowKind.inpaint,
        client,
        canvas=image,
        mask=mask,
        cond=prompt,
        inpaint=inpaint,
        perf=perf_settings,
    )
    run_and_save(qtapp, client, job, "test_fill_control_max_pixels", image, mask)


def test_outpaint_resolution_multiplier(qtapp, client):
    perf_settings = PerformanceSettings(batch_size=1, resolution_multiplier=0.8)
    image = Image.create(Extent(2048, 1024))
    beach = Image.load(image_dir / "beach_1536x1024.webp")
    image.draw_image(beach, (512, 0))
    mask = Mask.load(image_dir / "beach_outpaint_mask.png")
    prompt = ConditioningInput("photo of a beach and jungle, nature photography, tropical")
    params = automatic_inpaint(image.extent, mask.bounds, Arch.sd15, prompt)
    job = create(
        WorkflowKind.inpaint,
        client,
        canvas=image,
        mask=mask,
        cond=prompt,
        inpaint=params,
        perf=perf_settings,
    )
    run_and_save(qtapp, client, job, "test_outpaint_resolution_multiplier", image, mask)


@pytest.mark.parametrize("arch", [Arch.sdxl, Arch.flux])
def test_dynamic_cache(qtapp, local_client, arch):
    job = create(
        WorkflowKind.generate,
        local_client,
        canvas=Extent(1024, 800),
        cond=ConditioningInput("a photo of autumn leaves and dead trees"),
        style=default_style(local_client, arch),
        perf=PerformanceSettings(dynamic_caching=True),
    )
    run_and_save(qtapp, local_client, job, f"test_dynamic_cache_{arch.name}")


def test_lora(qtapp, client):
    files = FileLibrary.instance()
    lora = files.loras.add(File.local(test_dir / "data" / "animeoutlineV4_16.safetensors"))
    style = default_style(client, Arch.sd15)
    style.loras.append({"name": lora.id, "strength": 1.0})
    job = create(
        WorkflowKind.generate,
        client,
        files=files,
        canvas=Extent(512, 512),
        cond=ConditioningInput("manga, lineart, monochrome, sunflower field"),
        style=style,
    )
    run_and_save(qtapp, client, job, "test_lora")


def test_nsfw_filter(qtapp, client):
    params = {"cond": ConditioningInput("nude")}
    job = create(WorkflowKind.generate, client, canvas=Extent(512, 512), **params)
    job.nsfw_filter = 0.8
    run_and_save(qtapp, client, job, "test_nsfw_filter")


def test_translation(qtapp, client):
    cond = ConditioningInput("rote (kerze) auf einer fensterbank", style="photo", language="de")
    job = create(WorkflowKind.generate, client, canvas=Extent(512, 512), cond=cond)
    run_and_save(qtapp, client, job, "test_translation")


def test_custom_workflow(qtapp, local_client: Client):
    workflow_json = test_dir / "data" / "workflow-custom.json"
    workflow_dict = json.loads(workflow_json.read_text())
    workflow_graph = ComfyWorkflow.import_graph(workflow_dict, local_client.models.node_inputs)
    params = {
        "1. Prompt": "a painting of a forest",
        "2. Detail/2. Steps": 14,
        "2. Detail/4. CFG": 3.5,
    }
    job = WorkflowInput(
        WorkflowKind.custom,
        images=ImageInput.from_extent(Extent(512, 512)),
        sampling=SamplingInput("custom", "custom", 1, 1000, seed=1234),
        inpaint=InpaintParams(InpaintMode.fill, Bounds(0, 0, 512, 512)),
        custom_workflow=CustomWorkflowInput(workflow_graph.root, params),
    )
    assert job.images is not None
    job.images.initial_image = Image.create(Extent(512, 512))
    run_and_save(qtapp, local_client, job, "test_custom_workflow")


inpaint_benchmark: dict[str, tuple[InpaintMode, str, Bounds | None]] = {
    "tori": (InpaintMode.fill, "photo of tori, japanese garden", None),
    "bruges": (InpaintMode.fill, "photo of a canal in bruges, belgium", None),
    "apple-tree": (
        InpaintMode.expand,
        "children's illustration showing kids next to an apple tree with a ladder",
        Bounds(0, 416, 1024, 1024 - 416),
    ),
    "girl-cornfield": (
        InpaintMode.expand,
        "anime artwork showing a girl standing in a wheat field under a blue sky",
        Bounds(0, 0, 1200 - 400, 1272),
    ),
    "street": (InpaintMode.remove_object, "photo of a street in tokyo", None),
    "nature": (
        InpaintMode.add_object,
        "photo of a stony river bed surrounded by trees. on the right side there is a black bear",
        Bounds(420, 200, 604, 718),
    ),
    "park": (
        InpaintMode.add_object,
        "photo of a lady sitting on a bench in a park",
        Bounds(310, 370, 940, 1110),
    ),
    "superman": (
        InpaintMode.expand,
        "superman giving a speech at a congress hall filled with people",
        None,
    ),
    "woman-travel": (
        InpaintMode.fill,
        "photo of a woman in a trenchcoat at a brisk walk. she is pulling a trolley suitcase. it is spring, she is walking alongside a brick wall trimmed by a victorian style fence. tree branches can be seen reaching over the wall, with early blossom buds. the woman is seen in profile, she has long hair. her clothes are fashionable and she is wearing sunglasses.",
        Bounds(275, 400, 1024 - 275, 1280 - 400),
    ),
    "bistro-table": (InpaintMode.remove_object, "photo of an empty bistro table", None),
}


def run_inpaint_benchmark(
    qtapp, client, sdver: Arch, prompt_mode: str, scenario: str, seed: int, out_dir: Path
):
    mode, prompt, bounds = inpaint_benchmark[scenario]
    image = Image.load(image_dir / "inpaint" / f"{scenario}-image.webp")
    mask = Mask.load(image_dir / "inpaint" / f"{scenario}-mask.webp")
    if bounds:
        mask = Mask.crop(mask, bounds)
    cond = ConditioningInput(prompt if prompt_mode == "prompt" else "")
    style = default_style(client, sdver)
    cond, _, md = workflow.prepare_prompts(cond, style, 1, sdver, mode)
    params = detect_inpaint(mode, mask.bounds, sdver, cond, 1.0)
    params.blend = 30
    params.feather = min(81, max(33, int(0.1 * mask.bounds.extent.diagonal)))
    params.grow = 4 + params.feather // 2
    job = create(
        WorkflowKind.inpaint,
        client,
        style=style,
        canvas=image,
        mask=mask,
        cond=cond,
        inpaint=params,
        seed=seed,
    )
    result_name = f"benchmark_inpaint_{scenario}_{sdver.name}_{prompt_mode}_{seed}"
    run_and_save(qtapp, client, job, result_name, image, mask, output_dir=out_dir)
    return result_name, md


def test_inpaint_benchmark(pytestconfig, qtapp, client):
    if not pytestconfig.getoption("--benchmark"):
        pytest.skip("Only runs with --benchmark")
    if isinstance(client, CloudClient):
        pytest.skip("Inpaint benchmark runs local")
    print()

    output_dir = config.benchmark_dir / datetime.now(UTC).strftime("%Y%m%d-%H%M")
    output_dir.mkdir(parents=True, exist_ok=True)
    seeds = [4213, 897281]
    prompt_modes = ["prompt", "noprompt"]
    scenarios = inpaint_benchmark.keys()
    sdvers = [Arch.sdxl, Arch.zimage, Arch.flux2_4b]
    runs = itertools.product(sdvers, scenarios, prompt_modes, seeds)
    meta = {}

    for sdver, scenario, prompt_mode, seed in runs:
        mode, prompt, bounds = inpaint_benchmark[scenario]
        prompt_required = mode in [InpaintMode.add_object, InpaintMode.replace_background]
        if prompt_required and prompt_mode == "noprompt":
            continue

        print("-", scenario, "|", sdver.name, "|", prompt_mode, "|", seed)
        result_name, md = run_inpaint_benchmark(
            qtapp, client, sdver, prompt_mode, scenario, seed, output_dir
        )
        meta[result_name] = {
            "arch": sdver.name,
            "user_prompt": prompt if prompt_mode == "prompt" else "",
            "full_prompt": md["prompt_final"],
            "scenario": scenario,
            "image": f"{scenario}-image.webp",
            "mask": f"{scenario}-mask.webp",
            "seed": seed,
            "mode": mode.name,
            "bounds": [bounds.x, bounds.y, bounds.width, bounds.height] if bounds else None,
        }
    (output_dir / "meta.json").write_text(json.dumps(meta, indent=4))


# def test_reproduce(qtapp, client: Client):
#     json_text = """
#     {
#         "images": {
#             "extent": {
#                 "desired": [
#                 3000,
#                 2000
#                 ],
#                 "initial": [
#                 768,
#                 512
#                 ],
#                 "input": [
#                 3000,
#                 2000
#                 ],
#                 "target": [
#                 5760,
#                 3840
#                 ]
#             },
#             "hires_image": 1,
#             "hires_mask": 2,
#             "initial_image": 0
#         },
#         "conditioning": {
#             "control": [
#                 {
#                     "mode": "line_art",
#                     "image": 3,
#                     "range": [
#                         0,
#                         0.8
#                     ]
#                 }
#             ],
#             "negative": "<redacted>",
#             "positive": "<redacted>",
#             "style": "<redacted>"
#         },
#         "inpaint": {
#             "feather": 164,
#             "grow": 164,
#             "mode": "custom",
#             "target_bounds": [
#                 2008,
#                 1152,
#                 3752,
#                 2680
#             ],
#             "use_inpaint_model": true
#         },
#         "models": {
#             "checkpoint": "serenity_v21Safetensors.safetensors",
#             "vae": "Checkpoint Default",
#             "version": "sd15"
#         },
#         "sampling": {
#             "cfg_scale": 7,
#             "sampler": "dpmpp_2m",
#             "scheduler": "karras",
#             "seed": 473470401,
#             "total_steps": 20
#         },
#         "crop_upscale_extent": [
#         2904,
#         2072
#         ],
#         "kind": "inpaint",
#         "nsfw_filter": 0.8
#     }
#     """
#     workflow = json.loads(json_text)
#     initial = Image.create(Extent(768, 512))
#     hires = Image.create(Extent(2904, 2072))
#     mask = Mask.transparent(Bounds(0, 0, 3000, 2000))
#     lineart = Image.create(Extent(3000, 2000))
#     images = ImageCollection([initial, hires, mask.to_image(), lineart])
#     data, offsets = images.to_bytes()
#     workflow["image_data"] = {"bytes": data, "offsets": offsets}
#     input = WorkflowInput.from_dict(workflow)
#     run_and_save(qtapp, client, input, "test_reproduce")

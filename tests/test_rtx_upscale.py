import math

from ai_diffusion.backend import resources, workflow
from ai_diffusion.backend.client import ClientModels
from ai_diffusion.backend.comfy_workflow import ComfyRunMode
from ai_diffusion.image import DummyImage, Extent


def _upscale_nodes(extent: Extent, factor: float, quality="ULTRA"):
    image = DummyImage(extent)
    work = workflow.prepare_upscale_simple(image, resources.rtx_vsr_node, factor, quality)
    graph = workflow.create(work, ClientModels(), ComfyRunMode.runtime)
    return list(graph)


def test_rtx_upscale_uses_nvidia_node_and_quality():
    nodes = _upscale_nodes(Extent(640, 480), 2.0, "HIGH")
    rtx = next(node for node in nodes if node.type == resources.rtx_vsr_node)

    assert rtx.inputs["resize_type"] == "scale by multiplier"
    scale = rtx.inputs["resize_type.scale"]
    assert isinstance(scale, (int, float)) and math.isclose(scale, 2.0)
    assert rtx.inputs["quality"] == "HIGH"
    assert all(node.type not in ("UpscaleModelLoader", "ImageUpscaleWithModel") for node in nodes)
    assert all(node.type != "ImageScale" for node in nodes)


def test_rtx_upscale_matches_exact_document_size():
    nodes = _upscale_nodes(Extent(101, 103), 2.0)
    scale = next(node for node in nodes if node.type == "ImageScale")

    assert scale.inputs["width"] == 202
    assert scale.inputs["height"] == 206


def test_standard_upscaler_keeps_existing_workflow():
    image = DummyImage(Extent(640, 480))
    work = workflow.prepare_upscale_simple(image, "4x_model.pth", 2.0, "HIGH")
    graph = workflow.create(work, ClientModels(), ComfyRunMode.runtime)

    assert work.upscale is not None and work.upscale.rtx_quality == "ULTRA"
    assert any(node.type == "UpscaleModelLoader" for node in graph)
    assert any(node.type == "ImageUpscaleWithModel" for node in graph)
    assert all(node.type != resources.rtx_vsr_node for node in graph)

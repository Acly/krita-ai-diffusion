import json
import zlib
from collections.abc import Iterable
from copy import copy
from pathlib import Path

import pytest
from PyQt5.QtCore import Qt

from ai_diffusion import workflow
from ai_diffusion.api import CustomStyleInput, CustomWorkflowInput, ImageInput, WorkflowInput
from ai_diffusion.client import (
    CheckpointInfo,
    Client,
    ClientModels,
    JobInfoOutput,
    OutputBatchMode,
    TextOutput,
)
from ai_diffusion.comfy_workflow import ComfyNode, ComfyObjectInfo, ComfyWorkflow, Output
from ai_diffusion.connection import Connection, ConnectionState
from ai_diffusion.custom_workflow import (
    CustomParam,
    CustomWorkspace,
    ParamKind,
    SortedWorkflows,
    WorkflowCollection,
    WorkflowSource,
    workflow_parameters,
)
from ai_diffusion.image import Bounds, Extent, Image, ImageCollection, Mask
from ai_diffusion.jobs import Job, JobKind, JobParams, JobQueue
from ai_diffusion.resources import Arch
from ai_diffusion.style import Style
from ai_diffusion.util import PluginError

from .config import test_dir


class MockClient(Client):
    def __init__(self, node_defs: ComfyObjectInfo):
        self.models = ClientModels()
        self.models.node_inputs = node_defs

    @staticmethod
    async def connect(url: str, access_token: str = "") -> Client:
        return MockClient(ComfyObjectInfo({}))

    async def enqueue(self, work: WorkflowInput, front: bool = False) -> str:
        return ""

    async def listen(self):  # type: ignore
        return

    async def interrupt(self):
        pass

    async def cancel(self, job_ids: Iterable[str]):
        pass


def create_mock_connection(
    initial_workflows: dict[str, dict],
    node_defs: ComfyObjectInfo | None = None,
    state: ConnectionState = ConnectionState.connected,
):
    connection = Connection()
    connection._client = MockClient(node_defs or ComfyObjectInfo({}))
    connection._workflows = initial_workflows
    connection.state = state
    return connection


def _assert_has_workflow(
    collection: WorkflowCollection,
    name: str,
    source: WorkflowSource,
    graph: dict,
    file: Path | None = None,
):
    workflow = collection.find(name)
    assert (
        workflow is not None
        and workflow.source == source
        and workflow.workflow.root == graph
        and workflow.path == file
    )


def test_collection(tmp_path: Path):
    file1 = tmp_path / "file1.json"
    file1_graph = {"0": {"class_type": "F1", "inputs": {}}}
    file1.write_text(json.dumps(file1_graph))

    file2 = tmp_path / "file2.json"
    file2_graph = {"0": {"class_type": "F2", "inputs": {}}}
    file2.write_text(json.dumps(file2_graph))

    connection_graph = {"0": {"class_type": "C1", "inputs": {}}}
    connection_workflows = {"connection1": connection_graph}
    connection = create_mock_connection(connection_workflows, state=ConnectionState.disconnected)

    collection = WorkflowCollection(connection, tmp_path)
    events = []

    assert len(collection) == 0

    def on_loaded():
        events.append("loaded")

    collection.loaded.connect(on_loaded)
    doc_graph = {"0": {"class_type": "D1", "inputs": {}}}
    collection.add_from_document("doc1", doc_graph)

    connection.state = ConnectionState.connected
    assert len(collection) == 4
    assert events == ["loaded"]
    _assert_has_workflow(collection, "file1", WorkflowSource.local, file1_graph, file1)
    _assert_has_workflow(collection, "file2", WorkflowSource.local, file2_graph, file2)
    _assert_has_workflow(collection, "connection1", WorkflowSource.remote, connection_graph)
    _assert_has_workflow(collection, "doc1", WorkflowSource.document, doc_graph)

    def on_begin_insert(index, first, last):
        events.append(("begin_insert", first))

    def on_end_insert():
        events.append("end_insert")

    def on_data_changed(start, end):
        events.append(("data_changed", start.row()))

    collection.rowsAboutToBeInserted.connect(on_begin_insert)
    collection.rowsInserted.connect(on_end_insert)
    collection.dataChanged.connect(on_data_changed)

    connection2_graph = {"0": {"class_type": "C2", "inputs": {}}}
    connection_workflows["connection2"] = connection2_graph
    connection.workflow_published.emit("connection2")

    assert len(collection) == 5
    _assert_has_workflow(collection, "connection2", WorkflowSource.remote, connection2_graph)

    file1_index = collection.find_index("file1").row()
    file1_graph_changed = {"0": {"class_type": "F3", "inputs": {}}}
    collection.set_graph(collection.find_index("file1"), file1_graph_changed)
    _assert_has_workflow(collection, "file1", WorkflowSource.local, file1_graph_changed, file1)
    assert events == ["loaded", ("begin_insert", 4), "end_insert", ("data_changed", file1_index)]

    sorted = SortedWorkflows(collection)
    assert sorted[0].source is WorkflowSource.document
    assert sorted[1].source is WorkflowSource.remote
    assert sorted[2].source is WorkflowSource.remote
    assert sorted[3].name == "file1"
    assert sorted[4].name == "file2"


def make_dummy_graph(n: int = 42):
    return {
        "1": {
            "class_type": "ETN_Parameter",
            "inputs": {
                "name": "param1",
                "type": "number (integer)",
                "default": n,
                "min": 5,
                "max": 95,
            },
        }
    }


def test_files(tmp_path: Path):
    collection_folder = tmp_path / "workflows"

    collection = WorkflowCollection(create_mock_connection({}), collection_folder)
    assert len(collection) == 0

    file1 = tmp_path / "file1.json"
    file1.write_text(json.dumps(make_dummy_graph()))

    collection.import_file(file1)
    assert collection.find("file1") is not None

    collection.import_file(file1)
    assert collection.find("file1 (1)") is not None

    collection.save_as("file1", make_dummy_graph(77))
    assert collection.find("file1 (2)") is not None

    files = [
        collection_folder / "file1.json",
        collection_folder / "file1 (1).json",
        collection_folder / "file1 (2).json",
    ]
    assert all(f.exists() for f in files)

    collection.remove("file1 (1)")
    assert collection.find("file1 (1)") is None
    assert not (collection_folder / "file1 (1).json").exists()

    bad_file = tmp_path / "bad.json"
    bad_file.write_text("bad json")
    with pytest.raises(PluginError):
        collection.import_file(bad_file)


async def dummy_generate(workflow_input):
    return None


def test_workspace():
    connection_workflows = {"connection1": make_dummy_graph(42)}
    connection = create_mock_connection(connection_workflows)
    workflows = WorkflowCollection(connection)

    jobs = JobQueue()
    workspace = CustomWorkspace(workflows, dummy_generate, jobs)
    assert workspace.workflow_id == "connection1"
    assert workspace.workflow and workspace.workflow.id == "connection1"
    assert workspace.graph and workspace.graph.node(0).type == "ETN_Parameter"
    assert workspace.metadata[0].name == "param1"
    assert workspace.params == {"param1": 42}

    doc_graph = {
        "1": {
            "class_type": "ETN_Parameter",
            "inputs": {
                "name": "param2",
                "type": "number (integer)",
                "default": 23,
                "min": 5,
                "max": 95,
            },
        }
    }
    workspace.set_graph("doc1", doc_graph, document_name="doc1")
    assert workspace.workflow_id == "Embedded Workflow (doc1)"
    assert workspace.workflow and workspace.workflow.source is WorkflowSource.document
    assert workspace.graph and workspace.graph.node(0).type == "ETN_Parameter"
    assert workspace.metadata[0].name == "param2"
    assert workspace.params == {"param2": 23}

    doc_graph["1"]["inputs"]["default"] = 24
    doc_graph["2"] = {
        "class_type": "ETN_Parameter",
        "inputs": {"name": "param3", "type": "number (integer)", "default": 7, "min": 0, "max": 10},
    }
    workflows.set_graph(workflows.index(1), doc_graph)
    assert workspace.metadata[0].default == 24
    assert workspace.metadata[1].name == "param3"
    assert workspace.params == {"param2": 23, "param3": 7}


def test_import():
    graph = {
        "4": {"class_type": "A", "inputs": {"steps": 4, "float": 1.2, "string": "mouse"}},
        "zak": {"class_type": "C", "inputs": {"in": ["9", 1]}},
        "9": {"class_type": "B", "inputs": {"in": ["4", 0]}},
    }
    w = ComfyWorkflow.import_graph(graph, ComfyObjectInfo({}))
    assert w.node(0) == ComfyNode(0, "A", {"steps": 4, "float": 1.2, "string": "mouse"})
    assert w.node(1) == ComfyNode(1, "B", {"in": Output(0, 0)})
    assert w.node(2) == ComfyNode(2, "C", {"in": Output(1, 1)})
    assert w.node_count == 3
    assert w.guess_sample_count() == 4


def test_import_with_loop():
    graph = {
        "1": {"class_type": "A", "inputs": {"steps": 4, "float": 1.2, "string": "mouse"}},
        "2": {"class_type": "B", "inputs": {"in": ["1", 0]}},
        "3": {"class_type": "C", "inputs": {"in": ["5", 1]}},
        "4": {"class_type": "D", "inputs": {"in": ["3", 0]}},
        "5": {"class_type": "E", "inputs": {"in": ["4", 0]}},
    }
    w = ComfyWorkflow.import_graph(graph, ComfyObjectInfo({}))
    # Currently imports a partial graph, important thing is that it does not hang/crash
    assert w.node_count > 0


def test_import_ui_workflow():
    graph = json.loads((test_dir / "data" / "workflow-ui.json").read_text())
    object_info = json.loads((test_dir / "data" / "object_info.json").read_text())
    node_defs = ComfyObjectInfo(object_info)
    result = ComfyWorkflow.import_graph(graph, node_defs)

    expected_graph = json.loads((test_dir / "data" / "workflow-api.json").read_text())
    expected = ComfyWorkflow.import_graph(expected_graph, ComfyObjectInfo({}))
    assert result.root == expected.root


def test_parameters():
    node_defs = ComfyObjectInfo({
        "ChoiceNode": {"input": {"required": {"choice_param": [["a", "b", "c"], {}]}}},
        "ChoiceNodeV3": {
            "input": {
                "required": {
                    "choice_param": ["COMBO", {"options": ["a", "b", "c"], "default": "a"}]
                }
            }
        },
    })

    w = ComfyWorkflow(node_defs)
    w.add("ETN_Parameter", 1, name="int", type="number (integer)", default=4, min=0, max=10)
    w.add("ETN_Parameter", 1, name="bool", type="toggle", default=True)
    w.add("ETN_Parameter", 1, name="number", type="number", default=1.2, min=0.0, max=10.0)
    w.add("ETN_Parameter", 1, name="text", type="text", default="mouse")
    w.add("ETN_Parameter", 1, name="positive", type="prompt (positive)", default="p")
    w.add("ETN_Parameter", 1, name="negative", type="prompt (negative)", default="n")
    w.add("ETN_Parameter", 1, name="choice_unconnected", type="choice", default="z")
    choice_param = w.add("ETN_Parameter", 1, name="choice", type="choice", default="c")
    w.add("ChoiceNode", 1, choice_param=choice_param)
    choice_param_v3 = w.add("ETN_Parameter", 1, name="choice_v3", type="choice", default="c")
    w.add("ChoiceNodeV3", 1, choice_param=choice_param_v3)
    w.add("ETN_KritaImageLayer", 1, name="image")
    w.add("ETN_KritaMaskLayer", 1, name="mask")
    w.add("ETN_KritaStyle", 9, name="style", sampler_preset="live")  # type: ignore

    assert list(workflow_parameters(w)) == [
        CustomParam(ParamKind.number_int, "int", 4, 0, 10),
        CustomParam(ParamKind.toggle, "bool", True),
        CustomParam(ParamKind.number_float, "number", 1.2, 0.0, 10.0),
        CustomParam(ParamKind.text, "text", "mouse"),
        CustomParam(ParamKind.prompt_positive, "positive", "p"),
        CustomParam(ParamKind.prompt_negative, "negative", "n"),
        CustomParam(ParamKind.text, "choice_unconnected", "z"),
        CustomParam(ParamKind.choice, "choice", "c", choices=["a", "b", "c"]),
        CustomParam(ParamKind.choice, "choice_v3", "c", choices=["a", "b", "c"]),
        CustomParam(ParamKind.image_layer, "image"),
        CustomParam(ParamKind.mask_layer, "mask"),
        CustomParam(ParamKind.style, "style", "live"),
    ]


def test_parameter_order():
    params = [
        CustomParam(ParamKind.number_int, "Ant", 4, 0, 10),
        CustomParam(ParamKind.number_int, "Bee", 4, 0, 10),
        CustomParam(ParamKind.number_int, "3. Cat", 4, 0, 10),
        CustomParam(ParamKind.number_int, "1. Dolphin", 4, 0, 10),
        CustomParam(ParamKind.number_int, "4. Elephant", 4, 0, 10),
        CustomParam(ParamKind.number_int, "2. Fish/9. Salmon", 4, 0, 10),
        CustomParam(ParamKind.number_int, "2. Fish/8. Trout", 4, 0, 10),
        CustomParam(ParamKind.number_int, "1. Insect/Dragonfly", 4, 0, 10),
    ]
    assert sorted(params) == [
        CustomParam(ParamKind.number_int, "Ant", 4, 0, 10),
        CustomParam(ParamKind.number_int, "Bee", 4, 0, 10),
        CustomParam(ParamKind.number_int, "1. Dolphin", 4, 0, 10),
        CustomParam(ParamKind.number_int, "3. Cat", 4, 0, 10),
        CustomParam(ParamKind.number_int, "4. Elephant", 4, 0, 10),
        CustomParam(ParamKind.number_int, "1. Insect/Dragonfly", 4, 0, 10),
        CustomParam(ParamKind.number_int, "2. Fish/8. Trout", 4, 0, 10),
        CustomParam(ParamKind.number_int, "2. Fish/9. Salmon", 4, 0, 10),
    ]


def test_prepare_mask():
    connection_workflows = {"connection1": make_dummy_graph(42)}
    connection = create_mock_connection(connection_workflows)
    workflows = WorkflowCollection(connection)

    jobs = JobQueue()
    workspace = CustomWorkspace(workflows, dummy_generate, jobs)

    mask = Mask.rectangle(Bounds(10, 10, 40, 40), Bounds(10, 10, 40, 40))
    canvas_bounds = Bounds(0, 0, 100, 100)
    selection_bounds = Bounds(12, 12, 34, 34)
    selection_node = ComfyNode(0, "ETN_Selection", {"context": "automatic", "padding": 3})

    prepared_mask, bounds = workspace.prepare_mask(
        selection_node, copy(mask), selection_bounds, canvas_bounds
    )
    assert bounds == Bounds(6, 6, 48, 48)  # mask.bounds + padding // multiple of 8
    assert prepared_mask is not None
    assert prepared_mask.bounds == Bounds(4, 4, 40, 40)

    selection_node.inputs["context"] = "mask_bounds"
    prepared_mask, bounds = workspace.prepare_mask(
        selection_node, copy(mask), selection_bounds, canvas_bounds
    )
    assert bounds == Bounds(9, 9, 40, 40)  # selection_bounds + padding // multiple of 8
    assert prepared_mask is not None
    assert prepared_mask.bounds == Bounds(1, 1, 40, 40)

    selection_node.inputs["context"] = "entire_image"
    prepared_mask, bounds = workspace.prepare_mask(
        selection_node, copy(mask), selection_bounds, canvas_bounds
    )
    assert bounds == canvas_bounds
    assert prepared_mask is not None
    assert prepared_mask.bounds == mask.bounds


def test_text_output():
    connection_workflows = {"connection1": make_dummy_graph(42)}
    connection = create_mock_connection(connection_workflows, ComfyObjectInfo({}))
    workflows = WorkflowCollection(connection)

    output_events = []

    def on_output(outputs: dict):
        output_events.append(copy(outputs))

    text_messages = [
        TextOutput("1", "Food", "Dumpling", "text/plain"),
        TextOutput("2", "Drink", "Tea", "text/plain"),
        TextOutput("3", "Time", "Moonrise", "text/plain"),
        TextOutput("1", "Food", "Sweet Potato", "text/plain"),
    ]

    jobs = JobQueue()
    job_params = JobParams(Bounds(0, 0, 1, 1), "test")
    job1 = Job("job1", JobKind.diffusion, job_params)
    job2 = Job("job2", JobKind.diffusion, job_params)

    workspace = CustomWorkspace(workflows, dummy_generate, jobs)
    workspace.outputs_changed.connect(on_output)
    workspace.handle_output(job1, text_messages[0])
    workspace.handle_output(job1, text_messages[1])
    assert workspace.outputs == {"1": text_messages[0], "2": text_messages[1]}

    jobs.job_finished.emit(job1)

    workspace.handle_output(job2, text_messages[3])
    workspace.handle_output(job2, text_messages[2])
    jobs.job_finished.emit(job2)
    assert workspace.outputs == {"1": text_messages[3], "3": text_messages[2]}

    assert output_events == [
        {"1": text_messages[0]},  # show_output(0)
        {"1": text_messages[0], "2": text_messages[1]},  # show_output(1)
        # job_finished(job1) - no changes
        {"1": text_messages[3], "2": text_messages[1]},  # show_output(3)
        {"1": text_messages[3], "2": text_messages[1], "3": text_messages[2]},  # show_output(2)
        {"1": text_messages[3], "3": text_messages[2]},  # job_finished(job2)
    ]


def test_job_info_output():
    job = Job("job1", JobKind.diffusion, JobParams(Bounds(0, 0, 1, 1), "test"))
    job_anim = Job("job2", JobKind.animation, JobParams(Bounds(0, 0, 1, 1), "test"))
    output1 = JobInfoOutput(name="Name1", batch_mode=OutputBatchMode.images, resize_canvas=True)
    output2 = JobInfoOutput(name="Name2", batch_mode=OutputBatchMode.animation, resize_canvas=False)
    output3 = JobInfoOutput(name="Name3", batch_mode=OutputBatchMode.layers, resize_canvas=True)

    workspace = CustomWorkspace(
        WorkflowCollection(create_mock_connection({})), dummy_generate, JobQueue()
    )

    workspace.handle_output(job, output1)
    assert job.params.resize_canvas is True
    assert job.params.name == "Name1"
    assert job.kind == JobKind.diffusion

    workspace.handle_output(job, output2)
    assert job.params.resize_canvas is False
    assert job.params.name == "Name2"
    assert job.kind == JobKind.animation

    workspace.handle_output(job_anim, output3)
    assert job_anim.params.resize_canvas is True
    assert job_anim.params.name == "Name3"
    assert job_anim.kind == JobKind.diffusion
    assert job_anim.params.is_layered is True


def img_id(image: Image):
    data = image.to_bytes()
    hash = zlib.crc32(data)
    return f"{hash:08x}"


def test_expand():
    ext = ComfyWorkflow()
    in_img, width, height, seed, in_mask = ext.add("ETN_KritaCanvas", 5)  # type: ignore
    rgba = ext.apply_mask(in_img, in_mask)
    scaled = ext.add("ImageScale", 1, image=rgba, width=width, height=height)
    ext.add("ETN_KritaOutput", 1, images=scaled)
    inty = ext.add(
        "ETN_Parameter", 1, name="inty", type="number (integer)", default=4, min=0, max=10
    )
    numby = ext.add("ETN_Parameter", 1, name="numby", type="number", default=1.2, min=0.0, max=10.0)
    texty = ext.add("ETN_Parameter", 1, name="texty", type="text", default="mouse")
    booly = ext.add("ETN_Parameter", 1, name="booly", type="toggle", default=True)
    choicy = ext.add("ETN_Parameter", 1, name="choicy", type="choice", default="c")
    layer_img = ext.add("ETN_KritaImageLayer", 1, name="layer_img")
    layer_mask = ext.add("ETN_KritaMaskLayer", 1, name="layer_mask")
    stylie = ext.add("ETN_KritaStyle", 9, name="style", sampler_preset="auto")  # type: ignore
    ext.add(
        "Sink",
        1,
        seed=seed,
        inty=inty,
        numby=numby,
        texty=texty,
        booly=booly,
        choicy=choicy,
        layer_img=layer_img,
        layer_mask=layer_mask,
        model=stylie[0],
        clip=stylie[1],
        vae=stylie[2],
        positive=stylie[3],
        negative=stylie[4],
        sampler=stylie[5],
        scheduler=stylie[6],
        steps=stylie[7],
        guidance=stylie[8],
    )

    style = Style(Path("default.json"))
    style.checkpoints = ["checkpoint.safetensors"]
    style.style_prompt = "bee hive"
    style.negative_prompt = "pigoon"

    models = ClientModels()
    models.checkpoints = {
        "checkpoint.safetensors": CheckpointInfo("checkpoint.safetensors", Arch.sd15)
    }

    style_input = CustomStyleInput(
        models=style.get_models(models.checkpoints),
        sampling=workflow.sampling_from_style(style, 1.0, False),
        positive_prompt=style.style_prompt,
        negative_prompt=style.negative_prompt,
    )

    params = {
        "inty": 7,
        "numby": 3.4,
        "texty": "cat",
        "booly": False,
        "choicy": "b",
        "layer_img": Image.create(Extent(4, 4), Qt.GlobalColor.black),
        "layer_mask": Image.create(Extent(4, 4), Qt.GlobalColor.white),
        "style": style_input,
    }

    input = CustomWorkflowInput(workflow=ext.root, params=params)
    images = ImageInput.from_extent(Extent(4, 4))
    images.initial_image = Image.create(Extent(4, 4), Qt.GlobalColor.red)

    w = ComfyWorkflow()
    w = workflow.expand_custom(w, input, images, Bounds(0, 0, 4, 4), 123, models)

    expected = [
        ComfyNode(1, "ETN_LoadImageCache", {"id": img_id(images.initial_image)}),
        ComfyNode(2, "ETN_ApplyMaskToImage", {"image": Output(1, 0), "mask": Output(1, 1)}),
        ComfyNode(3, "ImageScale", {"image": Output(2, 0), "width": 4, "height": 4}),
        ComfyNode(4, "ETN_KritaOutput", {"images": Output(3, 0)}),
        ComfyNode(5, "ETN_LoadImageCache", {"id": img_id(params["layer_img"])}),
        ComfyNode(6, "ETN_LoadImageCache", {"id": img_id(params["layer_mask"])}),
        ComfyNode(7, "CheckpointLoaderSimple", {"ckpt_name": "checkpoint.safetensors"}),
        ComfyNode(
            8,
            "Sink",
            {
                "seed": 123,
                "inty": 7,
                "numby": 3.4,
                "texty": "cat",
                "booly": False,
                "choicy": "b",
                "layer_img": Output(5, 0),
                "layer_mask": Output(6, 1),
                "model": Output(7, 0),
                "clip": Output(7, 1),
                "vae": Output(7, 2),
                "positive": "bee hive",
                "negative": "pigoon",
                "sampler": "dpmpp_2m",
                "scheduler": "karras",
                "steps": 20,
                "guidance": 7.0,
            },
        ),
    ]
    for node in expected:
        assert node in w, f"Node {node} not found in\n{json.dumps(w.root, indent=2)}"


def test_expand_animation():
    ext = ComfyWorkflow()
    img_layer, img_layer_alpha = ext.add("ETN_KritaImageLayer", 2, name="image")
    mask_layer = ext.add("ETN_KritaMaskLayer", 1, name="mask")
    ext.add("Sink", 1, image=img_layer, image_alpha=img_layer_alpha, mask=mask_layer)

    in_images = ImageCollection([
        Image.create(Extent(4, 4), Qt.GlobalColor.black),
        Image.create(Extent(4, 4), Qt.GlobalColor.white),
    ])
    in_masks = ImageCollection([
        Mask.rectangle(Bounds(0, 0, 4, 4), Bounds(0, 0, 4, 4)).to_image(),
        Mask.rectangle(Bounds(1, 1, 3, 3), Bounds(1, 1, 3, 3)).to_image(),
    ])
    params = {
        "image": in_images,
        "mask": in_masks,
    }

    input = CustomWorkflowInput(workflow=ext.root, params=params)
    images = ImageInput.from_extent(Extent(4, 4))
    models = ClientModels()

    w = ComfyWorkflow()
    w = workflow.expand_custom(w, input, images, Bounds(0, 0, 4, 4), 123, models)

    expected = [
        ComfyNode(1, "ETN_LoadImageCache", {"id": img_id(in_images[0])}),
        ComfyNode(2, "ETN_LoadImageCache", {"id": img_id(in_images[1])}),
        ComfyNode(3, "ImageBatch", {"image1": Output(1, 0), "image2": Output(2, 0)}),
        ComfyNode(4, "MaskToImage", {"mask": Output(1, 1)}),
        ComfyNode(5, "MaskToImage", {"mask": Output(2, 1)}),
        ComfyNode(6, "ImageBatch", {"image1": Output(4, 0), "image2": Output(5, 0)}),
        ComfyNode(7, "ImageToMask", {"image": Output(6, 0), "channel": "red"}),
        ComfyNode(8, "ETN_LoadImageCache", {"id": img_id(in_masks[0])}),
        ComfyNode(9, "ETN_LoadImageCache", {"id": img_id(in_masks[1])}),
        ComfyNode(10, "MaskToImage", {"mask": Output(8, 1)}),
        ComfyNode(11, "MaskToImage", {"mask": Output(9, 1)}),
        ComfyNode(12, "ImageBatch", {"image1": Output(10, 0), "image2": Output(11, 0)}),
        ComfyNode(13, "ImageToMask", {"image": Output(12, 0), "channel": "red"}),
        ComfyNode(
            14,
            "Sink",
            {
                "image": Output(3, 0),
                "image_alpha": Output(7, 0),
                "mask": Output(13, 0),
            },
        ),
    ]
    for node in expected:
        assert node in w, f"Node {node} not found in\n{json.dumps(w.root, indent=2)}"


def test_expand_selection():
    ext = ComfyWorkflow()
    select, select_active, off_x, off_y = ext.add(
        "ETN_KritaSelection", 4, context="automatic", padding=2
    )
    canvas, width, height, _seed = ext.add("ETN_KritaCanvas", 4)
    ext.add(
        "Sink",
        1,
        image=canvas,
        width=width,
        height=height,
        mask=select,
        has_selection=select_active,
        offset_x=off_x,
        offset_y=off_y,
    )

    params = {}
    input = CustomWorkflowInput(workflow=ext.root, params=params)
    images = ImageInput.from_extent(Extent(8, 16))
    images.initial_image = Image.create(Extent(8, 16), Qt.GlobalColor.red)
    images.hires_mask = Image.create(Extent(8, 16), Qt.GlobalColor.green)
    bounds = Bounds(2, 3, 8, 16)  # selection from (2,2) to (6,6)
    models = ClientModels()

    w = ComfyWorkflow()
    w = workflow.expand_custom(w, input, images, bounds, 123, models)

    expected = [
        ComfyNode(1, "ETN_LoadImageCache", {"id": img_id(images.hires_mask)}),
        ComfyNode(2, "ETN_LoadImageCache", {"id": img_id(images.initial_image)}),
        ComfyNode(
            3,
            "Sink",
            {
                "image": Output(2, 0),
                "width": 8,
                "height": 16,
                "mask": Output(1, 1),
                "has_selection": True,
                "offset_x": 2,
                "offset_y": 3,
            },
        ),
    ]
    for node in expected:
        assert node in w, f"Node {node} not found in\n{json.dumps(w.root, indent=2)}"

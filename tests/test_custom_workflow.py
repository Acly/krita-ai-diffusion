import json
import pytest
from copy import copy
from pathlib import Path
from PyQt5.QtCore import Qt

from ai_diffusion.api import CustomWorkflowInput, ImageInput, WorkflowInput
from ai_diffusion.client import Client, ClientModels, CheckpointInfo, TextOutput
from ai_diffusion.connection import Connection, ConnectionState
from ai_diffusion.comfy_workflow import ComfyNode, ComfyWorkflow, Output
from ai_diffusion.custom_workflow import WorkflowSource, WorkflowCollection
from ai_diffusion.custom_workflow import SortedWorkflows, CustomWorkspace
from ai_diffusion.custom_workflow import CustomParam, ParamKind, workflow_parameters
from ai_diffusion.image import Image, Extent, ImageCollection
from ai_diffusion.jobs import JobQueue, Job, JobKind, JobParams
from ai_diffusion.style import Style
from ai_diffusion.resources import Arch
from ai_diffusion.image import Bounds
from ai_diffusion import workflow

from .config import test_dir


class MockClient(Client):
    def __init__(self, node_inputs: dict[str, dict]):
        self.models = ClientModels()
        self.models.node_inputs = node_inputs

    @staticmethod
    async def connect(url: str, access_token: str = "") -> Client:
        return MockClient({})

    async def enqueue(self, work: WorkflowInput, front: bool = False) -> str:
        return ""

    async def listen(self):  # type: ignore
        return

    async def interrupt(self):
        pass

    async def clear_queue(self):
        pass


def create_mock_connection(
    initial_workflows: dict[str, dict],
    node_inputs: dict[str, dict] | None = None,
    state: ConnectionState = ConnectionState.connected,
):
    connection = Connection()
    connection._client = MockClient(node_inputs or {})
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

    collection = WorkflowCollection(create_mock_connection({}, {}), collection_folder)
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
    with pytest.raises(RuntimeError):
        collection.import_file(bad_file)


async def dummy_generate(workflow_input):
    return None


def test_workspace():
    connection_workflows = {"connection1": make_dummy_graph(42)}
    connection = create_mock_connection(connection_workflows, {})
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
    w = ComfyWorkflow.import_graph(graph, {})
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
    w = ComfyWorkflow.import_graph(graph, {})
    # Currently imports a partial graph, important thing is that it does not hang/crash
    assert w.node_count > 0


def test_import_ui_workflow():
    graph = json.loads((test_dir / "data" / "workflow-ui.json").read_text())
    object_info = json.loads((test_dir / "data" / "object_info.json").read_text())
    node_inputs = {k: v.get("input") for k, v in object_info.items()}
    result = ComfyWorkflow.import_graph(graph, node_inputs)

    expected_graph = json.loads((test_dir / "data" / "workflow-api.json").read_text())
    expected = ComfyWorkflow.import_graph(expected_graph, {})
    assert result.root == expected.root


def test_parameters():
    node_inputs = {"ChoiceNode": {"required": {"choice_param": (["a", "b", "c"],)}}}

    w = ComfyWorkflow(node_inputs=node_inputs)
    w.add("ETN_Parameter", 1, name="int", type="number (integer)", default=4, min=0, max=10)
    w.add("ETN_Parameter", 1, name="bool", type="toggle", default=True)
    w.add("ETN_Parameter", 1, name="number", type="number", default=1.2, min=0.0, max=10.0)
    w.add("ETN_Parameter", 1, name="text", type="text", default="mouse")
    w.add("ETN_Parameter", 1, name="positive", type="prompt (positive)", default="p")
    w.add("ETN_Parameter", 1, name="negative", type="prompt (negative)", default="n")
    w.add("ETN_Parameter", 1, name="choice_unconnected", type="choice", default="z")
    choice_param = w.add("ETN_Parameter", 1, name="choice", type="choice", default="c")
    w.add("ChoiceNode", 1, choice_param=choice_param)
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


def test_text_output():
    connection_workflows = {"connection1": make_dummy_graph(42)}
    connection = create_mock_connection(connection_workflows, {})
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
    workspace = CustomWorkspace(workflows, dummy_generate, jobs)
    workspace.outputs_changed.connect(on_output)
    workspace.show_output(text_messages[0])
    workspace.show_output(text_messages[1])
    assert workspace.outputs == {"1": text_messages[0], "2": text_messages[1]}

    job_params = JobParams(Bounds(0, 0, 1, 1), "test")
    jobs.job_finished.emit(Job("job1", JobKind.diffusion, job_params))

    workspace.show_output(text_messages[3])
    workspace.show_output(text_messages[2])
    jobs.job_finished.emit(Job("job2", JobKind.diffusion, job_params))
    assert workspace.outputs == {"1": text_messages[3], "3": text_messages[2]}

    assert output_events == [
        {"1": text_messages[0]},  # show_output(0)
        {"1": text_messages[0], "2": text_messages[1]},  # show_output(1)
        # job_finished(job1) - no changes
        {"1": text_messages[3], "2": text_messages[1]},  # show_output(3)
        {"1": text_messages[3], "2": text_messages[1], "3": text_messages[2]},  # show_output(2)
        {"1": text_messages[3], "3": text_messages[2]},  # job_finished(job2)
    ]


def test_expand():
    ext = ComfyWorkflow()
    in_img, width, height, seed = ext.add("ETN_KritaCanvas", 4)
    scaled = ext.add("ImageScale", 1, image=in_img, width=width, height=height)
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
    stylie = ext.add("ETN_KritaStyle", 9, name="style", sampler_preset="live")  # type: ignore
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
    params = {
        "inty": 7,
        "numby": 3.4,
        "texty": "cat",
        "booly": False,
        "choicy": "b",
        "layer_img": Image.create(Extent(4, 4), Qt.GlobalColor.black),
        "layer_mask": Image.create(Extent(4, 4), Qt.GlobalColor.white),
        "style": style,
    }

    input = CustomWorkflowInput(workflow=ext.root, params=params)
    images = ImageInput.from_extent(Extent(4, 4))
    images.initial_image = Image.create(Extent(4, 4), Qt.GlobalColor.white)

    models = ClientModels()
    models.checkpoints = {
        "checkpoint.safetensors": CheckpointInfo("checkpoint.safetensors", Arch.sd15)
    }

    w = ComfyWorkflow()
    w = workflow.expand_custom(w, input, images, 123, models)
    expected = [
        ComfyNode(1, "ETN_LoadImageBase64", {"image": images.initial_image.to_base64()}),
        ComfyNode(2, "ImageScale", {"image": Output(1, 0), "width": 4, "height": 4}),
        ComfyNode(3, "ETN_KritaOutput", {"images": Output(2, 0)}),
        ComfyNode(4, "ETN_LoadImageBase64", {"image": params["layer_img"].to_base64()}),
        ComfyNode(5, "ETN_LoadMaskBase64", {"mask": params["layer_mask"].to_base64()}),
        ComfyNode(6, "CheckpointLoaderSimple", {"ckpt_name": "checkpoint.safetensors"}),
        ComfyNode(
            7,
            "Sink",
            {
                "seed": 123,
                "inty": 7,
                "numby": 3.4,
                "texty": "cat",
                "booly": False,
                "choicy": "b",
                "layer_img": Output(4, 0),
                "layer_mask": Output(5, 0),
                "model": Output(6, 0),
                "clip": Output(6, 1),
                "vae": Output(6, 2),
                "positive": "bee hive",
                "negative": "pigoon",
                "sampler": "euler",
                "scheduler": "sgm_uniform",
                "steps": 6,
                "guidance": 1.8,
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
        Image.create(Extent(4, 4), Qt.GlobalColor.black),
        Image.create(Extent(4, 4), Qt.GlobalColor.white),
    ])
    params = {
        "image": in_images,
        "mask": in_masks,
    }

    input = CustomWorkflowInput(workflow=ext.root, params=params)
    images = ImageInput.from_extent(Extent(4, 4))
    models = ClientModels()

    w = ComfyWorkflow()
    w = workflow.expand_custom(w, input, images, 123, models)
    expected = [
        ComfyNode(1, "ETN_LoadImageBase64", {"image": in_images[0].to_base64()}),
        ComfyNode(2, "ETN_LoadImageBase64", {"image": in_images[1].to_base64()}),
        ComfyNode(3, "ImageBatch", {"image1": Output(1, 0), "image2": Output(2, 0)}),
        ComfyNode(4, "MaskToImage", {"mask": Output(1, 1)}),
        ComfyNode(5, "MaskToImage", {"mask": Output(2, 1)}),
        ComfyNode(6, "ImageBatch", {"image1": Output(4, 0), "image2": Output(5, 0)}),
        ComfyNode(7, "ImageToMask", {"image": Output(6, 0), "channel": "red"}),
        ComfyNode(8, "ETN_LoadMaskBase64", {"mask": in_masks[0].to_base64()}),
        ComfyNode(9, "ETN_LoadMaskBase64", {"mask": in_masks[1].to_base64()}),
        ComfyNode(10, "MaskToImage", {"mask": Output(8, 0)}),
        ComfyNode(11, "MaskToImage", {"mask": Output(9, 0)}),
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

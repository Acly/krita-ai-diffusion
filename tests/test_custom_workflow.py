import json
import pytest
from pathlib import Path
from PyQt5.QtCore import Qt

from ai_diffusion.api import CustomWorkflowInput, ImageInput, SamplingInput
from ai_diffusion.connection import Connection
from ai_diffusion.comfy_workflow import ComfyNode, ComfyWorkflow, Output
from ai_diffusion.custom_workflow import CustomWorkflow, WorkflowSource, WorkflowCollection
from ai_diffusion.custom_workflow import SortedWorkflows, CustomWorkspace
from ai_diffusion.custom_workflow import CustomParam, ParamKind, workflow_parameters
from ai_diffusion.image import Image, Extent
from ai_diffusion import workflow

from .config import test_dir


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

    connection = Connection()
    connection_graph = {"0": {"class_type": "C1", "inputs": {}}}
    connection_workflows = {"connection1": connection_graph}
    connection._workflows = connection_workflows

    collection = WorkflowCollection(connection, tmp_path)
    assert len(collection) == 3
    _assert_has_workflow(collection, "file1", WorkflowSource.local, file1_graph, file1)
    _assert_has_workflow(collection, "file2", WorkflowSource.local, file2_graph, file2)
    _assert_has_workflow(collection, "connection1", WorkflowSource.remote, connection_graph)

    events = []

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

    assert len(collection) == 4
    _assert_has_workflow(collection, "connection2", WorkflowSource.remote, connection2_graph)

    file1_graph_changed = {"0": {"class_type": "F3", "inputs": {}}}
    collection.set_graph(collection.index(0), file1_graph_changed)
    _assert_has_workflow(collection, "file1", WorkflowSource.local, file1_graph_changed, file1)
    assert events == [("begin_insert", 3), "end_insert", ("data_changed", 0)]

    collection.add_from_document("doc1", {"0": {"class_type": "D1", "inputs": {}}})

    sorted = SortedWorkflows(collection)
    assert sorted[0].source is WorkflowSource.document
    assert sorted[1].source is WorkflowSource.remote
    assert sorted[2].source is WorkflowSource.remote
    assert sorted[3].name == "file1"
    assert sorted[4].name == "file2"


def make_dummy_graph(n: int = 42):
    return {
        "1": {
            "class_type": "ETN_IntParameter",
            "inputs": {"name": "param1", "default": n, "min": 5, "max": 95},
        }
    }


def test_files(tmp_path: Path):
    collection_folder = tmp_path / "workflows"

    collection = WorkflowCollection(Connection(), collection_folder)
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


def test_workspace():
    connection = Connection()
    connection_workflows = {"connection1": make_dummy_graph(42)}
    connection._workflows = connection_workflows
    workflows = WorkflowCollection(connection)

    workspace = CustomWorkspace(workflows)
    assert workspace.workflow_id == "connection1"
    assert workspace.workflow and workspace.workflow.id == "connection1"
    assert workspace.graph and workspace.graph.node(0).type == "ETN_IntParameter"
    assert workspace.metadata[0].name == "param1"
    assert workspace.params == {"param1": 42}

    doc_graph = {
        "1": {
            "class_type": "ETN_IntParameter",
            "inputs": {"name": "param2", "default": 23, "min": 5, "max": 95},
        }
    }
    workspace.set_graph("doc1", doc_graph)
    assert workspace.workflow_id == "doc1"
    assert workspace.workflow and workspace.workflow.source is WorkflowSource.document
    assert workspace.graph and workspace.graph.node(0).type == "ETN_IntParameter"
    assert workspace.metadata[0].name == "param2"
    assert workspace.params == {"param2": 23}

    doc_graph["1"]["inputs"]["default"] = 24
    doc_graph["2"] = {
        "class_type": "ETN_IntParameter",
        "inputs": {"name": "param3", "default": 7, "min": 0, "max": 10},
    }
    workflows.set_graph(workflows.index(1), doc_graph)
    assert workspace.metadata[0].default == 24
    assert workspace.metadata[1].name == "param3"
    assert workspace.params == {"param2": 23, "param3": 7}


def test_import():
    graph = {
        "4": {"class_type": "A", "inputs": {"int": 4, "float": 1.2, "string": "mouse"}},
        "zak": {"class_type": "C", "inputs": {"in": ["9", 1]}},
        "9": {"class_type": "B", "inputs": {"in": ["4", 0]}},
    }
    w = ComfyWorkflow.import_graph(graph, {})
    assert w.node(0) == ComfyNode(0, "A", {"int": 4, "float": 1.2, "string": "mouse"})
    assert w.node(1) == ComfyNode(1, "B", {"in": Output(0, 0)})
    assert w.node(2) == ComfyNode(2, "C", {"in": Output(1, 1)})


def test_import_ui_workflow():
    graph = json.loads((test_dir / "data" / "workflow-ui.json").read_text())
    object_info = json.loads((test_dir / "data" / "object_info.json").read_text())
    node_inputs = {k: v.get("input") for k, v in object_info.items()}
    result = ComfyWorkflow.import_graph(graph, node_inputs)

    expected_graph = json.loads((test_dir / "data" / "workflow-api.json").read_text())
    expected = ComfyWorkflow.import_graph(expected_graph, {})
    assert result.root == expected.root


def test_parameters():
    w = ComfyWorkflow()
    w.add("ETN_IntParameter", 1, name="int", default=4, min=0, max=10)
    w.add("ETN_BoolParameter", 1, name="bool", default=True)
    w.add("ETN_NumberParameter", 1, name="number", default=1.2, min=0.0, max=10.0)
    w.add("ETN_TextParameter", 1, name="text", type="general", default="mouse")
    w.add("ETN_TextParameter", 1, name="positive", type="prompt (positive)", default="p")
    w.add("ETN_TextParameter", 1, name="negative", type="prompt (negative)", default="n")
    w.add("ETN_KritaImageLayer", 1, name="image")
    w.add("ETN_KritaMaskLayer", 1, name="mask")

    assert list(workflow_parameters(w)) == [
        CustomParam(ParamKind.number_int, "int", 4, 0, 10),
        CustomParam(ParamKind.boolean, "bool", True),
        CustomParam(ParamKind.number_float, "number", 1.2, 0.0, 10.0),
        CustomParam(ParamKind.text, "text", "mouse"),
        CustomParam(ParamKind.prompt_positive, "positive", "p"),
        CustomParam(ParamKind.prompt_negative, "negative", "n"),
        CustomParam(ParamKind.image_layer, "image"),
        CustomParam(ParamKind.mask_layer, "mask"),
    ]


def test_expand():
    ext = ComfyWorkflow()
    in_img, width, height, seed = ext.add("ETN_KritaCanvas", 4)
    scaled = ext.add("ImageScale", 1, image=in_img, width=width, height=height)
    ext.add("ETN_KritaOutput", 1, images=scaled)
    inty = ext.add("ETN_IntParameter", 1, name="inty", default=4, min=0, max=10)
    numby = ext.add("ETN_NumberParameter", 1, name="numby", default=1.2, min=0.0, max=10.0)
    texty = ext.add("ETN_TextParameter", 1, name="texty", type="general", default="mouse")
    booly = ext.add("ETN_BoolParameter", 1, name="booly", default=True)
    layer_img = ext.add("ETN_KritaImageLayer", 1, name="layer_img")
    layer_mask = ext.add("ETN_KritaMaskLayer", 1, name="layer_mask")
    ext.add(
        "Sink",
        1,
        seed=seed,
        inty=inty,
        numby=numby,
        texty=texty,
        booly=booly,
        layer_img=layer_img,
        layer_mask=layer_mask,
    )

    params = {
        "inty": 7,
        "numby": 3.4,
        "texty": "cat",
        "booly": False,
        "layer_img": Image.create(Extent(4, 4), Qt.GlobalColor.black),
        "layer_mask": Image.create(Extent(4, 4), Qt.GlobalColor.white),
    }

    input = CustomWorkflowInput(workflow=ext.root, params=params)
    images = ImageInput.from_extent(Extent(4, 4))
    images.initial_image = Image.create(Extent(4, 4), Qt.GlobalColor.white)
    sampling = SamplingInput("", "", 1.0, 1000, seed=123)

    w = ComfyWorkflow()
    w = workflow.expand_custom(w, input, images, sampling)
    expected = [
        ComfyNode(1, "ETN_LoadImageBase64", {"image": images.initial_image.to_base64()}),
        ComfyNode(2, "ImageScale", {"image": Output(1, 0), "width": 4, "height": 4}),
        ComfyNode(3, "ETN_KritaOutput", {"images": Output(2, 0)}),
        ComfyNode(4, "ETN_LoadImageBase64", {"image": params["layer_img"].to_base64()}),
        ComfyNode(5, "ETN_LoadMaskBase64", {"mask": params["layer_mask"].to_base64()}),
        ComfyNode(
            6,
            "Sink",
            {
                "seed": 123,
                "inty": 7,
                "numby": 3.4,
                "texty": "cat",
                "booly": False,
                "layer_img": Output(4, 0),
                "layer_mask": Output(5, 0),
            },
        ),
    ]
    for node in expected:
        assert node in w, f"Node {node} not found in\n{json.dumps(w.root, indent=2)}"

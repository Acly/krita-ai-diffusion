import asyncio
import json

from enum import Enum
from copy import copy
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, NamedTuple, Literal, TYPE_CHECKING
from pathlib import Path
from PyQt5.QtCore import Qt, QObject, QUuid, QAbstractListModel, QSortFilterProxyModel, QModelIndex
from PyQt5.QtCore import pyqtSignal

from .api import WorkflowInput
from .client import TextOutput, ClientOutput
from .comfy_workflow import ComfyWorkflow, ComfyNode
from .connection import Connection, ConnectionState
from .image import Bounds, Image
from .jobs import Job, JobParams, JobQueue, JobKind
from .properties import Property, ObservableProperties
from .style import Styles
from .util import base_type_match, user_data_dir, client_logger as log
from .ui import theme
from . import eventloop

if TYPE_CHECKING:
    from .layer import LayerManager


class WorkflowSource(Enum):
    document = 0
    remote = 1
    local = 2


@dataclass
class CustomWorkflow:
    id: str
    source: WorkflowSource
    workflow: ComfyWorkflow
    path: Path | None = None

    @property
    def name(self):
        return self.id.removesuffix(".json")


class WorkflowCollection(QAbstractListModel):

    _icon_local = theme.icon("file-json")
    _icon_remote = theme.icon("web-connection")
    _icon_document = theme.icon("file-kra")

    loaded = pyqtSignal()

    def __init__(self, connection: Connection, folder: Path | None = None):
        super().__init__()
        self._connection = connection
        self._folder = folder or user_data_dir / "workflows"
        self._workflows: list[CustomWorkflow] = []
        self._pending_workflows: list[tuple[str, WorkflowSource, dict]] = []

        self._connection.state_changed.connect(self._handle_connection)
        self._connection.workflow_published.connect(self._process_remote_workflow)
        self._handle_connection(self._connection.state)

    def _handle_connection(self, state: ConnectionState):
        if state in (ConnectionState.connected, ConnectionState.disconnected):
            self.clear()

        if state is ConnectionState.connected:
            for id, source, graph in self._pending_workflows:
                self._process_workflow(id, source, graph)
            self._pending_workflows.clear()

            for file in self._folder.glob("*.json"):
                try:
                    self._process_file(file)
                except Exception as e:
                    log.exception(f"Error loading workflow from {file}: {e}")

            for wf in self._connection.workflows.keys():
                self._process_remote_workflow(wf)

            self.loaded.emit()

    def _node_inputs(self):
        return self._connection.client.models.node_inputs

    def _process_workflow(
        self, id: str, source: WorkflowSource, graph: dict, path: Path | None = None
    ):
        if self._connection.state is not ConnectionState.connected:
            self._pending_workflows.append((id, source, graph))
            return

        comfy_flow = ComfyWorkflow.import_graph(graph, self._node_inputs())
        workflow = CustomWorkflow(id, source, comfy_flow, path)
        idx = self.find_index(workflow.id)
        if idx.isValid():
            self._workflows[idx.row()] = workflow
            self.dataChanged.emit(idx, idx)
        else:
            self.append(workflow)
        return idx

    def _process_remote_workflow(self, id: str):
        graph = self._connection.workflows[id]
        self._process_workflow(id, WorkflowSource.remote, graph)

    def _process_file(self, file: Path):
        with file.open("r") as f:
            graph = json.load(f)
            self._process_workflow(file.stem, WorkflowSource.local, graph, file)

    def rowCount(self, parent=QModelIndex()):
        return len(self._workflows)

    def data(self, index: QModelIndex, role: int = 0):
        if role == Qt.ItemDataRole.DisplayRole:
            return self._workflows[index.row()].name
        if role == Qt.ItemDataRole.UserRole:
            return self._workflows[index.row()].id
        if role == Qt.ItemDataRole.DecorationRole:
            source = self._workflows[index.row()].source
            if source is WorkflowSource.document:
                return self._icon_document
            if source is WorkflowSource.remote:
                return self._icon_remote
            return self._icon_local

    def append(self, item: CustomWorkflow):
        end = len(self._workflows)
        self.beginInsertRows(QModelIndex(), end, end)
        self._workflows.append(item)
        self.endInsertRows()

    def add_from_document(self, id: str, graph: dict):
        self._process_workflow(id, WorkflowSource.document, graph)

    def remove(self, id: str):
        idx = self.find_index(id)
        if idx.isValid():
            wf = self._workflows[idx.row()]
            if wf.source is WorkflowSource.local and wf.path is not None:
                wf.path.unlink()
            self.beginRemoveRows(QModelIndex(), idx.row(), idx.row())
            self._workflows.pop(idx.row())
            self.endRemoveRows()

    def clear(self):
        if len(self._workflows) > 0:
            self.beginResetModel()
            self._workflows.clear()
            self.endResetModel()

    def set_graph(self, index: QModelIndex, graph: dict):
        wf = self._workflows[index.row()]
        wf.workflow = ComfyWorkflow.import_graph(graph, self._node_inputs())
        self.dataChanged.emit(index, index)

    def save_as(self, id: str, graph: dict):
        if self.find(id) is not None:
            suffix = 1
            while self.find(f"{id} ({suffix})"):
                suffix += 1
            id = f"{id} ({suffix})"

        self._folder.mkdir(exist_ok=True)
        path = self._folder / f"{id}.json"
        path.write_text(json.dumps(graph, indent=2))
        self._process_workflow(id, WorkflowSource.local, graph, path)
        return id

    def import_file(self, filepath: Path):
        try:
            with filepath.open("r") as f:
                graph = json.load(f)
                try:
                    ComfyWorkflow.import_graph(graph, self._node_inputs())
                except Exception as e:
                    raise RuntimeError(f"This is not a supported workflow file ({e})")
            return self.save_as(filepath.stem, graph)
        except Exception as e:
            raise RuntimeError(f"Error importing workflow from {filepath}: {e}")

    def find_index(self, id: str):
        for i, wf in enumerate(self._workflows):
            if wf.id == id:
                return self.index(i)
        return QModelIndex()

    def find(self, id: str):
        idx = self.find_index(id)
        if idx.isValid():
            return self._workflows[idx.row()]
        return None

    def get(self, id: str):
        result = self.find(id)
        if result is None:
            raise KeyError(f"Workflow {id} not found")
        return result

    def __getitem__(self, index: int):
        return self._workflows[index]

    def __len__(self):
        return len(self._workflows)


class SortedWorkflows(QSortFilterProxyModel):
    def __init__(self, workflows: WorkflowCollection):
        super().__init__()
        self._workflows = workflows
        self.setSourceModel(workflows)
        self.setSortCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        self.sort(0)

    def lessThan(self, left: QModelIndex, right: QModelIndex):
        l = self._workflows[left.row()]
        r = self._workflows[right.row()]
        if l.source is r.source:
            return l.name < r.name
        return l.source.value < r.source.value

    def __getitem__(self, index: int):
        idx = self.mapToSource(self.index(index, 0)).row()
        return self._workflows[idx]


class ParamKind(Enum):
    image_layer = 0
    mask_layer = 1
    number_int = 2
    number_float = 3
    toggle = 4
    text = 5
    prompt_positive = 6
    prompt_negative = 7
    choice = 8
    style = 9


class CustomParam(NamedTuple):
    kind: ParamKind
    name: str
    default: Any | None = None
    min: int | float | None = None
    max: int | float | None = None
    choices: list[str] | None = None


def workflow_parameters(w: ComfyWorkflow):
    text_types = ("text", "prompt (positive)", "prompt (negative)")
    for node in w:
        match (node.type, node.input("type", "")):
            case ("ETN_KritaStyle", _):
                name = node.input("name", "Style")
                yield CustomParam(ParamKind.style, name, node.input("sampler_preset", "auto"))
            case ("ETN_KritaImageLayer", _):
                name = node.input("name", "Image")
                yield CustomParam(ParamKind.image_layer, name)
            case ("ETN_KritaMaskLayer", _):
                name = node.input("name", "Mask")
                yield CustomParam(ParamKind.mask_layer, name)
            case ("ETN_Parameter", "number (integer)"):
                name = node.input("name", "Parameter")
                default = node.input("default", 0)
                min = node.input("min", -(2**31))
                max = node.input("max", 2**31)
                yield CustomParam(ParamKind.number_int, name, default=default, min=min, max=max)
            case ("ETN_Parameter", "number"):
                name = node.input("name", "Parameter")
                default = node.input("default", 0.0)
                min = node.input("min", 0.0)
                max = node.input("max", 1.0)
                yield CustomParam(ParamKind.number_float, name, default=default, min=min, max=max)
            case ("ETN_Parameter", "toggle"):
                name = node.input("name", "Parameter")
                default = node.input("default", False)
                yield CustomParam(ParamKind.toggle, name, default=default)
            case ("ETN_Parameter", type) if type in text_types:
                name = node.input("name", "Parameter")
                default = node.input("default", "")
                match type:
                    case "text":
                        yield CustomParam(ParamKind.text, name, default=default)
                    case "prompt (positive)":
                        yield CustomParam(ParamKind.prompt_positive, name, default=default)
                    case "prompt (negative)":
                        yield CustomParam(ParamKind.prompt_negative, name, default=default)
            case ("ETN_Parameter", "choice"):
                name = node.input("name", "Parameter")
                default = node.input("default", "")
                if choices := _get_choices(w, node):
                    yield CustomParam(ParamKind.choice, name, choices=choices, default=default)
                else:
                    yield CustomParam(ParamKind.text, name, default=default)
            case ("ETN_Parameter", unknown_type) if unknown_type != "auto":
                unknown = node.input("name", "?") + ": " + unknown_type
                log.warning(f"Custom workflow has an unsupported parameter type {unknown}")


def _get_choices(w: ComfyWorkflow, node: ComfyNode):
    connected, input_name = next(w.find_connected(node.output()), (None, ""))
    if connected:
        if input_type := w.input_type(connected.type, input_name):
            if isinstance(input_type[0], list):
                return input_type[0]
    return None


ImageGenerator = Callable[[WorkflowInput | None], Awaitable[None | Literal[False] | WorkflowInput]]


class CustomGenerationMode(Enum):
    regular = 0
    live = 1


class CustomWorkspace(QObject, ObservableProperties):

    workflow_id = Property("", setter="_set_workflow_id")
    params = Property({}, persist=True)
    mode = Property(CustomGenerationMode.regular, setter="_set_mode")
    is_live = Property(False, setter="toggle_live")
    has_result = Property(False)
    outputs = Property({})

    workflow_id_changed = pyqtSignal(str)
    graph_changed = pyqtSignal()
    params_changed = pyqtSignal(dict)
    mode_changed = pyqtSignal(CustomGenerationMode)
    is_live_changed = pyqtSignal(bool)
    result_available = pyqtSignal(Image)
    has_result_changed = pyqtSignal(bool)
    outputs_changed = pyqtSignal(dict)
    modified = pyqtSignal(QObject, str)

    _live_poll_rate = 0.1

    def __init__(self, workflows: WorkflowCollection, generator: ImageGenerator, jobs: JobQueue):
        super().__init__()
        self._workflows = workflows
        self._generator = generator
        self._workflow: CustomWorkflow | None = None
        self._graph: ComfyWorkflow | None = None
        self._metadata: list[CustomParam] = []
        self._last_input: WorkflowInput | None = None
        self._last_result: Image | None = None
        self._last_job: JobParams | None = None
        self._new_outputs: list[str] = []

        jobs.job_finished.connect(self._handle_job_finished)
        workflows.dataChanged.connect(self._update_workflow)
        workflows.loaded.connect(self._set_default_workflow)
        self._set_default_workflow()

    def _set_default_workflow(self):
        if not self.workflow_id and len(self._workflows) > 0:
            self.workflow_id = self._workflows[0].id
        else:
            current_index = self._workflows.find_index(self.workflow_id)
            if current_index.isValid():
                self._update_workflow(current_index, QModelIndex())

    def _update_workflow(self, idx: QModelIndex, _: QModelIndex):
        wf = self._workflows[idx.row()]
        if wf.id == self._workflow_id:
            self._workflow = wf
            self._graph = self._workflow.workflow
            self._metadata = list(workflow_parameters(self._graph))
            self.params = _coerce(self.params, self._metadata)
            self.graph_changed.emit()

    def _set_workflow_id(self, id: str):
        if self._workflow_id == id:
            return
        self._workflow_id = id
        self.workflow_id_changed.emit(id)
        self.modified.emit(self, "workflow_id")
        index = self._workflows.find_index(id)
        if index.isValid():  # might be invalid when loading document before connecting
            self._update_workflow(index, QModelIndex())

    def set_graph(self, id: str, graph: dict):
        if self._workflows.find(id) is None:
            id = "Document Workflow (embedded)"
            self._workflows.add_from_document(id, graph)
        self.workflow_id = id

    def import_file(self, filepath: Path):
        self.workflow_id = self._workflows.import_file(filepath)

    def save_as(self, id: str):
        assert self._graph, "Save as: no workflow selected"
        self.workflow_id = self._workflows.save_as(id, self._graph.root)

    def remove_workflow(self):
        if id := self.workflow_id:
            self._workflow_id = ""
            self._workflow = None
            self._graph = None
            self._metadata = []
            self._workflows.remove(id)

    def generate(self):
        eventloop.run(self._generator(None))

    def toggle_live(self, active: bool):
        if self._is_live != active:
            self._is_live = active
            self.is_live_changed.emit(active)
            if active:
                eventloop.run(self._continue_generating())

    def _set_mode(self, value: CustomGenerationMode):
        if self._mode != value:
            self._mode = value
            self.mode_changed.emit(value)
            self.is_live = False

    @property
    def workflow(self):
        return self._workflow

    @property
    def graph(self):
        return self._graph

    @property
    def metadata(self):
        return self._metadata

    @property
    def job_name(self):
        for param in self.metadata:
            if param.kind is ParamKind.prompt_positive:
                return str(self.params[param.name])
        return self.workflow_id or "Custom Workflow"

    def collect_parameters(self, layers: "LayerManager", bounds: Bounds):
        params = copy(self.params)
        for md in self.metadata:
            param = params.get(md.name)

            if md.kind is ParamKind.image_layer:
                if param is None and len(layers.images) > 0:
                    param = layers.images[0].id
                layer = layers.find(QUuid(param))
                if layer is None:
                    raise ValueError(f"Input layer for parameter {md.name} not found")
                params[md.name] = layer.get_pixels(bounds)
            elif md.kind is ParamKind.mask_layer:
                if param is None and len(layers.masks) > 0:
                    param = layers.masks[0].id
                layer = layers.find(QUuid(param))
                if layer is None:
                    raise ValueError(f"Input layer for parameter {md.name} not found")
                params[md.name] = layer.get_mask(bounds)
            elif md.kind is ParamKind.style:
                style = Styles.list().find(str(param))
                if style is None:
                    raise ValueError(f"Style {param} not found")
                params[md.name] = style
            elif param is None:
                raise ValueError(f"Parameter {md.name} not found")

        return params

    def show_output(self, output: ClientOutput | None):
        if isinstance(output, TextOutput):
            self._new_outputs.append(output.key)
            self.outputs[output.key] = output
            self.outputs_changed.emit(self.outputs)

    def _handle_job_finished(self, job: Job):
        to_remove = [k for k in self.outputs.keys() if k not in self._new_outputs]
        for key in to_remove:
            del self.outputs[key]
        if len(to_remove) > 0:
            self.outputs_changed.emit(self.outputs)
        self._new_outputs.clear()

        if job.kind is JobKind.live_preview:
            if len(job.results) > 0:
                self._last_result = job.results[0]
                self._last_job = job.params
                self.result_available.emit(self._last_result)
                self.has_result = True
            eventloop.run(self._continue_generating())

    async def _continue_generating(self):
        while self.is_live:
            new_input = await self._generator(self._last_input)
            if new_input is False:  # abort live generation
                self.is_live = False
                return
            elif new_input is None:  # no changes in input data
                await asyncio.sleep(self._live_poll_rate)
            else:  # frame was scheduled
                self._last_input = new_input
                return

    @property
    def live_result(self):
        assert self._last_result and self._last_job, "No live result available"
        return self._last_result, self._last_job


def _coerce(params: dict[str, Any], types: list[CustomParam]):
    def use(value, default):
        if default is None:
            return value
        if value is None or not base_type_match(value, default):
            return default
        return value

    return {t.name: use(params.get(t.name), t.default) for t in types}

from enum import Enum
from typing import Any, NamedTuple
from PyQt5.QtCore import pyqtSignal, QObject

from .comfy_workflow import ComfyWorkflow
from .connection import Connection
from .properties import Property, ObservableProperties


class ParamKind(Enum):
    image_layer = 0
    mask_layer = 1
    number_int = 2


class CustomParam(NamedTuple):
    kind: ParamKind
    name: str
    default: Any | None = None
    min: int | None = None
    max: int | None = None


def _gather_params(w: ComfyWorkflow):
    for node in w:
        match node.type:
            case "ETN_KritaImageLayer":
                name = node.input("name", "Image")
                yield CustomParam(ParamKind.image_layer, name)
            case "ETN_KritaMaskLayer":
                name = node.input("name", "Mask")
                yield CustomParam(ParamKind.mask_layer, name)
            case "ETN_IntParameter":
                name = node.input("name", "Parameter")
                default = node.input("default", 0)
                min = node.input("min", -(2**31))
                max = node.input("max", 2**31)
                yield CustomParam(ParamKind.number_int, name, default=default, min=min, max=max)


class CustomWorkspace(QObject, ObservableProperties):

    graph_id = Property("", persist=True)
    graph = Property({}, persist=True, setter="_set_graph")
    params = Property({}, persist=True)

    graph_id_changed = pyqtSignal(str)
    graph_changed = pyqtSignal(dict)
    params_changed = pyqtSignal(dict)
    modified = pyqtSignal(QObject, str)

    def __init__(self, connection: Connection):
        super().__init__()
        self._workflow: ComfyWorkflow | None = None
        self._metadata: list[CustomParam] = []
        self._connection = connection
        self._connection.workflow_published.connect(self._update_workflow)

        if len(connection.workflows) > 0:
            self._update_workflow(next(iter(connection.workflows.keys())))

    def _update_workflow(self, id: str):
        wf = self._connection.workflows[id]
        if not self.graph_id:
            self.graph_id = id
        if self.graph_id == id:
            self.graph = wf

    def _set_graph(self, graph: dict):
        self._workflow = ComfyWorkflow.import_graph(graph)
        self._metadata = list(_gather_params(self._workflow))
        self.params = _coerce(self.params, self._metadata)
        self._graph = self._workflow.root
        self.graph_changed.emit(self._graph)

    @property
    def workflow(self):
        return self._workflow

    @property
    def metadata(self):
        return self._metadata


def _coerce(params: dict[str, Any], types: list[CustomParam]):
    def use(value, default):
        if value is None or not type(value) == type(default):
            return default
        return value

    return {t.name: use(params.get(t.name), t.default) for t in types}

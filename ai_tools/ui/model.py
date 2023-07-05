import asyncio
import sys
import traceback
from enum import Flag
from typing import Optional, Callable
from PyQt5.QtCore import QObject, pyqtSignal
from ai_tools import (
    eventloop,
    Document,
    Image,
    Mask,
    Extent,
    Bounds,
    ImageCollection,
    workflow,
    Progress,
    Interrupted,
    NetworkError,
)
from .server import DiffusionServer, ServerState
import krita


class State(Flag):
    setup = 0
    generating = 1
    preview = 2


class Model(QObject):
    """ViewModel for diffusion workflows on a Krita document. Stores all inputs related to
    image generation. Goes through the following states:
    - setup: gather inputs for image generation
    - setup|generating: image generation in progress (allows cancellation)
    - preview: image generation complete, previewing results with options to apply or discard
    - preview|generating: generating additional results on the same inputs
    """

    _doc: Document
    _layer: Optional[krita.Node] = None
    _image: Optional[Image] = None
    _mask: Optional[Mask] = None
    _extent: Optional[Extent] = None
    _bounds: Optional[Bounds] = None

    changed = pyqtSignal()
    progress_changed = pyqtSignal()

    state = State.setup
    prompt = ""
    strength = 1.0
    progress = 0.0
    results: ImageCollection
    error = ""
    task: Optional[asyncio.Task] = None

    def __init__(self, document: Document):
        super().__init__()
        self._doc = document
        self.results = ImageCollection()

    @staticmethod
    def active():
        """Return the model for the currently active document."""
        return ModelRegistry.instance().model_for_active_document()

    def setup(self):
        """Retrieve the current image and selection mask as inputs for the next generation(s)."""
        self._mask = self._doc.create_mask_from_selection()
        if self._mask is not None or self.strength < 1.0:
            self._image = self._doc.get_image()
            self._bounds = self._mask.bounds if self._mask else Bounds(0, 0, *self._image.extent)
        else:
            self._extent = self._doc.extent
            self._bounds = Bounds(0, 0, *self._extent)

    async def _generate(self):
        try:
            assert State.generating not in self.state
            assert DiffusionServer.instance().state is ServerState.connected

            diffusion = DiffusionServer.instance().diffusion
            image, mask = self._image, self._mask
            progress = Progress(self.report_progress)
            self.state = self.state | State.generating
            self.progress = 0.0
            self.changed.emit()

            if image is None and mask is None:
                assert self._extent is not None and self.strength == 1
                generator = workflow.generate(diffusion, self._extent, self.prompt, progress)
            elif mask is None and self.strength < 1:
                assert image is not None
                generator = workflow.refine(diffusion, image, self.prompt, self.strength, progress)
            elif self.strength == 1:
                assert image is not None and mask is not None
                generator = workflow.inpaint(diffusion, image, mask, self.prompt, progress)
            else:
                assert image is not None and mask is not None and self.strength < 1
                generator = workflow.refine_region(
                    diffusion, image, mask, self.prompt, self.strength, progress
                )
            async for result in generator:
                self.state = State.preview | State.generating
                self.results.append(result)
                if self._layer is None:
                    self._layer = self._doc.insert_layer(
                        f"[Preview] {self.prompt}", result, self._bounds
                    )
                self.changed.emit()
            self.state = State.preview
            self.changed.emit()
        except Interrupted:
            self.reset()
        except asyncio.CancelledError:
            pass  # reset called by cancel()
        except NetworkError as e:
            self.report_error(e.message, f"[url={e.url}, code={e.code}]")
        except AssertionError as e:
            _, _, tb = sys.exc_info()
            traceback.print_tb(tb)
            self.report_error("Error: Internal assertion failed.")
        except Exception as e:
            self.report_error(str(e))
        finally:
            self.changed.emit()

    def generate(self):
        self.task = eventloop.run(self._generate())

    def cancel(self):
        assert State.generating in self.state and self.task is not None
        DiffusionServer.instance().interrupt()
        self.task.cancel()
        self.state = self.state & (~State.generating)
        self.reset()

    def report_progress(self, value):
        self.progress = value
        self.progress_changed.emit()

    def report_error(self, message: str, details: Optional[str] = None):
        print("[krita-ai-tools]", message, details)
        self.state = State.setup
        self.error = message
        self.changed.emit()

    def show_preview(self, index: int):
        self._doc.set_layer_pixels(self._layer, self.results[index], self._bounds)

    def apply_current_result(self):
        """Apply selected result by duplicating the preview layer and inserting it below.
        This allows to apply multiple results (eg. to combine them afterwards by erasing parts).
        """
        new_layer = self._layer
        self._layer = self._layer.duplicate()
        parent = new_layer.parentNode()
        parent.addChildNode(self._layer, new_layer)
        new_layer.setLocked(False)
        new_layer.setName(new_layer.name().replace("[Preview]", "[Generated]"))

    def reset(self):
        """Discard all results, cancel any running generation, and go back to the setup stage.
        Setup configuration like prompt and strength is not reset.
        """
        if self.state is (State.preview | State.generating) and not self.task.cancelled():
            self.cancel()
        if self._layer:
            self._layer.setLocked(False)
            self._layer.remove()
            self._layer = None
        self._image = None
        self._mask = None
        self._extent = None
        self.results = ImageCollection()
        self.state = State.setup
        self.progress = 0
        self.error = ""
        self.changed.emit()

    @property
    def is_active(self):
        return self._doc.is_active

    @property
    def is_valid(self):
        return self._doc.is_valid


class ModelRegistry(QObject):
    """Singleton that keeps track of all models (one per open image document) and notifies
    widgets when new ones are created."""

    _instance = None
    _models = []

    created = pyqtSignal(Model)

    def __init__(self):
        super().__init__()

    @classmethod
    def instance(cls):
        if cls._instance is None:
            cls._instance = ModelRegistry()
        return cls._instance

    def model_for_active_document(self):
        # Remove models for documents that have been closed
        self._models = [m for m in self._models if m.is_valid]

        # Find or create model for active document
        if Document.active() is not None:
            model = next((m for m in self._models if m.is_active), None)
            if model is None:
                model = Model(Document.active())
                self._models.append(model)
                self.created.emit(model)
            return model

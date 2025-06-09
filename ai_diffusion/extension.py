import sys
from pathlib import Path
from typing import Callable
from PyQt5.QtWidgets import QAction
from krita import Extension, Krita, DockWidgetFactory, DockWidgetFactoryBase, Window  # type: ignore

from . import eventloop, __version__
from .settings import settings
from .model import Workspace
from .ui import actions
from .ui.diffusion import ImageDiffusionWidget
from .ui.settings import SettingsDialog
from .root import root
from .util import client_logger as log


class AIToolsExtension(Extension):
    _actions: dict[str, QAction] = {}
    _settings_dialog: SettingsDialog

    def __init__(self, parent):
        super().__init__(parent)
        log.info(f"Extension initialized, Version: {__version__}, Python: {sys.version}")

        debugpy_path = Path(__file__).parent / "debugpy" / "src"
        if debugpy_path.exists():
            try:
                sys.path.insert(0, str(debugpy_path))
                import debugpy

                debugpy.listen(("127.0.0.1", 5678), in_process_debug_adapter=True)
                log.info("Developer mode: debugpy listening on port 5678")
            except ImportError:
                pass

        eventloop.setup()
        settings.load()
        root.init()
        self._settings_dialog = SettingsDialog(root.server)

        notifier = Krita.instance().notifier()
        notifier.setActive(True)
        notifier.applicationClosing.connect(self.shutdown)  # type: ignore

    def setup(self):
        eventloop.run(root.autostart(self._settings_dialog.connection.update_ui))

    def shutdown(self):
        root.server.terminate()
        eventloop.stop()

    def _create_action(self, window: Window, name: str, func: Callable[[], None]):
        action = window.createAction(f"ai_diffusion_{name}", "", "")
        action.triggered.connect(func)
        self._actions[name] = action

    def createActions(self, window):
        self._create_action(window, "settings", self._settings_dialog.show)
        self._create_action(window, "generate", actions.generate)
        self._create_action(window, "cancel", actions.cancel_active)
        self._create_action(window, "cancel_queued", actions.cancel_queued)
        self._create_action(window, "cancel_all", actions.cancel_all)
        self._create_action(window, "toggle_preview", actions.toggle_preview)
        self._create_action(window, "apply", actions.apply)
        self._create_action(window, "apply_alternative", actions.apply_alternative)
        self._create_action(window, "create_region", actions.create_region)
        self._create_action(
            window, "switch_workspace_generation", actions.set_workspace(Workspace.generation)
        )
        self._create_action(
            window, "switch_workspace_upscaling", actions.set_workspace(Workspace.upscaling)
        )
        self._create_action(window, "switch_workspace_live", actions.set_workspace(Workspace.live))
        self._create_action(
            window, "switch_workspace_graph", actions.set_workspace(Workspace.custom)
        )
        self._create_action(window, "toggle_workspace", actions.toggle_workspace)


Krita.instance().addExtension(AIToolsExtension(Krita.instance()))
Krita.instance().addDockWidgetFactory(
    DockWidgetFactory("imageDiffusion", DockWidgetFactoryBase.DockRight, ImageDiffusionWidget)  # type: ignore
)

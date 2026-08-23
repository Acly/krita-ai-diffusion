"""uitest.py: Automation helpers for design.py – wait for state, simulate input, take screenshots.

Used by scripts/design.py to support scripted UI testing and iteration. Can be driven
entirely from the command line:

    python scripts/design.py --wait connected --screenshot shot.png --exit

or extended with a custom Python script passed via --script:

    # my_test.py
    async def run(auto):
        await auto.wait_connected()
        auto.click("Open document")
        await auto.sleep(0.5)
        auto.type_text("TextPromptWidget", "a cute robot")
        auto.screenshot("shot.png")
"""

from __future__ import annotations

import asyncio
import importlib.util
import traceback
from collections.abc import Callable
from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QAbstractButton, QApplication, QWidget

from ai_diffusion import eventloop
from ai_diffusion.model.connection import ConnectionState
from ai_diffusion.model.root import root

WidgetQuery = str | QWidget | Callable[[QWidget], bool]


class Automation:
    """Context object passed to test scripts. Wraps the design preview window and
    provides primitives to wait for application state, interact with widgets, and
    render widgets to image files."""

    def __init__(self, dock: QWidget, window: QWidget):
        self.dock = dock  # the ImageDiffusionWidget
        self.window = window  # the top-level container (control pane + dock)
        self.root = root

    # -- waiting ------------------------------------------------------

    async def wait_for(self, condition: Callable[[], bool], timeout: float = 30.0, message=""):
        elapsed = 0.0
        while not condition():
            if elapsed >= timeout:
                raise TimeoutError(message or f"Condition not met within {timeout:.1f}s")
            await asyncio.sleep(0.05)
            elapsed += 0.05

    async def wait_state(self, state: ConnectionState, timeout: float = 30.0):
        await self.wait_for(
            lambda: root.connection.state is state,
            timeout,
            f"Connection did not reach state '{state.name}' within {timeout:.1f}s",
        )

    async def wait_connected(self, timeout: float = 30.0):
        await self.wait_state(ConnectionState.connected, timeout)

    async def wait_visible(self, query: WidgetQuery, timeout: float = 30.0):
        def is_visible():
            try:
                self.find(query)
                return True
            except LookupError:
                return False

        await self.wait_for(is_visible, timeout, f"Widget {query!r} not visible")

    async def sleep(self, seconds: float):
        await asyncio.sleep(seconds)

    # -- widget lookup --------------------------------------------------

    def find(
        self, query: WidgetQuery, parent: QWidget | None = None, visible: bool = True
    ) -> QWidget:
        """Find a single widget by objectName, class name, button text, or predicate.

        Only visible widgets are matched by default, since several workspaces may
        contain hidden instances of the same widget type. Pass ``visible=False``
        to include hidden widgets."""
        if isinstance(query, QWidget):
            return query
        matches = self.find_all(query, parent, visible)
        if not matches:
            raise LookupError(f"No widget matching {query!r}")
        return matches[0]

    def find_all(
        self, query: WidgetQuery, parent: QWidget | None = None, visible: bool = True
    ) -> list[QWidget]:
        parent = parent or self.window
        matcher = query if callable(query) else _matcher(str(query))
        return [
            w for w in parent.findChildren(QWidget) if matcher(w) and (not visible or w.isVisible())
        ]

    # -- actions --------------------------------------------------------

    # QTest methods are static, but the PyQt6 type stubs declare them as instance
    # methods, hence the type: ignore comments.

    def click(self, query: WidgetQuery):
        widget = self.find(query)
        QTest.mouseClick(widget, Qt.MouseButton.LeftButton)  # type: ignore

    def type_text(self, query: WidgetQuery, text: str, clear=True):
        """Simulate typing into a widget (QLineEdit, QPlainTextEdit, ...) via key events."""
        widget = self.find(query)
        widget.setFocus(Qt.FocusReason.ShortcutFocusReason)
        if clear:
            _clear(widget)
        QTest.keyClicks(widget, text)  # type: ignore

    def set_text(self, query: WidgetQuery, text: str):
        """Set text directly without simulating key events (faster, no key handling)."""
        widget = self.find(query)
        for method in ("setText", "setPlainText", "setCurrentText"):
            if hasattr(widget, method):
                getattr(widget, method)(text)
                return
        raise TypeError(f"Widget {widget} does not accept text input")

    # -- screenshots ------------------------------------------------------

    def screenshot(self, path: str | Path, target: WidgetQuery | None = None):
        """Render a widget (the ImageDiffusionWidget by default) to an image file."""
        widget = self.dock if target is None else self.find(target)
        path = Path(path)
        if path.parent != Path(""):
            path.parent.mkdir(parents=True, exist_ok=True)
        if not widget.grab().save(str(path)):
            raise RuntimeError(f"Failed to save screenshot to {path}")
        print(f"[uitest] screenshot saved to {path}")


def _matcher(query: str) -> Callable[[QWidget], bool]:
    def match(w: QWidget) -> bool:
        return (
            w.objectName() == query
            or type(w).__name__ == query
            or (isinstance(w, QAbstractButton) and w.text() == query)
        )

    return match


def _clear(widget: QWidget):
    if hasattr(widget, "clear"):
        widget.clear()  # type: ignore
    elif hasattr(widget, "selectAll"):
        widget.selectAll()  # type: ignore
        QTest.keyClick(widget, Qt.Key.Key_Delete)  # type: ignore


async def _run_script(auto: Automation, path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load script {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    run_fn = getattr(module, "run", None)
    if run_fn is None:
        raise RuntimeError(f"Script {path} must define a 'run(auto)' function")
    result = run_fn(auto)
    if asyncio.iscoroutine(result):
        await result


async def _run_steps(auto: Automation, args):
    for state_name in args.wait or []:
        print(f"[uitest] waiting for connection state '{state_name}'...")
        await auto.wait_state(ConnectionState[state_name], args.timeout)
    for script in args.script or []:
        print(f"[uitest] running script {script}...")
        await _run_script(auto, script)
    if args.screenshot:
        auto.screenshot(args.screenshot)


def has_steps(args) -> bool:
    return bool(args.wait or args.script or args.screenshot)


def start(app: QApplication, dock: QWidget, window: QWidget, args):
    """Schedule automation steps to run alongside the Qt event loop. If --exit was
    passed, the application quits once all steps are done (exit code 1 on failure)."""
    auto = Automation(dock, window)

    async def _run():
        exit_code = 0
        try:
            await _run_steps(auto, args)
            print("[uitest] all steps completed")
        except Exception:
            traceback.print_exc()
            exit_code = 1
        if args.exit:
            app.exit(exit_code)

    eventloop.run(_run())

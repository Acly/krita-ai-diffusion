"""Real Qt UI cases, run in a subprocess to isolate QApplication from QCoreApplication."""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import patch

from PyQt6.QtCore import QObject, Qt, pyqtSignal
from PyQt6.QtTest import QSignalSpy, QTest
from PyQt6.QtWidgets import QApplication, QLineEdit, QVBoxLayout, QWidget

root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))
app = QApplication.instance() or QApplication([])
# SIP stubs expose the QTest namespace helpers as instance methods.
qtest = cast(Any, QTest)

from ai_diffusion.backend.resources import Arch
from ai_diffusion.model.connection import ConnectionState
from ai_diffusion.style import Style
from ai_diffusion.ui import widget as widgets
from ai_diffusion.ui.search_combo import SearchableComboBox
from ai_diffusion.util import ensure


class SearchComboTests(unittest.TestCase):
    def setUp(self):
        self.window = QWidget()
        layout = QVBoxLayout(self.window)
        self.combo = SearchableComboBox(self.window)
        for label, key in [
            ("DreamPaw ★", "dream.json"),
            ("Nova Orange", "nova.json"),
            ("Orange Dream", "orange.json"),
            ("Nova Orange", "nova2.json"),
        ]:
            self.combo.addItem(label, key)
        layout.addWidget(self.combo)
        self.other = QLineEdit(self.window)
        layout.addWidget(self.other)
        self.window.show()
        app.processEvents()
        self.assertTrue(self.combo.isEditable(), "Style selector must accept search text")
        self.spy = QSignalSpy(self.combo.selection_committed)
        self.combo._editor.setFocus()
        app.processEvents()

    def tearDown(self):
        self.window.close()
        self.window.deleteLater()
        app.processEvents()

    def search(self, text):
        self.combo._editor.selectAll()
        qtest.keyClicks(self.combo._editor, text)
        app.processEvents()

    def test_contains_search_ignores_case_without_committing(self):
        self.search("oRAnGe")
        self.assertEqual(self.combo._completion.completionCount(), 3)
        self.assertEqual(len(self.spy), 0)
        self.assertEqual(self.combo.currentData(), "dream.json")

    def test_enter_commits_matching_style(self):
        self.search("nova")
        qtest.keyClick(self.combo._editor, Qt.Key.Key_Return)
        app.processEvents()
        self.assertEqual(self.combo.currentData(), "nova.json")
        self.assertEqual(len(self.spy), 1)
        self.assertEqual(self.combo.currentText(), "Nova Orange")

    def test_arrow_navigation_selects_second_duplicate(self):
        self.search("nova")
        popup = self.combo._popup
        qtest.keyClick(self.combo._editor, Qt.Key.Key_Down)
        qtest.keyClick(popup, Qt.Key.Key_Down)
        qtest.keyClick(popup, Qt.Key.Key_Return)
        app.processEvents()
        self.assertEqual(self.combo.currentData(), "nova2.json")
        self.assertEqual(len(self.spy), 1)

    def test_keyboard_events_on_focused_combo(self):
        self.combo.setFocus()
        app.processEvents()
        qtest.keyClicks(self.combo, "nova")
        app.processEvents()
        self.assertEqual(self.combo._completion.completionCount(), 2)
        qtest.keyClick(self.combo, Qt.Key.Key_Down)
        qtest.keyClick(self.combo, Qt.Key.Key_Down)
        qtest.keyClick(self.combo, Qt.Key.Key_Return)
        app.processEvents()
        self.assertEqual(self.combo.currentData(), "nova2.json")
        self.assertEqual(len(self.spy), 1)

    def test_escape_restores_starred_selected_style(self):
        self.search("orange")
        qtest.keyClick(self.combo._popup, Qt.Key.Key_Escape)
        app.processEvents()
        self.assertEqual(self.combo._editor.text(), "DreamPaw ★")
        self.assertEqual(self.combo.currentData(), "dream.json")
        self.assertEqual(len(self.spy), 0)

    def test_unmatched_enter_creates_nothing(self):
        self.search("nonexistent")
        qtest.keyClick(self.combo._editor, Qt.Key.Key_Return)
        app.processEvents()
        self.assertEqual(self.combo.count(), 4)
        self.assertEqual(self.combo._editor.text(), "DreamPaw ★")
        self.assertEqual(len(self.spy), 0)

    def test_focus_loss_cancels_search(self):
        self.search("nova")
        self.other.setFocus()
        app.processEvents()
        self.assertEqual(self.combo._editor.text(), "DreamPaw ★")
        self.assertEqual(len(self.spy), 0)

    def test_mouse_completion_keeps_duplicate_filename_identity(self):
        self.search("nova")
        popup = self.combo._popup
        index = ensure(popup.model()).index(1, 0)
        qtest.mouseClick(
            popup.viewport(), Qt.MouseButton.LeftButton, pos=popup.visualRect(index).center()
        )
        app.processEvents()
        self.assertEqual(self.combo.currentData(), "nova2.json")
        self.assertEqual(len(self.spy), 1)

    def test_normal_dropdown_still_selects(self):
        self.combo.showPopup()
        view = ensure(self.combo.view())
        index = ensure(view.model()).index(2, 0)
        qtest.mouseClick(
            view.viewport(), Qt.MouseButton.LeftButton, pos=view.visualRect(index).center()
        )
        app.processEvents()
        self.assertEqual(self.combo.currentData(), "orange.json")
        self.assertEqual(len(self.spy), 1)


class Registry(QObject):
    items: list[Style]
    default: Style
    changed = pyqtSignal()
    name_changed = pyqtSignal()

    def filtered(self):
        return self.items


class Settings(QObject):
    recent_styles: list[str]
    recent_styles_count: int
    changed = pyqtSignal(str, object)


class Connection(QObject):
    state: ConnectionState
    client_if_connected: None
    state_changed = pyqtSignal()


class SelectorIntegrationTests(unittest.TestCase):
    def setUp(self):
        registry = Registry()
        registry.items = []
        for i, name in enumerate(
            ["DreamPaw v10", "DreamPaw v20", "Nova Orange"] + [f"Style {i}" for i in range(1790)]
        ):
            style = Style(Path(f"style-{i}.json"))
            style.name = name
            registry.items.append(style)
        registry.default = registry.items[1]
        self.registry = registry
        settings = Settings()
        settings.recent_styles = [s.filename for s in registry.items[:2]]
        settings.recent_styles_count = 5
        connection = Connection()
        connection.state = ConnectionState.connected
        connection.client_if_connected = None
        self.services = (settings, connection)
        patches = [
            patch.object(widgets.Styles, "list", return_value=registry),
            patch.object(widgets, "settings", settings),
            patch.object(widgets, "root", SimpleNamespace(connection=connection)),
            patch.object(
                widgets, "filter_supported_styles", side_effect=lambda styles, client: styles
            ),
            patch.object(widgets, "resolve_arch", return_value=Arch.sdxl),
        ]
        for patcher in patches:
            patcher.start()
            self.addCleanup(patcher.stop)
        self.widget = widgets.StyleSelectWidget(None)
        self.widget.show()
        app.processEvents()

    def tearDown(self):
        self.widget.close()
        self.widget.deleteLater()
        app.processEvents()

    def test_recent_selection_restored_by_filename(self):
        self.assertEqual(self.widget._combo.currentData(), self.registry.default.filename)
        self.widget.update_styles()
        self.assertEqual(self.widget._combo.currentData(), self.widget.value.filename)

    def test_full_library_search_selects_model_once(self):
        combo = self.widget._combo
        self.assertTrue(combo.isEditable())
        spy = QSignalSpy(self.widget.value_changed)
        combo._editor.setFocus()
        combo._editor.selectAll()
        qtest.keyClicks(combo._editor, "dreampaw")
        app.processEvents()
        self.assertGreater(combo._completion.completionCount(), 0)
        self.assertEqual(len(spy), 0)
        selected = (
            ensure(combo._completion.completionModel()).index(0, 0).data(Qt.ItemDataRole.UserRole)
        )
        qtest.keyClick(combo._editor, Qt.Key.Key_Return)
        app.processEvents()
        self.assertEqual(self.widget.value.filename, selected)
        self.assertEqual(len(spy), 1)

    def test_mouse_selection_survives_immediate_recent_style_refresh(self):
        settings, _connection = self.services

        def remember(style):
            settings.recent_styles.insert(0, style.filename)
            settings.changed.emit("recent_styles", settings.recent_styles)

        self.widget.value_changed.connect(remember)
        combo = self.widget._combo
        combo._editor.setFocus()
        combo._editor.selectAll()
        qtest.keyClicks(combo._editor, "dreampaw")
        app.processEvents()
        popup = combo._popup
        index = ensure(popup.model()).index(0, 0)
        selected = index.data(Qt.ItemDataRole.UserRole)
        qtest.mouseClick(
            popup.viewport(), Qt.MouseButton.LeftButton, pos=popup.visualRect(index).center()
        )
        app.processEvents()
        self.assertEqual(self.widget.value.filename, selected)
        self.assertEqual(combo.currentData(), selected)
        self.assertEqual(combo.currentText(), self.widget.value.name + " ★")


if __name__ == "__main__":
    unittest.main()

from enum import Enum
from dataclasses import dataclass
import csv
from typing import cast

from PyQt5.QtWidgets import QApplication, QCompleter, QPlainTextEdit, QStyledItemDelegate, QStyle
from PyQt5.QtGui import QFont, QPalette, QPen, QColor, QFontMetrics, QTextCursor
from PyQt5.QtCore import Qt, QStringListModel, QSize, QRect, QAbstractProxyModel

from ..root import root
from ..settings import settings
from ..files import FileFilter
from ..util import ensure, plugin_dir, user_data_dir


class TagType(Enum):
    general = 0
    artist = 1
    copyright = 3
    character = 4
    meta = 5


@dataclass
class TagItem:
    tag: str
    type_: TagType
    count: int
    meta: str


class TagListModel(QStringListModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tags = []

    def setTags(self, tags):
        self.tags = tags
        super().setStringList([tag.tag for tag in tags])


def cursor_position(text: str, cursor: QTextCursor):
    pos_c16 = cursor.position()  # counted as 2-byte characters
    bytes_utf16 = text.encode("utf-16")
    byte_pos = 2 + pos_c16 * 2  # utf-16 text starts with 2-byte BOM
    text_until_pos = bytes_utf16[:byte_pos].decode("utf-16")
    return len(text_until_pos)


class TagCompleterDelegate(QStyledItemDelegate):
    def paint(self, painter, option, index):
        tag_item = self._tag_item(index)

        painter = ensure(painter)
        painter.save()
        if option.state & QStyle.StateFlag.State_Selected:
            painter.fillRect(option.rect, option.palette.highlight())
            text_color = option.palette.highlightedText().color()
        else:
            painter.fillRect(option.rect, self._background_color(tag_item.type_))
            text_color = option.palette.text().color()

        # Set up fonts
        normal_font = QFont(option.font)
        small_font = QFont(option.font)
        small_font.setPointSize(normal_font.pointSize() - 2)
        small_font.setItalic(True)

        # Calculate rectangles
        rect = option.rect
        meta_width = QFontMetrics(normal_font).width(tag_item.meta) + 10
        meta_rect = QRect(rect.right() - meta_width, rect.top(), meta_width, rect.height())

        # Draw the tag
        painter.setFont(normal_font)
        painter.setPen(QPen(text_color))
        painter.drawText(
            rect, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, tag_item.tag
        )

        # Draw the meta info
        painter.setFont(small_font)
        painter.setPen(QPen(text_color.lighter(150)))  # Slightly lighter color for meta info
        painter.drawText(
            meta_rect, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, tag_item.meta
        )

        painter.restore()

    def sizeHint(self, option, index):
        tag_item = self._tag_item(index)

        normal_font = QFont(option.font)
        small_font = QFont(option.font)
        small_font.setPointSize(normal_font.pointSize() - 2)

        tag_width = QFontMetrics(normal_font).width(tag_item.tag)
        meta_width = QFontMetrics(small_font).width(tag_item.meta)

        total_width = tag_width + meta_width + 10  # Add some padding
        size = super().sizeHint(option, index)

        return QSize(total_width, size.height())

    def _tag_item(self, index) -> TagItem:
        model = index.model()
        while isinstance(model, QAbstractProxyModel):
            model = model.sourceModel()
        assert isinstance(model, TagListModel)
        return model.tags[index.model().mapToSource(index).row()]

    def _background_color(self, tag_type):
        tag_colors = {
            TagType.general: QColor(Qt.GlobalColor.blue),
            TagType.artist: QColor(Qt.GlobalColor.red),
            TagType.copyright: QColor(Qt.GlobalColor.yellow),
            TagType.character: QColor(Qt.GlobalColor.green),
            TagType.meta: QColor(Qt.GlobalColor.cyan),
        }

        tag_color = tag_colors.get(tag_type, QColor(Qt.GlobalColor.white))

        # Get the default background color for dropdown items
        app = cast(QApplication, QApplication.instance())
        base_color = app.palette().color(QPalette.Base)

        # Blend the colors
        return self._blend_colors(base_color, tag_color, 0.2)

    # how is there not a native qt way to do this
    def _blend_colors(self, from_, to, factor):
        r = from_.red() * (1 - factor) + to.red() * factor
        g = from_.green() * (1 - factor) + to.green() * factor
        b = from_.blue() * (1 - factor) + to.blue() * factor
        return QColor(int(r), int(g), int(b))


# Ensure there's only one of these globally. It gets pretty big if we have one per widget.
_tag_model = TagListModel([])
_tag_files = None


class PromptAutoComplete:
    def __init__(self, widget: QPlainTextEdit):
        self._widget = widget
        self._completer = QCompleter()
        self._completer.activated.connect(self._insert_completion)
        self._completer.setCompletionMode(QCompleter.CompletionMode.PopupCompletion)
        self._completer.setFilterMode(Qt.MatchFlag.MatchContains)
        self._completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        self._completer.setWidget(widget)
        self._popup = ensure(self._completer.popup())
        self._lora_delegate = ensure(self._popup.itemDelegate())
        self._completion_prefix = ""
        self._completion_suffix = ""

        self._lora_model = FileFilter(root.files.loras)
        self._lora_model.available_only = True

        self._reload_tag_model()
        settings.changed.connect(self._reload_tag_model)

    def _reload_tag_model(self):
        global _tag_model
        global _tag_files

        tag_files = settings.tag_files

        if _tag_files == tag_files:
            return

        all_tags = []
        plugin_tags_path = plugin_dir / "tags"
        user_tags_path = user_data_dir / "tags"
        for tag_name in tag_files:
            plugin_tag_path = plugin_tags_path / f"{tag_name}.csv"
            user_tag_path = user_tags_path / f"{tag_name}.csv"
            tag_path = user_tag_path if user_tag_path.is_file() else plugin_tag_path
            if not tag_path.is_file():
                # formerly active file that was deleted
                continue

            with tag_path.open("r", encoding="utf-8") as f:
                csv_reader = csv.reader(f)
                for tag, type_str, count, _aliases in csv_reader:
                    if type_str.isdigit():  # skip header rows if they exist
                        tag = tag.replace("_", " ")
                        try:
                            tag_type = TagType(int(type_str))
                        except Exception:  # default to general category if category unrecognised
                            tag_type = TagType(0)
                        count = int(count)
                        count_str = str(count)
                        if count > 1_000_000:
                            count_str = f"{count / 1_000_000:.0f}m"
                        elif count > 1_000:
                            count_str = f"{count / 1_000:.0f}k"
                        meta = f"{tag_name} {count_str}"
                        all_tags.append(TagItem(tag, tag_type, count, meta))

        sorted_tags = sorted(all_tags, key=lambda x: x.count, reverse=True)
        seen = set()
        unique_tags = [a for a in sorted_tags if a.tag not in seen and seen.add(a.tag) is None]

        _tag_model.setTags(unique_tags)
        _tag_files = tag_files

    def _current_text(self, separators=" >\n") -> str:
        text = self._widget.toPlainText()
        start = pos = cursor_position(text, self._widget.textCursor())
        while pos > 0 and (text[pos - 1] not in separators or pos > 1 and text[pos - 2] == "\\"):
            pos -= 1
        return text[pos:start]

    def check_completion(self):
        prefix = self._current_text()
        name = prefix.removeprefix("<lora:")
        lora_mode = len(prefix) > len(name)

        if lora_mode:
            self._completer.setModel(self._lora_model)
            self._completion_prefix = name
            self._completion_suffix = ">"
            self._popup.setItemDelegate(self._lora_delegate)
        else:
            # fall through to tag search
            self._completion_prefix = prefix = self._current_text(separators="()>,\n").lstrip()
            name = prefix.replace("\\(", "(").replace("\\)", ")")
            if not name.startswith("<") and len(name.rstrip()) > 2:
                self._completer.setModel(_tag_model)
                self._popup.setItemDelegate(TagCompleterDelegate())
                self._completion_suffix = ""
            else:
                self._popup.hide()
                return
        self._completer.setCompletionPrefix(name)
        rect = self._widget.cursorRect()
        self._popup.setCurrentIndex(ensure(self._completer.completionModel()).index(0, 0))
        scrollbar = ensure(self._popup.verticalScrollBar())
        rect.setWidth(self._popup.sizeHintForColumn(0) + scrollbar.sizeHint().width())
        self._completer.complete(rect)

    def _insert_completion(self, completion):
        triggers = ""
        if self._current_text().startswith("<lora:"):
            if file := root.files.loras.find(f"{completion}.safetensors"):
                triggers = " " + file.meta("lora_triggers", "")
        else:  # tag completion
            # escape () in tags so they won't be interpreted as prompt weights
            completion = completion.replace("(", "\\(").replace(")", "\\)")
        text = self._widget.toPlainText()
        initial_cursor = self._widget.textCursor()
        pos = cursor_position(text, initial_cursor)
        start_pos = pos - len(self._completion_prefix)  # pos in python string
        start_cursor_pos = initial_cursor.position() - len(self._completion_prefix)  # pos in utf-16
        fill = completion + self._completion_suffix + triggers
        text = text[:start_pos] + fill + text[pos:]
        self._widget.setPlainText(text)
        cursor = self._widget.textCursor()
        cursor.setPosition(start_cursor_pos + len(fill))
        self._widget.setTextCursor(cursor)

    @property
    def is_active(self):
        return self._popup.isVisible()

    action_keys = [
        Qt.Key.Key_Enter,
        Qt.Key.Key_Return,
        Qt.Key.Key_Up,
        Qt.Key.Key_Down,
        Qt.Key.Key_Tab,
        Qt.Key.Key_Backtab,
    ]

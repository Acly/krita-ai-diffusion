import csv
from enum import Enum
from pathlib import Path

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QStandardItemModel, QStandardItem
from PyQt5.QtWidgets import QCompleter


class DanbooruTagCategory(Enum):
    General = 0
    Artist = 1
    Copyright = 3
    Character = 4
    Meta = 5


class TagCompleteItem(QStandardItem):
    CountRole = Qt.ItemDataRole.UserRole + 1
    CategoryRole = Qt.ItemDataRole.UserRole + 2
    AliasRole = Qt.ItemDataRole.UserRole + 3

    def __init__(self, text, count, category, alias):
        super().__init__(text)
        self.setData(count, TagCompleteItem.CountRole)
        self.setData(category, TagCompleteItem.CategoryRole)
        self.setData(alias, TagCompleteItem.AliasRole)


def create_completer():
    tags_folder = Path(__file__).parent.parent / "tags"
    danbooru_csv_file = tags_folder / "danbooru.csv"

    model = QStandardItemModel()
    with open(danbooru_csv_file, mode="r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)

        for text, category, count, alias in reader:
            model.appendRow(
                TagCompleteItem(
                    text, int(count), DanbooruTagCategory(int(category)), alias.split(",")
                )
            )

    completer = QCompleter()
    completer.setModel(model)
    return completer
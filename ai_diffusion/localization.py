import json
from pathlib import Path
from typing import NamedTuple

from .util import user_data_dir, client_logger as log


class Language(NamedTuple):
    id: str
    name: str
    path: Path

    @staticmethod
    def from_file(filepath: Path):
        try:
            with filepath.open(encoding="utf-8") as f:
                lang = json.load(f)
                return Language(lang["id"], lang["name"], filepath)
        except Exception as e:
            log.warning(f"Not a valid language file: {filepath}: {e}")
            return None


class Localization:
    _id: str
    _name: str
    _translations: dict[str, str]

    available: list[Language]
    current: "Localization"

    def __init__(self, id: str = "en", name: str = "English", translations: dict = {}):
        self._id = id
        self._name = name
        self._translations = translations

    def translate(self, key: str, **kwargs):
        translation = self._translations.get(key, key) or key
        if kwargs:
            try:
                translation = translation.format(**kwargs)
            except Exception as e:
                log.error(f"Failed to format translation for {key}: {e}")
                translation = key.format(**kwargs)
        return translation

    @property
    def id(self):
        return self._id

    @staticmethod
    def load(id: str, filepath: Path):
        try:
            with filepath.open(encoding="utf-8") as f:
                lang = json.load(f)
                return Localization(id, lang["name"], lang["translations"])
        except Exception as e:
            raise Exception(f"Could not load language file for {id} at {filepath}: {e}")

    @staticmethod
    def init(settings_path: Path | None = None):
        language = "en"
        settings_path = settings_path or user_data_dir / "settings.json"
        if settings_path.exists():
            try:
                with settings_path.open(encoding="utf-8") as f:
                    settings = json.load(f)
                    language = settings.get("language", "en")
            except Exception as e:
                log.warning(f"Could not read language settings: {e}")

        language_file = Path(__file__).parent / "language" / f"{language}.json"
        try:
            return Localization.load(language, language_file)
        except Exception as e:
            log.warning(str(e))
            return Localization()

    @staticmethod
    def scan():
        dir = Path(__file__).parent / "language"
        langs = (Language.from_file(f) for f in dir.iterdir() if f.suffix == ".json")
        return [lang for lang in langs if lang]


Localization.current = Localization.init()
Localization.available = Localization.scan()


def translate(key: str, **kwargs):
    return Localization.current.translate(key, **kwargs)

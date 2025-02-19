from __future__ import annotations
from typing import Any, Callable, cast

import csv
import random
import re
import requests
from pathlib import Path
from bs4 import BeautifulSoup
from PyQt5.QtWidgets import (
    QAction,
    QSlider,
    QWidget,
    QPlainTextEdit,
    QLabel,
    QMenu,
    QSpinBox,
    QToolButton,
    QComboBox,
    QHBoxLayout,
    QSizePolicy,
    QStyle,
    QStyleOption,
    QWidgetAction,
    QCheckBox,
    QGridLayout,
    QPushButton,
    QFrame,
    QScrollBar,
)
from PyQt5.QtGui import (
    QDesktopServices,
    QGuiApplication,
    QFontMetrics,
    QKeyEvent,
    QMouseEvent,
    QPalette,
    QTextCursor,
    QPainter,
    QPaintEvent,
    QKeySequence,
)
from PyQt5.QtCore import Qt, QMetaObject, QSize, pyqtSignal, QEvent, QUrl
from krita import Krita

from ..style import Style, Styles
from ..root import root
from ..client import filter_supported_styles, resolve_arch
from ..properties import Binding, Bind, bind, bind_combo
from ..jobs import JobState, JobKind
from ..model import Model, Workspace, SamplingQuality, ProgressKind, ErrorKind, Error, no_error
from ..text import edit_attention, select_on_cursor_pos
from ..localization import translate as _
from ..util import ensure
from ..workflow import apply_strength, snap_to_percent
from ..settings import Settings, settings
from .autocomplete import PromptAutoComplete
from .theme import SignalBlocker
from . import actions, theme


class QueuePopup(QMenu):
    _model: Model
    _connections: list[QMetaObject.Connection]

    def __init__(self, supports_batch=True, parent: QWidget | None = None):
        super().__init__(parent)
        self._connections = []

        palette = self.palette()
        self.setObjectName("QueuePopup")
        self.setStyleSheet(
            f"""
            QWidget#QueuePopup {{
                background-color: {palette.window().color().name()}; 
                border: 1px solid {palette.dark().color().name()};
            }}"""
        )

        self._layout = QGridLayout()
        self.setLayout(self._layout)

        batch_label = QLabel(_("Batches"), self)
        batch_label.setVisible(supports_batch)
        self._layout.addWidget(batch_label, 0, 0)
        batch_layout = QHBoxLayout()
        self._batch_slider = QSlider(Qt.Orientation.Horizontal, self)
        self._batch_slider.setMinimum(1)
        self._batch_slider.setMaximum(10)
        self._batch_slider.setSingleStep(1)
        self._batch_slider.setPageStep(1)
        self._batch_slider.setVisible(supports_batch)
        self._batch_slider.setToolTip(_("Number of jobs to enqueue at once"))
        self._batch_label = QLabel("1", self)
        self._batch_label.setVisible(supports_batch)
        batch_layout.addWidget(self._batch_slider)
        batch_layout.addWidget(self._batch_label)
        self._layout.addLayout(batch_layout, 0, 1)

        self._seed_label = QLabel(_("Seed"), self)
        self._layout.addWidget(self._seed_label, 1, 0)
        self._seed_input = QSpinBox(self)
        self._seed_check = QCheckBox(self)
        self._seed_check.setText(_("Fixed"))
        self._seed_input.setMinimum(0)
        self._seed_input.setMaximum(2**31 - 1)
        self._seed_input.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self._seed_input.setToolTip(
            _(
                "The seed controls the random part of the output. A fixed seed value will always produce the same result for the same inputs."
            )
        )
        self._randomize_seed = QToolButton(self)
        self._randomize_seed.setIcon(theme.icon("random"))
        seed_layout = QHBoxLayout()
        seed_layout.addWidget(self._seed_check)
        seed_layout.addWidget(self._seed_input)
        seed_layout.addWidget(self._randomize_seed)
        self._layout.addLayout(seed_layout, 1, 1)

        resolution_multiplier_label = QLabel(_("Resolution"), self)
        self._resolution_multiplier_slider = QSlider(Qt.Orientation.Horizontal, self)
        self._resolution_multiplier_slider.setRange(3, 15)
        self._resolution_multiplier_slider.setValue(10)
        self._resolution_multiplier_slider.setSingleStep(1)
        self._resolution_multiplier_slider.setPageStep(1)
        self._resolution_multiplier_slider.setToolTip(Settings._resolution_multiplier.desc)
        self._resolution_multiplier_slider.valueChanged.connect(self._set_resolution_multiplier)
        self._resolution_multiplier_display = QLabel("1.0 x", self)
        self._resolution_multiplier_display.setAlignment(Qt.AlignmentFlag.AlignRight)
        self._resolution_multiplier_display.setMinimumWidth(20)
        resolution_multiplier_layout = QHBoxLayout()
        resolution_multiplier_layout.addWidget(self._resolution_multiplier_slider)
        resolution_multiplier_layout.addWidget(self._resolution_multiplier_display)
        self._layout.addWidget(resolution_multiplier_label, 2, 0)
        self._layout.addLayout(resolution_multiplier_layout, 2, 1)

        enqueue_label = QLabel(_("Enqueue"), self)
        self._queue_front_combo = QComboBox(self)
        self._queue_front_combo.addItem(_("in Front (new jobs first)"), True)
        self._queue_front_combo.addItem(_("at the Back"), False)
        self._layout.addWidget(enqueue_label, 3, 0)
        self._layout.addWidget(self._queue_front_combo, 3, 1)

        cancel_label = QLabel(_("Cancel"), self)
        self._layout.addWidget(cancel_label, 4, 0)
        self._cancel_active = self._create_cancel_button(_("Active"), actions.cancel_active)
        self._cancel_queued = self._create_cancel_button(_("Queued"), actions.cancel_queued)
        self._cancel_all = self._create_cancel_button(_("All"), actions.cancel_all)
        cancel_layout = QHBoxLayout()
        cancel_layout.addWidget(self._cancel_active)
        cancel_layout.addWidget(self._cancel_queued)
        cancel_layout.addWidget(self._cancel_all)
        self._layout.addLayout(cancel_layout, 4, 1)

        self._model = root.active_model

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model: Model):
        Binding.disconnect_all(self._connections)
        self._model = model
        self._randomize_seed.setEnabled(self._model.fixed_seed)
        self._seed_input.setEnabled(self._model.fixed_seed)
        self._batch_label.setText(str(self._model.batch_count))
        self._connections = [
            bind(self._model, "batch_count", self._batch_slider, "value"),
            model.batch_count_changed.connect(lambda v: self._batch_label.setText(str(v))),
            bind(self._model, "seed", self._seed_input, "value"),
            bind(self._model, "fixed_seed", self._seed_check, "checked", Bind.one_way),
            self._seed_check.toggled.connect(lambda v: setattr(self._model, "fixed_seed", v)),
            self._model.fixed_seed_changed.connect(self._seed_input.setEnabled),
            self._model.fixed_seed_changed.connect(self._randomize_seed.setEnabled),
            self._randomize_seed.clicked.connect(self._model.generate_seed),
            model.resolution_multiplier_changed.connect(self._update_resolution_multiplier),
            bind_combo(self._model, "queue_front", self._queue_front_combo),
            model.jobs.count_changed.connect(self._update_cancel_buttons),
        ]

    def _create_cancel_button(self, name: str, action: Callable[[], None]):
        button = QToolButton(self)
        button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        button.setText(name)
        button.setIcon(theme.icon("cancel"))
        button.setEnabled(False)
        button.clicked.connect(action)
        return button

    def _update_cancel_buttons(self):
        has_active = self._model.jobs.any_executing()
        has_queued = self._model.jobs.count(JobState.queued) > 0
        self._cancel_active.setEnabled(has_active)
        self._cancel_queued.setEnabled(has_queued)
        self._cancel_all.setEnabled(has_active or has_queued)

    def _update_resolution_multiplier(self):
        slider_value = round(self.model.resolution_multiplier * 10)
        if self._resolution_multiplier_slider.value() != slider_value:
            self._resolution_multiplier_slider.setValue(slider_value)

    def _set_resolution_multiplier(self, value: int):
        self.model.resolution_multiplier = value / 10
        self._resolution_multiplier_display.setText(f"{(value / 10):.1f} x")

    def mouseReleaseEvent(self, a0: QMouseEvent | None) -> None:
        if parent := cast(QWidget, self.parent()):
            parent.close()
        return super().mouseReleaseEvent(a0)


class QueueButton(QToolButton):
    def __init__(self, supports_batch=True, parent: QWidget | None = None):
        super().__init__(parent)
        self._model = root.active_model
        self._connect_model()

        self._popup = QueuePopup(supports_batch)
        popup_action = QWidgetAction(self)
        popup_action.setDefaultWidget(self._popup)
        self.addAction(popup_action)

        self.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self._update()

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model: Model):
        if self._model != model:
            Binding.disconnect_all(self._connections)
            self._model = model
            self._popup.model = model
            self._connect_model()

    def _connect_model(self):
        self._connections = [
            self._model.jobs.count_changed.connect(self._update),
            self._model.progress_kind_changed.connect(self._update),
        ]

    def _update(self):
        count = self._model.jobs.count(JobState.queued)
        queued_msg = _("{count} jobs queued.", count=count)
        cancel_msg = _("Click to cancel.")

        if self._model.progress_kind is ProgressKind.upload:
            self.setIcon(theme.icon("queue-upload"))
            self.setToolTip(_("Uploading models.") + f" {queued_msg} {cancel_msg}")
            count += 1
        elif self._model.jobs.any_executing():
            self.setIcon(theme.icon("queue-active"))
            if count > 0:
                self.setToolTip(_("Generating image.") + f" {queued_msg} {cancel_msg}")
            else:
                self.setToolTip(_("Generating image.") + f" {cancel_msg}")
            count += 1
        else:
            self.setIcon(theme.icon("queue-inactive"))
            self.setToolTip(_("Idle."))
        self.setText(f"{count} ")

    def sizeHint(self) -> QSize:
        original = super().sizeHint()
        width = original.height() * 0.75 + self.fontMetrics().width(" 99 ") + 20
        return QSize(int(width), original.height())

    def paintEvent(self, a0):
        _paint_tool_drop_down(self, self.text())

def download_site_icons():
    """Download site icons if they don't exist locally"""
    import requests
    from pathlib import Path
    
    icons_dir = Path(__file__).parent.parent / "resources"
    icons_dir.mkdir(exist_ok=True)
    
    icons = {
        'e621': {
            'url': 'https://e621.net/packs/static/main-logo-2653c015c5870ec4ff08.svg',
            'file': 'e621-icon.svg'
        },
        'danbooru': {
            'url': 'https://danbooru.donmai.us/favicon.svg',
            'file': 'danbooru-icon.svg'
        }
    }
    
    for site, info in icons.items():
        icon_path = icons_dir / info['file']
        if not icon_path.exists():
            try:
                response = requests.get(info['url'])
                if response.ok:
                    icon_path.write_text(response.text)
                    print(f"Downloaded {site} icon")
                else:
                    print(f"Failed to download {site} icon: {response.status_code}")
            except Exception as e:
                print(f"Error downloading {site} icon: {e}")

def create_site_icon(site_name: str) -> QIcon:
    """Create a QIcon from site SVG"""
    from PyQt5.QtSvg import QSvgRenderer
    from PyQt5.QtCore import QSize, Qt
    from PyQt5.QtGui import QIcon, QPixmap, QPainter
    
    icon_path = Path(__file__).parent.parent / "resources" / f"{site_name}-icon.svg"
    
    # Download icons if they don't exist
    if not icon_path.exists():
        download_site_icons()
    
    if not icon_path.exists():
        return QIcon()  # Return empty icon if download failed
        
    # Create icon at multiple sizes
    icon = QIcon()
    renderer = QSvgRenderer(str(icon_path))
    
    for size in [16, 24, 32, 48]:
        pixmap = QPixmap(QSize(size, size))
        pixmap.fill(Qt.transparent)
        renderer.render(QPainter(pixmap))
        icon.addPixmap(pixmap)
        
    return icon

class StyleSelectWidget(QWidget):
    _value: Style
    _styles: list[Style]

    value_changed = pyqtSignal(Style)
    quality_changed = pyqtSignal(SamplingQuality)

    def __init__(self, parent: QWidget | None, show_quality=False):
        super().__init__(parent)
        self._value = Styles.list().default
        self._e621_artists = []
        self._danbooru_artists = []
        self._artist_history = []  # Store prompt history for undo
        self._last_artist = None
        self._load_artists()

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)

        # Style combo box
        self._combo = QComboBox(self)
        self.update_styles()
        self._combo.currentIndexChanged.connect(self.change_style)
        layout.addWidget(self._combo, 3)

        # Quality selector
        if show_quality:
            self._quality_combo = QComboBox(self)
            self._quality_combo.addItem(_("Fast"), SamplingQuality.fast.value)
            self._quality_combo.addItem(_("Quality"), SamplingQuality.quality.value)
            self._quality_combo.currentIndexChanged.connect(self.change_quality)
            layout.addWidget(self._quality_combo, 1)

        # Favorite artists dropdown
        self._favorite_artists = QComboBox(self)
        self._favorite_artists.setToolTip(_("Favorite artists"))
        self._favorite_artists.currentIndexChanged.connect(self._insert_favorite_artist)
        self._load_favorite_artists()
        layout.addWidget(self._favorite_artists)

        # Star button for favorites
        self._star_button = QToolButton(self)
        self._star_button.setIcon(Krita.instance().icon('star-shape'))
        self._star_button.setAutoRaise(True)
        self._star_button.setToolTip(_("Add last used artist to favorites"))
        self._star_button.clicked.connect(self._add_to_favorites)
        layout.addWidget(self._star_button)

        # Random e621 artist button
        random_e621_artist = QToolButton(self)
        random_e621_artist.setIcon(theme.icon("random"))
        random_e621_artist.setAutoRaise(True)
        random_e621_artist.setToolTip(_("Insert random e621 artist"))
        random_e621_artist.clicked.connect(lambda: self.insert_random_artist("e621"))
        layout.addWidget(random_e621_artist)

        # E621 random tags button
        self._e621_tags = QToolButton(self)
        self._e621_tags.setIcon(create_site_icon('e621'))
        self._e621_tags.setAutoRaise(True)
        self._e621_tags.setToolTip(_("Get tags from random e621 image"))
        self._e621_tags.clicked.connect(lambda: self.get_random_tags("e621"))
        layout.addWidget(self._e621_tags)

        # Random Danbooru artist button
        random_danbooru_artist = QToolButton(self)
        random_danbooru_artist.setIcon(theme.icon("random"))
        random_danbooru_artist.setAutoRaise(True)
        random_danbooru_artist.setToolTip(_("Insert random Danbooru artist"))
        random_danbooru_artist.clicked.connect(lambda: self.insert_random_artist("danbooru"))
        layout.addWidget(random_danbooru_artist)

        # Danbooru random tags button 
        self._danbooru_tags = QToolButton(self)
        self._danbooru_tags.setIcon(create_site_icon('danbooru'))
        self._danbooru_tags.setAutoRaise(True)
        self._danbooru_tags.setToolTip(_("Get tags from random Danbooru image"))
        self._danbooru_tags.clicked.connect(lambda: self.get_random_tags("danbooru"))
        layout.addWidget(self._danbooru_tags)

        # Settings button
        settings = QToolButton(self)
        settings.setIcon(theme.icon("settings"))
        settings.setAutoRaise(True)
        settings.clicked.connect(self.show_settings)
        layout.addWidget(settings)

        # Solo filter checkbox
        self._solo_check = QCheckBox(_("Solo only"), self)
        self._solo_check.setChecked(Settings._require_solo.default)
        self._solo_check.toggled.connect(self._toggle_solo)
        layout.addWidget(self._solo_check)

        Styles.list().changed.connect(self.update_styles)
        Styles.list().name_changed.connect(self.update_styles)
        root.connection.state_changed.connect(self.update_styles)

        self.installEventFilter(self)

    def _load_artists(self):
        """Initialize separate artist lists for e621 and danbooru"""
        self._e621_artists = []
        self._danbooru_artists = []
        self._artist_history = []
        self._last_artist = None

        try:
            # Load e621 artists
            e621_file = Path(__file__).parent.parent / "e621_artist_webui.csv"
            if e621_file.exists():
                with open(e621_file, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    self._e621_artists.extend([row['trigger'] for row in reader])
                print(f"Loaded {len(self._e621_artists)} e621 artists")

            # Load danbooru artists
            danbooru_file = Path(__file__).parent.parent / "danbooru_artist_webui.csv"
            if danbooru_file.exists():
                with open(danbooru_file, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    self._danbooru_artists.extend([row['trigger'] for row in reader])
                print(f"Loaded {len(self._danbooru_artists)} Danbooru artists")

        except Exception as e:
            print(f"Error loading artists: {e}")

    def _load_favorite_artists(self):
        """Load favorite artists from settings"""
        self._favorite_artists.clear()
        self._favorite_artists.addItem(_("Favorite artists"))
        
        # Get the favorites list from settings
        favorites = settings.favorite_artists
        if favorites and isinstance(favorites, list):
            self._favorite_artists.addItems(favorites)

    def _add_to_favorites(self):
        """Add current artist to favorites"""
        if not self._last_artist:
            return
            
        try:
            current_favorites = settings.favorite_artists or []
            
            if self._last_artist not in current_favorites:
                current_favorites.append(self._last_artist)
                settings.favorite_artists = current_favorites
                settings.save()
                self._load_favorite_artists()
        except Exception as e:
            print(f"Error saving favorite artist: {e}")

    def _insert_favorite_artist(self, index):
        """Insert selected favorite artist into prompt"""
        if index > 0:  # Skip the first item which is just the label
            model = root.model_for_active_document()
            if model:
                current_prompt = model.regions.positive
                self._artist_history.append(current_prompt)
                
                artist = self._favorite_artists.currentText()
                new_prompt = re.sub(r'artist:[^,]+,?\s*', '', current_prompt).strip()
                new_prompt = f"artist:{artist}, {new_prompt}" if new_prompt else f"artist:{artist}, "
                model.regions.positive = new_prompt
                self._last_artist = artist
                
            self._favorite_artists.setCurrentIndex(0)  # Reset selection

    def insert_random_artist(self):
        if not self._artists:
            return

        model = root.model_for_active_document()
        if not model:
            return

        current_prompt = model.regions.positive
        # Remove any existing artist: tags - match until comma or end of string
        new_prompt = re.sub(r'artist:[^,]+,?\s*', '', current_prompt).strip()
        # Insert new random artist at the beginning
        random_artist = random.choice(self._artists)
        new_prompt = f"artist:{random_artist}, " + new_prompt if new_prompt else f"artist:{random_artist}, "
        model.regions.positive = new_prompt
        self._last_artist = random_artist

    def insert_random_artist(self, source="e621"):
        """Insert a random artist from specified source (e621 or danbooru)"""
        artists = self._e621_artists if source == "e621" else self._danbooru_artists
        if not artists:
            print(f"No artists loaded from {source}")
            return

        model = root.model_for_active_document()
        if not model:
            return

        current_prompt = model.regions.positive
        self._artist_history.append(current_prompt)
        
        # Remove any existing artist: tags
        new_prompt = re.sub(r'artist:[^,]+,?\s*', '', current_prompt).strip()
        
        # Insert new random artist at the beginning
        random_artist = random.choice(artists)
        new_prompt = f"artist:{random_artist}, {new_prompt}" if new_prompt else f"artist:{random_artist}, "
        model.regions.positive = new_prompt
        self._last_artist = random_artist

    def _toggle_solo(self, checked):
        settings.require_solo = checked
        settings.save()

    def eventFilter(self, obj, event):
        if event.type() == QEvent.KeyPress:
            if event.key() == Qt.Key_Z and event.modifiers() == Qt.ControlModifier:
                if self._artist_history:
                    model = root.model_for_active_document()
                    if model:
                        model.regions.positive = self._artist_history.pop()
                    return True
        return super().eventFilter(obj, event)

    def _format_tag(self, tag: str, category: str) -> str:
        """
        Format a tag based on its category and content.
        
        Args:
            tag: The tag to format
            category: The tag category (artist, copyright, character, species, general)
        
        Returns:
            The formatted tag string
        """
        # Remove leading/trailing whitespace
        tag = tag.strip()
        
        # For artist category, add artist: prefix
        if category == 'artist':
            # Remove existing artist: prefix if present to avoid duplication
            if tag.startswith('artist:'):
                tag = tag[7:]
            tag = f"artist:{tag}"
        
        # Escape parentheses in any category
        if '(' in tag or ')' in tag:
            tag = tag.replace('(', '\\(').replace(')', '\\)')
            
        return tag

    def _process_tag_list(self, tag_elements, category: str, unwanted_tags: list[str]) -> list[str]:
        """
        Process a list of tag elements and format them appropriately.
        
        Args:
            tag_elements: The BS4 elements containing tags
            category: The tag category 
            unwanted_tags: List of tags to exclude
            
        Returns:
            List of formatted tag strings
        """
        tags = []
        for tag_element in tag_elements:
            tag = tag_element.text.strip()
            if tag and tag not in unwanted_tags:
                formatted_tag = self._format_tag(tag, category)
                tags.append(formatted_tag)
        return tags

    def _try_get_random_tags(self, session, site="e621", max_attempts=10):
        """Try to get random tags from specified site."""
        for attempt in range(max_attempts):
            try:
                # Select appropriate URL and selectors based on site
                if site == "e621":
                    url = "https://e621.net/posts/random"
                    tag_selector = '#tag-list li a.search-tag'
                    category_selector = '#tag-list > ul.{}-tag-list > li a.search-tag'
                else:  # danbooru
                    url = "https://danbooru.donmai.us/posts/random"
                    tag_selector = '.tag-list li a'
                    category_selector = '.{}-tag-list li a'

                print(f"\nAttempt {attempt + 1} for {site}")
                print(f"Requesting URL: {url}")
                
                random_page = session.get(url, allow_redirects=True)
                if not random_page.ok:
                    print(f"Failed to fetch random post from {site} - Status {random_page.status_code}")
                    continue
                
                print(f"Got page response: {random_page.url}")
                    
                soup = BeautifulSoup(random_page.text, 'html.parser')
                
                # Debug tag selectors
                print(f"\nUsing tag selector: {tag_selector}")
                tag_elements = soup.select(tag_selector)
                print(f"Found {len(tag_elements)} total tags")
                
                if not tag_elements:
                    print("No tags found with selector")
                    # Print a small portion of the HTML for debugging
                    print("Page content preview:")
                    print(random_page.text[:500])
                    continue
                
                # Print found tags for debugging
                tag_texts = [tag.text.strip() for tag in tag_elements]
                print("\nFound tags:", tag_texts[:10], "..." if len(tag_texts) > 10 else "")
                
                if settings.require_solo:
                    if 'solo' not in tag_texts or 'duo' in tag_texts:
                        print("Skipping - solo requirement not met")
                        continue

                # Get enabled categories
                enabled_categories = []
                tag_categories = settings.tag_categories or {
                    'artist': False,
                    'copyright': False,
                    'character': False,
                    'species': False,
                    'general': True
                }
                
                print("\nEnabled categories:", tag_categories)
                
                for cat, enabled in tag_categories.items():
                    if enabled:
                        if site == "danbooru" and cat == "species":
                            continue  # Danbooru doesn't have species category
                        category_name = f"{cat}-tag" if site == "danbooru" else cat
                        enabled_categories.append(category_name)
                        
                if not enabled_categories:
                    enabled_categories = ['general'] if site == "e621" else ['general-tag']
                
                print("Processing categories:", enabled_categories)
                
                # Collect and format tags from enabled categories
                tags_output = []
                unwanted = settings.unwanted_tags if settings.unwanted_tags is not None else []
                
                for category in enabled_categories:
                    selector = category_selector.format(category)
                    print(f"\nChecking category {category} with selector: {selector}")
                    tag_elements = soup.select(selector)
                    print(f"Found {len(tag_elements)} tags in {category}")
                    
                    category_name = category.replace('-tag', '') if site == "danbooru" else category
                    formatted_tags = self._process_tag_list(tag_elements, category_name, unwanted)
                    print(f"After processing: {len(formatted_tags)} tags")
                    tags_output.extend(formatted_tags)
                    
                if not tags_output:
                    print("No valid tags found after filtering")
                    continue
                    
                print("\nFinal tags:", tags_output)

                # Update the prompt
                model = root.model_for_active_document()
                if not model:
                    print("No active document model")
                    return False

                # Save current prompt for undo
                current_prompt = model.regions.positive
                self._artist_history.append(current_prompt)

                # Keep existing artist tag if present
                artist_tags = ""
                artist_match = re.match(r'(artist:[^,]+,\s*)', current_prompt)
                if artist_match:
                    artist_tags = artist_match.group(1)
                    self._last_artist = artist_match.group(1).replace("artist:", "").replace(",", "").strip()
                
                # Update prompt with new tags
                new_prompt = artist_tags + ', '.join(tags_output)
                print("\nFinal prompt:", new_prompt)
                
                model.regions.positive = new_prompt.strip()
                return True
                
            except Exception as e:
                print(f"\nError in attempt {attempt + 1}:")
                print(f"Type: {type(e).__name__}")
                print(f"Error: {str(e)}")
                continue
        
        print("Failed all attempts to get tags")
        return False

    def get_random_tags(self, site="e621"):
        """Get random tags from specified site"""
        try:
            model = root.model_for_active_document()
            if not model:
                return

            if site == "e621":
                # E621 works fine with regular requests
                session = requests.Session()
                session.headers.update({
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                })
                success = self._try_get_random_tags_requests(session, site)
            else:
                # Try to import curl_cffi with better error handling
                try:
                    import sys
                    print("Python path:", sys.path)  # Debug: show Python path
                    import curl_cffi
                    print("curl_cffi version:", curl_cffi.__version__)  # Debug: show version
                    from curl_cffi import requests as curl_requests
                    success = self._try_get_random_tags_curl(curl_requests, site)
                except ImportError as e:
                    print(f"Import error details: {str(e)}")  # Debug: show exact import error
                    print("Attempted import from:", sys.path)  # Debug: show where Python looked
                    print("curl_cffi is not installed. Please install with: pip install curl_cffi")
                    return False
                    
            if not success:
                print(f"Unable to fetch valid tags from {site}")
                
        except Exception as e:
            print(f"Error in get_random_tags: {e}")
            return

    def _try_get_random_tags_requests(self, session, site="e621", max_attempts=10):
        """Try to get random tags using regular requests."""
        for attempt in range(max_attempts):
            try:
                print(f"\nAttempt {attempt + 1} for {site}")
                
                if site == "e621":
                    url = "https://e621.net/posts/random"
                    tag_selector = '#tag-list li a.search-tag'
                    category_selector = '#tag-list > ul.{}-tag-list > li a.search-tag'
                else:
                    url = "https://danbooru.donmai.us/posts/random"
                    tag_selector = '#tag-list li a.search-tag'
                    category_selector = '#tag-list > ul.{}-tag-list li a.search-tag'
                
                print(f"Requesting URL: {url}")
                response = session.get(url, allow_redirects=True)
                
                if not response.ok:
                    print(f"Failed to fetch random post - Status {response.status_code}")
                    print("Response:", response.text[:500])
                    continue
                    
                print(f"Got page response: {response.url}")
                
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Debug tag selectors
                print(f"\nUsing tag selector: {tag_selector}")
                tag_elements = soup.select(tag_selector)
                print(f"Found {len(tag_elements)} total tags")
                
                if not tag_elements:
                    print("No tags found with selector")
                    print("Page content preview:")
                    print(response.text[:500])
                    continue
                
                # Print found tags for debugging
                tag_texts = [tag.text.strip() for tag in tag_elements]
                print("\nFound tags:", tag_texts[:10], "..." if len(tag_texts) > 10 else "")
                
                if settings.require_solo:
                    if 'solo' not in tag_texts or 'duo' in tag_texts:
                        print("Skipping - solo requirement not met")
                        continue

                # Get enabled categories
                enabled_categories = []
                tag_categories = settings.tag_categories or {
                    'artist': False,
                    'copyright': False,
                    'character': False,
                    'species': False,
                    'general': True
                }
                
                print("\nEnabled categories:", tag_categories)
                
                for cat, enabled in tag_categories.items():
                    if enabled:
                        if site == "danbooru" and cat == "species":
                            continue  # Danbooru doesn't have species category
                        if site == "danbooru":
                            enabled_categories.append(f"{cat}-tag")
                        else:
                            enabled_categories.append(cat)
                        
                if not enabled_categories:
                    enabled_categories = ['general'] if site == "e621" else ['general-tag']
                
                print("Processing categories:", enabled_categories)
                
                # Collect and format tags from enabled categories
                tags_output = []
                unwanted = settings.unwanted_tags if settings.unwanted_tags is not None else []
                
                for category in enabled_categories:
                    selector = category_selector.format(category)
                    print(f"\nChecking category {category} with selector: {selector}")
                    tag_elements = soup.select(selector)
                    print(f"Found {len(tag_elements)} tags in {category}")
                    
                    category_name = category.replace('-tag', '') if site == "danbooru" else category
                    formatted_tags = self._process_tag_list(tag_elements, category_name, unwanted)
                    print(f"After processing: {len(formatted_tags)} tags")
                    tags_output.extend(formatted_tags)
                    
                if not tags_output:
                    print("No valid tags found after filtering")
                    continue
                    
                print("\nFinal tags:", tags_output)

                # Update the prompt
                model = root.model_for_active_document()
                if not model:
                    print("No active document model")
                    return False

                current_prompt = model.regions.positive
                self._artist_history.append(current_prompt)

                artist_tags = ""
                artist_match = re.match(r'(artist:[^,]+,\s*)', current_prompt)
                if artist_match:
                    artist_tags = artist_match.group(1)
                    self._last_artist = artist_match.group(1).replace("artist:", "").replace(",", "").strip()
                
                new_prompt = artist_tags + ', '.join(tags_output)
                print("\nFinal prompt:", new_prompt)
                
                model.regions.positive = new_prompt.strip()
                return True
                    
            except Exception as e:
                print(f"\nError in attempt {attempt + 1}:")
                print(f"Type: {type(e).__name__}")
                print(f"Error: {str(e)}")
                continue
                
        print("Failed to get valid tags after all attempts")
        return False

    def _try_get_random_tags_curl(self, curl_requests, site="danbooru", max_attempts=10):
        """Try to get random tags using curl requests."""
        for attempt in range(max_attempts):
            try:
                print(f"\nAttempt {attempt + 1} for {site}")
                url = "https://danbooru.donmai.us/posts/random"
                print(f"Requesting URL: {url}")
                
                response = curl_requests.get(url, impersonate="chrome")
                if not response.ok:
                    print(f"Failed to fetch random post - Status {response.status_code}")
                    continue
                    
                print(f"Got page response: {response.url}")
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Danbooru specific selectors for different tag categories
                category_selectors = {
                    'artist': '.artist-tag-list li a.search-tag',
                    'copyright': '.copyright-tag-list li a.search-tag',
                    'character': '.character-tag-list li a.search-tag', 
                    'general': '.general-tag-list li a.search-tag'
                }
                
                # Get solo tag status first
                all_tags = []
                for selector in category_selectors.values():
                    tag_elements = soup.select(selector)
                    all_tags.extend([tag.text.strip() for tag in tag_elements])
                
                if settings.require_solo:
                    if 'solo' not in all_tags or any(x in all_tags for x in ['duo', 'group']):
                        print("Skipping - solo requirement not met")
                        continue
                
                # Get enabled categories from settings
                tag_categories = settings.tag_categories or {
                    'artist': False,
                    'copyright': False, 
                    'character': False,
                    'species': False,
                    'general': True
                }
                
                print("\nEnabled categories:", tag_categories)
                
                # Process each enabled category
                tags_output = []
                unwanted = settings.unwanted_tags if settings.unwanted_tags is not None else []
                
                for category, enabled in tag_categories.items():
                    if not enabled or category == 'species':  # Skip species for Danbooru
                        continue
                        
                    selector = category_selectors.get(category)
                    if not selector:
                        continue
                        
                    print(f"\nChecking category {category} with selector: {selector}")
                    tag_elements = soup.select(selector)
                    print(f"Found {len(tag_elements)} tags in {category}")
                    
                    formatted_tags = self._process_tag_list(tag_elements, category, unwanted)
                    print(f"After processing: {len(formatted_tags)} tags")
                    tags_output.extend(formatted_tags)
                    
                if not tags_output:
                    print("No valid tags found after filtering")
                    continue
                    
                print("\nFinal tags:", tags_output)

                # Update the prompt
                model = root.model_for_active_document()
                if not model:
                    print("No active document model")
                    return False

                # Save current prompt for undo
                current_prompt = model.regions.positive
                self._artist_history.append(current_prompt)

                # Keep existing artist tag if present
                artist_tags = ""
                artist_match = re.match(r'(artist:[^,]+,\s*)', current_prompt)
                if artist_match:
                    artist_tags = artist_match.group(1)
                    self._last_artist = artist_match.group(1).replace("artist:", "").replace(",", "").strip()
                
                # Join tags and update prompt
                new_prompt = artist_tags + ', '.join(tags_output)
                print("\nFinal prompt:", new_prompt)
                
                model.regions.positive = new_prompt.strip()
                return True
                
            except Exception as e:
                print(f"\nError in attempt {attempt + 1}:")
                print(f"Type: {type(e).__name__}")
                print(f"Error: {str(e)}")
                continue
        
        print("Failed all attempts to get tags")
        return False

    def update_styles(self):
        comfy = root.connection.client_if_connected
        self._styles = filter_supported_styles(Styles.list().filtered(), comfy)
        with SignalBlocker(self._combo):
            self._combo.clear()
            for style in self._styles:
                icon = theme.checkpoint_icon(resolve_arch(style, comfy))
                self._combo.addItem(icon, style.name, style.filename)
            if self._value in self._styles:
                self._combo.setCurrentText(self._value.name)
            elif len(self._styles) > 0:
                self._value = self._styles[0]
                self._combo.setCurrentIndex(0)

    def change_style(self):
        style = self._styles[self._combo.currentIndex()]
        if style != self._value:
            self._value = style
            self.value_changed.emit(style)

    def change_quality(self):
        quality = SamplingQuality(self._quality_combo.currentData())
        self.quality_changed.emit(quality)

    def show_settings(self):
        from .settings import SettingsDialog
        SettingsDialog.instance().show(self._value)

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, style: Style):
        if style != self._value:
            self._value = style
            self._combo.setCurrentText(style.name)


class ResizeHandle(QWidget):
    """A small resize handle that appears at the bottom of the prompt widget."""

    handle_dragged = pyqtSignal(int)

    def __init__(self, parent: QWidget):
        super().__init__(parent)
        self.setCursor(Qt.CursorShape.SizeVerCursor)
        self.setFixedSize(22, 8)
        self._dragging = False

    def mousePressEvent(self, a0: QMouseEvent | None) -> None:
        if ensure(a0).button() == Qt.MouseButton.LeftButton:
            self._dragging = True

    def mouseReleaseEvent(self, a0: QMouseEvent | None) -> None:
        self._dragging = False

    def mouseMoveEvent(self, a0: QMouseEvent | None) -> None:
        if not self._dragging:
            return
        y_pos = self.mapToParent(ensure(a0).pos()).y()
        self.handle_dragged.emit(y_pos)

    def paintEvent(self, a0: QPaintEvent | None) -> None:
        if not self.isVisible():
            return
        painter = QPainter(self)
        painter.setPen(self.palette().color(QPalette.ColorRole.PlaceholderText).lighter(100))
        painter.setBrush(painter.pen().color())
        w, h = self.width(), self.height()
        for i, x in enumerate(range(2, w - 1, 3)):
            y = 2 * h // 3 if i % 2 == 0 else h // 3
            painter.drawEllipse(x - 1, y - 1, 2, 2)


class TextPromptWidget(QPlainTextEdit):
    activated = pyqtSignal()
    text_changed = pyqtSignal(str)
    handle_dragged = pyqtSignal(int)

    def __init__(self, line_count=2, is_negative=False, parent=None):
        super().__init__(parent)
        self._line_count = line_count
        self._is_negative = is_negative

        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setTabChangesFocus(True)
        self.setFrameStyle(QFrame.Shape.NoFrame)

        self._completer = PromptAutoComplete(self)
        self.textChanged.connect(self.notify_text_changed)

        self._resize_handle: ResizeHandle | None = None

        palette: QPalette = self.palette()
        self._base_color = palette.color(QPalette.ColorRole.Base)
        self.is_negative = is_negative
        self.line_count = line_count

    def event(self, e: QEvent | None):
        assert e is not None
        # Ctrl+Backspace should be handled by QPlainTextEdit, not Krita.
        if e.type() == QEvent.Type.ShortcutOverride:
            assert isinstance(e, QKeyEvent)
            if e.matches(QKeySequence.StandardKey.DeleteStartOfWord):
                e.accept()
        return super().event(e)

    def keyPressEvent(self, e: QKeyEvent | None):
        assert e is not None
        if self._completer.is_active and e.key() in PromptAutoComplete.action_keys:
            e.ignore()
            return

        self.handle_weight_adjustment(e)

        if e.key() == Qt.Key.Key_Return and e.modifiers() == Qt.KeyboardModifier.ShiftModifier:
            self.activated.emit()
        else:
            super().keyPressEvent(e)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        if self._resize_handle:
            self._place_resize_handle()

    def focusOutEvent(self, e):
        super().focusOutEvent(e)
        if scroll := self.verticalScrollBar():
            scroll.triggerAction(QScrollBar.SliderAction.SliderToMinimum)

    def focusNextPrevChild(self, next):
        if self._completer.is_active:
            return False
        return super().focusNextPrevChild(next)

    def notify_text_changed(self):
        self._completer.check_completion()
        self.text_changed.emit(self.text)

    @property
    def text(self):
        return self.toPlainText()

    @text.setter
    def text(self, value: str):
        if value == self.text:
            return
        with SignalBlocker(self):  # avoid auto-completion on non-user input
            self.setPlainText(value)

    @property
    def is_resizable(self):
        return self._resize_handle is not None

    @is_resizable.setter
    def is_resizable(self, value: bool):
        if value and self._resize_handle is None:
            self._resize_handle = ResizeHandle(self)
            self._resize_handle.handle_dragged.connect(self.handle_dragged)
            self._place_resize_handle()
            self._resize_handle.show()
        if not value and self._resize_handle is not None:
            self._resize_handle.handle_dragged.disconnect(self.handle_dragged)
            self._resize_handle.deleteLater()
            self._resize_handle = None

    def _place_resize_handle(self):
        if self._resize_handle:
            rect = self.geometry()
            self._resize_handle.move(
                (rect.width() - self._resize_handle.width()) // 2,
                rect.height() - self._resize_handle.height(),
            )

    @property
    def line_count(self):
        return self._line_count

    @line_count.setter
    def line_count(self, value: int):
        self._line_count = value
        fm = QFontMetrics(ensure(self.document()).defaultFont())
        self.setFixedHeight(fm.lineSpacing() * value + 10)

    @property
    def is_negative(self):
        return self._is_negative

    @is_negative.setter
    def is_negative(self, value: bool):
        self._is_negative = value
        if not value:
            self.setPlaceholderText(_("Describe the content you want to see, or leave empty."))
        else:
            self.setPlaceholderText(_("Describe content you want to avoid."))

        if value:
            self.setContentsMargins(0, 2, 0, 2)
            self.setFrameStyle(QFrame.Shape.StyledPanel)
            self.setStyleSheet("QFrame { background: rgba(255, 0, 0, 15); }")
        else:
            self.setFrameStyle(QFrame.Shape.NoFrame)

    @property
    def has_focus(self):
        return self.hasFocus()

    @has_focus.setter
    def has_focus(self, value: bool):
        if value:
            self.setFocus()

    def move_cursor_to_end(self):
        cursor = self.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self.setTextCursor(cursor)

    def handle_weight_adjustment(self, event: QKeyEvent):
        """Handles Ctrl + (arrow key up / arrow key down) attention weight adjustment."""
        if event.key() in [Qt.Key.Key_Up, Qt.Key.Key_Down] and (
            event.modifiers() & Qt.Modifier.CTRL
        ):
            cursor = self.textCursor()
            text = self.toPlainText()

            if cursor.hasSelection():
                start = cursor.selectionStart()
                end = cursor.selectionEnd()
            else:
                start, end = select_on_cursor_pos(text, cursor.position())

            target_text = text[start:end]
            text_after_edit = edit_attention(target_text, event.key() == Qt.Key.Key_Up)
            text = text[:start] + text_after_edit + text[end:]
            self.setPlainText(text)
            cursor = self.textCursor()
            cursor.setPosition(min(start + len(text_after_edit), len(text)))
            cursor.setPosition(min(start, len(text)), QTextCursor.KeepAnchor)
            self.setTextCursor(cursor)


class StrengthSnapping:
    model: Model

    def __init__(self, model: Model):
        self.model = model

    def get_steps(self) -> tuple[int, int]:
        is_live = self.model.workspace is Workspace.live
        if self.model.workspace is Workspace.animation:
            is_live = self.model.animation.sampling_quality is SamplingQuality.fast
        return self.model.style.get_steps(is_live=is_live)

    def nearest_percent(self, value: int) -> int | None:
        _, max_steps = self.get_steps()
        steps, start_at_step = self.apply_strength(value)
        return snap_to_percent(steps, start_at_step, max_steps=max_steps)

    def apply_strength(self, value: int) -> tuple[int, int]:
        min_steps, max_steps = self.get_steps()
        strength = value / 100
        return apply_strength(strength, steps=max_steps, min_steps=min_steps)


# SpinBox variant that allows manually entering strength values,
# but snaps to model_steps on step actions (scrolling, arrows, arrow keys).
class StrengthSpinBox(QSpinBox):
    snapping: StrengthSnapping | None

    def __init__(self, parent=None):
        super().__init__(parent)
        self.snapping = None
        # for manual input
        self.setMinimum(1)
        self.setMaximum(100)

    def stepBy(self, steps):
        value = max(self.minimum(), min(self.maximum(), self.value() + steps))
        if self.snapping is not None:
            # keep going until we hit a new snap point
            current_point = self.nearest_snap_point(self.value())
            while self.nearest_snap_point(value) == current_point and value > 1:
                value += 1 if steps > 0 else -1
            value = self.nearest_snap_point(value)
        self.setValue(value)

    def nearest_snap_point(self, value: int) -> int:
        assert self.snapping
        return self.snapping.nearest_percent(value) or (int(value / 5) * 5)


class StrengthWidget(QWidget):
    _model: Model | None = None
    _value: int = 100

    value_changed = pyqtSignal(float)

    def __init__(self, slider_range: tuple[int, int] = (1, 100), prefix=True, parent=None):
        super().__init__(parent)
        self._layout = QHBoxLayout()
        self._layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(self._layout)

        self._slider = QSlider(Qt.Orientation.Horizontal, self)
        self._slider.setMinimum(slider_range[0])
        self._slider.setMaximum(slider_range[1])
        self._slider.setValue(self._value)
        self._slider.setSingleStep(5)
        self._slider.valueChanged.connect(self.slider_changed)

        self._input = StrengthSpinBox(self)
        self._input.setValue(self._value)
        if prefix:
            self._input.setPrefix(_("Strength") + ": ")
        self._input.setSuffix("%")
        self._input.setSpecialValueText(_("Off"))
        self._input.valueChanged.connect(self.notify_changed)

        settings.changed.connect(self.update_suffix)

        self._layout.addWidget(self._slider)
        self._layout.addWidget(self._input)

    def slider_changed(self, value: int):
        if self._input.snapping is not None:
            value = self._input.snapping.nearest_percent(value) or value
        self.notify_changed(value)

    def notify_changed(self, value: int):
        if self._update_value(value):
            self.value_changed.emit(self.value)

    def _update_value(self, value: int):
        with SignalBlocker(self._slider), SignalBlocker(self._input):
            self._slider.setValue(value)
            self._input.setValue(value)
        if value != self._value:
            self._value = value
            self.update_suffix()
            return True
        return False

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model: Model):
        if self._model:
            self._model.style_changed.disconnect(self.update_suffix)
            self._model.animation.sampling_quality_changed.disconnect(self.update_suffix)
        self._model = model
        self._model.style_changed.connect(self.update_suffix)
        self._model.animation.sampling_quality_changed.connect(self.update_suffix)
        self._input.snapping = StrengthSnapping(self._model)
        self.update_suffix()

    @property
    def value(self):
        return self._value / 100

    @value.setter
    def value(self, value: float):
        if value == self.value:
            return
        self._update_value(round(value * 100))

    def update_suffix(self):
        if not self._input.snapping or not settings.show_steps:
            self._input.setSuffix("%")
            return

        steps, start_at_step = self._input.snapping.apply_strength(self._value)
        self._input.setSuffix(f"% ({steps - start_at_step}/{steps})")


class WorkspaceSelectWidget(QToolButton):
    _icons = {
        Workspace.generation: theme.icon("workspace-generation"),
        Workspace.upscaling: theme.icon("workspace-upscaling"),
        Workspace.live: theme.icon("workspace-live"),
        Workspace.animation: theme.icon("workspace-animation"),
        Workspace.custom: theme.icon("workspace-custom"),
    }

    _value = Workspace.generation

    def __init__(self, parent):
        super().__init__(parent)

        menu = QMenu(self)
        menu.addAction(self._create_action(_("Generate"), Workspace.generation))
        menu.addAction(self._create_action(_("Upscale"), Workspace.upscaling))
        menu.addAction(self._create_action(_("Live"), Workspace.live))
        menu.addAction(self._create_action(_("Animation"), Workspace.animation))
        menu.addAction(self._create_action(_("Graph"), Workspace.custom))

        self.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
        self.setMenu(menu)
        self.setPopupMode(QToolButton.InstantPopup)
        self.setToolTip(
            _("Switch between workspaces: image generation, upscaling, live preview and animation.")
        )
        self.setMinimumWidth(int(self.sizeHint().width() * 1.6))
        self.value = Workspace.generation

    def paintEvent(self, a0):
        _paint_tool_drop_down(self)

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, workspace: Workspace):
        self._value = workspace
        self.setIcon(self._icons[workspace])

    def _create_action(self, name: str, workspace: Workspace):
        action = QAction(name, self)
        action.setIcon(self._icons[workspace])
        action.setIconVisibleInMenu(True)
        action.triggered.connect(actions.set_workspace(workspace))
        return action


class GenerateButton(QPushButton):
    def __init__(self, kind: JobKind, parent: QWidget):
        super().__init__(parent)
        self.model = root.active_model
        self._operation = _("Generate")
        self._kind = kind
        self._cost = 0
        self._cost_icon = theme.icon("interstice")
        self._seed_icon = theme.icon("seed")
        self._resolution_icon = theme.icon("resolution-multiplier")
        self.setAttribute(Qt.WidgetAttribute.WA_Hover)

    @property
    def operation(self):
        return self._operation

    @operation.setter
    def operation(self, value: str):
        self._operation = value
        self.update()

    def minimumSizeHint(self):
        fm = self.fontMetrics()
        return QSize(fm.width(self._operation) + 40, 12 + int(1.3 * fm.height()))

    def enterEvent(self, a0: QEvent | None):
        if client := root.connection.client_if_connected:
            if client.user:
                self._cost = self.model.estimate_cost(self._kind)

    def leaveEvent(self, a0: QEvent | None):
        self._cost = 0

    def paintEvent(self, a0: QPaintEvent | None) -> None:
        opt = QStyleOption()
        opt.initFrom(self)
        opt.state |= QStyle.StateFlag.State_Sunken if self.isDown() else 0
        painter = QPainter(self)
        fm = self.fontMetrics()
        style = ensure(self.style())
        rect = self.rect()
        pixmap = self.icon().pixmap(int(fm.height() * 1.3))
        is_hover = int(opt.state) & QStyle.StateFlag.State_MouseOver
        element = QStyle.PrimitiveElement.PE_PanelButtonCommand
        vcenter = Qt.AlignmentFlag.AlignVCenter
        content_width = fm.width(self._operation) + 5 + pixmap.width()
        content_rect = rect.adjusted(int(0.5 * (rect.width() - content_width)), 0, 0, 0)
        style.drawPrimitive(element, opt, painter, self)
        style.drawItemPixmap(painter, content_rect, vcenter, pixmap)
        content_rect = content_rect.adjusted(pixmap.width() + 5, 0, 0, 0)
        style.drawItemText(painter, content_rect, vcenter, self.palette(), True, self._operation)

        cost_width = 0
        if is_hover and self._cost > 0:
            pixmap = self._cost_icon.pixmap(fm.height())
            text_width = fm.width(str(self._cost))
            cost_width = text_width + 16 + pixmap.width()
            cost_rect = rect.adjusted(rect.width() - cost_width, 0, 0, 0)
            painter.setOpacity(0.3)
            painter.drawLine(
                cost_rect.left(), cost_rect.top() + 6, cost_rect.left(), cost_rect.bottom() - 6
            )
            painter.setOpacity(0.7)
            cost_rect = cost_rect.adjusted(6, 0, 0, 0)
            style.drawItemText(painter, cost_rect, vcenter, self.palette(), True, str(self._cost))
            cost_rect = cost_rect.adjusted(text_width + 4, 0, 0, 0)
            style.drawItemPixmap(painter, cost_rect, vcenter, pixmap)

        seed_width = 0
        if is_hover and self.model.fixed_seed:
            pixmap = self._seed_icon.pixmap(fm.height())
            seed_width = pixmap.width() + 4
            seed_rect = rect.adjusted(rect.width() - cost_width - seed_width, 0, 0, 0)
            style.drawItemPixmap(painter, seed_rect, vcenter, pixmap)

        if is_hover and self.model.resolution_multiplier != 1.0:
            pixmap = self._resolution_icon.pixmap(fm.height())
            resolution_rect = rect.adjusted(
                rect.width() - cost_width - seed_width - pixmap.width() - 4, 0, 0, 0
            )
            style.drawItemPixmap(painter, resolution_rect, vcenter, pixmap)


class ErrorBox(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._error = no_error
        self._original_error = ""

        self.setObjectName("errorBox")
        self.setFrameStyle(QFrame.Shape.StyledPanel)

        self._label = QLabel(self)
        self._label.setWordWrap(True)
        self._label.setOpenExternalLinks(True)
        self._label.setTextFormat(Qt.TextFormat.RichText)

        self._copy_button = QToolButton(self)
        self._copy_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
        self._copy_button.setIcon(Krita.instance().icon("edit-copy"))
        self._copy_button.setToolTip(_("Copy error message to clipboard"))
        self._copy_button.setAutoRaise(True)
        self._copy_button.clicked.connect(self._copy_error)

        self._recharge_button = QToolButton(self)
        self._recharge_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self._recharge_button.setText(_("Charge"))
        self._recharge_button.setIcon(theme.icon("interstice"))
        self._recharge_button.clicked.connect(self._recharge)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.addWidget(self._label)
        layout.addWidget(self._copy_button)
        layout.addWidget(self._recharge_button)

        self.reset()

    def reset(self, color: str = theme.red):
        self._copy_button.setVisible(False)
        self._recharge_button.setVisible(False)
        self._label.setStyleSheet(f"color: {color};")
        if color == theme.red:
            self.setStyleSheet("QFrame#errorBox { border: 1px solid #a01020; }")
        else:
            self.setStyleSheet(None)
        self.setVisible(False)

    @property
    def error(self):
        return self._error

    @error.setter
    def error(self, error: Error):
        self.reset()
        self._error = error
        self._original_error = error.message if error else ""
        if error.kind is ErrorKind.insufficient_funds:
            self._show_payment_error(error.data)
        elif error:
            self._show_error(error.message)

    def _show_error(self, text: str):
        if text.count("\n") > 3:
            lines = text.split("\n")
            n = 1
            text = lines[-n]
            while n < len(lines) and text.strip() == "":
                n += 1
                text = lines[-n]
        if len(text) > 60 * 3:
            text = text[: 60 * 2] + " [...] " + text[-60:]
        self._label.setText(text)
        if text != self._original_error:
            self._label.setToolTip(self._original_error)
        self._copy_button.setVisible(True)
        self.setVisible(True)

    def _show_payment_error(self, data: dict[str, Any] | None):
        self.reset(theme.yellow)
        message = "Insufficient funds"
        if data:
            message = _(
                "Insufficient funds - generation would cost {cost} tokens. Remaining tokens: {tokens}",
                cost=data["cost"],
                tokens=data["credits"],
            )
        self._label.setText(message)
        self._recharge_button.setVisible(True)
        self.setVisible(True)

    def _copy_error(self):
        if clipboard := QGuiApplication.clipboard():
            clipboard.setText(self._original_error)

    def _recharge(self):
        QDesktopServices.openUrl(QUrl("https://www.interstice.cloud/user"))


def create_wide_tool_button(icon_name: str, text: str, parent=None):
    button = QToolButton(parent)
    button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
    button.setIcon(theme.icon(icon_name))
    button.setToolTip(text)
    button.setAutoRaise(True)
    icon_height = button.iconSize().height()
    button.setIconSize(QSize(int(icon_height * 1.25), icon_height))
    return button


def create_framed_label(text: str, parent=None):
    frame = QFrame(parent)
    frame.setFrameStyle(QFrame.Shape.StyledPanel | QFrame.Shadow.Plain)
    label = QLabel(parent=frame)
    label.setText(text)
    frame_layout = QHBoxLayout()
    frame_layout.setContentsMargins(4, 2, 4, 2)
    frame_layout.addWidget(label)
    frame.setLayout(frame_layout)
    return frame, label


def _paint_tool_drop_down(widget: QToolButton, text: str | None = None):
    opt = QStyleOption()
    opt.initFrom(widget)
    painter = QPainter(widget)
    style = ensure(widget.style())
    rect = widget.rect()
    pixmap = widget.icon().pixmap(int(rect.height() * 0.75))
    element = QStyle.PrimitiveElement.PE_Widget
    if int(opt.state) & QStyle.StateFlag.State_MouseOver:
        element = QStyle.PrimitiveElement.PE_PanelButtonCommand
    style.drawPrimitive(element, opt, painter, widget)
    style.drawItemPixmap(
        painter,
        rect.adjusted(4, 0, 0, 0),
        Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
        pixmap,
    )
    if text:
        text_rect = rect.adjusted(pixmap.width() + 4, 0, 0, 0)
        style.drawItemText(
            painter, text_rect, Qt.AlignmentFlag.AlignVCenter, widget.palette(), True, text
        )
    painter.translate(int(0.5 * rect.width() - 10), 0)
    style.drawPrimitive(QStyle.PrimitiveElement.PE_IndicatorArrowDown, opt, painter)

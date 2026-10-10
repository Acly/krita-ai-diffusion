import os
import subprocess
import sys
from pathlib import Path


def test_style_search_ui():
    # The main test process owns a QCoreApplication. Widgets need a separate
    # QApplication, so run the real keyboard/mouse cases in their own process.
    result = subprocess.run(
        [sys.executable, str(Path(__file__).with_name("ui_style_search.py"))],
        env={**os.environ, "QT_QPA_PLATFORM": "offscreen"},
        capture_output=True,
        check=False,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr

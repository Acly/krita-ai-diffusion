"""Generative AI plugin for Krita"""

__version__ = "1.43.0"

import importlib.util

if not importlib.util.find_spec(".websockets.src", "ai_diffusion"):
    raise ImportError(
        "Could not find websockets module. This indicates that it was not installed with the"
        " plugin. Please make sure to download a plugin release package (NOT just the source!). You"
        " can find the latest release package here:"
        " https://github.com/Acly/krita-ai-diffusion/releases"
    )


# The following imports depend on the code running inside Krita, so they cannot be imported in tests.
_krita_spec = importlib.util.find_spec("krita")
if _krita_spec is not None:
    origin = getattr(_krita_spec, "origin", "") or ""
    # Avoid treating local helper modules named `krita.py` (e.g. tooling nodes)
    # as the actual Krita application module.
    if not origin.endswith("krita.py"):
        from .extension import AIToolsExtension as AIToolsExtension

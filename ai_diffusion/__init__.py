"""Generative AI plugin for Krita"""

__version__ = "1.50.0"

import importlib.util

if not importlib.util.find_spec(".websockets.src", "ai_diffusion"):
    raise ImportError(
        "Could not find websockets module. This indicates that it was not installed with the"
        " plugin. Please make sure to download a plugin release package (NOT just the source!). You"
        " can find the latest release package here:"
        " https://github.com/Acly/krita-ai-diffusion/releases"
    )

# The following imports depend on the code running inside Krita, so the cannot be imported in tests.
if importlib.util.find_spec("krita"):
    import krita

    krita_ver = krita.Krita.instance().version()
    if not krita_ver.startswith("5"):
        raise ImportError(f"This Plugin is for Krita 5.x, but you are using Krita {krita_ver}.")

    from .extension import AIToolsExtension as AIToolsExtension

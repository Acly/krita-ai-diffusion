# Image generation plugin for Krita

Krita is an image editor and painting app. `krita-ai-diffusion` is a Python plugin that adds
image generation functionality via diffusion models.


## Overview

The plugin runs within Krita's embedded Python interpreter. It may not use 3rd party
libraries except for Qt and the websockets library.

### Inference
* `api.py` has data structures for an inference request (`WorkflowInput`) - everything relevant to image generation MUST be contained here
* `comfy_client.py` is a HTTP/WebSocket client that connects to a ComfyUI server to fulfill requests
* `cloud_client.py` is a HTTP client that connects to an image generation service to fulfill requests
* `workflow.py` transforms `WorkflowInput` into ComfyUI workflows
* `server.py` contains an installer for a ComfyUI server that will run in the background alongside the plugin
* `resources.py` is a central location that enumerates supported AI models and extensions

### UI
The UI is separated into distinct workspaces:
* "Generation" for launching asynchronous image jobs and viewing their results
* "Live" for automatically generating preview output after every change
* "Upscale" for diffusion-based super-resolution tasks
* "Custom/Graph" for importing and running custom user ComfyUI workflows
* "Animation" for batch-processing image frames

Code is separated into:
* `ai_diffusion/model`: Model classes which hold observable UI state and implement actions
* `ai_diffusion/ui`: Qt widgets for the user interface
* Persistence layer which loads/stores state in files or Krita documents
  * `persistence.py`, `settings.py`, `files.py`, ...

### Image manipulation
Helpers and tools for Krita's objects (`document.py`, `layer.py`) and
general image algorithms (`image.py`, `resolution.py`).


## Commands

Activate the virtual environment with `source .venv/bin/activate` for all commands.

Always run checks, formatting and linting:
```
ruff check
ruff format
pyright
```

### Tests

Run tests with these priorities:
1. specific files with `pytest tests/{test_file.py}`
2. all fast (no inference) tests with `pytest tests --ci`
3. *only after changing workflows* (includes inference): `pytest tests/test_workflow.py`
4. *only after changing cloud client* (includes inference): `pytest tests/test_workflow.py --cloud`
5. *only after changing installer* (slow tests): `pytest tests/test_server.py --test-install`


## Code Guidelines

* Avoid docstrings/comments for small functions and intuitive code
* Use `snake_case` (lowercase) for enums and constants
* Put imports at the beginning of the file unless there is a good reason not to

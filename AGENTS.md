# Image generation plugin for Krita

Krita is an image editor and painting app. `krita-ai-diffusion` is a Python plugin that adds
image generation functionality via diffusion models.


## Overview

The plugin runs within Krita's embedded Python interpreter. It may not use 3rd party
libraries except for Qt and the websockets library.

### Architecture

* `ai_diffusion/backend/` - code related to run AI models via ComfyUI
  * `api.py` - serializable data structures for an inference request (`WorkflowInput`) - everything relevant to image generation MUST be contained here
  * `comfy_client.py` - HTTP/WebSocket client that connects to a ComfyUI server to fulfill requests
  * `cloud_client.py` - HTTP client that connects to an image generation service to fulfill requests
  * `workflow.py` - transforms `WorkflowInput` into ComfyUI workflows
  * `server.py` - installer for a ComfyUI server that will run in the background alongside the plugin
  * `resources.py` - central location that enumerates supported AI models and extensions
* `ai_diffusion/ui/` - Qt widgets that make up the UI
  * `generation.py` - workspace for launching asynchronous image jobs and viewing their results in a history list
  * `live.py` - workspace for automatically generating preview output after every change
  * `upscale.py` - workspace for diffusion-based super-resolution tasks
  * `custom_workflow.py` - workspace for importing and running custom user ComfyUI workflows
  * `animation.py` - for batch-processing image frames
* `ai_diffusion/model/` - classes which hold the observable app state and actions, often there is a 1:1 relationship between `model/` and `ui/` files
  * `connection.py` - manages the backend clients, shared across all documents
  * `model.py` - the document model, almost all state is kept per opened document
  * `jobs.py` - queued and finished diffusion jobs
  * `control.py` - state related to control layers (images used as input to ControlNet models)
  * `region.py` - state related to regional prompts (images whose alpha coverage is used as attention masks for diffusion)
* `ai_diffusion/` - Krita wrapper classes, persistence layer, image manipulation, and general helpers and utils
  * `document.py` - used to interact with Krita documents
  * `persistence.py` - stores/loads document state to .kra files
  * `settings.py` - stores plugin settings in settings.json
  * `style.py` - stores `Style` objects, each references a diffusion model along with parameters, LoRA, and default prompts
  * `files.py` - lists of files, both local filesystem or remote on a server
  * `image.py` - `Image`, `Mask`, `Extent` and `Bounds` objects, image manipulation
  * `text.py` - text prompt processing tools


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

### UI Tests

There are no dedicated tests for UI. The following command can be used as a quick check
that the UI code runs without errors:
```
python scripts/design.py --exit
```

The design preview can also be scripted to test and iterate on UI (see `scripts/uitest.py`):
```
# wait for server connection, then render the ImageDiffusionWidget to a file and exit
python scripts/design.py --wait connected --screenshot shot.png --exit

# run a custom script that drives the UI (clicks, typing, waits, screenshots)
python scripts/design.py --script my_test.py --exit
```
A script is a Python file defining `run(auto)` (may be async), where `auto` provides
`wait_connected()/wait_for()/wait_visible()`, `click()/type_text()/set_text()`
(widgets are found by objectName, class name or button text), and `screenshot()`.
Example:
```python
async def run(auto):
    await auto.wait_connected()
    auto.click("Open document")
    auto.click("Add layer")
    await auto.wait_visible("TextPromptWidget")
    auto.type_text("TextPromptWidget", "a cute robot")
    auto.screenshot("shot.png")
```

### Creating a worktree

1. create a worktree of the `krita-ai-diffusion` repository (this repository)
2. create a worktree of the `diffusion-service` repository located in `service` folder if present.
   place it inside the worktree from step 1, as a folder named `service`, so that the structure matches.
3. symlink the `scripts/downloads` folder into the new worktree.
4. symlink `service/web/.env.local` and `service/api/.dev.vars` if they exist.
5. create a fresh venv with `uv` inside the new worktree. match python version and packages from the venv
   in the main clone rather than installing from requirements.


## Code Guidelines

* Avoid docstrings/comments for small functions and intuitive code
* Use `snake_case` (lowercase) for enums and constants
* Put imports at the beginning of the file unless there is a good reason not to
* Omit type annotations that can be inferred (arguments with default, return types)

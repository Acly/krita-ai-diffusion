# Contributing guide

Patches and contributions are very welcome!

## Reporting issues

If you are facing an issue or want to report a bug, please check [existing issues](https://github.com/Acly/krita-ai-diffusion/issues).

Having a look at the log files can provide additional information for troubleshooting. You can find them in the `.logs` subfolder of the plugin installation folder (`ai_diffusion`). There is also a link in the plugin's connection settings.

When you open a new issue, please attach the log files. Other useful information to include: OS (Windows/Linux/Mac), Krita version, Plugin version, GPU vendor.

## Translations (Localization)

You can create or improve a translation for the plugin interface into your language.
Language files are stored in the `ai_diffusion/langauge` folder. Each translation has its own
file, typically named using a language code (eg. `en.json` for English).
If no file for your language exists, you can use the `new_language.json.template` file and
rename it. It will show up in the plugin's language settings after a restart.

You can check [existing translations here](https://github.com/Acly/krita-ai-diffusion/tree/main/ai_diffusion/language) - this might be more up-to-date than your local installation!

To edit a localization file, open the file in a text editor and provide translations for
each of the english text strings. For example, to provide a German translation, it could look
like this:
```json
{
  "id": "de",
  "name": "Deutsch",
  "translations": {
    "(Custom)": "(Benutzerdefiniert)",
    "<No text prompt>": "<Keine Text-Eingabe>",
    "Active": "Aktiv",
    "Add Content": "Inhalte hinzufügen",
    "Could not find LoRA '{lora}' used by sampler preset '{name}'": "LoRA '{lora}' konnte nicht gefunden werden, wird aber von Sampler '{name}' benutzt",
    ...
  }
}
```

**Important:** `{placeholders}` must be left unmodified! They will be replaced with actual content during runtime.

To update an existing translation (eg. after the plugin has been updated and new text was added)
simply search for entries which are `null`. These are valid, but not translated yet.

## Contributing code

For bigger changes, it makes sense to create an issue first to discuss a proposal before time is comitted.

You can submit your changes by opening a [pull request](https://github.com/Acly/krita-ai-diffusion/pulls).

### Plugin development

The easiest way to run a development version of the plugin is to use symlinks:
1. `git clone` the repository into a location of your choice
1. `git submodule update --init`
1. in the pykrita folder where Krita expects plugins:
   * create a symlink to the `ai_diffusion` folder
   * create a symlink to `ai_diffusion.desktop`

### Code formatting

The codebase uses [ruff](https://docs.astral.sh/ruff/) for linting. You can
use an IDE integration, or check locally by running in the repository root:
```
ruff format
ruff check
```

### Code style

Code style follows the official Python recommendations. Only exception: no `ALL_CAPS`.

### Type checking

Type annotations should be used where types can't be inferred. Basic type checks are enabled for the project and should not report errors.

The `Krita` module is special in that it is usually only available when running inside Krita. To make type checking work an interface file is located in `scripts/typeshed`.

You can run `pyright` from the repository root to perform type checks on the entire codebase. This is also done by the CI.

### Debug

The project includes a `launch.json` for VSCode which is configured to attach to
a running Krita process. This allows to use the visual debugger for exceptions,
breakpoints, inspecting and stepping through the code. Start debugging via the
"Run and Debug" tab (F5).

The way it works is:
1. `debugpy` is added to the `ai_diffusion` folder as a git submodule to make it
   available inside Krita's embedded Python
1. `extension.py` starts a debug server if the `debugpy` module is present
   (skipped for release deployments)
2. VSCode (or more generally any `debugpy` client) attaches to the server

You can also add breakpoints inside the code with `import debugpy; debugpy.breakpoint()`.

### Tests

There are tests, although with some caveats currently.

To install dependencies for tests run:
```
pip install -r requirements.txt
```
Tests are run from the project root via pytest:
```
pytest tests
```
Some tests require a running ComfyUI server. This should be automated... but for now it's not.

### What is tested
Generating images is tested. Because it takes a lot of time the number of tests is limited. Images are not compared in most cases, as they tend to frequently change with updates to dependencies.

Functionality which uses Krita's API is _not_ tested. It just doesn't work outside Krita without a comprehensive mock.

UI is _not_ tested. Because UI.

Everything else has tests. Mostly. If effort is reasonable, tests are expected. They help being confident about making changes.

### Testing the installer

Testing changes to the installer is annoying because of the file sizes involved. There are some things that help. You can preload model files with the following script:
```
python scripts/download_models.py --minimal scripts/downloads
```
This will download the minimum required models and store them in `scripts/downloads`.

The following command does some automated testing for installation and upgrade. It starts a local file server which pulls preloaded models, so it's reasonably fast and doesn't download the entire internet.
```
pytest tests/test_server.py --test-install -vs
```
You can also run the file server manually. Then you can start Krita with the `HOSTMAP` environment variable set, and it will map HuggingFace & civit.ai links to localhost.
```
python scripts/file_server.py

HOSTMAP=1 /your/krita/install/krita

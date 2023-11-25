# Contributing guide

Patches and contributions are very welcome!

## Reporting issues

If you are facing an issue or want to report a bug, please check [existing issues](https://github.com/Acly/krita-ai-diffusion/issues).

Having a look at the log files can provide additional information for troubleshooting. You can find them in the `.logs` subfolder of the plugin installation folder (`ai_diffusion`). There is also a link in the plugin's connection settings.

When you open a new issue, please attach the log files. Other useful information to include: OS (Windows/Linux/Mac), Krita version, Plugin version, GPU vendor.

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

The codebase uses [black](https://github.com/psf/black) for formatting. The project root contains a `pyproject.toml` to configure the line length, it should be picked up automatically.

### Code style

Code style follows the official Python recommendations. Only exception: no `ALL_CAPS`.

### Type checking

Type annotations should be used where types can't be inferred. Basic type checks are enabled for the project and should not report errors.

The `Krita` module is special in that it is usually only available when running inside Krita. To make type checking work, include `scripts/typeshed` in your `PYTHONPATH`.

Configuration for VSCode with Pylance (.vscode/settings.json):
```
{
  "python.analysis.typeCheckingMode": "basic",
  "python.analysis.exclude": [
    "scripts/typeshed/**",
    "ai_diffusion/websockets/**"
  ]
}
```

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
Generating images is tested. Because it takes a lot of time the number of tests is limited. Because it's very random, images are not compared (but this can be solved with consistent installation and fixed seeds).

Functionality which uses Krita's API is _not_ tested. It just doesn't work outside Krita without a comprehensive mock.

UI is _not_ tested. Because UI.

Everything else has tests. Mostly. If effort is reasonable, tests are expected. They help being confident about making changes.

### Testing the installer

Testing changes to the installer is annoying because of the file sizes involved. There are some things that help. You can preload all model files with the following script:
```
python scripts/docker.py
```
This _can_ be used to build a docker image afterwards, but it's not necessary for testing.

The following command does some automated testing for installation and upgrade. It starts a local file server which pulls preloaded models, so it's reasonably fast and doesn't download the entire internet.
```
pytest tests/test_server.py --test-install -vs
```
You can also run the file server manually. Then you can start Krita with the `HOSTMAP` environment variable set, and it will map HuggingFace & civit.ai links to localhost.
```
python scripts/file_server.py

HOSTMAP=1 /your/krita/install/krita
```
Note that the mock file server likes to transmit corrupted files if they are very large (eg. SDXL checkpoint)... not sure why (?)

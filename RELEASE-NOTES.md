# 1.53.0-reloaded.5 — Unofficial Combined Build

An independent community build by **Teutonick**, combining **Acly's Krita AI Diffusion
1.53.0** with **FatGuy84's Krita AI Reloaded** and Krea 2 Edit V2 integration.
This is not an official release of either upstream project or Krea, and is not
endorsed by their authors.

## Download

- [Installable plugin ZIP](https://github.com/Teutonick/krita-ai-diffusion/releases/download/v1.53.0-reloaded.5/krita_ai_diffusion-1.53.0-reloaded.5.zip)
- [Exact corresponding source](https://github.com/Teutonick/krita-ai-diffusion/releases/download/v1.53.0-reloaded.5/krita_ai_diffusion-1.53.0-reloaded.5-source.zip)
- [SHA-256 checksums](https://github.com/Teutonick/krita-ai-diffusion/releases/download/v1.53.0-reloaded.5/SHA256SUMS-1.53.0-reloaded.5.txt)

Import the plugin ZIP through **Tools → Scripts → Import Python Plugin from File**,
then fully restart Krita. Configure workloads and optional components in Managed Server.

## What's included

- Upstream 1.53.0 and Reloaded's visual browsers, prompt tools, wildcard/batch workflows,
  history search/favorites/ratings, export integrations and other interface improvements.
- Krea 2 Turbo, Identity Edit V2, General Edit V2 and Remix integration.
- Managed installation of selected Krea Edit nodes and LoRA Manager dependencies;
  optional download/launch of the official Eagle installer.
- Protection against accidental value changes when scrolling over closed settings dropdowns.
- Session-based media storage and startup cleanup of abandoned plugin-owned sessions.

## Fixes in this build

- Preserve the required empty `preview_url` metadata field: LoRA Manager no longer rejects restored models after a restart.
- Always supplement the manager library with ComfyUI's actual LoRA file list, including models with missing metadata.
- Show a neutral LoRA icon when an online preview is unavailable; retain model selection and subfolder paths.
- Invalidate incomplete library caches written by the previous build. Preview resolution uses the configured CivitAI host and SHA-256 without requiring a metadata sidecar.

- Concurrent requests to start the managed server now share one startup task.
- If the default port is occupied, the managed server chooses an available port without
  stopping the other application. Explicitly requested busy ports produce a clear error.
- ComfyUI's SQLite database now receives an explicit session-specific `--database-url`;
  `--user-directory` alone did not isolate its default database.
- Older Krea UI workflows use server-declared defaults for newly added widgets instead
  of failing with an IndexError.

## Krea 2 Edit V2 model terms

**GPLv3 covers the plugin code, not the model weights.**
The setup dialog presents the V2 License, Krea 2 Community License and NOTICE and
requires explicit acknowledgement before downloading V2 weights. Complete copies
are included in the plugin and installed beside the models.

Free use of the V2 models is non-commercial. Commercial use of the models or their
outputs requires a separate written license from **Lars Bouaraba**; additional Krea
licensing may apply. Accepting the setup dialog does not grant commercial rights.
[Official model terms](https://huggingface.co/conradlocke/krea2-identity-edit/blob/main/LICENSE_V2.md)
· [Commercial licensing contact](mailto:l.bouaraba@gmail.com).

## Media retention

Automatic browser thumbnails stay in memory. Temporary transfers, animation frames,
managed-server media and LoRA Manager preview files use a session folder that is cleaned
on normal shutdown or recovered at the next startup after a crash. Metadata and favorites
may be cached without preview files. Explicit exports and `.kra` history remain.
External/cloud servers and arbitrary custom workflows require session-scoped approval.
Unidentified files in global TEMP and existing user model/output folders are not deleted.

## Source, licenses and credits

Both the plugin ZIP and the exact source ZIP contain the same source manifest. The source
archive includes modified sources, build scripts, the bundled WebSocket dependency and
license notices. No model weights or Eagle executable are embedded in these archives.

The combined plugin code is **GPLv3**. Separately downloaded components and models retain
their own licenses. See the included THIRD-PARTY-NOTICES.md and model license documents.

Thank you to **Acly and the Krita AI Diffusion contributors** for the original plugin,
and to **FatGuy84 and the Krita AI Reloaded contributors** for the workflow and UI
improvements. Credit also belongs to Lars Bouaraba / conradlocke, willmiao and the
contributors to the integrated components.

Report combined-build issues to [this fork](https://github.com/Teutonick/krita-ai-diffusion/issues).
Please reproduce an issue in the official build before reporting it upstream.

Modified on **2026-11-03**. Validation details and remaining acceptance limits are
included in BUILD-REPORT.md. GPU image-generation acceptance is not claimed by these notes.

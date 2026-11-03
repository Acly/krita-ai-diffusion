# Combined build report

Version: `1.53.0-reloaded.5`. Prepared 2026-11-03. No commit or publication performed.

This is Teutonick's unofficial combined fork of Acly's Krita AI Diffusion 1.53.0 and FatGuy84's Reloaded improvements, with Krea 2 integration. See THIRD-PARTY-NOTICES.md and SOURCE-MANIFEST.json for attribution and exact source provenance.

## Changes

- Preserves upstream 1.53, Reloaded UI/features, managed optional-component installation, Krea Turbo/Edit/Remix and Identity/General Edit V2.
- Closed settings dropdowns ignore mouse-wheel changes and forward scrolling to their enclosing scroll area.
- Session-owned temporary media, crash recovery and active-instance protection; automatic preview images stay in RAM or temporary session storage. Manual exports, preferences, model weights and .kra contents retain their intended persistence. Unowned historical files are not deleted.
- Krea V2 setup presents English terms, official document links and two explicit acknowledgements. Bundled, hash-verified LICENSE_V2.md, LICENSE.pdf and NOTICE are copied beside the weights before download. CLI installation also requires explicit acceptance. Acceptance does not grant commercial rights.
- README, About and release notes identify this unofficial combined fork and separately credit Acly and FatGuy84. Matching source and plugin archives contain the same source manifest; checksums accompany both archives.
- Server startup is single-flight. An occupied default port selects an available port; an explicitly requested occupied port produces an actionable error. Existing processes are not killed.
- ComfyUI's SQLite database is explicitly redirected into the session's user directory, avoiding the shared default database.
- Legacy Krea UI workflows receive defaults for newly added node widgets instead of failing with IndexError during import.

## Validation

- Ruff check and format passed. Pyright: zero errors.
- Available CI suite: 333 passed, 111 skipped, 9 deselected. Windows TEMP/TMP redirected into the working copy. Seven deselected cases require a dedicated live test-server fixture; two require working standalone PyQt TLS.
- Real isolated ComfyUI CPU startup passed with an occupied default port and two concurrent start requests: exactly one server process started. Dedicated session database arguments were applied. Both bundled Krea workflows imported against actual node definitions (26 and 23 nodes). The test server stopped and its session directory was removed.
- Qt terms dialog checked: both acknowledgements are required, and withdrawing either disables acceptance. Dialog rendered and visually reviewed.
- Existing Identity V2 weights on the user's server were checked in full: SHA-256 c53843f9451963a6cb3c22dc4975205056b93a6575fb9347dd2bc6e055ef9877 matches the pinned official artifact. No re-download is required for that file.
- Required workflow integration attempt: eight passed, then stopped because the test style requests checkpoint 'not-found', absent on the available server. This does not establish diffusion inference acceptance.
- Required fresh-installer integration attempt stopped at standalone PyQt TLS initialization while downloading ComfyUI. Full first-run installation inside Krita remains unverified.
- Earlier isolated CPU media smoke test generated a 64x64 test image through ETN RAM transfer; standard SaveImage output stayed in the session and was removed at cleanup. This was not model inference.

## Remaining acceptance

Import the plugin ZIP into Krita and restart Krita completely. Verify first-run installation in Krita, actual Krea V2 generation/editing, online previews and Eagle export. GPU generation and a fresh complete installation are not claimed as validated. Eagle's own installation/activation remains interactive. No existing user server was terminated or modified by the isolated startup test.

## LoRA library hotfix (.5)

The supplied logs show LoRA Manager skipping models because restored sidecars were missing its mandatory preview_url field. Metadata retention now keeps that field as an empty string, preserving the schema while discarding local preview paths. The manager list is supplemented by the standard ComfyUI file list, with case/separator normalization to avoid duplicate entries and correct preservation of subfolders. Unknown architecture labels no longer hide otherwise unclassified models. Missing previews use the neutral LoRA icon. Existing CivitAI SHA-256 resolution remains independent of sidecar files; preview bytes stay in RAM. Regression checks cover incomplete/unavailable manager responses and schema-preserving session restoration.

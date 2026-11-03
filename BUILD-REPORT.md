# Combined build report

Version: `1.53.0-reloaded.3`. Built 2026-11-03. No commit created.

Sources:
- Local fork: `748133b` (Krea Turbo/Edit/Remix changes preserved).
- Acly 1.53.0: `0217cd2197fcadbd70d7e63af25e29cc21cb7c8b`.
- Reloaded ux/improvements: `08082efef6f2d4a90a80c32a90895a8c0e3f913d`.
- Krea Edit nodes V2: `13ef60f20828949becada815d3cbc371eb5ccaab`.
- ComfyUI-LoRA-Manager: `5f607e5c0fc9db196592de7e58da994eddf1fd2e`.
- V2 weights: Hugging Face `conradlocke/krea2-identity-edit`, revision `f33b26291c640e8ec50b0da420bfee3c9a660daa`. SHA-256 values are in presets/models.json.

Included:
- Upstream 1.53 runtime/server requirements and all Reloaded plugin code: visual pickers, prompt/LoRA/recipe/wildcard tools, history search/favorites/ratings, batching/looping, AI Select, reference names, AnimaDex/CivitAI/Prompt AI integrations, Save to Eagle, MOSS support.
- Native Krea Turbo, Identity Edit V2, General Edit V2 and Remix styles.
- Managed-server Additional components tab, persistent selection, automatic node ZIP extraction and Python dependency installation in the managed server venv.
- Component version markers, retry and folder rollback on dependency failure.
- Krea model selection automatically selects nodes, base model, encoder and VAE dependencies. Existing weights are reused. V1 LoRAs do not count as V2.
- Bundled Reloaded Krea workflows updated to V2 weights and copied into the workflow library during component installation.
- Official Eagle installer download/launch; Eagle's own installer, trial/license activation and library selection remain interactive. Eagle must run locally for export. No Eagle redistributable is bundled.
- Upstream self-update blocked for this fork version so the combined features are not overwritten.

Validation:
- Ruff check passed; Ruff format passed; Pyright: 0 errors.
- Available CI suite: 302 passed, 111 skipped, 9 deselected. TEMP and TMP point into .appdata/test-temp. Tests ran outside sandbox after sandbox execution stalled.
- The 9 deselected cases are 7 live ComfyUI tests (server fixture absent) and 2 download tests (standalone PyQt TLS initialization failed). An unfiltered CI run reproduced those environment failures; no claim that the unfiltered suite is green.
- Required workflow inference attempt: 8 passed, then connection refused at 127.0.0.1:8188; stopped at first error.
- Required installer integration attempt stopped when standalone PyQt failed TLS initialization downloading ComfyUI. Full first-run installation has NOT been validated inside Krita.
- Real Qt ServerWidget instantiated and rendered; component-selection signals and automatic Krea node selection passed.
- Both pinned custom-node ZIPs downloaded and archive root names verified. Eagle Windows installer and both pinned V2 LoRA URLs returned HTTP 200. Model weights were not downloaded and no GPU generation was run.
- ZIP integrity, expected runtime files, bundled workflows, JSON parsing and Python syntax checked after packaging.

Remaining acceptance:
Import ZIP into Krita, restart, choose Managed Server workloads/components, complete an actual installation, then generate/edit with Krea V2 and test LoRA metadata and Eagle export. External servers, account/API-key integrations and Eagle licensing cannot be provisioned as an unattended managed-server dependency.

Settings wheel fix: closed combo boxes ignore wheel input throughout SettingsDialog, even when focused or created dynamically. Wheel input is forwarded to the enclosing scroll area. Open dropdowns and keyboard/click selection remain available.


Media retention update (.3):
- Session-owned media directories with QLockFile ownership, crash recovery, retrying cleanup and no traversal of symlinks/junctions or unowned export folders.
- Managed ComfyUI input/output/temp/user directories and TEMP/TMP/TMPDIR are isolated in the session. LoRA Manager uses LORA_MANAGER_SETTINGS_DIR and centralized temporary sidecars/examples. Only model metadata JSON/preferences are preserved; preview files are not copied to metadata caches.
- Online-only LoRA/checkpoint thumbnails; no local file fallback, no HTTP disk cache. Video frame extraction uses pipes rather than global TEMP media files. Prompt thumbnails are in memory, AnimaDex no longer writes thumbnail caches. Automatic image/workflow debug dumps disabled.
- Animation/live-recording intermediate frames and Eagle staging confined to the session. Manual image exports and .kra history unchanged.
- External/cloud connections and custom workflow execution require explicit approval per session; external manager model downloads blocked to prevent uncontrolled persistent previews.
- Real isolated ComfyUI+LoRA Manager CPU smoke test passed: centralized sidecar/example paths confirmed via API; a 64x64 image generated and transferred via ETN RAM cache without an output file; standard SaveImage output confined to the session; session removed after server shutdown. No diffusion model inference or existing server changes.
- Additional tests exercise crash recovery, active-instance protection, user-export preservation, metadata-only preference restoration, online preview resolution, RAM-only thumbnails, network cache settings, transfer exceptions and denial without approval.
- No global TEMP cleanup or blanket deletion of existing user model/output directories. Older unmarked files cannot safely be attributed to the plugin and are not removed.

Final .3 validation: 317 passed, 111 skipped, 9 deselected in the available CI suite; Ruff check/format and Pyright passed. The deselected pre-existing cases require the separate live ComfyUI test fixture or working standalone PyQt TLS; the real isolated backend smoke test described above passed.

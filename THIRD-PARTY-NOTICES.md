# Third-party notices

## Plugin code

This is an unofficial combined build maintained in
[Teutonick/krita-ai-diffusion](https://github.com/Teutonick/krita-ai-diffusion).
The combined plugin is distributed under the GNU General Public License, version 3;
see [LICENSE](LICENSE). Modified on 2026-11-03. Original copyright, license and
warranty notices remain applicable. No endorsement by the upstream authors is implied.

| Source | Contribution | License / pinned source |
| --- | --- | --- |
| Acly and the Krita AI Diffusion contributors | Original plugin, managed server installer and upstream 1.53.0 | GPLv3; [0217cd2197fcadbd70d7e63af25e29cc21cb7c8b](https://github.com/Acly/krita-ai-diffusion/tree/0217cd2197fcadbd70d7e63af25e29cc21cb7c8b) |
| FatGuy84 and the Krita AI Reloaded contributors | Visual browsers, prompt/batch tools, history/export changes, Krea custom workflows and other Reloaded additions | GPLv3; [08082efef6f2d4a90a80c32a90895a8c0e3f913d](https://github.com/FatGuy84/krita-ai-reloaded/tree/08082efef6f2d4a90a80c32a90895a8c0e3f913d) |
| Teutonick | Combined integration, installation changes, media retention policy and subsequent fixes | GPLv3; exact modified sources accompany each release |
| Aymeric Augustin and the websockets contributors | Bundled WebSocket library | BSD 3-Clause; its original LICENSE is retained in ai_diffusion/websockets/LICENSE; source revision 624a36cc9c1ea1369971387d7bb23533b2797350 |

## Independently downloaded components

These components are not relicensed by the plugin and retain their own notices:

- **ComfyUI-LoRA-Manager** — willmiao and contributors, GPLv3. Managed node revision
  `5f607e5c0fc9db196592de7e58da994eddf1fd2e`.
  [Project and license](https://github.com/willmiao/ComfyUI-Lora-Manager).
- **ComfyUI-Krea2Edit nodes** — Lars Bouaraba and contributors, Apache License 2.0.
  Managed node revision `13ef60f20828949becada815d3cbc371eb5ccaab`.
  [Project and license](https://github.com/lbouaraba/comfyui-krea2edit).
- **Eagle** — a separate proprietary desktop application. Its official installer and
  own license/trial process are used; its executable is not bundled in the plugin ZIP.
  [Eagle](https://eagle.cool/).
- **ComfyUI, other node packs, runtimes and model weights** — downloaded components
  retain their respective licenses and requirements. Their inclusion in setup is not
  a statement that every component or model is licensed under GPLv3.

## Krea 2 Edit V2 model documents

Identity Edit V2 and General Edit V2 are distributed by Lars Bouaraba / conradlocke.
The Krea 2 Identity Edit V2 License applies in addition to the Krea 2 Community License.
Free use is non-commercial; commercial use of the models or their outputs requires
separate written permission from Lars Bouaraba, and additional Krea licensing may apply.

The unmodified model license texts are included under
`ai_diffusion/licenses/krea2-v2/` for reference, with pinned source URLs and SHA-256 hashes
in `manifest.json`. These documents do not change the license of the plugin code.
Copies are installed in `models/loras/krea2-edit-v2-licenses/` when a user explicitly
acknowledges the model terms. Acknowledgement is not a commercial license grant.

Model document revision: `f33b26291c640e8ec50b0da420bfee3c9a660daa`.
[Official model repository](https://huggingface.co/conradlocke/krea2-identity-edit).

## Exact corresponding source

The source archive shipped with a release is the authoritative source snapshot for
that release. `SOURCE-MANIFEST.json` is identical in the source and plugin archives
and records the hashes of the included source files. Build scripts and dependency
notices are included. The GPLv3 license and its warranty disclaimer accompany both.

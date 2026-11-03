# Combined build: 1.53 + Reloaded + Krea Edit V2

Install `krita_ai_diffusion-1.53.0-reloaded.1.zip` using Krita's **Tools → Scripts → Import Python Plugin from File**, then restart Krita and enable the plugin if needed.

In the plugin's **Managed Server** setup:

- **Workloads**: select your base models. Selecting Krea enables its Edit V2 nodes automatically and downloads the text encoder, VAE and selected model weights.
- **Additional components**: select **LoRA Browser metadata** to install ComfyUI-LoRA-Manager and its Python dependencies automatically. Existing components can be added later using the same Install button.
- **Krea 2 Identity Edit V2** and **General Edit V2** are separate LoRAs; select the matching style for the task. Old V1 files remain usable with your custom styles, but do not satisfy a V2 download.
- **Eagle**: select **Download and launch Eagle setup**. No separate download is needed. Complete Eagle's own installer, trial/license activation and library selection, then keep it running locally for Save to Eagle. Already installed Eagle needs no additional plugin dependency.

Automatic component installation manages the plugin's own local ComfyUI server. An external/remote ComfyUI server must be maintained by its owner. LoRA metadata, CivitAI, AnimaDex and Prompt AI integrations keep their Reloaded configuration requirements (accounts/API keys/local services where applicable).

Bundled Krea custom workflows are copied to the plugin's workflow library when Krea nodes are installed. The native Krea styles work in the regular generation workspace.

Updates must come from this combined fork's ZIP. The upstream updater is prevented from replacing this build and losing the added features.

Sources: [Acly 1.53.0](https://github.com/Acly/krita-ai-diffusion/releases/tag/v1.53.0), [Reloaded](https://github.com/FatGuy84/krita-ai-reloaded/tree/08082efef6f2d4a90a80c32a90895a8c0e3f913d), [Krea Edit V2](https://github.com/lbouaraba/comfyui-krea2edit/tree/13ef60f20828949becada815d3cbc371eb5ccaab). See [BUILD-REPORT.md](BUILD-REPORT.md) for validation and limitations.

---

<h1>
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="branding/wordmark-dark.svg">
    <img width="330" src="branding/wordmark-light.svg" alt="Krita AI Reloaded">
  </picture>
</h1>

<p>
  <a href="https://github.com/Acly/krita-ai-diffusion"><img alt="Based on upstream v1.53.0" src="https://img.shields.io/badge/upstream-v1.53.0-17120C?style=for-the-badge&labelColor=241C13&color=FFB020"></a>
  <a href="LICENSE"><img alt="License GPL-3.0" src="https://img.shields.io/badge/license-GPL--3.0-17120C?style=for-the-badge&labelColor=241C13&color=FF7A1A"></a>
  <a href="https://ko-fi.com/fatguy84"><img alt="Support on Ko-fi" src="https://img.shields.io/badge/support-Ko--fi-17120C?style=for-the-badge&labelColor=241C13&color=FFC24D"></a>
  <a href="https://www.youtube.com/@KritaAIReloaded"><img alt="Videos on YouTube" src="https://img.shields.io/badge/videos-YouTube-17120C?style=for-the-badge&labelColor=241C13&color=FF6A00"></a>
</p>

> **Generate faster. Browse visually. Run it again.**

A usability-focused fork of the [Krita AI Diffusion plugin](https://github.com/Acly/krita-ai-diffusion)
with numerous quality-of-life improvements for prompt handling, LoRA management
and batch generation workflows.
Feature videos and a full tour are on the
[YouTube channel](https://www.youtube.com/@KritaAIReloaded).

<p>
  <a href="#sequential-wildcards--batch-control">Wildcards</a> ·
  <a href="#lora-browser">LoRAs</a> ·
  <a href="#recipe-browser">Recipes</a> ·
  <a href="#animadex-browser">AnimaDex</a> ·
  <a href="#civitai-browser">CivitAI</a> ·
  <a href="#style-picker">Styles</a> ·
  <a href="#history--export">History</a> ·
  <a href="#custom-workflows">Workflows</a> ·
  <a href="#installation">Install</a>
</p>

All credit for the plugin itself goes to [Acly](https://github.com/Acly) and the
upstream contributors — this fork only layers UX improvements on top and tracks
upstream releases (currently based on v1.53.0).

If this fork saves you time, you can support it on [Ko-fi](https://ko-fi.com/fatguy84)
or [GitHub Sponsors](https://github.com/sponsors/FatGuy84) — and please consider
[sponsoring Acly](https://github.com/sponsors/Acly) too, who built the plugin this is based on.

**This fork is for you if:**

* you work with large LoRA collections and want to browse them visually
  (previews, tags, favorites) instead of memorizing file names
* you generate batches and want systematic control over which prompt/LoRA
  combination each batch item uses
* you want prompt history actions, batch sizing and image export to be less
  clicky and more predictable

## At a glance

| Browse | Batch | Organise |
| --- | --- | --- |
| **LoRA, Recipe, Style, Checkpoint and Wildcard pickers** with live search, previews and favorites | **Sequential `[[a\|b]]` wildcards**, Cartesian batches, up to 1000 per run, Loop Generate | **History** with search, favorites, 1–5 star ratings and filters, plus Save to Eagle |
| **CivitAI and AnimaDex browsers** — search and download without leaving Krita | **Local prompt tools** via Ollama: enhance, batch prompts, generate wildcard files | **Faster startup** through disk-cached model lists, and DLSS5 Enhance on results, canvas or selection — tiled for any size |

<br>

## What's different from upstream

### Sequential wildcards & batch control

Upstream only has random wildcards `{a|b|c}`, where each generation picks one
option at random. This fork adds a **sequential** variant using double
brackets: `[[a|b|c]]`. Instead of picking randomly, batch item 1 gets `a`,
item 2 gets `b`, item 3 gets `c`, item 4 wraps back to `a`, and so on — useful
for systematically running through a fixed set of variations instead of
hoping the dice land right.

![Sequential wildcards, auto batch-count, and preview size](media/demo-wildcards.gif)

Multiple `[[...]]` groups in the same prompt combine into a **Cartesian
product** across the whole batch. For example:

```
[[black|white]] cat, [[sitting|jumping]]
```

generates all 4 combinations (black+sitting, black+jumping, white+sitting,
white+jumping) as batch items 1–4, then repeats. A button next to the batch
count (the "a×b" icon) reads the prompt and sets the batch count to
the exact number of combinations for you, so you don't have to count by hand.
The same icon also sits **next to Loop Generate**: one click there sets the
batch count to the combinations *and* starts generating. It does nothing when
the prompt has no sequential wildcards, so it never fires a stray batch.

`<lora:name:weight>` tags work inside `[[...]]` groups too, and are switched
correctly per batch item — each image in the batch is generated with its own
LoRA set rather than all of them sharing whatever LoRA the first prompt
evaluation picked (which is what happens upstream if you try this).

![Two sequential wildcard groups with per-item LoRA switching and auto batch-count](media/demo-wildcards-lora-batch.gif)

Other batch changes:
* **Batch count up to 1000** (upstream caps at 10), with a spinbox you can
  type into directly instead of only dragging a tiny slider.
* **Loop Generate**: a toggle button (circular arrow icon) next to Generate.
  Turning it on starts a batch immediately, and every time the queue empties
  it automatically enqueues another one — keeps running unattended (e.g.
  overnight) until you toggle it off again. If generation fails outright
  (bad prompt, disconnected server, etc.) it turns itself back off instead of
  spinning uselessly.
* **Strength Sweep**: the ⇅ button next to the Strength slider. Pick From, To
  and Step (e.g. 40% → 75% in steps of 5, or downwards with From > To), plus a
  fixed seed (prefilled with the current one). The dialog shows how many images
  that makes; with a fixed seed one image per value is queued, so strength is the
  only difference. Turn the fixed seed off to get *Images per step* (1–50) with
  random seeds. Your previous strength and batch count are restored.
* **Pause queue**: a pause toggle in the queue group, before the ✕ and queue buttons. The running generation
  finishes, queued jobs (batches, Loop Generate) wait until you click again.
  ComfyUI cannot freeze a sampler mid-run, so this holds the *queue*, not the
  current image. Local ComfyUI only.
* **Cancel current generation**: the ✕ button between Pause and the queue button stops the
  image that is generating right now; queued jobs (batches, Loop Generate) carry
  on with the next one. Only enabled while a job is running.
* **Unload models**: *Unload Models* in the queue popup (click the queue button)
  frees VRAM and RAM on the ComfyUI server, e.g. before switching to another
  GPU-heavy app. A running generation finishes first; the next job loads its
  models again. A bar next to it shows the GPU's current VRAM use (refreshed
  every 2 s while the popup is open; counts other apps on the GPU too).
  Local ComfyUI only.
* **Preview while generating**: intermediate sampler results show on the canvas
  in the preview layer, replaced by the real result when the job finishes.
  Toggle with the eye button between Loop and Pause (or Settings → *Preview While Generating*).
  Update rate is set in Settings → *Preview Update Interval* (0.1–5.0 s, default 0.3 s).
  Picking a result in the history pauses previews so they don't pull the canvas away. Uses ComfyUI's own
  previews (low resolution, quality depends on the server's preview method).

  ![Loop Generate keeps enqueuing batches until toggled off](media/demo-loop-generate.gif)

### File-based Wildcards

`__name__` in a prompt picks a random line from a `.txt` file in your
`wildcards/` folder (one option per line), matching the convention used by
collections like [sd-wildcards](https://github.com/mattjaybe/sd-wildcards) —
drop such a repo's `wildcards/` folder in directly and it works with no
extra setup. `__folder/name__` addresses a file in a subfolder. A new
**Wildcards** button next to LoRA/Recipe in the prompt toolbar opens a
browser: search, a line-count + preview tooltip per file, a Reload button
for files you just added, and an "Open Folder" button (creates the folder
if it doesn't exist yet).

Three insert modes:
* **Normal** `__name__` — a live reference, re-read from the file and
  picked at random every time you generate. Edit the file later and it
  picks that up automatically.
* **Random** `{a|b|c}` — expands the file's lines directly into the prompt
  as a regular random wildcard group (a frozen snapshot at insert time, no
  live file dependency).
* **Sequential** `[[a|b|c]]` — same expansion, but cycles through every
  line across the batch instead of picking randomly, so you can sweep a
  whole wildcard file systematically in one batch (something a plain
  `__name__` reference can't do, since file wildcards only pick randomly).

Unknown/misspelled `__name__` references are left visible in the evaluated
prompt rather than silently disappearing, so a typo or missing file is easy
to spot instead of just producing a slightly-off image.

The browser has two tabs. **Library** lists the files and edits them in place
(create, rename, edit, save, delete - deleted files go to the system trash). **Generate** writes a new one with the local
language model ([Prompt AI](#prompt-ai)): name a category — hairstyles, hair colors, poses, outfits —
and it produces a list of interchangeable options, which you can prune before
saving. Entries are written in rounds rather than one call each, since an entry
is only a few words and per-call model overhead would dominate. Saving switches
to the Library tab with the new file selected, ready to edit.

### Named references (`@Name`)

Multi-image edit models (Qwen Edit Plus, Flux 2 Klein, Krea 2) can take
several reference images, but the prompt has to point at them by number —
"image 2", "Picture 3" — and that number shifts whenever you add, remove or
reorder a reference layer. This fork lets you **name** each Reference control
layer (the small *Name* field next to the layer picker, shown for edit models)
and mention it in the prompt instead:

```
Add @hat and @scarf to the girl
```

You can also skip the control layer: `@name` that matches no named reference
picks up the **Krita layer** of that name (spaces become `_`, so "Red Hat" is
`@Red_Hat`) and sends it as an extra reference. Typing `@` opens a completion
list with reference names and, for edit models, layer names. This only works
with models that take reference images — SD, SDXL, Illustrious, Anima & co.
have no way to receive them, so there `@name` just becomes the plain name.

On generate, each `@name` is rewritten to the token the model expects,
counting images in the order they are sent (the canvas first when editing,
then the reference layers in the order they are listed, then regional ones) — `image 2` for
Flux 2 / Krea 2, `Picture 2` for Qwen Edit Plus. Names are case-insensitive and
saved with the document.

Models that get a single stitched image (Flux Kontext, Qwen Edit) or no image
numbering at all get the plain name, so the same prompt still reads naturally.
On edit models an `@name` that matches neither a reference nor a layer is left as typed and logged as a
warning, and `a@b.com`-style text is never touched. Wildcards and
`<lora:...>` tags are unaffected.

### Prompt AI

An **AI** button next to the prompt rewrites it with a local language model
through [Ollama](https://ollama.com). Everything runs on your machine: no
account, no cloud service, nothing leaves your network. You need Ollama
installed and at least one model pulled (`ollama pull ...`). Turn it on under
**Settings → Prompt AI**, enter the server URL (default `127.0.0.1:11434`),
press **Connect** and pick a model from the list.

Clicking the button runs **Enhance**; the arrow next to it opens the rest:

* **Enhance** / **Enhance selection** - improve the whole prompt, or only the
  text selected in the prompt field
* **Add detail only** - keeps your wording and adds to it
* **Rewrite from scratch** - a new prompt for the same subject
* **Split into sections** - sorts a long prompt under `# Quality & Style`,
  `# Character`, `# Clothing`, `# Pose & Expression`, `# Environment`,
  `# Lighting` and `# Camera` comment headings, so you can find and edit one
  part; LoRA tags go under a last `# Loras` heading, one per line. Nothing is
  rewritten or dropped: weights, LoRA tags and wildcards stay as they are, and
  anything the model leaves out is put back under `# Other`
* **Variations as sequential wildcard** - writes several alternatives as one
  `[[a|b|c]]` group (count set by **Variation Count** in the settings)
* **Describe the image** / **Describe current selection** - captions the
  canvas, or the selected area of it, as a prompt; needs a vision model
* **Modify with instruction...** - a free-form change such as "make it night,
  remove the hat". Diffusion models handle "no hat" in the positive prompt
  badly, so the dialog has a **Move exclusions to negative prompt** checkbox:
  whatever the instruction removes ("hat") is added to the negative prompt,
  without duplicating terms already there
* **Prompt batch for generation...** - see below
* **Revert** - steps back through earlier versions of the prompt

LoRA tags `<lora:...>`, file wildcards `__name__` and wildcard groups `{a|b}`
and `[[a|b]]` are taken out before the prompt goes to the model and put back
afterwards, so the model cannot mangle or drop them.

**Prompt Batch** builds a pool of different prompts before you generate, so a
batch or Loop Generate doesn't repeat the same one. **Vary current prompt**
varies what is already there; **Random for a theme** invents scenes from a
short idea. Set the **Count**, then output the result as a **Sequential group
`[[a|b|c]]`** in the prompt or as a **Wildcard file** in your `wildcards/`
folder. **Set batch count** matches the batch to the number of prompts, and
**Generate images when finished** starts right away. Writing wildcard lists
by category lives in the [Wildcards](#file-based-wildcards) browser's
**Generate** tab.

The model is told how to write for the active style's base model: tag lists
for SD 1.5, SD XL, Illustrious, Pony and Anima; natural language for Flux,
Qwen, Z-Image, Krea 2 and others; edit instructions for Flux Kontext and
Qwen Edit. A fourth profile, **Video (LTX / motion)**, is there to map a
family to yourself. The profiles are in `presets/prompt_enhance.json`. To change them, copy that file to the
settings folder (**Open Settings folder**) as `prompt_enhance.json`, edit it
and press **Reload Profiles** - entries there replace the defaults per
profile. A profile can also name its own `model`, and a `describe_model` for
image description (`qwen2.5vl:7b` by default).

If Ollama shares the GPU with ComfyUI, **Free Image Model First** unloads the
diffusion models before each call, and **Keep Model Loaded** (0 by default)
drops the language model again right after, so image generation gets its
VRAM back.

### LoRA Browser

Click the **LoRA** button in the prompt field to open a visual LoRA picker
instead of typing `<lora:...>` tags from memory or scrolling a giant
dropdown. It talks to [ComfyUI-Lora-Manager](https://github.com/willmiao/ComfyUI-Lora-Manager)
if you have it installed, and falls back to a plain file-name list otherwise
(no crash, just fewer features — no previews/tags/trigger words).

![LoRA browser: multi-select, wildcard-group insert, add to prompt](media/demo-lora-browser.gif)

With Lora Manager available you get:
* **Preview thumbnails** for every LoRA, loaded lazily as you scroll so
  opening the browser with hundreds of LoRAs doesn't stall — including
  **video previews** (first frame extracted via `ffmpeg` if it's on PATH,
  falls back to a play-icon placeholder otherwise)
* **Search** by name, **tag filter** (alphabetical), and a **base-model
  filter** with full names (SD 1.5 / SD XL / Illustrious / Flux Kontext /
  Anima / etc.), matching the labels used in the style editor
* **Favorites**, synced both ways with Lora Manager — right-click a LoRA to
  add/remove it as a favorite from directly inside Krita, no need to switch
  to the Lora Manager web UI
* **Commercial-use badge**: a small $ square in the bottom-right corner of
  each thumbnail — green if CivitAI's license allows commercial use of
  generated images, red if it doesn't, grey if unknown. A gold ★ in the
  top-right marks favorites, and a base-model name tag sits bottom-left.
  Right-click → **Open on CivitAI** jumps straight to the model page.
* **Sort by name or date added** — a dropdown next to the filters, useful
  for finding LoRAs you just downloaded without scrolling the whole list
* **Content filter** (All / Safe Only / Hide Explicit), based on CivitAI's
  content rating of the preview image (Safe Only hides R and above, Hide
  Explicit hides only X/XXX)
* **Trigger words** pulled from CivitAI metadata — insert them alongside the
  LoRA tag with one click, either a specific phrase group or all of them
* **Adjustable thumbnail size** via a slider (up to 384px), and the whole
  list is cached to disk so reopening the browser is instant instead of
  re-querying the server. With **Preload LoRA List** (Settings → Lora Manager,
  on by default) the list is fetched in the background right after connecting,
  so even the first open doesn't wait. A selection you've made survives the list
  reloading as more LoRAs stream in or filters change, instead of silently
  dropping and forcing you to re-find what you picked.
* **Reload list** (re-fetches everything from Lora Manager) and **Scan server**
  (picks up newly added LoRA files without leaving the browser). The scan only
  fetches what is new: ComfyUI re-reads its file lists without inspecting any
  model, and Lora Manager's list is read newest first until nothing new turns
  up and the count matches — 1 request when nothing changed instead of ~100
  for a 10k library. A new file with an old date (e.g. copied) is still found;
  if files were removed it falls back to the full reload.
* **Insert position**: choose whether the LoRA goes at the end of the prompt,
  the start, or at your cursor
* **Copy** button next to *Add to Prompt* — puts the same tags on the
  clipboard instead of inserting them, handy for pasting elsewhere

**Multi-select** (Ctrl/Shift-click) lets you pick several LoRAs at once and
insert them together — as a **random** `{a|b}` group, a **sequential**
`[[a|b]]` group (matching the wildcard syntax above), or **separate** tags
(all applied at once, no wildcard) — with each LoRA's trigger words carried
along automatically. This is the fast way to set up "try LoRA A, then B,
then C" batches without manually typing out the wildcard syntax.

The dialog is non-modal, so Krita stays fully usable while it's open — no
need to close it before painting or switching layers.

### Recipe Browser

Click the **Recipe** button next to the LoRA browser button to open a picker for
[Lora Manager's Recipes](https://github.com/willmiao/ComfyUI-Lora-Manager) —
saved prompt + LoRA-stack combinations. Same visual browser as the LoRA
picker (previews, search, base-model filter, favorites, sort by name/date
added), but applying a recipe fills in a whole prompt setup at once instead
of a single LoRA:

![Recipe browser: apply a saved prompt+LoRA combo, then generate](media/demo-recipe-browser.gif)

* **Add to Prompt** appends the recipe's prompt and LoRA tags to whatever
  you already have, on a new line
* **Replace Prompt** overwrites the current positive/negative prompt with
  the recipe's
* LoRAs missing from your local library are flagged (⚠) instead of silently
  failing at generation time

The dialog is non-modal, same as the LoRA browser.

It also works the other way round: right-click a result in the history →
**"Save as Recipe"** sends the prompt, negative prompt, sampler settings and
LoRA stack straight to Lora Manager as a new recipe, so a generation you
like becomes reusable without retyping anything.

### AnimaDex Browser

With an Anima style selected, click the **AnimaDex** button in the prompt toolbar to browse
[AnimaDex](https://animadex.net) ([source](https://github.com/zetaneko/AnimaDex)) —
a catalogue of ~36k anime characters and artists the Anima model knows,
each with a sample render and the trigger phrase that reproduces it.

* Search by name, series or tag (comma-separated terms must all match),
  filter by series, gender, hair colour/length, eye colour, or artwork score
  for artists
* **Add to Prompt** inserts the trigger in the form the
  [Anima model card](https://huggingface.co/circlestone-labs/Anima) asks
  for: artists get their required `@` prefix, and danbooru disambiguators are
  escaped (`sabo \(one piece\)`) - unescaped, ComfyUI would read the
  parentheses as a weight and drop them. Wildcard export and "Copy" use the
  same form. Tick
  **+ tags** to also insert the character's appearance tags. With several
  selected, pick the format like in the LoRA browser: **Random** `{a|b}`
  (one per generation), **Sequential** `[[a|b]]` (cycles through the batch,
  combines with other groups into a Cartesian product) or **Separate** (all
  at once)
* **Save as Wildcard…** writes the triggers of the selected entries into a
  wildcard file - or, with nothing selected, of every result of the current
  search (e.g. every Touhou character as `__animadex/touhou__`), ready for
  sequential wildcards and Cartesian batches. **+ tags** applies here too
* Right-click: copy the trigger, open the danbooru page, or open a character's
  Anima LoRA on CivitAI

**Offline catalogue.** Without setup the browser reads the live site, 36
results at a time. For instant search, click **Import…** and paste an export
token from animadex.net → Account → *Offline dataset export*. Only the
catalogue metadata is downloaded (a few MB); thumbnails are fetched as you
browse and cached on disk, instead of the multi-GB image set. The token is
used once and not stored — **Update** later fetches only what changed.
LoRA links, votes and artist categories exist only on the live site; the
**Source** dropdown switches between the two.

Settings → Integrations → AnimaDex hides the button or points it at a
self-hosted instance (e.g. `127.0.0.1:5000`).

**Blank thumbnails or "closed the connection" on import?** Images and the
offline export are served from a second host, `blobs.animadex.net`. Some
antivirus web filters block that host by name while letting `animadex.net`
through — confirmed with Bitdefender's Online Threat Prevention. Your browser
still loads the images because it hides the host name via Encrypted Client
Hello, which Krita's Qt can't do, so only Krita is affected. The browser
window shows a warning when this happens; the fix is adding
`blobs.animadex.net` to the filter's exceptions (Bitdefender: Protection →
Online Threat Prevention → Settings → Manage exceptions). Check what the
filter reported first — if it flags the host as malware or phishing rather
than a content category, think twice.

### CivitAI Browser

The **CivitAI** tab — in the LoRA browser for LoRAs, in the style dialog for
checkpoints — searches civitai.com without leaving Krita and downloads straight
into your local library. The plugin only searches — the download itself is handed to
[ComfyUI-Lora-Manager](https://github.com/willmiao/ComfyUI-Lora-Manager), which
already knows your folder layout and writes metadata and preview images next to
the file, so a downloaded model shows up fully described in the LoRA browser.

* **Filters**: search text, LoRA vs Checkpoint, sort order and time period, plus
  a base-model filter that is pre-set to the architecture of your current
  checkpoint. Pony, Illustrious and NoobAI are separate labels on CivitAI even
  though they are SDXL underneath, so picking "SD XL" includes them.
* **Tag filter**: a dropdown filled with CivitAI's own tag vocabulary (the 100 most
  used, listed alphabetically), and you can type your own. A tag
  the site does not know is silently ignored by its search — the browser checks the
  entered tag against the vocabulary and says so instead of showing you unfiltered
  results that look like a match.
* **Download location**: a *Save to* row picks the model root (as configured in
  Lora Manager) and a subfolder — either an existing one from your library or a
  new path you type, which Lora Manager creates. Both are remembered, and leaving
  them empty keeps Lora Manager's own default layout.
* **Already in your library** is unmistakable on the tile: the preview is dimmed
  and gets a coloured banner — green *installed* for this exact version (matched by
  file hash against Lora Manager), blue *update* when you have a different version
  of the same model. The **In library / Not in library** dropdown filters on it, so
  a search can be narrowed to models you do not have yet.
* **Selling images**: the `$` badge is green when the model's CivitAI license
  permits selling generated images and red when it does not. Only the `Image`
  license grants that — `Sell` covers reselling the model itself, and
  `Rent`/`RentCivit` only allow running it on a generation service. The
  **"Sellable images only"** checkbox filters the results down to the green ones,
  and the tooltip always shows the raw license values.
* **Refresh** re-runs the search and drops everything the dialog caches — previews,
  the tag list, the download folders and the set of models already in your library —
  so a model downloaded elsewhere shows up as installed without reopening.
* **Multi-select and batch download**: ctrl/shift-click several tiles and download
  them in one go. The bar shows the combined size and how many of the selected
  models you already have; those and early-access ones are skipped instead of
  failing. Progress counts through the batch (`3/12`), cancel stops the running
  transfer and the rest of the queue, and a summary reports what got through.
  Downloads run one after another — Lora Manager has no server-side queue worker,
  and parallel transfers would only split the same bandwidth.
* **Downloads** show progress, speed and a cancel button. Duplicate downloads are
  refused by Lora Manager, early-access models are marked and cannot be
  downloaded (they need a purchase on CivitAI first).
* **Content rating**: the filter starts at whatever *Settings → Integrations →
  CivitAI Content Filter* is set to, and changing it in the browser updates that
  setting, so it stays where you left it.
* **Site**: *Settings → Integrations → CivitAI Site* switches between
  `civitai.red` (default) and `civitai.com`. Same API, same models, same images.
* **API key** (optional, *Settings → Integrations → CivitAI API Key*, created under
  Account settings → API Keys): raises the rate limits and gives
  access to content that is hidden from anonymous requests. Note it is stored in
  plain text in the plugin's `settings.json`, like the other connection settings.
* **Previews** are pulled as CDN thumbnails (`anim=false`), which also means
  animated and video previews arrive as a still frame — no ffmpeg needed, and a
  tile costs ~50 KB instead of the multi-megabyte original. Where CivitAI's search
  endpoint returns no image at all (it does that for models rated R and above), the
  browser fetches one from the model's detail endpoint for visible tiles only.

The dialogs are organised as tabs rather than windows opening on top of windows:
the style dialog holds **Styles**, **Checkpoints** and **CivitAI**, and the LoRA
browser holds **Library** and **CivitAI**. Tabs of one window share their state —
a checkpoint downloaded in the CivitAI tab appears in the Checkpoints tab, and
styles created there land in the style list behind the first tab.

### Style Picker

If you have more than a couple dozen style presets, the stock dropdown
becomes unusable — hundreds of entries in one scrolling list with no way to
search. This fork replaces it with the same kind of searchable, non-modal
picker dialog as the LoRA browser. Click the style name/icon button
(where the dropdown used to be) to open it.

![Style browser: search and pick from hundreds of presets](media/demo-style-browser.gif)

* **Live search** by style name or checkpoint file name
* **Base-model family filter**, including a **"Base Model Family" field**
  added to every style (editable in the style editor's advanced checkpoint
  section, next to Architecture). Illustrious and Pony checkpoints use the
  exact same architecture as SDXL — there is no way to tell them apart from
  the model weights themselves — so this field defaults to a guess based on
  the checkpoint's file name ("Auto") but can be overridden by hand from a
  dropdown if the guess is wrong, and that's what the picker's filter uses
* **Favorite styles**: right-click a style or hit `F` to star it; favorites
  get pinned in their own section above the existing "Recently Used" list
  (upstream already sorts by recent use — this just adds the same idea for
  styles you always come back to, regardless of recency)
* **Checkpoint thumbnails**: each style shows the actual preview image of its
  checkpoint (pulled from Lora Manager, video previews included) with a
  base-model name tag, instead of just a generic architecture icon — falls
  back to the icon when there's no preview. Toggle between a **List** and a
  **Grid** view, with a size slider (up to 512px).
* **Sort by name or date added** (the "Recently Used" section keeps its own
  recency order regardless of the sort dropdown, since that's its purpose)
* **Actions on selection**: buttons to add/remove a favorite or **delete** a
  style (user styles only, built-ins are protected), plus a Reload button.
* **Generate across selected styles**: Ctrl/Shift-click several styles, set
  a seed (or leave it random), and hit **Generate across** — runs the
  current prompt once per style with the *same* seed, so the style is the
  only variable and results land grouped by style in the history for an
  easy comparison. Tick **Grid** to get one extra layer once all are done,
  with the results side by side and the style name above each, scaled to fit
  the canvas; tick **Layers** to also put every result on the canvas as
  `[style name] prompt (seed)`. LoRAs written in the prompt that belong to
  another architecture than a style's checkpoint are left out for that style
  (a dialog lists them before anything is queued).

#### Create Styles from Checkpoints

The **Create from Checkpoints…** button in the style picker opens a checkpoint
browser (same visual grid as the LoRA browser — previews, search, base-model
filter, favorites, sort) where you can **multi-select checkpoints and generate
one style per checkpoint in a single step**. Pick an existing style as a
**template** and every new style copies its settings (sampler, steps, etc.),
swapping in the checkpoint. Names come from the checkpoint's title (with its
version), and the Base Model Family is filled in automatically from Lora
Manager's metadata. Checkpoints the plugin's model list leaves out — added
after connecting, or ones the server couldn't classify (base model "unknown")
— are still found via ComfyUI's loader file lists, and the style stores the
architecture Lora Manager reports so it keeps working after a reconnect. A
**Scan server** button looks for new checkpoint files (checkpoints only, no
full model re-inspection), and if a checkpoint still isn't found, a dialog
names it and offers **Scan server and retry** for just those — Generate
across keeps its seed on the retry. A green ✓ badge marks checkpoints
that already have a style, and a **"Hide checkpoints with a style"** filter
narrows the browser down to ones that don't yet. Same **content filter**
(All / Safe Only / Hide Explicit) as the LoRA browser.

The same browser also has a **Generate across** button — like the style
sweep above, but over raw checkpoints instead of full styles, useful for a
quick look before committing to creating styles for all of them. Only the
checkpoint changes; sampler, steps and every other setting come from the
current style. The same **Grid** / **Layers** options apply (labels are the
short checkpoint name, e.g. `[flux1-dev] prompt (seed)`). The style's LoRAs
and the prompt's LoRAs that belong to another architecture than a checkpoint
are left out for that one — Lora Manager's base model decides, else the
current style's architecture — and a style pinned to an architecture falls
back to auto-detect for checkpoints it doesn't fit. Still works best within
one base model family, since sampler settings rarely carry over.

### Faster startup

**Model list disk cache**: upstream re-queries the server for every single
checkpoint/LoRA/etc. on every Krita startup, which is slow with large model
libraries (700+ entries easily takes a while). This fork caches the
discovered model list to disk per server URL, so subsequent startups load
instantly from cache. Click the Refresh button in the connection settings
after installing new models to force a full re-scan — or use **Scan server**
in the LoRA or Checkpoints browser, which only re-reads the file lists and is
much faster.

### History & export

* **Batch grouping**: results from one Generate click (or one Loop Generate
  cycle) are grouped under a single header in the history, even when
  wildcards make each image's evaluated prompt different — upstream/earlier
  versions of this fork would otherwise split a wildcard batch into a
  separate header per image. Click a header to select every image in that
  batch at once (for bulk favorite/rating/delete via the context menu);
  right-click it for **Collapse/Expand Batch** and **Select Batch**.
* **Split prompt actions**: upstream's "Copy Prompt" context-menu action both
  copied to clipboard *and* overwrote your current prompt field in one click,
  which is surprising if you only wanted one of those. Now there are four
  separate entries: Apply/Copy × raw/evaluated prompt.
* **Copy Image to Clipboard**: puts the generated image itself on the clipboard,
  at full resolution, for pasting into another application.
* **Keyboard shortcuts** while the history has focus, all of them listed next to
  their entry in the context menu: `Enter` apply to canvas, `Space` toggle
  selection, `F` favorite, `0`-`5` rating, `S` apply style, `P` apply prompt,
  `R` save as recipe, `Ctrl+C` copy image, `Ctrl+Shift+C` copy evaluated prompt,
  `Ctrl+S` save image, `Del` discard.
* **Apply Style / Strength / Seed**: these entries take the settings an image was
  generated with and make them current. Upstream calls them "Copy Style" and so
  on, which reads like they go to the clipboard — they never did. Apply Style is
  greyed out when the style has since been deleted.
* **Mark Style as Favorite**: right-click a generated image to add (or remove) the
  style it was made with to your favorite styles, without opening the style
  picker. Greyed out when the style no longer exists.
* **Search the generation history**: a search box above the history list
  filters by prompt text live as you type. A scope dropdown next to it
  narrows the match to "All Prompts", "Raw Only" (the pattern you typed,
  including wildcard syntax), or "Evaluated Only" (wildcards resolved) —
  useful when a wildcard-resolved word matches results you didn't mean to
  include, or vice versa.

  ![Filtering history by prompt text and search scope](media/demo-filter-prompts.gif)

* **Favorite / Applied filters**: two toggle buttons above the history —
  one shows only images you've starred as favorites, the other shows only
  images you've actually applied to the canvas. Combine with the search box
  to narrow down a long history fast.
* **Favorite images**: right-click a result or hit `F` to mark it as a
  favorite — shown as a white star badge in the corner of the thumbnail
  (distinct from the existing "applied to canvas" badge in the opposite
  corner). Favorites are saved into the document and survive closing and
  reopening it.

  ![Marking and filtering by favorite](media/demo-favorites.gif)

* **1-5 star rating**: press `1`–`5` while a result is selected to rate it
  (pressing the same number again clears it), or right-click → "Set Rating".
  Shown as a row of yellow stars in the bottom-left corner of the thumbnail.
  A "Rating" dropdown above the history filters to an *exact* rating (not
  "N and up") — useful for triaging a big batch: rate as you review, then
  filter to e.g. only 4★ results. Ratings are saved into the document like
  favorites.

  ![Rating images and filtering by exact rating](media/demo-rating-filter.gif)
* **Preview size slider**: a slider above the history resizes all thumbnails
  live, from small (fit more on screen) to large (see detail without
  clicking through). Thumbnails are always regenerated from the
  full-resolution result, so they stay sharp at any size.
* **Generation time** tracked per job and shown in its tooltip, e.g.
  "13 s". It counts from when the server starts executing the job to
  completion - time spent waiting in the server queue (or uploading LoRAs) is
  left out. Stored with the history, older entries simply show none. The
  tooltip also shows the image size in pixels and how much space the result
  takes in the document history.
* **Model name in layer names**: applied results are named after the model
  that made them, e.g. `[flux1-dev] [Generated] a red fox in snow (1234)` -
  folder, file extension and precision tags like `fp8` or `Q8_0` are dropped.
  Handy when comparing models side by side in the layer stack. Turn it off in
  Settings > Interface > "Model Name in Layer Names".
* **Large results are kept in the document**: results wider or taller than
  WebP allows (16383 px) are stored as JPEG (PNG with transparency) instead
  of being dropped, one failed store no longer stops all later results from
  being saved, and a single result bigger than "Stored History Size" no
  longer wipes the stored history.
* **Save to Eagle**: if you use [Eagle](https://eagle.cool) to organize
  reference images, right-click a result → "Save to Eagle" sends it straight
  into your library via Eagle's local API, with the prompt as the item
  title, full generation metadata (seed, sampler, LoRAs, etc.) as the
  annotation, and the style/LoRA names as tags — no manual export/import
  round-trip through the filesystem. If you rated the image (1-5★) in
  Krita, that rating carries over to Eagle's own star rating.

  ![Sending a rated image to Eagle and reviewing it there](media/demo-eagle-export.gif)
* **DLSS5 Enhance**: right-click a result → "DLSS5 Enhance" → Cinematic /
  Default / Natural runs it through NVIDIA DLSS 5 neural rendering
  (reconstructed skin, hair and lighting detail) at the same resolution. The
  enhanced image lands in the history as its own entry, with the source's
  prompt and metadata, ready to apply in the same place. Works on a
  multi-selection. The entry only appears when the server has the
  [ComfyUI-DLSS5-Enhancer](https://github.com/Blueforcer/ComfyUI-DLSS5-Enhancer)
  nodes, which need Windows and an RTX 30 series GPU or newer on the machine
  running ComfyUI.

  The same works on the canvas: the arrow menu next to **Generate** ends
  with **DLSS5 Enhance Canvas** (the whole visible image) and, while there is
  a selection, **DLSS5 Enhance Selection**, each with the same three looks.
  The selection variant enhances the area around the selection and blends
  the result back in through the selection mask, so only the selected part
  changes and soft edges stay soft.

  Images larger than DLSS 5 accepts (7680×4320) are enhanced in overlapping
  tiles. DLSS tone maps each tile on its own, so a pass over the whole image
  at the largest size DLSS accepts first sets the overall look: each tile
  keeps only its fine detail and takes brightness and colour gradients from
  that pass, which removes the tone steps at the seams. Tiles overlap by
  256 px. Each pass restarts the DLSS worker, so expect a few seconds per
  tile. This also works for canvases wider than 16384 px.

  The settings a result was made with are kept with it: the history entry
  and layer are named like `[DLSS5 Natural, M, 1.0, 18 tiles]`, and saved
  images, Eagle exports and "Info to Clipboard" record area, style,
  intensity, tone, structure, skin, model preset and tile count.
* **Generation status line**: while a job runs, a line under the progress
  bar says what the server is doing — sampling step 12/30, loading models,
  decoding, DLSS5 pass 3 of 10 — and for how long. Hidden while idle.
* **Generation mode in metadata**: saved PNGs and the history tooltip now
  record *how* an image was made — Generate, Refine, Inpaint (Fill / Add
  Content / Remove Content / Replace Background / …), Upscale, etc. — not
  just the prompt and sampler settings. For Generate (Custom), the
  Seamless/Focus/Edit/Fill/Context settings used are recorded too.

### AI selection

The arrow menu next to **Generate** has an **AI Select** submenu that runs a
segmentation model on the ComfyUI server and hands the result back to Krita:

* **Select Subject** — the main foreground object of the visible image
  becomes the selection, with soft edges where the model is unsure.
* **Select by Text...** — type what to select (`sky`, `water`, `the red car`)
  and that becomes the selection. Uses CLIPSeg, which is fast but coarse:
  edges follow a low-resolution grid, so treat it as a starting point and
  refine with Krita's selection tools. It always selects the closest match,
  even when the prompt names something that isn't in the picture.
* **Remove Background** — adds a transparency mask to the active layer that
  hides everything but its subject. Nothing is erased: paint on the mask to
  fix the edges, or delete it to get the background back.

All three are also Krita actions (Settings → Configure Krita → Keyboard
Shortcuts → Scripts), so you can give them shortcuts. They run in the queue
like any other job; the submenu only shows when the server has the nodes:

* Subject / Remove Background: ComfyUI's built-in **RemoveBackground** when a
  model is in `models/background_removal`, otherwise **easy imageRemBg** from
  [ComfyUI-Easy-Use](https://github.com/yolain/ComfyUI-Easy-Use) with its
  RMBG-1.4 model (`models/rembg/RMBG-1.4.pth`).
* Select by Text: **BatchCLIPSeg** from
  [ComfyUI-KJNodes](https://github.com/kijai/ComfyUI-KJNodes); it downloads
  the CLIPSeg model (about 300 MB) to `models/clip_seg` on first use.

### Custom workflows

Ready-made Krita-adapted ComfyUI workflows live in
[`custom_workflows/`](custom_workflows/), for the
[Krea 2 Identity Edit](https://huggingface.co/conradlocke/krea2-identity-edit)
model. This requires the
[comfyui-krea2edit](https://github.com/lbouaraba/comfyui-krea2edit) custom
node pack installed on your ComfyUI server (the model uses dual conditioning
that stock ComfyUI nodes can't provide).

* **Krea2 Identity Edit** — single reference: the current Krita canvas is
  both the scene and the identity source for the edit.
* **Krea2 Identity Edit (Character + Scene)** — two references: the canvas
  provides the scene, and a second, selectable Krita layer provides the
  person/character reference to composite in. Handy for "put this character
  into this scene" style edits.

Both workflows expose the prompt, `grounding_px` (edit strength/precision),
and **two stackable LoRA slots** as regular Krita parameters — the LoRA
slots are populated from a dropdown of every LoRA your server knows about,
same as a normal style, so you can layer character/style LoRAs on top of the
edit without touching the underlying ComfyUI graph.

To use them, copy the `.json` files from `custom_workflows/` into
`%APPDATA%\krita\ai_diffusion\workflows\`, then pick the workflow from the
Custom workspace's workflow dropdown in Krita.

Krea 2 also works as an **edit model in the regular workspaces** (ported from
upstream [PR #2589](https://github.com/Acly/krita-ai-diffusion/pull/2589),
without its managed-install parts). With the same node pack installed and the
Identity Edit LoRA in `models/loras`, pick the **Krea 2 Identity Edit** style
(or Krea 2 Turbo, which links to it) and use Edit / Refine as with Flux 2:
the canvas is the first reference, and up to one Reference control layer
adds a second — its strength sets the reference boost. `<layer:Name>` in the
prompt is rewritten to `image N`, as for Flux 2. Without the node pack,
Krea 2 edits fail with a missing-node error from ComfyUI; plain Krea 2
text-to-image is unaffected.

## Installation

Same as upstream — see the [Plugin Installation Guide](https://docs.interstice.cloud/installation).
To use this fork instead of the official release, clone this repository and
link/copy the `ai_diffusion` folder into Krita's `pykrita` directory
(the bundled `websockets` library must be added from an official release
package, it is not part of the repository).

Some features require additional software:
* LoRA Browser metadata: [ComfyUI-Lora-Manager](https://github.com/willmiao/ComfyUI-Lora-Manager)
* Save to Eagle: [Eagle](https://eagle.cool) running locally
* Krea2 workflows: [comfyui-krea2edit](https://github.com/lbouaraba/comfyui-krea2edit) nodes
* DLSS5 Enhance: [ComfyUI-DLSS5-Enhancer](https://github.com/Blueforcer/ComfyUI-DLSS5-Enhancer)
  nodes and their runtime (Windows, RTX 30+)
* AI Select: [ComfyUI-Easy-Use](https://github.com/yolain/ComfyUI-Easy-Use)
  (or a model for ComfyUI's built-in background removal) and
  [ComfyUI-KJNodes](https://github.com/kijai/ComfyUI-KJNodes) for Select by Text

---

## Original plugin

Everything else — features, hardware requirements, setup, ComfyUI server options and
the gallery — is documented upstream:

* [Krita AI Diffusion on GitHub](https://github.com/Acly/krita-ai-diffusion)
* [Plugin documentation](https://docs.interstice.cloud)

This fork is licensed under GPL-3.0, like the original.

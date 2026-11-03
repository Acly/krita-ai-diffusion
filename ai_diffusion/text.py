from __future__ import annotations

import random
import re
from pathlib import Path
from typing import NamedTuple

from .backend.api import ConditioningInput, InpaintMode, LoraInput, RegionInput, WorkflowKind
from .files import FileCollection, FileSource
from .localization import translate as _
from .model.jobs import JobParams
from .util import PluginError
from .util import client_logger as log
from .wildcards import WildcardLibrary

# Functions to convert between position in Python str objects (unicode) and
# index in QString char16 arrays (used in eg. QTextCursor).


def char16_len(text: str):
    return len(text.encode("utf-16")) // 2 - 1  # subtract BOM


def char16_index_to_str_index(text: str, c16_index: int):
    bytes_utf16 = text.encode("utf-16")
    byte_pos = 2 + c16_index * 2  # utf-16 text starts with 2-byte BOM
    text_until_pos = bytes_utf16[:byte_pos].decode("utf-16")
    return len(text_until_pos)


def str_index_to_char16_index(text: str, index: int):
    return char16_len(text[:index])


# Prompt processing utilities


class LoraId(NamedTuple):
    file: str
    name: str

    @staticmethod
    def normalize(original: str | None):
        if original is None:
            return LoraId("", "<Invalid LoRA>")
        return LoraId(original, original.replace("\\", "/").removesuffix(".safetensors"))


pattern_comment = re.compile(r"(?<!\\)#(?![0-9a-fA-F]{6}).*")
pattern_lora = re.compile(r"<lora:([^:<>]+)(?::(-?[^:<>]*))?>", re.IGNORECASE)
pattern_layer = re.compile(r"<layer:([^>]+)>", re.IGNORECASE)
# @Name mention of a named reference image. Not preceded by a word char, so e-mail
# addresses and the like stay untouched.
pattern_reference_name = re.compile(r"(?<![\w@])@(\w+(?:-\w+)*)")
pattern_weight_expr = re.compile(r"\([^:()]+:(-?[\d.]+)\)")
pattern_wildcard = re.compile(r"(\{[^{}]+\|[^{}]+\})")
pattern_seq_wildcard = re.compile(r"\[\[((?:(?!\]\]).)*)\]\]", re.DOTALL)
pattern_seq_file_wildcard = re.compile(r"__seq:([\w\-./]+?)__")
# Matches either kind of sequential group ([[a|b]] or __seq:name__), so both can be
# combined into a single ordered Cartesian product below.
pattern_seq_combined = re.compile(r"\[\[((?:(?!\]\]).)*)\]\]|__seq:([\w\-./]+?)__", re.DOTALL)
pattern_file_wildcard = re.compile(r"__([\w\-./]+?)__")


def format_option_group(options: list[str], sequential: bool):
    """[[a|b]] or {a|b} with brackets and pipes each on their own line."""
    joined = "\n|\n".join(options)
    open_, close = ("[[", "]]") if sequential else ("{", "}")
    return f"{open_}\n{joined}\n{close}"


def strip_prompt_comments(prompt: str):
    """Strip comments (text after #) from the prompt, unless the # is escaped with a backslash,
    or it's a hex color code."""
    lines = prompt.splitlines()
    stripped_lines = [pattern_comment.sub("", line).replace(r"\#", "#").rstrip() for line in lines]
    return "\n".join(stripped_lines).strip()


def merge_prompt(prompt: str, style_prompt: str, language: str = ""):
    if language and prompt:
        prompt = f"lang:{language} {prompt} lang:en "
    if style_prompt == "":
        return prompt
    elif "{prompt}" in style_prompt:
        return style_prompt.replace("{prompt}", prompt)
    elif prompt == "":
        return style_prompt
    return f"{prompt}, {style_prompt}"


def extract_loras(prompt: str, lora_files: FileCollection):
    loras: list[LoraInput] = []

    def replace(match: re.Match[str]):
        lora_file = None
        input = match[1].lower()

        for file in lora_files:
            if file.source is not FileSource.unavailable:
                lora_filename = Path(file.id).stem.lower()
                lora_normalized = file.name.lower()
                if input in (lora_filename, lora_normalized):
                    lora_file = file
                    break

        if not lora_file:
            error = _("LoRA not found") + f": {input}"
            log.warning(error)
            raise PluginError(error)

        lora_strength: float = lora_file.meta("lora_strength", 1.0)
        if match[2]:
            try:
                lora_strength = float(match[2])
            except ValueError:
                error = _("Invalid LoRA strength for") + f" {input}: {lora_strength}"
                log.warning(error)
                raise ValueError(error)

        loras.append(LoraInput(lora_file.id, lora_strength))
        return ""

    prompt = pattern_lora.sub(replace, prompt)
    return prompt.strip(), loras


def _extract_layers(prompt: str, start_index=1):
    layer_index = start_index
    layer_names: dict[str, int] = {}

    for match in pattern_layer.finditer(prompt):
        name = match[1]
        idx = layer_names.get(name)
        if idx is None:
            idx = layer_index
            layer_index += 1
        layer_names[name] = idx
    return layer_names


def extract_layers(cond: ConditioningInput, region: RegionInput | None = None):
    start_index = 2 if cond.edit_reference else 1
    start_index += sum(1 if c.mode.is_ip_adapter else 0 for c in cond.control)
    if region is None:
        return _extract_layers(cond.positive, start_index)
    else:
        start_index += sum(1 if c.mode.is_ip_adapter else 0 for c in region.control)
        return _extract_layers(region.positive, start_index)


def replace_layers(prompt: str, layer_mapping: dict[str, int], replacement="Picture {}"):
    def replace(match: re.Match[str]):
        return replacement.format(layer_mapping[match[1]])

    return pattern_layer.sub(replace, prompt).strip()


def sanitize_reference_name(name: str):
    return re.sub(r"[^\w\-]", "", re.sub(r"\s+", "_", name.strip()))


def reference_name_mapping(names: list[str], start_index=1) -> dict[str, int]:
    """Map reference names (case-insensitive) to their image number in the prompt.
    `names` holds one entry per input image in order, empty for unnamed ones."""
    mapping: dict[str, int] = {}
    for index, name in enumerate(names, start_index):
        key = sanitize_reference_name(name).lower()
        if key and key not in mapping:
            mapping[key] = index
    return mapping


def extract_reference_mentions(prompt: str):
    mentions: dict[str, str] = {}
    for match in pattern_reference_name.finditer(strip_prompt_comments(prompt)):
        mentions.setdefault(match[1].lower(), match[1])
    return list(mentions.values())


def replace_reference_names(prompt: str, mapping: dict[str, int], replacement=""):
    """Rewrite @Name mentions to the token the model uses for its input images (eg.
    "image {}"). Without a replacement (model has no numbered images) every mention becomes
    the plain name. Otherwise unknown names are left as they are."""

    def replace(match: re.Match[str]):
        name = match[1]
        if not replacement:
            return name
        index = mapping.get(name.lower())
        if index is None:
            log.warning(f"Prompt mentions unknown reference @{name}")
            return match[0]
        return replacement.format(index)

    return pattern_reference_name.sub(replace, prompt)


def eval_wildcards(text: str, seed: int, batch_index: int = 0):
    rng = random.Random(seed)
    wildcard_library = WildcardLibrary.instance()

    def replace_random(match: re.Match[str]):
        options = match[1].strip("{} ").split("|")
        return rng.choice(options).strip()

    def replace_file(match: re.Match[str]):
        options = wildcard_library.get(match[1])
        if not options:
            return match[0]  # unknown/empty file - leave the __name__ visible rather
            # than silently deleting it, so a typo or missing file is obvious
        return rng.choice(options)

    # Cartesian product: each [[...]] or __seq:name__ group gets its own stride
    # dimension, in order of appearance, so a batch cycles through every combination
    def seq_group_options(match: re.Match[str]) -> list[str]:
        if match.group(1) is not None:
            return [o.strip() for o in match.group(1).split("|")]
        options = wildcard_library.get(match.group(2))
        return options or [match.group(0)]  # unknown/empty file - leave tag visible

    seq_matches = list(pattern_seq_combined.finditer(text))
    if seq_matches:
        option_counts = [max(1, len(seq_group_options(m))) for m in seq_matches]
        strides: list[int] = []
        stride = 1
        for count in option_counts:
            strides.append(stride)
            stride *= count

        _group = [0]

        def replace_sequential(match: re.Match[str]):
            options = seq_group_options(match)
            idx = (batch_index // strides[_group[0]]) % len(options)
            _group[0] += 1
            return options[idx]

        text = pattern_seq_combined.sub(replace_sequential, text)

    for __ in range(10):
        prev = text
        text = pattern_file_wildcard.sub(replace_file, text)
        text = pattern_wildcard.sub(replace_random, text)
        if text == prev:
            break

    return text


def select_current_parenthesis_block(
    text: str, cursor_pos: int, open_brackets: list[str], close_brackets: list[str]
) -> tuple[int, int] | None:
    """Select the current parenthesis block that the cursor points to."""
    # Ensure cursor position is within valid range
    cursor_pos = max(0, min(cursor_pos, len(text)))

    # Find the nearest '(' before the cursor
    start = -1
    for open_bracket in open_brackets:
        start = max(start, text.rfind(open_bracket, 0, cursor_pos))

    # If '(' is found, find the corresponding ')' after the cursor
    end = -1
    if start != -1:
        open_parens = 1
        for i in range(start + 1, len(text)):
            if text[i] in open_brackets:
                open_parens += 1
            elif text[i] in close_brackets:
                open_parens -= 1
                if open_parens == 0:
                    end = i
                    break

    # Return the indices only if both '(' and ')' are found
    if start != -1 and end >= cursor_pos:
        return start, end + 1
    else:
        return None


def select_current_word(text: str, cursor_pos: int) -> tuple[int, int]:
    """Select the word the cursor points to."""
    delimiters = r".,\/!?%^*;:{}=`~()<> " + "\t\r\n"
    start = end = cursor_pos

    # seek backward to find beginning
    while start > 0 and text[start - 1] not in delimiters:
        start -= 1

    # seek forward to find end
    while end < len(text) and text[end] not in delimiters:
        end += 1

    return start, end


def select_on_cursor_pos(text: str, cursor_pos: int) -> tuple[int, int]:
    """Return a range in the text based on the cursor_position."""
    return select_current_parenthesis_block(
        text, cursor_pos, ["(", "<"], [")", ">"]
    ) or select_current_word(text, cursor_pos)


class ExprNode:
    def __init__(self, type, value, weight=1.0, children=None):
        self.type = type  # 'text' or 'expr'
        self.value = value  # text or sub-expression
        self.weight = weight  # weight for 'expr' nodes
        self.children = children if children is not None else []  # child nodes

    def __repr__(self):
        if self.type == "text":
            return f"Text('{self.value}')"
        else:
            assert self.type == "expr"
            return f"Expr({self.children}, weight={self.weight})"


def parse_expr(expression: str) -> list[ExprNode]:
    """
    Parses following attention syntax language.
    expr = text | (expr:number)
    expr = text + expr | expr + text
    """

    def parse_segment(segment):
        match = re.match(r"^[([{<](.*?):(-?[\d.]+)[\]})>]$", segment, flags=re.DOTALL)
        if match:
            inner_expr = match.group(1)
            number = float(match.group(2))
            return ExprNode("expr", None, weight=number, children=parse_expr(inner_expr))
        else:
            return ExprNode("text", segment)

    segments = []
    stack = []
    start = 0
    bracket_pairs = {"(": ")", "<": ">"}

    for i, char in enumerate(expression):
        if char in bracket_pairs:
            if not stack:
                if start != i:
                    segments.append(ExprNode("text", expression[start:i]))
                start = i

            stack.append(bracket_pairs[char])
        elif stack and char == stack[-1]:
            stack.pop()
            if not stack:
                node = parse_segment(expression[start : i + 1])
                if node.type == "expr":
                    segments.append(node)
                    start = i + 1
                else:
                    stack.append(char)

    if start < len(expression):
        remaining_text = expression[start:].strip()
        if remaining_text:
            segments.append(ExprNode("text", remaining_text))

    return segments


def edit_attention(text: str, positive: bool) -> str:
    """Edit the attention of text within the prompt."""
    if text == "":
        return text

    segments = parse_expr(text)
    if len(segments) == 1 and segments[0].type == "expr":
        attention_string = text[1 : text.rfind(":")]
        weight = segments[0].weight
        open_bracket = text[0]
        close_bracket = text[-1]
    elif text[0] == "<":
        attention_string = text[1:-1]
        weight = 1.0
        open_bracket = "<"
        close_bracket = ">"
    else:
        attention_string = text
        weight = 1.0
        open_bracket = "("
        close_bracket = ")"

    weight = weight + 0.1 * (1 if positive else -1)
    weight = max(weight, -2.0)
    weight = min(weight, 2.0)

    return (
        attention_string
        if weight == 1.0 and open_bracket == "("
        else f"{open_bracket}{attention_string}:{weight:.1f}{close_bracket}"
    )


_workflow_kind_text = {
    WorkflowKind.generate: "Generate",
    WorkflowKind.refine: "Refine",
    WorkflowKind.inpaint: "Inpaint",
    WorkflowKind.refine_region: "Refine Region",
    WorkflowKind.upscale_simple: "Upscale",
    WorkflowKind.upscale_tiled: "Upscale",
    WorkflowKind.control_image: "Control Image",
    WorkflowKind.custom: "Custom",
    WorkflowKind.dlss5_enhance: "DLSS5 Enhance",
}

_inpaint_mode_text = {
    InpaintMode.fill: "Fill",
    InpaintMode.expand: "Expand",
    InpaintMode.add_object: "Add Content",
    InpaintMode.remove_object: "Remove Content",
    InpaintMode.replace_background: "Replace Background",
    InpaintMode.custom: "Custom",
}


def create_mode_label(params: JobParams) -> str:
    label = _workflow_kind_text.get(params.workflow_kind, params.workflow_kind.name)
    if params.workflow_kind in (WorkflowKind.inpaint, WorkflowKind.refine_region):
        detail = _inpaint_mode_text.get(params.inpaint_mode) if params.inpaint_mode else None
        if detail:
            label = f"{label} ({detail})"
    return label


digital_source_type = "http://cv.iptc.org/newscodes/digitalsourcetype/"


def create_ai_generated_xmp(workflow_kind: WorkflowKind):
    source_type = "trainedAlgorithmicMedia"
    if workflow_kind is not WorkflowKind.generate:
        source_type = "compositeWithTrainedAlgorithmicMedia"
    return f"""<?xpacket begin="\ufeff" id="W5M0MpCehiHzreSzNTczkc9d"?>
<x:xmpmeta xmlns:x="adobe:ns:meta/">
 <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about=""
   xmlns:Iptc4xmpExt="http://iptc.org/std/Iptc4xmpExt/2008-02-29/"
   Iptc4xmpExt:DigitalSourceType="{digital_source_type}{source_type}"/>
 </rdf:RDF>
</x:xmpmeta>
<?xpacket end="w"?>"""


# creates the img text metadata for embedding in PNG files in style like Automatic1111
def create_img_metadata(params: JobParams):
    meta = params.metadata

    prompt = meta.get("prompt_final", meta.get("prompt", ""))
    neg_prompt = meta.get("negative_prompt_final", meta.get("negative_prompt", ""))
    sampler = meta.get("sampler", "")
    steps = meta.get("steps", 0)
    cfg_scale = meta.get("guidance", 0.0)
    model = meta.get("checkpoint", "Unknown")
    seed = params.seed
    width = params.bounds.width
    height = params.bounds.height
    strength = meta.get("strength", None)
    loras = meta.get("loras", [])

    # Embed LoRAs in the prompt
    lora_tags = ""
    for lora in loras:
        if isinstance(lora, dict):
            name = lora.get("name")
            weight = lora.get("weight", 0.0)
        elif isinstance(lora, (list, tuple)) and len(lora) >= 2:
            name, weight = lora[0], lora[1]
        else:
            continue
        if weight != 0:
            lora_tags += f" <lora:{name}:{weight}>"

    full_prompt = f"{prompt.strip()}{lora_tags}"

    # Construct output
    lines = []
    lines.append(full_prompt)
    lines.append(f"Negative prompt: {neg_prompt}")
    lines.append(
        f"Steps: {steps}, Sampler: {sampler}, CFG scale: {cfg_scale}, Seed: {seed}, Size: {width}x{height}, Model hash: unknown, Model: {model}"
    )

    if strength is not None and strength != 1.0:
        lines[-1] += f", Denoising strength: {strength}"

    if mode := create_mode_label(params):
        lines[-1] += f", Mode: {mode}"

    custom_inpaint = meta.get("custom_inpaint")
    if custom_inpaint:
        if custom_inpaint.get("seamless"):
            lines[-1] += ", Seamless: On"
        if custom_inpaint.get("focus"):
            lines[-1] += ", Focus: On"
        if custom_inpaint.get("edit"):
            lines[-1] += ", Edit: On"
        fill = custom_inpaint.get("fill")
        if fill and fill != "none":
            lines[-1] += f", Fill: {fill.capitalize()}"
        context = custom_inpaint.get("context")
        if context:
            lines[-1] += f", Context: {context}"

    # Separate "DLSS5 x: y" keys, as A1111-style parsers split parameters on ", "
    if isinstance(dlss5 := meta.get("dlss5"), dict):
        for key, label in _dlss5_metadata_keys:
            if key in dlss5:
                lines[-1] += f", DLSS5 {label}: {dlss5[key]}"

    return "\n".join(lines)


_dlss5_metadata_keys = [
    ("area", "area"),
    ("style", "style"),
    ("intensity", "intensity"),
    ("tone", "tone"),
    ("structure", "structure"),
    ("skin", "skin"),
    ("model_preset", "model"),
    ("tiles", "tiles"),
]

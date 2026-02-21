from __future__ import annotations

import random
import re
from pathlib import Path
from typing import NamedTuple

from .api import LoraInput
from .files import FileCollection, FileSource
from .jobs import JobParams
from .localization import translate as _
from .util import PluginError
from .util import client_logger as log

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


def strip_prompt_comments(prompt: str):
    """Strip comments (text after #) from the prompt, unless the # is escaped with a backslash."""
    lines = prompt.splitlines()
    stripped_lines = [
        re.sub(r"(?<!\\)#.*", "", line).replace(r"\#", "#").rstrip() for line in lines
    ]
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


_pattern_lora = re.compile(r"<lora:([^:<>]+)(?::(-?[^:<>]*))?>", re.IGNORECASE)
_pattern_layer = re.compile(r"<layer:([^>]+)>", re.IGNORECASE)
_pattern_wildcard = re.compile(r"(\{[^{}]+\|[^{}]+\})")


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

    prompt = _pattern_lora.sub(replace, prompt)
    return prompt.strip(), loras


def extract_layers(prompt: str, replacement="Picture {}", start_index=1):
    layer_index = start_index
    layer_names: list[str] = []

    def replace(match: re.Match[str]):
        nonlocal layer_index
        replacement_text = replacement.format(layer_index)
        layer_index += 1
        layer_names.append(match[1])
        return replacement_text

    prompt = _pattern_layer.sub(replace, prompt)
    return prompt.strip(), layer_names


def eval_wildcards(text: str, seed: int):
    rng = random.Random(seed)

    def replace(match: re.Match[str]):
        wildcard_name = match[1]
        options = wildcard_name.split("|")
        return rng.choice(options).strip("{} ")

    for __ in range(10):
        prev = text
        text = _pattern_wildcard.sub(replace, text)
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

    return "\n".join(lines)

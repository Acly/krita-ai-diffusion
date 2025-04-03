import re
import random
from typing import Dict

from .text import remove_comments_and_normalize, find_matching_brace
from .util import client_logger as log
from .wildcards import process_wildcards, find_wildcard_files, read_wildcard_file

max_iterations = 100  # Maximum iterations for processing dynamic prompts
max_recursion_depth = 10  # Maximum recursion depth for nested templates

_var_name_pattern = r"[a-zA-Z0-9_]+"
_var_ref_re = re.compile(rf"\${{({_var_name_pattern})(?::([^}}]+))?}}")
_weight_re = re.compile(r"^(\d*\.?\d+)::(.*)")
_wildcard_re = re.compile(r"^__([a-zA-Z0-9_*/]+)__$")
_count_spec_re = re.compile(r"^(\d*-?\d*)(\$\$)")
_immediate_var_re = re.compile(r"\${([a-zA-Z0-9_]+)=!")
_regular_var_re = re.compile(r"\${([a-zA-Z0-9_]+)=")
_inner_var_re = re.compile(r"^\${([a-zA-Z0-9_]+)=(.*)}$")
_var_empty_re = re.compile(r"\${=")
_separator_re = re.compile(r"\$\$(.*?)\$\$")


def evaluate_dynamic_prompt(prompt: str) -> str:
    """Process dynamic prompt templates and evaluate them."""
    if not prompt:
        return prompt

    variables: Dict[str, str] = {}
    prompt = remove_comments_and_normalize(prompt)
    prev_prompt = ""
    iteration = 0

    while prev_prompt != prompt and iteration < max_iterations:
        prev_prompt = prompt
        iteration += 1
        try:
            prompt = _process_variables(prompt, variables)
            prompt = process_variants(prompt)
            prompt = process_wildcards(prompt, variables)
        except Exception as e:
            error_msg = f"Template processing error in iteration {iteration}: {str(e)}"
            log.warning(error_msg)
            raise Exception(error_msg) from e

    if iteration >= max_iterations:
        error_msg = f"Dynamic prompt processing reached maximum iterations ({max_iterations})"
        log.warning(error_msg)
        raise Exception(error_msg)

    return prompt


def _process_variables(text: str, variables: Dict[str, str]) -> str:
    """Process variable definitions and references."""
    if not text:
        return ""

    if _var_empty_re.search(text):
        raise Exception("Variable name cannot be empty")

    immediate_vars = set()

    # Process immediate variables
    i = 0
    while i < len(text):
        imm_match = _immediate_var_re.search(text[i:])
        if not imm_match:
            break

        var_name = imm_match.group(1)
        if not var_name:
            raise Exception("Variable name cannot be empty")

        var_start = i + imm_match.start()
        content_start = i + imm_match.end()

        try:
            var_end, _ = find_matching_brace(text[content_start:], 0)
            var_end += content_start
        except Exception:
            i = content_start + 1
            continue

        var_value = text[content_start : var_end - 1]
        processed_value = var_value

        if "{" in processed_value:
            processed_value = process_variants(processed_value)
        if "__" in processed_value:
            processed_value = process_wildcards(processed_value, variables)

        variables[var_name] = processed_value
        immediate_vars.add(var_name)
        text = text[:var_start] + text[var_end:]
        i = var_start

    # Process regular variables
    iteration = 0
    while iteration < max_iterations:
        processed = False
        i = 0
        while i < len(text):
            reg_match = _regular_var_re.search(text[i:])
            if not reg_match:
                break

            var_name = reg_match.group(1)
            if not var_name or var_name in immediate_vars:
                i += reg_match.end()
                continue

            var_start = i + reg_match.start()
            content_start = i + reg_match.end()

            try:
                var_end, _ = find_matching_brace(text[content_start:], 0)
                var_end += content_start
            except Exception:
                i = content_start + 1
                continue

            var_value = text[content_start : var_end - 1]

            if "${" in var_value:
                temp_vars = variables.copy()
                inner_match = _inner_var_re.match(var_value)

                if inner_match:
                    inner_name, inner_value = inner_match.groups()
                    inner_def = f"${{{inner_name}={inner_value}}}"
                    _process_variables(inner_def, temp_vars)
                    var_value = temp_vars.get(inner_name, "")
                else:
                    processed_value = _process_variables(var_value, temp_vars)
                    ref_match = _var_ref_re.match(processed_value)
                    if ref_match and ref_match.group(1) in temp_vars:
                        var_value = temp_vars[ref_match.group(1)]
                    else:
                        var_value = processed_value

                variables.update({
                    k: v for k, v in temp_vars.items() if k != var_name and k not in variables
                })
                processed = True

            variables[var_name] = var_value
            text = text[:var_start] + text[var_end:]
            i = var_start

        if not processed:
            break
        iteration += 1

    if iteration >= max_iterations:
        raise Exception(
            f"Variable definition processing reached maximum iterations ({max_iterations})"
        )

    # Process variable references
    def replace_var_ref(match):
        var_name = match.group(1)
        has_default = match.group(2) is not None
        default_value = match.group(2) if has_default else ""

        if var_name not in variables:
            if not has_default:
                raise Exception(f"Variable '${{{var_name}}}' is not defined")
            return default_value

        value = variables[var_name]
        if var_name not in immediate_vars:
            if "${" in value:
                temp_vars = variables.copy()
                temp_vars.pop(var_name, None)
                value = _process_variables(value, temp_vars)
                variables[var_name] = value
            if "{" in value:
                value = process_variants(value)
            if "__" in value:
                value = process_wildcards(value, variables)

        return value

    iteration = 0
    prev_text = ""
    while prev_text != text and iteration < max_iterations:
        prev_text = text
        text = _var_ref_re.sub(replace_var_ref, text)
        iteration += 1

    if iteration >= max_iterations:
        raise Exception(
            f"Variable reference processing reached maximum iterations ({max_iterations})"
        )

    return text


def process_variants(text: str, recursion_depth: int = 0) -> str:
    """Process variant patterns {option1|option2|option3}."""
    if not text:
        return ""

    if recursion_depth >= max_recursion_depth:
        raise Exception(f"Maximum recursion depth ({max_recursion_depth}) reached")

    result = ""
    i = 0
    depth = 0

    while i < len(text):
        if text[i] == "{":
            depth += 1
            if depth == 1:
                try:
                    j, has_pipe = find_matching_brace(text[i + 1 :], 0)
                    j += i + 1
                    variant_text = text[i:j]
                    result += (
                        _replace_variant(variant_text, recursion_depth)
                        if has_pipe
                        else variant_text
                    )
                    i = j
                    depth -= 1
                    continue
                except Exception as e:
                    raise Exception(f"Unmatched opening brace at position {i}") from e
        elif text[i] == "}":
            depth -= 1
            if depth < 0:
                raise Exception(f"Unmatched closing brace at position {i}")
        result += text[i]
        i += 1

    if depth > 0:
        raise Exception("Unmatched opening brace")
    elif depth < 0:
        raise Exception("Unmatched closing brace")

    return result


def _replace_variant(variant_text: str, recursion_depth: int) -> str:
    """Process a single variant pattern."""
    if not variant_text or len(variant_text) < 2:
        return variant_text

    if recursion_depth >= max_recursion_depth:
        raise Exception(f"Maximum recursion depth ({max_recursion_depth}) reached")

    inner = variant_text[1:-1]
    selection_count = 1
    separator = ", "

    # Handle count specification and separator
    count_match = _count_spec_re.match(inner)
    if count_match:
        count_spec = count_match.group(1)
        sep_match = _separator_re.search(inner)
        if sep_match:
            separator = sep_match.group(1)
            options_start = inner.find("$$", inner.find("$$") + 2) + 2
        else:
            options_start = inner.find("$$") + 2
        inner = inner[options_start:]

        if "-" in count_spec:
            parts = count_spec.split("-")
            lower = int(parts[0]) if parts[0] else 1
            upper = int(parts[1]) if len(parts) > 1 and parts[1] else None
        else:
            selection_count = int(count_spec) if count_spec else 1
            lower = upper = selection_count
    else:
        lower = upper = 1

    # Split options and handle wildcards
    options = []
    start = 0
    depth = 0
    for i, char in enumerate(inner):
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
        elif char == "|" and depth == 0:
            options.append(inner[start:i].strip())
            start = i + 1
    options.append(inner[start:].strip())

    # Expand wildcards and process weights
    weighted_options = []
    for opt in options:
        wildcard_match = _wildcard_re.match(opt)
        if wildcard_match:
            files = find_wildcard_files(wildcard_match.group(1))
            if files:
                values = read_wildcard_file(random.choice(files))
                if values:
                    weighted_options.extend((1.0, val) for val in values)
                    continue

        weight_match = _weight_re.match(opt)
        weight = float(weight_match.group(1)) if weight_match else 1.0
        text = weight_match.group(2) if weight_match else opt
        weighted_options.append((weight, text))

    # Select options
    max_options = len(weighted_options)
    if upper is None or upper > max_options:
        upper = max_options
    lower = min(lower, upper)
    selection_count = random.randint(lower, upper) if lower < upper else lower

    if selection_count >= max_options:
        selected = [opt[1] for opt in weighted_options]
    else:
        selected = []
        remaining = list(range(max_options))
        weights = [opt[0] for opt in weighted_options]

        while len(selected) < selection_count and remaining:
            total = sum(weights[i] for i in remaining)
            if total <= 0:
                break

            r = random.uniform(0, total)
            cumulative = 0

            for idx in remaining:
                cumulative += weights[idx]
                if cumulative >= r:
                    selected.append(weighted_options[idx][1])
                    remaining.remove(idx)
                    break

    # Process nested variants
    processed = [
        process_variants(opt, recursion_depth + 1) if "{" in opt else opt for opt in selected
    ]

    return separator.join(processed)

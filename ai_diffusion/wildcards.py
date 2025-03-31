import re
import random
import glob
import os
from pathlib import Path
from typing import List, Dict

from .util import user_data_dir, plugin_dir, client_logger as log

_custom_wildcard_dirs: List[Path] = []

# Compile regex patterns once
_param_re = re.compile(r"([a-zA-Z0-9_]+)=([^,]+)")
_valid_name_re = re.compile(r"^[a-zA-Z0-9_*/?!\[\]-]+$")
_invalid_bracket_content_re = re.compile(r"[^a-zA-Z0-9!^-]")
_param_default_re = re.compile(r"\${([a-zA-Z0-9_]+):([^}]+)}")
_simple_var_re = re.compile(r"\${([a-zA-Z0-9_]+)}")
_param_wildcard_re = re.compile(r"__(.*?)\(([^)]*)\)__")
_simple_wildcard_re = re.compile(r"__(.*?)__")


def set_wildcard_dirs(dirs: List[Path]) -> List[Path]:
    """Set custom wildcard directories with highest search priority."""
    global _custom_wildcard_dirs
    previous = _custom_wildcard_dirs.copy()
    _custom_wildcard_dirs = dirs
    return previous


def reset_wildcard_dirs() -> None:
    """Reset custom wildcard directories."""
    global _custom_wildcard_dirs
    _custom_wildcard_dirs = []


def _raise_error(msg: str) -> None:
    """Helper to log warning and raise exception."""
    log.warning(msg)
    raise Exception(msg)


def _is_valid_wildcard_name(name: str) -> bool:
    """Validate wildcard name for security and syntax."""
    if not name or ".." in name or name.startswith("/") or "\\" in name:
        return False

    if not _valid_name_re.match(name):
        return False

    # Validate bracket pairs and their contents
    stack = []
    i = 0
    while i < len(name):
        if name[i] == "[":
            stack.append(i)
        elif name[i] == "]":
            if not stack:
                return False
            start = stack.pop()
            content = name[start + 1 : i]

            if not content or _invalid_bracket_content_re.search(content):
                return False

            if content.startswith(("!", "^")):
                if not content[1:]:
                    return False
        i += 1

    return not stack


def process_wildcards(text: str, variables: Dict[str, str], recursion_depth: int = 0) -> str:
    """Process wildcard patterns __name__ and __name(param=value)__."""

    if not text:
        return ""

    def replace_param_wildcard(match):
        name, params_str = match.group(1), match.group(2)

        if not _is_valid_wildcard_name(name):
            _raise_error(f"Invalid wildcard name: {name}")

        params = {}
        if params_str:
            parts = [p.strip() for p in params_str.split(",")]
            if any(p and "=" not in p for p in parts):
                _raise_error(f"Invalid parameter syntax in wildcard: __{name}({params_str})__")

            params = {m.group(1): m.group(2).strip() for m in _param_re.finditer(params_str)}

        return _process_wildcard_value(name, params, variables, recursion_depth)

    def replace_simple_wildcard(match):
        name = match.group(1)
        if not _is_valid_wildcard_name(name):
            _raise_error(f"Invalid wildcard name: {name}")
        return _process_wildcard_value(name, variables, variables, recursion_depth)

    try:
        text = _param_wildcard_re.sub(replace_param_wildcard, text)
        text = _simple_wildcard_re.sub(replace_simple_wildcard, text)
    except Exception:
        raise

    return text


def _process_wildcard_value(
    wildcard_name: str, params: Dict[str, str], variables: Dict[str, str], recursion_depth: int
) -> str:
    """Process a wildcard, handling parameters and nested wildcards."""
    from .dynamic_prompts import process_variants  # avoid circular import

    if not _is_valid_wildcard_name(wildcard_name):
        _raise_error(f"Invalid wildcard name: {wildcard_name}")

    matching_files = find_wildcard_files(wildcard_name)
    if not matching_files:
        _raise_error(f"Wildcard not found: {wildcard_name}")

    # Handle directory/** wildcards
    if wildcard_name.endswith("/**"):
        all_values = []
        for file_path in matching_files:
            try:
                all_values.extend(read_wildcard_file(file_path))
            except Exception:
                continue
        if not all_values:
            _raise_error(f"No usable entries found in directory wildcard: {wildcard_name}")
        selected_value = random.choice(all_values)
    else:
        file_path = random.choice(matching_files)
        values = read_wildcard_file(file_path)
        if not values:
            _raise_error(f"Empty wildcard file: {wildcard_name} ({file_path})")
        selected_value = random.choice(values)

    # Process parameters and variables
    param_vars = params.copy()

    # Apply explicit parameters
    for name, value in param_vars.items():
        selected_value = selected_value.replace(f"${{{name}}}", value)

    # Handle parameter defaults
    def replace_default_params(match):
        name, default = match.group(1), match.group(2)
        return (
            param_vars.get(name)
            or (variables.get(name) if name not in param_vars else None)
            or default
        )

    selected_value = _param_default_re.sub(replace_default_params, selected_value)

    # Handle remaining variables
    def replace_simple_vars(match):
        name = match.group(1)
        return match.group(0) if name in param_vars else variables.get(name, "")

    selected_value = _simple_var_re.sub(replace_simple_vars, selected_value)

    # Process nested wildcards and variants
    if "{" in selected_value:
        selected_value = process_variants(selected_value, recursion_depth + 1)
    if "__" in selected_value:
        selected_value = process_wildcards(selected_value, variables, recursion_depth + 1)

    return selected_value


def find_wildcard_files(wildcard_name: str) -> List[Path]:
    """Find all wildcard files matching the given name pattern."""
    if not _is_valid_wildcard_name(wildcard_name):
        _raise_error(f"Invalid wildcard name: {wildcard_name}")

    # Build search paths in priority order (highest to lowest)
    wildcard_paths = _custom_wildcard_dirs + [user_data_dir / "wildcards", plugin_dir / "wildcards"]

    is_directory_wildcard = wildcard_name.endswith("/**")
    base_dir = wildcard_name[:-3] if is_directory_wildcard else wildcard_name
    has_glob_chars = any(c in wildcard_name for c in "*?[")
    has_dir_path = "/" in wildcard_name

    def search_for_matches(dir_path: Path) -> List[Path]:
        if not dir_path.exists():
            return []

        matches = set()

        if is_directory_wildcard:
            base_path = dir_path / base_dir
            if base_path.exists() and base_path.is_dir():
                for root, _, files in os.walk(str(base_path), followlinks=True):
                    matches.update(Path(os.path.join(root, f)) for f in files if f.endswith(".txt"))
            return list(matches)

        if not has_glob_chars:
            file_path = dir_path / f"{wildcard_name}.txt"
            if file_path.exists():
                matches.add(file_path)
            return list(matches)

        # Handle glob patterns
        for pattern in [
            str(dir_path / f"{wildcard_name}.txt"),
            *([] if has_dir_path else [str(dir_path / f"*/{wildcard_name}.txt")]),
            str(dir_path / f"**/{wildcard_name}.txt"),
        ]:
            matches.update(Path(p) for p in glob.glob(pattern, recursive=True))

        return list(matches)

    # Track files by their relative path, using priority to resolve duplicates
    result_map = {}  # relative_path -> absolute_path
    for priority, wildcard_dir in enumerate(wildcard_paths):
        for match in search_for_matches(wildcard_dir):
            try:
                rel_path = match.relative_to(wildcard_dir)
                if rel_path not in result_map:  # Higher priority paths are processed first
                    result_map[rel_path] = match
            except ValueError:
                continue  # Skip if relative_to fails

    return list(result_map.values())


def read_wildcard_file(file_path: Path) -> List[str]:
    """Read contents from a wildcard file."""
    if not file_path or not file_path.exists():
        _raise_error(f"Wildcard file does not exist: {file_path}")
        return []

    if file_path.suffix.lower() != ".txt":
        _raise_error(f"Unsupported wildcard file extension: {file_path.suffix}")
        return []

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip() and not line.strip().startswith("#")]
    except Exception as e:
        _raise_error(f"Error reading wildcard file {file_path}: {str(e)}")
        return []

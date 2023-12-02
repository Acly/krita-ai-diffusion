import re
from typing import Tuple


def select_current_parenthesis_block(
    text: str, cursor_pos: int, open_bracket: str, close_bracket: str
) -> Tuple[int, int] | None:
    """ """
    # Ensure cursor position is within valid range
    cursor_pos = max(0, min(cursor_pos, len(text)))

    # Find the nearest '(' before the cursor
    start = text.rfind(open_bracket, 0, cursor_pos)

    # If '(' is found, find the corresponding ')' after the cursor
    end = -1
    if start != -1:
        open_parens = 1
        for i in range(start + 1, len(text)):
            if text[i] == open_bracket:
                open_parens += 1
            elif text[i] == close_bracket:
                open_parens -= 1
                if open_parens == 0:
                    end = i
                    break

    # Return the indices only if both '(' and ')' are found
    if start != -1 and end != -1:
        return (start, end + 1)
    else:
        return None


def select_current_word(text: str, cursor_pos: int) -> Tuple[int, int]:
    """ """
    delimiters = r".,\/!?%^*;:{}=`~() " + "\t\r\n"
    start = end = cursor_pos

    # seek backward to find beginning
    while start > 0 and text[start - 1] not in delimiters:
        start -= 1

    # seek forward to find end
    while end < len(text) and text[end] not in delimiters:
        end += 1

    return start, end


def select_on_cursor_pos(text: str, cursor_pos: int) -> Tuple[int, int]:
    """ """
    return (
        select_current_parenthesis_block(text, cursor_pos, "(", ")")
        or select_current_parenthesis_block(text, cursor_pos, "[", "]")
        or select_current_parenthesis_block(text, cursor_pos, "<", ">")
        or select_current_word(text, cursor_pos)
    )


def edit_attention(text: str, start: int, end: int, positive: bool) -> str:
    """ """
    target_text = text[start:end]
    if target_text == "":
        return text

    pattern = r"(.+?):(\s*\d*\.?\d+\s*)"
    match = (
        re.match("^\\(" + pattern + "\\)$", target_text)
        or re.search("^\\[" + pattern + "\\]$", target_text)
        or re.search("^<" + pattern + ">$", target_text)
    )
    if match:
        attention_string = match.group(1)
        weight = float(match.group(2))
        open_bracket = target_text[0]
        close_bracket = target_text[-1]
    else:
        attention_string = target_text
        weight = 1.0
        open_bracket = "("
        close_bracket = ")"

    weight = weight + 0.1 * (1 if positive else -1)
    weight = max(weight, 0.0)
    weight = min(weight, 2.0)

    content = (
        attention_string
        if weight == 1.0
        else f"{open_bracket}{attention_string}:{weight:.1f}{close_bracket}"
    )

    return text[:start] + content + text[end:]

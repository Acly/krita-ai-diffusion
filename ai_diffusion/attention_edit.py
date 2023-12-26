import re
from typing import Tuple, List


def select_current_parenthesis_block(
    text: str, cursor_pos: int, open_brackets: list[str], close_brackets: list[str]
) -> Tuple[int, int] | None:
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


def select_current_word(text: str, cursor_pos: int) -> Tuple[int, int]:
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


def select_on_cursor_pos(text: str, cursor_pos: int) -> Tuple[int, int]:
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


def parse_expr(expression: str) -> List[ExprNode]:
    """
    Parses following attention syntax language.
    expr = text | (expr:number)
    expr = text + expr | expr + text
    """

    def parse_segment(segment):
        match = re.match(r"^[([{<](.*?):(-?[\d.]+)[\]})>]$", segment)
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

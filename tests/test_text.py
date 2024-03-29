from ai_diffusion.text import merge_prompt, extract_loras, edit_attention, select_on_cursor_pos
from ai_diffusion.api import LoraInput


def test_merge_prompt():
    assert merge_prompt("a", "b") == "a, b"
    assert merge_prompt("", "b") == "b"
    assert merge_prompt("a", "") == "a"
    assert merge_prompt("", "") == ""
    assert merge_prompt("a", "b {prompt} c") == "b a c"
    assert merge_prompt("", "b {prompt} c") == "b  c"


def test_extract_loras():
    loras = [
        "/path/to/Lora-One.safetensors",
        "Lora-two.safetensors",
    ]

    assert extract_loras("a ship", loras) == ("a ship", [])
    assert extract_loras("a ship <lora:lora-one>", loras) == (
        "a ship",
        [LoraInput(loras[0], 1.0)],
    )
    assert extract_loras("a ship <lora:LoRA-one>", loras) == (
        "a ship",
        [LoraInput(loras[0], 1.0)],
    )
    assert extract_loras("a ship <lora:lora-one:0.0>", loras) == (
        "a ship",
        [LoraInput(loras[0], 0.0)],
    )
    assert extract_loras("a ship <lora:lora-two:0.5>", loras) == (
        "a ship",
        [LoraInput(loras[1], 0.5)],
    )
    assert extract_loras("a ship <lora:lora-two:-1.0>", loras) == (
        "a ship",
        [LoraInput(loras[1], -1.0)],
    )

    try:
        extract_loras("a ship <lora:lora-three>", loras)
    except Exception as e:
        assert str(e).startswith("LoRA not found")

    try:
        extract_loras("a ship <lora:lora-one:test-invalid-str>", loras)
    except Exception as e:
        assert str(e).startswith("Invalid LoRA strength")


class TestEditAttention:
    def test_empty_selection(self):
        assert edit_attention("", positive=True) == ""

    def test_adjust_from_1(self):
        assert edit_attention("bar", positive=True) == "(bar:1.1)"

    def test_adjust_to_1(self):
        assert edit_attention("(bar:1.1)", positive=False) == "bar"

    def test_upper_bound(self):
        assert edit_attention("(bar:2.0)", positive=True) == "(bar:2.0)"
        assert edit_attention("(bar:1.95)", positive=True) == "(bar:2.0)"

    def test_lower_bound(self):
        assert edit_attention("(bar:-1.95)", positive=False) == "(bar:-2.0)"
        assert edit_attention("(bar:-2.0)", positive=False) == "(bar:-2.0)"

    def test_single_digit(self):
        assert edit_attention("(bar:0)", positive=True) == "(bar:0.1)"
        assert edit_attention("(bar:.1)", positive=True) == "(bar:0.2)"

    def test_nested(self):
        assert edit_attention("(foo:1.0), bar", positive=True) == "((foo:1.0), bar:1.1)"
        assert (
            edit_attention("(1girl:1.1), foo, (bar:1.3)", positive=True)
            == "((1girl:1.1), foo, (bar:1.3):1.1)"
        )
        assert (
            edit_attention("((foo:1.5), (bar:0.9), baz:1.4)", positive=True)
            == "((foo:1.5), (bar:0.9), baz:1.5)"
        )
        assert edit_attention("(:):1.0)", positive=True) == "(:):1.1)"

    def test_invalid_weight(self):
        assert edit_attention("(foo:bar)", positive=True) == "((foo:bar):1.1)"

    def test_no_weight(self):
        assert edit_attention("(foo)", positive=True) == "((foo):1.1)"

    def test_angle_bracket(self):
        assert edit_attention("<bar:1.0>", positive=True) == "<bar:1.1>"
        assert edit_attention("<foo:bar:1.0>", positive=True) == "<foo:bar:1.1>"
        assert edit_attention("<foo:bar:1.1>", positive=False) == "<foo:bar:1.0>"
        assert edit_attention("<foo:bar:0.0>", positive=False) == "<foo:bar:-0.1>"


class TestSelectOnCursorPos:
    def test_word_selection(self):
        assert select_on_cursor_pos("(foo:1.3), bar, baz", 12) == (11, 14)
        assert select_on_cursor_pos("((foo):1.3), bar, baz", 14) == (13, 16)
        assert select_on_cursor_pos("foo, bar, baz", 5) == (5, 8)
        assert select_on_cursor_pos("foo, bar, baz", 5) == (5, 8)

    def test_range_selection(self):
        assert select_on_cursor_pos("(foo:1.3), bar, baz", 1) == (0, 9)
        assert select_on_cursor_pos("foo, (bar:1.1), baz", 6) == (5, 14)
        assert select_on_cursor_pos("foo, (bar:1.1) <bar:baz:1.0>", 16) == (15, 28)

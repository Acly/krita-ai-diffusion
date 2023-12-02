from ai_diffusion.attention_edit import edit_attention


def edit_whole(text: str, positive: bool) -> str:
    return edit_attention(text, start=0, end=len(text), positive=positive)


class TestEditAttention:
    def test_empty_selection(self):
        assert edit_attention("foo", start=1, end=1, positive=True) == "foo"

    def test_adjust_from_1(self):
        assert (
            edit_attention("foo, bar, foobar", start=5, end=8, positive=True)
            == "foo, (bar:1.1), foobar"
        )

    def test_adjust_to_1(self):
        assert (
            edit_attention("foo, (bar:1.1), foobar", start=5, end=14, positive=False)
            == "foo, bar, foobar"
        )

    def test_upper_bound(self):
        assert edit_whole("(bar:2.0)", positive=True) == "(bar:2.0)"
        assert edit_whole("(bar:1.95)", positive=True) == "(bar:2.0)"

    def test_lower_bound(self):
        assert edit_whole("(bar:0.0)", positive=False) == "(bar:0.0)"
        assert edit_whole("(bar:0.01)", positive=False) == "(bar:0.0)"

    def test_single_digit(self):
        assert edit_whole("(bar:0)", positive=True) == "(bar:0.1)"
        assert edit_whole("(bar:.1)", positive=True) == "(bar:0.2)"

    def test_square_bracket(self):
        assert edit_whole("[bar:1.0]", positive=True) == "[bar:1.1]"
        assert edit_whole("[foo:bar:1.0]", positive=True) == "[foo:bar:1.1]"

    def test_angle_bracket(self):
        assert edit_whole("<bar:1.0>", positive=True) == "<bar:1.1>"
        assert edit_whole("<foo:bar:1.0>", positive=True) == "<foo:bar:1.1>"

    def test_nested(self):
        assert edit_whole("(foo:1.0), bar", positive=True) == "((foo:1.0), bar:1.1)"

    def test_invalid_weight(self):
        assert edit_whole("(foo:bar)", positive=True) == "((foo:bar):1.1)"

    def test_no_weight(self):
        assert edit_whole("(foo)", positive=True) == "((foo):1.1)"

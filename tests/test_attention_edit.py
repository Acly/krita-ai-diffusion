from ai_diffusion.attention_edit import edit_attention


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
        assert edit_attention("(bar:0.0)", positive=False) == "(bar:0.0)"
        assert edit_attention("(bar:0.01)", positive=False) == "(bar:0.0)"

    def test_single_digit(self):
        assert edit_attention("(bar:0)", positive=True) == "(bar:0.1)"
        assert edit_attention("(bar:.1)", positive=True) == "(bar:0.2)"

    def test_square_bracket(self):
        assert edit_attention("[bar:1.0]", positive=True) == "[bar:1.1]"
        assert edit_attention("[foo:bar:1.0]", positive=True) == "[foo:bar:1.1]"

    def test_angle_bracket(self):
        assert edit_attention("<bar:1.0>", positive=True) == "<bar:1.1>"
        assert edit_attention("<foo:bar:1.0>", positive=True) == "<foo:bar:1.1>"

    def test_nested(self):
        assert edit_attention("(foo:1.0), bar", positive=True) == "((foo:1.0), bar:1.1)"
        assert (
            edit_attention("(1girl:1.1), foo, (bar:1.3)", positive=True)
            == "((1girl:1.1), foo, (bar:1.3):1.1)"
        )
        assert edit_attention("(:):1.0)", positive=True) == "(:):1.1)"

    def test_invalid_weight(self):
        assert edit_attention("(foo:bar)", positive=True) == "((foo:bar):1.1)"

    def test_no_weight(self):
        assert edit_attention("(foo)", positive=True) == "((foo):1.1)"

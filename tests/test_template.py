"""Tests for PromptTemplate."""

import pytest
from prompt_eval.template import PromptTemplate


class TestVariableExtraction:
    def test_single_variable(self):
        tpl = PromptTemplate("Hello {{ name }}!")
        assert tpl.variables == ["name"]

    def test_multiple_variables(self):
        tpl = PromptTemplate("{{ a }} and {{ b }} then {{ c }}")
        assert tpl.variables == ["a", "b", "c"]

    def test_deduplicated(self):
        tpl = PromptTemplate("{{ x }} + {{ x }} = 2{{ x }}")
        assert tpl.variables == ["x"]

    def test_no_variables(self):
        tpl = PromptTemplate("No placeholders here.")
        assert tpl.variables == []

    def test_whitespace_tolerance(self):
        tpl = PromptTemplate("{{a}} and {{  b  }}")
        assert tpl.variables == ["a", "b"]


class TestRender:
    def test_basic_render(self):
        tpl = PromptTemplate("Say {{ word }}")
        assert tpl.render(word="hello") == "Say hello"

    def test_multiple_vars(self):
        tpl = PromptTemplate("{{ greeting }}, {{ name }}!")
        result = tpl.render(greeting="Hi", name="Alice")
        assert result == "Hi, Alice!"

    def test_missing_variable_raises(self):
        tpl = PromptTemplate("{{ a }} and {{ b }}")
        with pytest.raises(ValueError, match="Missing template variables"):
            tpl.render(a="x")

    def test_extra_kwargs_are_ignored(self):
        tpl = PromptTemplate("Hello {{ name }}")
        result = tpl.render(name="Bob", unused="data")
        assert result == "Hello Bob"

    def test_numeric_values(self):
        tpl = PromptTemplate("Count: {{ n }}")
        assert tpl.render(n=42) == "Count: 42"


class TestPartial:
    def test_fills_some_variables(self):
        tpl = PromptTemplate("{{ a }} and {{ b }}")
        partial = tpl.partial(a="X")
        assert partial.variables == ["b"]
        assert partial.render(b="Y") == "X and Y"

    def test_preserves_name(self):
        tpl = PromptTemplate( {{ x }}", name="my-tpl")
        partial = tpl.partial(x="done")
        assert partial.name == "my-tpl"


class TestMisc:
    def test_frozen(self):
        tpl = PromptTemplate("{{ x }}")
        with pytest.raises(AttributeError):
            tpl.template = "new"  # type: ignore

    def test_str(self):
        tpl = PromptTemplate("Hello {{ name }}", name="greeting")
        s = str(tpl)
        assert "greeting" in s
        assert "Hello" in s

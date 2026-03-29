"""Tests for evaluation reporters."""

import json

from prompt_eval.template import PromptTemplate
from prompt_eval.judges import ExactMatch
from prompt_eval.runner import EvalRunner, EvalCase
from prompt_eval.reporter import ConsoleReporter, JSONReporter, MarkdownReporter


def echo_llm(prompt: str) -> str:
    return prompt.strip().split()[-1]


def _make_summary():
   """Create a simple summary for reporter tests."""
    template = PromptTemplate("Say {{ word }}")
    runner = EvalRunner(template=template, judge=ExactMatch(), llm_fn=echo_llm)
    cases = [
        EvalCase(inputs={"word": "hello"}, expected="hello"),
        EvalCase(inputs={"word": "world"}, expected="world"),
    ]
    return runner.run(cases)


class TestConsoleReporter:
    def test_output_contains_key_info(self):
        summary = _make_summary()
        text = ConsoleReporter().report(summary, output=None)
        assert "EVALUATION RESULTS" in text
        assert "Pass rate" in text
        assert "2" in text

    def test_shows_pass_markers(self):
        summary = _make_summary()
        text = ConsoleReporter().report(summary, output=None)
        assert "[+]" in text  # passing cases


class TestJSONReporter:
    def test_valid_json(self):
        summary = _make_summary()
        text = JSONReporter().report(summary)
        data = json.loads(text)
        assert data["total"] == 2
        assert data["passed"] == 2
        assert len(data["results"]) == 2

    def test_result_structure(self):
        summary = _make_summary()
        data = json.loads(JSONReporter().report(summary))
        result = data["results"][0]
        assert "score" in result
        assert "inputs" in result
        assert "expected" in result
        assert "response" in result


class TestMarkdownReporter:
    def test_contains_table(self):
        summary = _make_summary()
        text = MarkdownReporter().report(summary)
        assert "| # |" in text
        assert "| Score |" in text

    def test_contains_statistics(self):
        summary = _make_summary()
        text = MarkdownReporter().report(summary)
        assert "Mean" in text
        assert "Median" in text

    def test_contains_header(self):
        summary = _make_summary()
        text = MarkdownReporter().report(summary)
        assert "## Evaluation Results" in text

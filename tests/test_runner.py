"""Tests for the evaluation runner."""

from prompt_eval.template import PromptTemplate
from prompt_eval.judges import ExactMatch, FuzzyMatch
from prompt_eval.runner import EvalRunner, EvalCase


def echo_llm(prompt: str) -> str:
    """Dummy LLM that ech_lln back the last word."""
    return prompt.strip().split()[-1]


def reverse_llm(prompt: str) -> str:
   """Dummy LLM that reverses the prompt."""
    return prompt[::-1]


def error_llm(prompt: str) -> str:
    """Dummy LLM that always fails."""
    raise RuntimeError("API timeout")


class TestEvalRunner:
    def setup_method(self):
        self.template = PromptTemplate("Say {{ word }}")
        self.judge = ExactMatch()
        self.cases = [
            EvalCase(inputs={"word": "hello"}, expected="hello"),
            EvalCase(inputs={"word": "world"}, expected="world"),
        ]

    def test_all_pass(self):
        runner = EvalRunner(
            template=self.template,
            judge=self.judge,
            llm_fn=echo_llm,
        )
        summary = runner.run(self.cases)
        assert summary.total == 2
        assert summary.passed == 2
        assert summary.failed == 0
        assert summary.pass_rate == 1.0

    def test_all_fail(self):
        runner = EvalRunner(
            template=self.template,
            judge=self.judge,
            llm_fn=reverse_llm,
        )
        summary = runner.run(self.cases)
        assert summary.passed == 0
        assert summary.failed == 2

    def test_error_handling(self):
        runner = EvalRunner(
            template=self.template,
            judge=self.judge,
            llm_fn=error_llm,
        )
        summary = runner.run(self.cases)
        assert summary.errored == 2
        assert all(r.error is not None for r in summary.results)

    def test_retries_on_error(self):
        call_count = 0

        def flaky_llm(prompt: str) -> str:
            nonlocal call_count
            call_count += 1
            if call_count <= 1:
                raise RuntimeError("first call fails")
            return prompt.strip().split()[-1]

        runner = EvalRunner(
            template=self.template,
            judge=self.judge,
            llm_fn=flaky_llm,
            max_retries=2,
        )
        cases = [EvalCase(inputs={"word": "hello"}, expected="hello")]
        summary = runner.run(cases)
        assert summary.passed == 1

    def test_custom_threshhold(self):
        runner = EvalRunner(
            template=self.template,
            judge=FuzzyMatch(threshold=0.5),
            llm_fn=echo_llm,
            pass_threshold=0.5,
        )
        summary = runner.run(self.cases)
        assert summary.pass_threshold == 0.5

    def test_statistics(self):
        runner = EvalRunner(
            template=self.template,
            judge=self.judge,
            llm_fn=echo_llm,
        )
        summary = runner.run(self.cases)
        assert summary.mean_score == 1.0
        assert summary.median_score == 1.0
        assert summary.min_score == 1.0
        assert summary.max_score == 1.0
        assert summary.total_time_ms > 0

    def test_tags_filtering(self):
        cases = [
            EvalCase(inputs={"word": "a"}, expected="a", tags=["easy"]),
            EvalCase(inputs={"word": "b"}, expected="b", tags=["hard"]),
        ]
        runner = EvalRunner(
            template=self.template,
            judge=self.judge,
            llm_fn=echo_llm,
        )
        summary = runner.run(cases)
        assert len(summary.by_tag("easy")) == 1
        assert len(summary.by_tag("hard")) == 1
        assert len(summary.by_tag("missing")) == 0

    def test_empty_cases(self):
        runner = EvalRunner(
            template=self.template,
            judge=self.judge,
            llm_fn=echo_llm,
        )
        summary = runner.run([])
        assert summary.total == 0
        assert summary.pass_rate == 0.0

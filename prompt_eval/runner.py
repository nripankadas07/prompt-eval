"""
Evaluation runner â the core orchestration engine.

Takes a list of eval cases, runs each through a prompt template + LLM,
scores the output with one or more judges, and collects results into
a summary with aggregate statistics.

Usage::

    runner = EvalRunner(
        template=PromptTemplate("Translate to French: {{ text }}"),
        judge=ExactMatch(ignore_case=True),
        llm_fn=my_openai_call,
    )
    summary = runner.run(cases)
"""

from __future__ import annotations

import time
import statistics
from dataclasses import dataclass, field
from typing import Any, Callable

from .template import PromptTemplate
from .judges import Judge, Score


# ââ Data classes ââââââââââââââââââââââââââââââââââââââââââââââââââââââ

@dataclass
class EvalCase:
    """A single evaluation test case."""

    inputs: dict[str, str]
    expected: str
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalResult:
    """Result of evaluating a single case."""

    case: EvalCase
    prompt: str
    response: str
    score: Score
    latency_ms: float
    error: str | None = None


@dataclass
class EvalSummary:
    """Aggregate statistics across all evaluated cases."""

    results: list[EvalResult]
    total: int
    passed: int
    failed: int
    errored: int
    mean_score: float
    median_score: float
    min_score: float
    max_score: float
    std_dev: float
    mean_latency_ms: float
    total_time_ms: float
    pass_threshold: float

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total if self.total > 0 else 0.0

    def by_tag(self, tag: str) -> list[EvalResult]:
        """Filter results by tag."""
        return [r for r in self.results if tag in r.case.tags]

    def failing(self) -> list[EvalResult]:
        """Return only failing results."""
        return [r for r in self.results if r.score.value < self.pass_threshold]


# ââ Runner ââââââââââââââââââââââââââââââââââââââââââââââââââââââââ

class EvalRunner:
    """Orchestrates prompt evaluation across a set of test cases.

    Parameters
    ----------
    template : PromptTemplate
        The prompt template to evaluate.
    judge : Judge
        The judge (or composite) that scores responses.
    llm_fn : callable
        ``(prompt: str) -> str`` â calls the LLM and returns its text.
    pass_threshold : float
        Score >= this value counts as a pass (default 0.7).
    max_retries : int
        Retries on error per case (default 0).
    """

    def __init__(
        self,
        template: PromptTemplate,
        judge: Judge,
        llm_fn: Callable[[str], str],
        *,
        pass_threshold: float = 0.7,
        max_retries: int = 0,
    ):
        self.template = template
        self.judge = judge
        self.llm_fn = llm_fn
        self.pass_threshold = pass_threshold
        self.max_retries = max_retries

    def evaluate_one(self, case: EvalCase) -> EvalResult:
        """Evaluate a single case."""
        prompt = self.template.render(**case.inputs)
        error = None
        response = ""
        latency_ms = 0.0

        for attempt in range(1 + self.max_retries):
            try:
                t0 = time.perf_counter()
                response = self.llm_fn(prompt)
                latency_ms = (time.perf_counter() - t0) * 1000
                error = None
                break
            except Exception as e:
                error = f"attempt {attempt + 1}: {e}"
                response = ""
                latency_ms = 0.0

        if error:
            return EvalResult(
                case=case,
                prompt=prompt,
                response=response,
                score=Score(value=0.0, reason=f"error: {error}"),
                latency_ms=latency_ms,
                error=error,
            )

        score = self.judge.score(response, case.expected)
        return EvalResult(
            case=case,
            prompt=prompt,
            response=response,
            score=score,
            latency_ms=latency_ms,
        )

    def run(self, cases: list[EvalCase]) -> EvalSummary:
        """Run evaluation across all cases and return summary statistics."""
        t0 = time.perf_counter()
        results = [self.evaluate_one(c) for c in cases]
        total_time = (time.perf_counter() - t0) * 1000

        scores = [r.score.value for r in results]
        latencies = [r.latency_ms for r in results]

        passed = sum(1 for s in scores if s >= self.pass_threshold)
        errored = sum(1 for r in results if r.error is not None)
        failed = len(results) - passed - errored

        return EvalSummary(
            results=results,
            total=len(results),
            passed=passed,
            failed=failed,
            errored=errored,
            mean_score=statistics.mean(scores) if scores else 0.0,
            median_score=statistics.median(scores) if scores else 0.0,
            min_score=min(scores) if scores else 0.0,
            max_score=max(scores) if scores else 0.0,
            std_dev=statistics.stdev(scores) if len(scores) > 1 else 0.0,
            mean_latency_ms=statistics.mean(latencies) if latencies else 0.0,
            total_time_ms=total_time,
            pass_threshold=self.pass_threshold,
        )

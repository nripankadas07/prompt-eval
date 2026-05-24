"""Prompt regression demo that runs without API keys."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prompt_eval import Contains, EvalCase, EvalRunner, PromptTemplate


def deterministic_llm(prompt: str) -> str:
    if "refund" in prompt.lower():
        return "category: billing"
    if "crash" in prompt.lower():
        return "category: bug"
    return "category: other"


def main() -> None:
    template = PromptTemplate("Classify this support ticket: {{ ticket }}")
    cases = [
        EvalCase({"ticket": "I need a refund"}, expected="billing", tags=["billing"]),
        EvalCase({"ticket": "The app crashes on launch"}, expected="bug", tags=["bug"]),
    ]
    summary = EvalRunner(template, Contains(ignore_case=True), deterministic_llm).run(cases)
    print(f"pass_rate={summary.pass_rate:.0%} mean_score={summary.mean_score:.2f}")
    for result in summary.results:
        print(f"{result.case.tags[0]} -> {result.response} ({result.score.value:.1f})")


if __name__ == "__main__":
    main()

"""Exit non-zero if the no-key prompt regression demo fails."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prompt_eval import Contains, EvalCase, EvalRunner, PromptTemplate


def deterministic_llm(prompt: str) -> str:
    return "category: billing" if "refund" in prompt.lower() else "category: bug"


def main() -> None:
    summary = EvalRunner(
        PromptTemplate("Classify: {{ ticket }}"),
        Contains(ignore_case=True),
        deterministic_llm,
    ).run([
        EvalCase({"ticket": "refund request"}, "billing"),
        EvalCase({"ticket": "app crash"}, "bug"),
    ])
    if summary.pass_rate < 1.0:
        raise SystemExit(1)
    print("prompt regression passed")


if __name__ == "__main__":
    main()

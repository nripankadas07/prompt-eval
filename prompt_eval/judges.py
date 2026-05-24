"""Pluggable judges for prompt-eval."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Callable, Iterable


@dataclass
class Score:
    value: float
    reason: str = ""
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 <= self.value <= 1.0:
            raise ValueError(f"score out of [0,1]: {self.value}")


class Judge:
    def score(self, output: str, expected: str) -> Score:
        raise NotImplementedError

    def __call__(self, output: str, expected: str) -> Score:
        return self.score(output, expected)


class ExactMatch(Judge):
    def __init__(self, *, ignore_case: bool = False) -> None:
        self.ignore_case = ignore_case

    def score(self, output: str, expected: str) -> Score:
        left = output.strip()
        right = expected.strip()
        if self.ignore_case:
            left = left.lower()
            right = right.lower()
        passed = left == right
        return Score(1.0 if passed else 0.0, metadata={"passed": passed})


class Contains(Judge):
    def __init__(self, *, ignore_case: bool = False) -> None:
        self.ignore_case = ignore_case

    def score(self, output: str, expected: str) -> Score:
        haystack = output
        needle = expected
        if self.ignore_case:
            haystack = haystack.lower()
            needle = needle.lower()
        passed = needle in haystack
        return Score(1.0 if passed else 0.0, metadata={"passed": passed})


class FuzzyMatch(Judge):
    def __init__(self, threshold: float = 0.8) -> None:
        self.threshold = threshold

    def score(self, output: str, expected: str) -> Score:
        ratio = SequenceMatcher(None, output, expected).ratio()
        return Score(
            ratio,
            reason=f"ratio={ratio:.3f}",
            metadata={"passed": ratio >= self.threshold},
        )


class RegexMatch(Judge):
    def __init__(self, pattern: str) -> None:
        self.regex = re.compile(pattern)

    def score(self, output: str, expected: str = "") -> Score:
        passed = self.regex.search(output) is not None
        return Score(1.0 if passed else 0.0, metadata={"passed": passed})


class SemanticSimilarity(Judge):
    def __init__(self, embed_fn: Callable[[str], list[float]] | None = None) -> None:
        self.embed_fn = embed_fn

    def score(self, output: str, expected: str) -> Score:
        if self.embed_fn is None:
            left = set(output.split())
            right = set(expected.split())
            if not left and not right:
                return Score(1.0, reason="jaccard=1.000")
            value = len(left & right) / max(len(left | right), 1)
            return Score(value, reason=f"jaccard={value:.3f}")

        first = self.embed_fn(output)
        second = self.embed_fn(expected)
        dot = sum(a * b for a, b in zip(first, second))
        norm_a = sum(a * a for a in first) ** 0.5
        norm_b = sum(b * b for b in second) ** 0.5
        if norm_a == 0.0 or norm_b == 0.0:
            return Score(0.0, reason="cosine=0.000")
        value = max(0.0, min(1.0, dot / (norm_a * norm_b)))
        return Score(value, reason=f"cosine={value:.3f}")


class LLMJudge(Judge):
    def __init__(self, llm_fn: Callable[[str], str]) -> None:
        self.llm_fn = llm_fn

    def score(self, output: str, expected: str) -> Score:
        prompt = f"Score this output from 0 to 1.\nExpected: {expected}\nOutput: {output}"
        raw = self.llm_fn(prompt).strip()
        try:
            parsed = json.loads(raw)
            value = float(parsed.get("score", 0.0))
            reason = str(parsed.get("reason", raw))
        except (json.JSONDecodeError, TypeError, ValueError, AttributeError):
            match = re.search(r"(?<!\d)(?:0(?:\.\d+)?|1(?:\.0+)?)(?!\d)", raw)
            value = float(match.group(0)) if match else 0.0
            reason = raw
        return Score(max(0.0, min(1.0, value)), reason=reason)


class CompositeJudge(Judge):
    def __init__(
        self,
        judges: Iterable[Judge | tuple[Judge, float]],
        weights: Iterable[float] | None = None,
    ) -> None:
        items = list(judges)
        if weights is None and all(isinstance(item, tuple) for item in items):
            pairs = [(item[0], float(item[1])) for item in items]  # type: ignore[index]
        else:
            judge_list = [item for item in items if isinstance(item, Judge)]
            weight_list = list(weights) if weights is not None else [1.0 / len(judge_list)] * len(judge_list)
            pairs = list(zip(judge_list, weight_list))

        if not pairs:
            raise ValueError("at least one judge is required")

        total_weight = sum(weight for _, weight in pairs)
        if abs(total_weight - 1.0) > 1e-9:
            raise ValueError("Weights must sum to 1.0")

        self.pairs = pairs

    def score(self, output: str, expected: str) -> Score:
        total = 0.0
        reasons: list[str] = []
        for judge, weight in self.pairs:
            score = judge.score(output, expected)
            total += score.value * weight
            reasons.append(f"{type(judge).__name__}={score.value:.3f}")
        return Score(total, reason=", ".join(reasons))

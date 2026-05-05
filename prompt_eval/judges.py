"""Pluggable judges for prompt-eval."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Iterable, Optional
import re as _re

@dataclass
class Score:
    value: float
    reason: str = ""
    def __post_init__(self):
        if not 0.0 <= self.value <= 1.0:
            raise ValueError(f"score out of [0,1]: {self.value}")

class Judge:
    def __call__(self, output: str, expected: str) -> Score:
        raise NotImplementedError

class ExactMatch(Judge):
    def __call__(self, output, expected):
        return Score(1.0 if output == expected else 0.0)

class Contains(Judge):
    def __call__(self, output, expected):
        return Score(1.0 if expected in output else 0.0)

class FuzzyMatch(Judge):
    def __init__(self, threshold: float = 0.8):
        self.threshold = threshold
    def __call__(self, output, expected):
        from difflib import SequenceMatcher
        r = SequenceMatcher(None, output, expected).ratio()
        return Score(r if r >= self.threshold else 0.0, reason=f"ratio={r:.3f}")

class RegexMatch(Judge):
    def __init__(self, pattern: str):
        self.regex = _re.compile(pattern)
    def __call__(self, output, expected):
        return Score(1.0 if self.regex.search(output) else 0.0)

class SemanticSimilarity(Judge):
    def __init__(self, embed_fn: Optional[Callable[[str], list[float]]] = None):
        self.embed_fn = embed_fn
    def __call__(self, output, expected):
        if self.embed_fn is None:
            a, b = set(output.split()), set(expected.split())
            if not a and not b: return Score(1.0)
            r = len(a & b) / max(len(a | b), 1)
            return Score(r, reason=f"jaccard={r:.3f}")
        v1, v2 = self.embed_fn(output), self.embed_fn(expected)
        dot = sum(x*y for x, y in zip(v1, v2))
        n1 = (sum(x*x for x in v1)) ** 0.5 or 1.0
        n2 = (sum(x*x for x in v2)) ** 0.5 or 1.0
        cos = max(0.0, min(1.0, dot / (n1 * n2)))
        return Score(cos, reason=f"cosine={cos:.3f}")

class LLMJudge(Judge):
    def __init__(self, llm_fn: Callable[[str], str]):
        self.llm_fn = llm_fn
    def __call__(self, output, expected):
        prompt = f"Score this output 0-1.\nExpected: {expected}\nOutput: {output}"
        s = self.llm_fn(prompt).strip()
        try:
            v = float(s.split()[0])
        except (ValueError, IndexError):
            v = 0.0
        return Score(max(0.0, min(1.0, v)), reason=f"llm={s!r}")

class CompositeJudge(Judge):
    def __init__(self, judges: Iterable[Judge], weights: Optional[Iterable[float]] = None):
        self.judges = list(judges)
        self.weights = list(weights) if weights else [1.0] * len(self.judges)
        if len(self.weights) != len(self.judges):
            raise ValueError("weights must match number of judges")
    def __call__(self, output, expected):
        total = wsum = 0.0
        for j, w in zip(self.judges, self.weights):
            total += j(output, expected).value * w
            wsum += w
        return Score(total / wsum if wsum else 0.0)

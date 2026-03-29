"""
prompt-eval â Lightweight LLM prompt evaluation framework.

Evaluate, compare, and score prompts against ground-truth datasets
using pluggable judges (exact match, fuzzy, semantic, LLM-as-judge).
"""

__version__ = "0.1.0"

from .template import PromptTemplate
from .judges import (
    Judge,
    ExactMatch,
    Contains,
    FuzzyMatch,
    RegexMatch,
    SemanticSimilarity,
    LLMJudge,
    CompositeJudge,
)
from .runner import EvalRunner, EvalCase, EvalResult, EvalSummary
from .reporter import Reporter, ConsoleReporter, JSONReporter, MarkdownReporter

__all__ = [
    "PromptTemplate",
    "Judge",
    "ExactMatch",
    "Contains",
    "FuzzyMatch",
    "RegexMatch",
    "SemanticSimilarity",
    "LLMJudge",
    "CompositeJudge",
    "EvalRunner",
    "EvalCase",
    "EvalResult",
    "EvalSummary",
    "Reporter",
    "ConsoleReporter",
    "JSONReporter",
    "MarkdownReporter",
]

"""Tests for evaluation judges."""

import pytest
from prompt_eval.judges import (
    Score,
    ExactMatch,
    Contains,
    FuzzyMatch,
    RegexMatch,
    SemanticSimilarity,
    LLMJudge,
    CompositeJudge,
)


# ââ Score ââââââââââââââââââââââââââââââââââââââââââââââââââââââââ

class TestScore:
    def test_valid_score(self):
        s = Score(value=0.5, reason="ok")
        assert s.value == 0.5

    def test_boundary_scores(self):
        Score(value=0.0)
        Score(value=1.0)

    def test_out_of_range_raises(self):
        with pytest.raises(ValueError):
            Score(value=1.5)
        with pytest.raises(ValueError):
            Score(value=-0.1)


# ââ ExactMatch âââââââââââââââââââââââââââââââââââââââââââââââââââ

class TestExactMatch:
    def test_exact(self):
        j = ExactMatch()
        assert j.score("hello", "hello").value == 1.0

    def test_mismatch(self):
        j = ExactMatch()
        assert j.score("hello", "world").value == 0.0

    def test_strips_whitespace(self):
        j = ExactMatch()
        assert j.score("  hello  ", "hello").value == 1.0

    def test_case_sensitive_by_default(self):
        j = ExactMatch()
        assert j.score("Hello", "hello").value == 0.0

    def test_ignore_case(self):
        j = ExactMatch(ignore_case=True)
        assert j.score("Hello", "hello").value == 1.0


# ââ Contains âââââââââââââââââââââââââââââââââââââââââââââââââââââ

class TestContains:
    def test_found(self):
        j = Contains()
        assert j.score("The cat sat on the mat", "cat").value == 1.0

    def test_not_found(self):
        j = Contains()
        assert j.score("The dog ran", "cat").value == 0.0

    def test_ignore_case(self):
        j = Contains(ignore_case=True)
        assert j.score("HELLO world", "hello").value == 1.0


# ââ FuzzyMatch âââââââââââââââââââââââââââââââââââââââââââââââââââ

class TestFuzzyMatch:
    def test_identical_strings(self):
        j = FuzzyMatch()
        assert j.score("hello", "hello").value == 1.0

    def test_similar_strings(self):
        j = FuzzyMatch()
        s = j.score("hello world", "helo world")
        assert s.value > 0.8

    def test_dissimilar_strings(self):
        j = FuzzyMatch()
        s = j.score("abc", "xyz")
        assert s.value < 0.5

    def test_threshold_in_metadata(self):
        j = FuzzyMatch(threshold=0.9)
        s = j.score("abc", "abc")
        assert s.metadata["passed"] is True


# ââ RegexMatch âââââââââââââââââââââââââââââââââââââââââââââââââââ

class TestRegexMatch:
    def test_matches(self):
        j = RegexMatch(r"\d{3}-\d{4}")
        assert j.score("Call 555-1234 now").value == 1.0

    def test_no_match(self):
        j = RegexMatch(r"\d{3}-\d{4}")
        assert j.score("No numbers here").value == 0.0


# ââ SemanticSimilarity âââââââââââââââââââââââââââââââââââââââââââ

class TestSemanticSimilarity:
    def test_identical_text(self):
        j = SemanticSimilarity()
        s = j.score("the quick brown fox", "the quick brown fox")
        assert s.value == 1.0

    def test_similar_text(self):
        j = SemanticSimilarity()
        s = j.score("the quick brown fox jumps", "the fast brown fox leaps")
        assert s.value > 0.3  # BOW baseline

    def test_custom_embed_fn(self):
        def dummy_embed(text: str) -> list[float]:
            return [len(text), len(text.split())]

        j = SemanticSimilarity(embed_fn=dummy_embed)
        s = j.score("hello world", "hello world")
        assert s.value == pytest.approx(1.0)

    def test_zero_vectors(self):
        def zero_embed(text: str) -> list[float]:
            return [0.0, 0.0]

        j = SemanticSimilarity(embed_fn=zero_embed)
        s = j.score("a", "b")
        assert s.value == 0.0


# ââ LLMJudge ââââââââââââââââââââââââââââââââââââââââââââââââââââ

class TestLLMJudge:
    def test_parses_json_response(self):
        def mock_llm(prompt: str) -> str:
            return '{"score": 0.85, "reason": "good answer"}'

        j = LLMJudge(llm_fn=mock_llm)
        s = j.score("response text", "expected text")
        assert s.value == 0.85
        assert "good answer" in s.reason

    def test_fallback_parsing(self):
        def mock_llm(prompt: str) -> str:
            return "I'd rate this 0.7 out of 1"

        j = LLMJudge(llm_fn=mock_llm)
        s = j.score("response", "expected")
        assert s.value == 0.7

    def test_unparseable_returns_zero(self):
        def mock_llm(prompt: str) -> str:
            return "This is unparseable"

        j = LLMJudge(llm_fn=mock_llm)
        s = j.score("response", "expected")
        assert s.value == 0.0


# ââ CompositeJudge ââââââââââââââââââââââââââââââââââââââââââ

class TestCompositeJudge:
    def test_weighted_average(self):
        j = CompositeJudge([
            (ExactMatch(), 0.5),
            (Contains(), 0.5),
        ])
        # "hello" exact matches "hello" (1.0 * 0.5)
        # "hello" contains "hello" (1.0 * 0.5)
        s = j.score("hello", "hello")
        assert s.value == 1.0

    def test_partial_scores(self):
        j = CompositeJudge([
            (ExactMatch(), 0.5),
            (Contains(), 0.5),
        ])
        # "hello world" != "hello" (0.0 * 0.5)
        # "hello world" contains "hello" (1.0 * 0.5)
        s = j.score("hello world", "hello")
        assert s.value == pytest.approx(0.5)

    def test_weights_must_sum_to_one(self):
        with pytest.raises(ValueError, match="Weights must sum to 1.0"):
            CompositeJudge([(ExactMatch(), 0.3), (Contains(), 0.3)])

    def test_reason_lists_components(self):
        j = CompositeJudge([
            (ExactMatch(), 0.5),
            (Contains(), 0.5),
        ])
        s = j.score("hello", "hello")
        assert "ExactMatch" in s.reason
        assert "Contains" in s.reason

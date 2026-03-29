"""
Result reporters â format evaluation summaries for humans and machines.

* **ConsoleReporter** â coloured terminal output
* **JSONReporter** â structured JSON for CI pipelines
* **MarkdownReporter** â pretty tables for docs / PR comments
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import TextIO
import sys

from .runner import EvalSummary


class Reporter(ABC):
    """Base class for evaluation reporters."""

    @abstractmethod
    def report(self, summary: EvalSummary, output: TextIO | None = None) -> str:
        """Format and optionally write the summary. Returns the formatted string."""


class ConsoleReporter(Reporter):
    """Human-readable terminal output with pass/fail indicators."""

    def report(self, summary: EvalSummary, output: TextIO | None = None) -> str:
        out = output or sys.stdout
        lines: list[str] = []

        lines.append("")
        lines.append("=" * 60)
        lines.append(f"  EVALUATION RESULTS  ({summary.total} cases)")
        lines.append("=" * 60)
        lines.append("")

        for i, r in enumerate(summary.results, 1):
            status = "PASS" if r.score.value >= summary.pass_threshold else "FAIL"
            marker = "+" if status == "PASS" else "-"
            lines.append(
                f"  [{marker}] Case {i:>3d}  score={r.score.value:.2f}  "
                f"latency={r.latency_ms:.0f}ms"
            )
            if r.error:
                lines.append(f"           ERROR: {r.error}")
            if r.score.reason:
                lines.append(f"           reason: {r.score.reason}")

        lines.append("")
        lines.append("-" * 60)
        lines.append(f"  Pass rate:    {summary.pass_rate:.1%}  "
                      f"({summary.passed}/{summary.total})")
        lines.append(f"  Mean score:   {summary.mean_score:.3f}")
        lines.append(f"  Median score: {summary.median_score:.3f}")
        lines.append(f"  Std dev:      {summary.std_dev:.3f}")
        lines.append(f"  Score range:  [{summary.min_score:.2f}, {summary.max_score:.2f}]")
        lines.append(f"  Mean latency: {summary.mean_latency_ms:.0f}ms")
        lines.append(f"  Total time:   {summary.total_time_ms:.0f}ms")
        lines.append(f"  Threshold:    {summary.pass_threshold}")
        lines.append("-" * 60)
        lines.append("")

        text = "\n".join(lines)
        out.write(text)
        return text


class JSONReporter(Reporter):
    """Machine-readable JSON output for CI/CD integration."""

    def report(self, summary: EvalSummary, output: TextIO | None = None) -> str:
        data = {
            "total": summary.total,
            "passed": summary.passed,
            "failed": summary.failed,
            "errored": summary.errored,
            "pass_rate": round(summary.pass_rate, 4),
            "mean_score": round(summary.mean_score, 4),
            "median_score": round(summary.median_score, 4),
            "std_dev": round(summary.std_dev, 4),
            "min_score": round(summary.min_score, 4),
            "max_score": round(summary.max_score, 4),
            "mean_latency_ms": round(summary.mean_latency_ms, 1),
            "total_time_ms": round(summary.total_time_ms, 1),
            "pass_threshold": summary.pass_threshold,
            "results": [
                {
                    "inputs": r.case.inputs,
                    "expected": r.case.expected,
                    "response": r.response,
                    "score": round(r.score.value, 4),
                    "reason": r.score.reason,
                    "latency_ms": round(r.latency_ms, 1),
                    "tags": r.case.tags,
                    "error": r.error,
                }
                for r in summary.results
            ],
        }

        text = json.dumps(data, indent=2, ensure_ascii=False)
        if output:
            output.write(text)
        return text


class MarkdownReporter(Reporter):
    """Markdown table output â ideal for PR comments and docs."""

    def report(self, summary: EvalSummary, output: TextIO | None = None) -> str:
        lines: list[str] = []

        lines.append(f"## Evaluation Results ({summary.total} cases)")
        lines.append("")
        lines.append(f"**Pass rate:** {summary.pass_rate:.1%} "
                      f"({summary.passed}/{summary.total}) | "
                      f"**Mean score:** {summary.mean_score:.3f} | "
                      f"**Threshold:** {summary.pass_threshold}")
        lines.append("")
        lines.append("| # | Score | Status | Latency | Reason |")
        lines.append("|---|-------|--------|---------|--------|")

        for i, r in enumerate(summary.results, 1):
            status = "PASS" if r.score.value >= summary.pass_threshold else "FAIL"
            lines.append(
                f"| {i} | {r.score.value:.2f} | {status} | "
                f"{r.latency_ms:.0f}ms | {r.score.reason[:50]} |"
            )

        lines.append("")
        lines.append("### Statistics")
        lines.append("")
        lines.append(f"| Metric | Value |")
        lines.append(f"|--------|-------|")
        lines.append(f"| Mean | {summary.mean_score:.3f} |")
        lines.append(f"| Median | {summary.median_score:.3f} |")
        lines.append(f"| Std Dev | {summary.std_dev:.3f} |")
        lines.append(f"| Min | {summary.min_score:.2f} |")
        lines.append(f"| Max | {summary.max_score:.2f} |")
        lines.append(f"| Mean Latency | {summary.mean_latency_ms:.0f}ms |")
        lines.append("")

        text = "\n".join(lines)
        if output:
            output.write(text)
        return text

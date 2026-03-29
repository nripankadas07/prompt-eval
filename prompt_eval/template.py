"""
Prompt template engine with variable substitution and validation.

Supports Jinja-style ``{{ variable }}`` placeholders. Templates
are immutable after creation â calling ``.render()`` returns a new
string without mutating the template.

Usage::

    tpl = PromptTemplate("Summarise {{ text }} in {{ style }} style.")
    prompt = tpl.render(text="War and Peace", style="haiku")
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


_VAR_PATTERN = re.compile(r"\{\{\s*(\w+)\s*\}\}")


@dataclass(frozen=True)
class PromptTemplate:
    """An immutable prompt template with ``{{ var }}`` placeholders."""

    template: str
    name: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    # ââ introspection âââââââââââââââââââââââââââââââââââââââââââââ

    @property
    def variables(self) -> list[str]:
        """Return ordered, deduplicated list of variable names."""
        seen: set[str] = set()
        result: list[str] = []
        for match in _VAR_PATTERN.finditer(self.template):
            var = match.group(1)
            if var not in seen:
                seen.add(var)
                result.append(var)
        return result

    # ââ rendering ââââââââââââââââââââââââââââââââââââââââââââââââ

    def render(self, **kwargs: Any) -> str:
        """Substitute variables and return the final prompt string.

        Raises ``ValueError`` if required variables are missing.
        """
        missing = set(self.variables) - set(kwargs.keys())
        if missing:
            raise ValueError(
                f"Missing template variables: {', '.join(sorted(missing))}"
            )

        def _replace(match: re.Match) -> str:
            return str(kwargs[match.group(1)])

        return _VAR_PATTERN.sub(_replace, self.template)

    def partial(self, **kwargs: Any) -> "PromptTemplate":
        """Return a new template with some variables already filled in."""
        def _replace(match: re.Match) -> str:
            var = match.group(1)
            if var in kwargs:
                return str(kwargs[var])
            return match.group(0)  # leave unfilled vars as-is

        new_template = _VAR_PATTERN.sub(_replace, self.template)
        return PromptTemplate(
            template=new_template,
            name=self.name,
            metadata={**self.metadata, "_partial_from": self.template},
        )

    # ââ convenience ââââââââââââââââââââââââââââââââââââââââââââââ

    def __str__(self) -> str:
        label = f" ({self.name})" if self.name else ""
        return f"PromptTemplate{label}: {self.template[:80]}..."

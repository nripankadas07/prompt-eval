# prompt-eval

A lightweight framework for evaluating LLM prompts against ground-truth datasets. Plug in any LLM, score with multiple judges, and get structured reports â all with zero heavy dependencies.

## Why This Exists

If you're iterating on prompts, you need to know whether version B is actually better than version A. Manual spot-checking doesn't scale. This framework automates the boring parts: rendering prompt templates, calling your LLM, scoring outputs with pluggable judges, and summarising results with statistics that tell you whether the difference is real.

## Quick Start

```python
from prompt_eval import PromptTemplate, ExactMatch, EvalRunner, EvalCase

# 1. Define your prompt template
template = PromptTemplate("Translate '{{ text }}' to French.")

# 2. Pick a judge (or compose several)
judge = ExactMatch(ignore_case=True)

# 3. Bring your own LLM
def my_llm(prompt: str) -> str:
    return call_openai(prompt)  # or Anthropic, local model, etc.

# 4. Define test cases
cases = [
    EvalCase(inputs={"text": "hello"}, expected="bonjour"),
    EvalCase(inputs={"text": "goodbye"}, expected="au revoir"),
]

# 5. Run evaluation
runner = EvalRunner(template=template, judge=judge, llm_fn=my_llm)
summary = runner.run(cases)

print(f"Pass rate: {summary.pass_rate:.0%t")
print(f"Mean score: {summary.mean_score:.3f}")
```

## Built-in Judges

| Judge | What It Does | Use When |
|-------|-------------|----------|
| `ExactMatch` | Binary 0/1 on string equality | Factual lookups, classification |
| `Contains` | 1.0 if expected text appears in response | Keyword extraction |
| `FuzzyMatch` | Normalised edit-distance ratio | Near-exact matching |
| `RegexMatch` | 1.0 if pattern matches response | Structured output validation |
| `SemanticSimilarity` | Cosine similarity (custom embeddings or BOW fallback) | Paraphrase tolerance |
| `LLMJudge` | Another LLM grades the response | Open-ended quality assessment |
| `CompositeJudge` | Weighted average of multiple judges | Balanced evaluation |

## Compose Judges

```python
from prompt_eval import CompositeJudge, ExactMatch, FuzzyMatch, SemanticSimilarity

judge = CompositeJudge([
    (ExactMatch(), 0.3),
    (FuzzyMatch(threshold=0.8), 0.3),
    (SemanticSimilarity(), 0.4),
])
```

## Prompt Templates

Templates use `{{ variable }}` syntax with validation:

```python
from prompt_eval import PromptTemplate

tpl = PromptTemplate(
    "You are a {{ role }}. Summarise: {{ text }}",
    name="summariser-v2",
)

# Inspect variables
print(tpl.variables)  # ["role", "text"]

# Render with validation (raises on missing vars)
prompt = tpl.render(role="editor", text="War and Peace")

# Partial application
partial = tpl.partial(role="editor")
prompt = partial.render(text="War and Peace")
```

## Reporters

```python
from prompt_eval import ConsoleReporter, JSONReporter, MarkdownReporter

# Terminal output with pass/fail markers
ConsoleReporter().report(summary)

# Structured JSON for CI pipelines
json_str = JSONReporter().report(summary)

# Markdown tables for PR comments
md = MarkdownReporter().report(summary)
```

## Architecture

```
prompt_eval/
âââ template.py    # Prompt template engine with {{ var }} substitution
âââ judges.py      # Pluggable scoring: exact, fuzzy, semantic, LLM, composite
âââ runner.py      # Orchestration: template â LLM â judge â statistics
âââ reporter.py    # Output formatting: console, JSON, markdown
```

The design is intentionally simple â four modules, no framework, no magic. Each piece is independently testable and replaceable.

## Testing

```bash
pip install -e ".[dev]"
pytest
```

## License

MIT

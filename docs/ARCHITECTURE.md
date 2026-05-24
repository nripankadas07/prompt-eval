# Architecture

`prompt-eval` thesis: unit tests for prompts with templates, deterministic fixtures, judges, and reports.

```mermaid
flowchart LR
    A0["EvalCase"]
    A1["PromptTemplate"]
    A2["LLM Function"]
    A3["Judge"]
    A4["EvalSummary"]
    A5["Reporter"]
    A0 --> A1
    A1 --> A2
    A2 --> A3
    A3 --> A4
    A4 --> A5
```

## Design Rules

- Keep the public API small enough to inspect in one sitting.
- Make demos run locally without network credentials.
- Put correctness checks in tests, conformance scripts, or benchmark scripts
  instead of relying on README claims.
- Prefer explicit failure modes over surprising implicit behavior.

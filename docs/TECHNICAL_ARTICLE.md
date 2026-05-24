# prompt-eval: unit tests for prompts

This is a launch-ready technical article draft for the repository. It is meant
to explain the idea, not inflate traction.

## The Problem

Prompt edits can silently regress behavior. Manual spot checks do not scale and are hard to review.

## The Core Idea

Treat a prompt as code: render it, run a model function, score the output, and fail CI when behavior drifts.

## No-Key Demos

The default examples use deterministic local functions so the evaluation contract is visible without API access.

## Limitations

Built-in judges are simple. Open-ended quality still needs domain-specific rubrics and carefully reviewed LLM judges.

## Try It

Run the README demo from a clean checkout. If the demo needs credentials, it is
not a good flagship demo.

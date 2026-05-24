# Release Readiness

`prompt-eval` is source-checkout ready. It is not claimed as published to PyPI or
npm unless a package registry release is verified later.

## Required Before Tagging

- [ ] README quick demo works from a clean checkout.
- [ ] Tests pass locally and in GitHub Actions.
- [ ] Type/lint/build checks pass.
- [ ] `python -m build` passes.
- [ ] CHANGELOG and ROADMAP reflect the shipped surface.
- [ ] No fake package, benchmark, badge, download, adoption, or production-use
  claim is present.

## Release Notes Template

```markdown
## vX.Y.Z

### Added

### Changed

### Fixed

### Verification
- tests:
- type/lint/build:
- benchmark/conformance script:
```

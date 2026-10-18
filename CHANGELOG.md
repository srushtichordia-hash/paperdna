# Changelog

All notable changes to PaperDNA are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/); versions follow
[SemVer](https://semver.org/) for the Python API surface.

## [Unreleased]

## [0.1.0] — 2026-10-18

First public MVP release. PaperDNA turns a scientific claim (or
abstract) into an auditable map of related findings: ranked papers,
predicted claim relationships with uncertainty, experimental-condition
context, verbatim evidence with offsets, and a knowledge graph — via a
local web app (React + FastAPI) and a Python pipeline.

### Added

- **Corpus ingestion** — validation, normalization, deduplication for
  OpenAlex/arXiv-style records (`paperdna.corpus`).
- **Semantic retrieval** — optional sentence-transformers embedder
  with `PaperIndex`, plus documented Precision@k/Recall@k metrics
  (`paperdna.retrieval`).
- **Candidate-claim extraction** — deterministic rule-based extraction
  preserving verbatim sentences and exact abstract offsets
  (`paperdna.claims`).
- **NLI claim-relationship classification** — optional cross-encoder
  backend with preserved uncertainty and uncalibrated scores
  (`paperdna.nli`).
- **Condition analysis** — deterministic extraction/comparison of
  experimental conditions (`paperdna.conditions`).
- **Evidence attribution** — verbatim evidence with verified offsets,
  stable claim ids, and explicit unverified/unavailable states
  (`paperdna.evidence`).
- **Knowledge graph** — validated nodes/edges with deterministic
  column-based layout (`paperdna.graph`).
- **Web application** — FastAPI backend (`paperdna.api`) + React/
  TypeScript/Vite frontend (`web/`); lexical-fallback honesty in both
  degraded and full modes; ML models are explicit opt-ins, never
  downloaded implicitly.
- **Evaluation harness** (`paperdna.eval`) — hand-labeled synthetic
  benchmarks with dev/test splits, deterministic offline runners for
  retrieval P@k/R@k, NLI macro-F1/per-class metrics, condition
  correctness, and an evidence-traceability audit. Methodology and
  results: [`docs/evaluation.md`](docs/evaluation.md).
- **CI** — GitHub Actions workflow running the backend suite, frontend
  tests, type check, and production build on every push/PR.
- Documentation: setup, run, evaluation, demo walkthrough, release
  checklist, per-module limitations.

### Known limitations

- Benchmarks are small, synthetic, and single-annotator; they measure
  regression safety, not scientific model quality.
- Real-model evaluation tiers (semantic P@k/R@k, cross-encoder
  macro-F1) require explicit optional installs and were not run in the
  release environment.
- NLI scores are uncalibrated; textual relationships are not
  scientific truth.

[Unreleased]: https://github.com/srushtichordia-hash/paperdna/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/srushtichordia-hash/paperdna/releases/tag/v0.1.0

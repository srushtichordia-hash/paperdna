# MVP release checklist (Issue #11)

This checklist maps 1:1 to the acceptance criteria of
[Issue #11](https://github.com/srushtichordia-hash/paperdna/issues/11).
Every item is verifiable from a clean checkout.

## Acceptance criteria

### ☑ Evaluation data and methodology are documented

- [x] Benchmarks: `tests/fixtures/retrieval_eval.json` (13 papers,
      19 queries), `nli_pairs.json` (24 pairs), `conditions_eval.json`
      (6 pairs) — synthetic, hand-labeled, self-describing.
- [x] Methodology, splits, metric definitions, and honesty framing:
      [`docs/evaluation.md`](evaluation.md).

### ☑ Metrics are calculated from actual predictions and labels

- [x] Retrieval P@k/R@k from the real lexical ranker's actual
      retrieved order (`docs/evaluation/reports/retrieval_test.json`).
- [x] NLI macro-F1/per-class from a non-circular baseline's actual
      predictions (`reports/nli_overlap_test.json`); the circular
      scripted system is flagged `circular: true` and separated.
- [x] Reports regenerate byte-identically:
      `python -m paperdna.eval all --out report.json`.

### ☑ Known limitations and failure cases are reported honestly

- [x] Concrete failure cases listed per system in
      [`docs/evaluation.md`](evaluation.md) (retrieval R@1 = 0.5 on
      the multi-relevant query; all six NLI baseline misses named).
- [x] Per-module limitations in the README; unverified items stated as
      unverified (real-model tiers, real corpora, calibration).

### ☑ Tests and setup instructions are reproducible

- [x] Full suite offline and deterministic: `pytest` (576 passed,
      2 skipped — the two `slow` real-model tests are opt-in).
- [x] Frontend: `npm test`, `npm run typecheck`, `npm run build`.
- [x] Evidence audit: 0 violations over 38 generated evidence sides
      (`tests/evaluation/`), with negative tests proving detection.
- [x] End-to-end integration tests across all modules
      (`tests/evaluation/test_integration_e2e.py`).
- [x] Setup and demo walkthrough: [`docs/demo.md`](demo.md),
      README "Web application" section.
- [x] CI (`.github/workflows/ci.yml`) runs backend + frontend checks
      on every push and pull request.

### ☑ The MVP release checklist is complete

- [x] Version 0.1.0, [`CHANGELOG.md`](../CHANGELOG.md), MIT
      [`LICENSE`](../LICENSE).
- [x] Model licenses and data sources documented (README
      "Model licenses & data sources").
- [x] Screenshots of the running app (`docs/screenshots/`).
- [x] Evaluation reports committed as release evidence
      (`docs/evaluation/reports/`).

## Pre-tag steps (maintainer)

- [ ] All CI checks green on the release commit.
- [ ] `pytest` + frontend checks pass locally.
- [ ] `python -m paperdna.eval all` reproduces the committed reports.
- [ ] Tag `v0.1.0` and create a GitHub release with the changelog
      entry.

## Deliberately out of scope for MVP

- Real-model evaluation runs (require explicit model downloads; exact
  commands in `docs/evaluation.md`).
- External/deployment hosting (local-first by design; no paid
  services).
- Real-corpus benchmarks (would require licensed scientific datasets
  and multi-annotator labeling).

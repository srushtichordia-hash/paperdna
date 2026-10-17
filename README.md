# PaperDNA

An NLP research discovery engine that maps agreement, disagreement, and condition-dependent scientific findings.

> **Status:** early foundation. Repository architecture, configuration, testing,
> and local corpus ingestion/validation are in place. NLP models, network
> ingestion from live sources, retrieval, and the UI are not implemented yet.

## Requirements

- Python >= 3.10
- `pip` and `venv` (included with standard Python installations)

## Setup

Clone the repository and create a virtual environment:

```bash
git clone https://github.com/srushtichordia-hash/paperdna.git
cd paperdna
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

## Installation

Install the package in editable mode with development dependencies:

```bash
pip install -e ".[dev]"
```

This installs PaperDNA with no runtime dependencies; `pytest` is the only
development dependency.

To use real embedding models for semantic retrieval, install the optional
`embed` extra (this pulls in PyTorch and sentence-transformers, several
hundred MB):

```bash
pip install -e ".[embed]"
```

## Running the tests

```bash
pytest
```

The corpus tests live under `tests/corpus/` and run against a small local
fixture (`tests/fixtures/sample_corpus.jsonl`); they are fully offline and
deterministic. To run only the corpus tests:

```bash
pytest tests/corpus
```

## Corpus schema

Ingestion turns raw metadata payloads into validated `PaperRecord` objects.
All text fields are Unicode-normalized and whitespace-collapsed.

| Field       | Required | Type               | Notes                                    |
| ----------- | -------- | ------------------ | ---------------------------------------- |
| `source`    | yes      | string             | Origin system, e.g. `arxiv`              |
| `source_id` | yes      | string             | Stable ID within the source              |
| `title`     | yes      | string             | Non-empty after normalization            |
| `abstract`  | yes      | string             | Non-empty after normalization            |
| `authors`   | yes      | list or CSV string | Non-empty after normalization            |
| `published` | no       | string             | `YYYY` or a real `YYYY-MM-DD` date       |
| `url`       | no       | string             | Must be `http://` or `https://`          |
| `doi`       | no       | string             | Normalized to lowercase canonical form   |

Malformed or incomplete records are **rejected, never raised**: each rejected
record is retained with its original payload and a reason. Duplicate detection
uses, in order: normalized DOI, then `(source, source_id)`, then a
conservative `(title, year, first author)` fallback. The fallback reduces
missed cross-source duplicates but can, in rare cases, merge distinct papers
that share all three attributes — every duplicate is reported with the
retained record it matched and the key that matched, so this is auditable.

Ingestion reports precise statistics (`loaded`, `validated`, `rejected`,
`unique`, `duplicates`) with the invariants
`loaded == validated + rejected` and `validated == unique + duplicates`.

Example:

```python
from paperdna.corpus import ingest_records, load_jsonl

result = ingest_records(load_jsonl("tests/fixtures/sample_corpus.jsonl"))
print(result.stats)  # {'loaded': 9, 'validated': 6, 'rejected': 3, 'unique': 3, 'duplicates': 3}
```

### Planned ingestion sources

No network ingestion exists yet. When it is added, the intended metadata
sources are the **arXiv API** (free, no API key, ~3 s between requests) and
**OpenAlex** (free, CC0-licensed, generous rate limits) — both provide stable
IDs, abstracts, and DOIs compatible with this schema.

## Configuration

PaperDNA reads configuration from environment variables with the `PAPERDNA_`
prefix and safe defaults, so no setup is required to get started. It does not
parse `.env` files itself — export the variables in your shell or use a tool of
your choice.

| Variable            | Default | Description                          |
| ------------------- | ------- | ------------------------------------ |
| `PAPERDNA_LOG_LEVEL`| `INFO`  | Logging level (`DEBUG`, `INFO`, ...) |

Example:

```bash
export PAPERDNA_LOG_LEVEL=DEBUG
```

See [`.env.example`](.env.example) for a documented template. Never commit real
secrets; `.env` files are gitignored.

## Semantic retrieval

`paperdna.retrieval` embeds paper records and retrieves the most similar
ones for a query, using cosine similarity.

- **Model:** [`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
  (Apache-2.0 license, 384 dimensions, ~90 MB). Selected as a strong
  quality/speed trade-off for titles and abstracts.
- **Optional dependency:** the core package and the default test suite work
  without PyTorch or sentence-transformers. The `SentenceTransformerEmbedder`
  loads the model only when explicitly instantiated — never at import time.
- **First run:** instantiating `SentenceTransformerEmbedder` downloads the
  model from Hugging Face (~90 MB) into the local HF cache
  (`~/.cache/huggingface`); later runs are offline.
- **Embedding text format:** `"{title}\n\n{abstract}"` (stable and
  documented in `paperdna.retrieval.preprocess`).
- **Identity & duplicates:** papers are identified by
  `(source.casefold(), source_id)`; duplicates at index build are dropped
  (first occurrence kept, count reported in `index.stats`), and records are
  never mutated.
- **Invalid vectors** (wrong dimension, NaN/inf, zero norm) raise
  `InvalidVectorError`; ties in ranking are broken deterministically by
  corpus order.

```python
from paperdna.corpus import ingest_records, load_jsonl
from paperdna.retrieval import PaperIndex, SentenceTransformerEmbedder

result = ingest_records(load_jsonl("tests/fixtures/sample_corpus.jsonl"))
index = PaperIndex.build(result.papers, SentenceTransformerEmbedder())
for hit in index.search("sparse attention for long documents", top_k=3):
    print(f"{hit.score:.3f}  {hit.record.source}:{hit.record.source_id}  {hit.record.title}")
```

### Retrieval evaluation

`paperdna.retrieval.eval` provides `precision_at_k` / `recall_at_k` with
documented conventions (positive `k` required; fewer-than-`k` retrieved
divides by `k`; empty relevance sets score `0.0`).
`tests/fixtures/retrieval_eval.json` is a small hand-labeled
**sanity-check** dataset — useful for catching regressions, not evidence of
scientific model quality.

### Testing retrieval offline

All retrieval unit tests use deterministic fake embedders; no model or
network access is needed:

```bash
pytest tests/retrieval
pytest -m "not slow"          # exclude the optional real-model smoke test
```

The real-model smoke test runs only when the `embed` extra is installed
(and is marked `slow`).

## Project layout

```
src/paperdna/           Package source (config, logging)
src/paperdna/corpus/    Corpus schema, validation, normalization, deduplication
src/paperdna/retrieval/ Embedding interface, cosine similarity, search index, eval
tests/                  Pytest test suite (smoke, corpus, retrieval tests)
tests/fixtures/         Local offline test fixtures
pyproject.toml          Project metadata, dependencies, and tool config
.env.example            Example environment configuration
```

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE).

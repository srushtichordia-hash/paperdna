# PaperDNA

An NLP research discovery engine that maps agreement, disagreement, and condition-dependent scientific findings.

> **Status:** early foundation. Repository architecture, configuration, testing,
> corpus ingestion/validation, semantic retrieval, baseline candidate-claim
> extraction, pairwise NLI relationship estimation, experimental-condition
> analysis, and evidence attribution/traceability are in place. Evidence
> graphs, network ingestion from live sources, and the UI are not implemented
> yet.

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

## Claim extraction

`paperdna.claims` extracts **candidate** finding sentences from abstracts
using a lightweight, fully offline baseline: rule-based sentence
segmentation (with character-offset traceability) followed by lexical cue
classification.

- **Selection:** sentences matching finding cues (`we find/show/observe`,
  `our results suggest/indicate`, `results show`, `is/are associated
  with`, `we conclude/present`) become candidate claims; sentences matching
  methods cues (`we trained`, `dataset`, `first,`) or background cues
  (`previous studies`, `it is known`, `prior work`) are excluded.
- **Cautious language is preserved:** uncertainty markers (`may`, `suggests`,
  `appears`, `is associated with`, `consistent with`) are *recorded* on each
  claim (`hedges`), never removed or upgraded to definitive statements.
- **Traceability:** every claim stores the verbatim sentence plus character
  offsets, so `abstract[claim.start:claim.end] == claim.sentence` always
  holds; source identity (`source`, `source_id`) is copied from the paper
  record. Records are never mutated and evidence text is never fabricated.
- **Deterministic and offline:** no models, no network, no downloads.
- **Evaluation:** `tests/fixtures/claims_eval.json` is a small hand-labeled
  **sanity-check** set — useful for catching regressions, not evidence of
  scientific model quality.

### Limitations

This baseline does **not** understand scientific content or truth. It is a
lexical heuristic: paraphrased findings without the listed cues are missed
(false negatives), sentences using similar wording are over-selected (false
positives), and unusual abbreviations or capitalization can cause
over-/under-segmentation. It cannot distinguish hedged findings from hedged
background, and it is not a substitute for later NLI- or model-based stages.
Treat its output as auditable candidates for further analysis.

```python
from paperdna.claims import extract_claims

result = extract_claims(record)  # record: PaperRecord
for claim in result.claims:
    print(claim.sentence, claim.hedges)  # exact source text + uncertainty markers
```

## Claim relationships (NLI)

`paperdna.nli` estimates the textual relationship between two candidate
claims: **entailment** (support), **contradiction**, or **neutral**.

- **Model:** [`cross-encoder/nli-deberta-v3-base`](https://huggingface.co/cross-encoder/nli-deberta-v3-base)
  — Apache-2.0, 184M parameters, 512-token inputs, ~several hundred MB
  download on first use (cached by Hugging Face). Trained on SNLI +
  MultiNLI (**general-domain**; scientific claim pairs are
  out-of-distribution, so expect degraded accuracy versus its reported
  92.4/90.0 general benchmarks).
- **Label order is never assumed.** The classifier resolves the model's
  own `id2label` config (with SciFact-style synonyms like
  `SUPPORTS`/`REFUTES`/`NOT_ENOUGH_INFO`); the model-card default order
  applies only when a config declares no labels, and un-mappable labels
  raise rather than guess.
- **Premise/hypothesis convention:** `classify_pair(premise, hypothesis)`
  scores in that direction only — NLI is not symmetric.
- **Uncertainty is preserved:** every result carries all three scores and
  its confidence; `uncertain=True` when confidence is below the threshold
  (default 0.6, a heuristic — softmax scores are **not** calibrated
  probabilities) or when either claim is hedged. Uncertain pairs keep
  their label but are flagged for review, never presented as established
  facts.
- **Traceability:** both claims' source identities and exact evidence
  sentences are copied into every `PairClassification`; truncation of long
  inputs is recorded; sources are never mutated.
- **Optional dependency:** install with `pip install -e ".[nli]"` (same
  sentence-transformers stack as the `embed` extra). The model loads only
  when `CrossEncoderNliClassifier` is instantiated — never at import time
  or in ordinary tests.

```python
from paperdna.nli import ClaimPair, CrossEncoderNliClassifier, classify_pair

result = classify_pair(pair.premise, pair.hypothesis, CrossEncoderNliClassifier())
print(result.label, result.confidence, result.uncertain)
```

### Evaluation

`classification_report(y_true, y_pred)` reports per-class
precision/recall/F1 and macro-F1 over the three relationships.
`tests/fixtures/nli_pairs.json` is a small hand-labeled **sanity-check**
set (support, contradiction, neutral, hedged, paraphrase, low-confidence
pairs) — useful for catching regressions, not evidence of model quality.

### Limitations

NLI models estimate *textual* relationships, not scientific truth. Scores
are uncalibrated; the model is domain-general; hedged or subtle scientific
claims (qualified support, condition-dependent findings) are handled
weakly; premise/hypothesis direction matters and reversing it can flip the
label; a `neutral` label means "no clear textual relationship", not "the
claims are unrelated in reality". All NLI output should be treated as
decision support for human review.

## Experimental conditions

`paperdna.conditions` extracts experimental-condition mentions from candidate
claims and compares them across two claims — a deterministic, offline,
stdlib-only baseline of transparent regex rules (no models, no downloads).

- **Categories:** `dataset`, `population`, `model`, `metric`, `setting`,
  `design` (fixed, documented in `paperdna.conditions.CONDITION_CATEGORIES`).
- **Traceability:** every `ConditionMention` stores the verbatim matched span
  with character offsets into the claim's sentence
  (`sentence[mention.start:mention.end] == mention.value`) and copies the
  claim's source identity. Conditions are only ever *found* in text — never
  inferred, completed, or fabricated; categories with no textual support are
  simply absent.
- **Ambiguity is flagged, not hidden:** generic wording ("our dataset",
  "various benchmarks") is extracted with `ambiguous=True` so it cannot be
  silently treated as comparable.
- **Comparison statuses** (per category): `matching`, `differing`, `missing`,
  `uncertain`. **`missing` is not `differing`** — an absent condition means
  the text does not say, which is *not* evidence that the studies used
  different conditions. `uncertain` covers ambiguous wording on either side.
  Normalized (casefolded, whitespace-collapsed) value sets are compared for
  equality; mentions are deduplicated per field; `stats` counts always sum
  to the six categories.
- **Context, never override:** `contextualize_nli` embeds the NLI
  `PairClassification` unchanged and adds human-readable notes. A
  contradiction is **never downgraded** to neutral because a condition
  differs; differences are framed as contextual evidence whose scientific
  impact requires interpretation.
- **Evaluation:** `tests/fixtures/conditions_eval.json` is a small
  hand-labeled **sanity-check** set (matching, conflicting, missing,
  ambiguous, incomplete) — useful for catching regressions, not evidence of
  scientific model quality.

```python
from paperdna.conditions import compare_conditions, contextualize_nli, extract_conditions

premise = extract_conditions(premise_claim)        # CandidateClaim
hypothesis = extract_conditions(hypothesis_claim)  # CandidateClaim
comparison = compare_conditions(premise, hypothesis)
print(comparison.summary)  # e.g. "differing: dataset, metric; missing: population, ..."

result = contextualize_nli(classification, comparison)  # PairClassification
print(result.classification.label)  # unchanged by conditions
for note in result.notes:
    print(note)
```

### Limitations

Condition extraction is deliberately conservative and incomplete: it is a
lexical baseline, so paraphrased or unusually phrased conditions are missed
(false negatives), everyday words can be over-matched (false positives),
and it cannot resolve coreference ("on it"), abbreviations outside the rule
lists, or implicit conditions. Even when two claims extract the same
condition value, that does not prove the underlying experiments are
identical; a differing condition does **not** prove it explains divergent
findings — correlation of wording is not causation of scientific
disagreement. `missing` always means "not stated in the text", never "the
studies differ". Treat all condition output as auditable context for human
interpretation, not as an automated verdict.

## Evidence attribution & traceability

`paperdna.evidence` links every predicted relationship to the exact source
sentences behind it, keeping **observed evidence** and **model
interpretation** strictly separate.

```python
from paperdna.evidence import build_attribution, attribution_to_dict

attribution = build_attribution(
    premise_claim, hypothesis_claim,          # CandidateClaim objects
    records=ingestion_result.papers,          # enables URL/title + verification
    classification=classification,            # PairClassification, embedded unchanged
    conditions=comparison,                    # optional ConditionComparison
)
print(attribution.premise.verification)       # "verified"
print(attribution.premise.paper.url)          # or None — never fabricated
print(attribution.interpretation.label)       # model output, not observed text
```

### How to trace an output back to its source

1. Take `attribution.premise.claim.claim_id` — a stable reference of the
   form `"{source}:{source_id}:{start}-{end}"`.
2. `source` + `source_id` identify the paper; `attribution.premise.paper`
   carries its `title`, `url`, `doi`, and `published` **when the source
   provided them** (`None` otherwise — missing metadata is never invented).
3. `start`/`end` are character offsets into that paper's abstract, so
   `abstract[claim.start:claim.end] == sentence` holds for verified
   evidence — the reader can re-check any quote against the original text.
4. `attribution_to_dict(...)` produces a JSON-safe dict (exact round-trip
   via `attribution_from_dict`) for storage or display.

### Verification statuses

| Status              | Meaning                                                                    |
| ------------------- | -------------------------------------------------------------------------- |
| `verified`          | Abstract available; `abstract[start:end] == sentence` checked              |
| `verified_by_search`| Offsets **inferred** from a unique sentence occurrence (classification-only path) |
| `unverified`        | Sentence verbatim, but no abstract to check (record missing, or the sentence occurs multiple times) — reason stated |
| `unavailable`       | No usable evidence text (e.g. empty sentence) — clearly indicated, never fabricated |

`unverified`/`unavailable` evidence always carries an
`unavailable_reason`. Structurally invalid offsets, spans that fail
verification against a *known* abstract, and source-identity mismatches
between claims and the classification passed alongside them are
**rejected** (`InvalidSpanError`, `AttributionMismatchError`) rather than
silently accepted.

The classification-only path (`attribute_from_classification`) covers the
case where the original `CandidateClaim` objects are gone: sentences come
verbatim from the `PairClassification`, and offsets are inferred only on a
unique abstract occurrence — repeated sentences leave offsets `None`
(claim ID suffix `unknown`) instead of guessing.

```python
from paperdna.evidence import attribute_from_classification

attribution = attribute_from_classification(classification, records=records)
```

### Observed vs. model-generated

`SentenceEvidence` holds only copied source text and metadata.
`RelationInterpretation` (label, scores, confidence, condition summary,
notes) is typed separately and documented as model output — an NLI
prediction or condition difference is **never** represented as an observed
scientific fact, and conditions never rewrite the NLI label. The original
`PairClassification` is embedded unchanged.

### Limitations

Traceability here is *textual identity*, not scientific grounding:
verification proves a sentence came from the abstract it was extracted
from, not that the sentence is true or correctly interpreted. A
`PairClassification` on its own carries no offsets, so the
classification-only path can verify only by searching the abstract, and
infers offsets solely on a unique occurrence. Claim IDs are deterministic
but tied to the current claim extractor's segmentation — a future change
to segmentation would change IDs. If a source revises an abstract after
extraction, verification fails loudly rather than silently passing.
URL/DOI/title availability depends entirely on upstream metadata. The
rule-based extraction feeding this stage remains heuristic (see claim
extraction limitations); attribution makes its outputs auditable, it does
not upgrade their quality.

## Research evidence knowledge graph

`paperdna.graph` turns the pipeline's outputs into one navigable,
validated graph of **papers, claims, evidence sentences, and
experimental conditions**, with explicit typed relationships. It is
pure data: deterministic ids, referential integrity, and round-trippable
JSON — no network calls, no models, no new dependencies.

```python
from paperdna.graph import build_graph, graph_to_json, graph_from_json
from paperdna.graph import supporting_relations, contradicting_relations

graph = build_graph(
    papers=ingestion_result.papers,        # PaperRecord (or PaperRef) iterables
    claims=all_claims,                     # CandidateClaim iterables
    attributions=attributions,             # RelationAttribution from paperdna.evidence
    conditions=[(claim, extracted)],       # (CandidateClaim, ClaimConditions) pairs
)
claim_id = "arxiv:2101.00001:27-80"        # a claim node id (see identity below)
supporting_relations(graph, claim_id)      # -> ((edge, other_claim), ...)
contradicting_relations(graph, claim_id, direction="both")
text = graph_to_json(graph)                # deterministic bytes for storage
graph2 = graph_from_json(text)             # exact round-trip: graph2 == graph
```

The lower-level `GraphBuilder` offers `add_paper`, `add_claim`,
`add_claim_ref`, `add_evidence`, `add_claim_conditions`, `add_relation`,
and `add_attribution` (each idempotent; `build()` validates).

### Schema

Nodes (all immutable, all with deterministic ids):

| Node       | Id format                                            | Carries |
| ---------- | ---------------------------------------------------- | ------- |
| `paper`    | `paper:{source.casefold()}:{source_id}`              | title, url, doi, published (each `None`-able), authors (`()` when unknown) |
| `claim`    | `{source}:{source_id}:{start}-{end}` (offsets part `unknown` when unavailable — same format as `evidence.claim_id_for`) | verbatim sentence, abstract offsets, cue/hedges (`None` when not extracted) |
| `evidence` | `evidence:{claim_id}`                                | verbatim sentence, verification status, unavailability reason, offsets |
| `condition`| `condition:{claim_id}:{start}-{end}` (sentence-relative offsets) | category, verbatim value, ambiguous flag, global offsets when known |

Edges (typed, directed where meaningful):

| Edge                | Direction                       | Meaning |
| ------------------- | ------------------------------- | ------- |
| `claim_paper`       | claim → paper                   | `EXTRACTED_FROM`: provenance of the claim |
| `claim_evidence`    | claim → evidence                | `HAS_EVIDENCE`: one observed evidence side |
| `claim_condition`   | claim → condition               | `HAS_CONDITION`: one extracted condition mention |
| `claim_relation`    | premise claim → hypothesis claim | `CLAIM_RELATION`: the predicted NLI relationship |

### Relationship semantics: predictions, not facts

A `claim_relation` edge is **model-generated output**: it stores the
NLI label (`entailment`/`contradiction`/`neutral` — the raw label is
never renamed), all three softmax scores, confidence, the
`uncertain`/`hedged`/`truncated` flags, the model name, and — when
provided — the upstream `ConditionComparison` embedded unchanged.
Query helpers group by label (`supporting_relations` = entailment,
`contradicting_relations`, `neutral_relations`); these names describe
what the *model predicted*, never verified scientific agreement.
`RELATION_ROLES` documents the label → role mapping. An evidence-only
attribution (no classification) produces an edge with `label=None`,
excluded from the label-specific queries. NLI is directional: the edge
source is the premise, the target the hypothesis.

### Provenance guarantees

- Evidence nodes preserve the exact sentence, the upstream verification
  status (`verified` / `verified_by_search` / `unverified` /
  `unavailable`) and the stated `unavailable_reason`; offsets are kept
  when known and stay `None` (id suffix `unknown`) when not — never
  guessed.
- Missing metadata is preserved as missing: paper title/url/doi and
  claim cue/hedges are `None` (authors `()`) unless an upstream record
  provided them; a later, richer add *enriches* a stub node
  (order-independently, never overwriting conflicting values).
- Confidence scores and condition comparisons ride through unchanged
  from `paperdna.nli` and `paperdna.conditions`.
- Ids are stable and insertion-order-independent; re-adding identical
  data is a no-op; conflicting payloads under an existing id raise
  `GraphConflictError`; edges referencing nonexistent nodes, wrong
  endpoint types, or invalid labels/scores raise `GraphValidationError`
  (listing every problem).

### Serialization

`graph_to_dict` produces a JSON-safe dict (`format`/`version` envelope,
`nodes`/`edges` lists in canonical order) and `graph_to_json` the
byte-deterministic text form: equal graphs always serialize identically,
and `graph_from_dict`/`graph_from_json` round-trip exactly
(`from(to(g)) == g`). Deserialization is strict: malformed fields,
unknown types or versions, and id/payload disagreements raise
`ValueError`; dangling references raise `GraphValidationError`.

### Limitations

The graph is a *textual* evidence map, not a knowledge base of truth:
NLI labels are uncalibrated model predictions and a `neutral` or
`contradiction` label is not a scientific verdict. Claim ids depend on
the current extractor's segmentation (same caveat as evidence
attribution), and abstracts are not stored in the graph — re-verification
needs the corpus. Condition nodes exist only for claims whose
`ClaimConditions` were supplied; a comparison-only edge stores field
statuses, not mentions. Disconnected components are normal (every
classified pair forms its own island until linked); `connected_components`
reports them and queries handle them gracefully. Visualization is
deliberately out of scope here — the deterministic JSON is designed as
its input.

## Project layout

```
src/paperdna/           Package source (config, logging)
src/paperdna/corpus/    Corpus schema, validation, normalization, deduplication
src/paperdna/retrieval/ Embedding interface, cosine similarity, search index, eval
src/paperdna/claims/    Candidate claim extraction (segmentation, cue rules)
src/paperdna/nli/       Pairwise claim-relationship classification (NLI)
src/paperdna/conditions/ Experimental-condition extraction and comparison
src/paperdna/evidence/   Evidence attribution and sentence traceability
src/paperdna/graph/      Research evidence knowledge graph (schema, builder, queries, serialization)
tests/                  Pytest test suite (smoke, corpus, retrieval, claims, nli, conditions, evidence, graph tests)
tests/fixtures/         Local offline test fixtures
pyproject.toml          Project metadata, dependencies, and tool config
.env.example            Example environment configuration
```

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE).

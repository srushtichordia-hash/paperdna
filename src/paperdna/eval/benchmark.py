"""Loaders for PaperDNA's hand-labeled evaluation benchmarks.

The benchmark data lives in ``tests/fixtures/`` and is entirely
synthetic and hand-authored: it exists to catch regressions and to
measure the deterministic offline baselines honestly. It is NOT a
sample of real scientific literature and NOT evidence of model quality
on real corpora.

Both loaders validate structure eagerly (unknown splits, duplicate
names, unresolvable paper references raise ``ValueError``) so a broken
fixture fails loudly instead of silently skewing metrics.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from paperdna.claims.schema import CandidateClaim
from paperdna.corpus.schema import PaperRecord

# tests/fixtures, resolved from src/paperdna/eval/benchmark.py.
DEFAULT_FIXTURES_DIR = Path(__file__).resolve().parents[3] / "tests" / "fixtures"
DEFAULT_RETRIEVAL_BENCHMARK = DEFAULT_FIXTURES_DIR / "retrieval_eval.json"
DEFAULT_NLI_PAIRS = DEFAULT_FIXTURES_DIR / "nli_pairs.json"
DEFAULT_CONDITIONS_PAIRS = DEFAULT_FIXTURES_DIR / "conditions_eval.json"

VALID_SPLITS = ("dev", "test")


def _load_json(path: Path) -> dict:
    if not path.is_file():
        raise ValueError(
            f"benchmark file not found: {path} — run from a full checkout, "
            "or pass an explicit --*-benchmark path"
        )
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a JSON object at the top level")
    return data


@dataclass(frozen=True)
class BenchmarkPaper:
    """One synthetic paper in the retrieval benchmark."""

    paper_id: str  # "{source}:{source_id}", e.g. "arxiv:2101.00001"
    source: str
    source_id: str
    title: str
    abstract: str

    def to_record(self) -> PaperRecord:
        return PaperRecord(
            source=self.source,
            source_id=self.source_id,
            title=self.title,
            abstract=self.abstract,
            authors=("Benchmark Author",),
        )


@dataclass(frozen=True)
class RetrievalQuery:
    """One hand-labeled retrieval query."""

    query: str
    relevant: tuple[str, ...]
    split: str
    anchor: bool


@dataclass(frozen=True)
class LabeledPair:
    """One hand-labeled claim pair for relation classification."""

    name: str
    premise: CandidateClaim
    hypothesis: CandidateClaim
    expected_label: str
    scripted_scores: dict[str, float]
    expected_uncertain: bool
    split: str


def load_retrieval_benchmark(
    path: Path = DEFAULT_RETRIEVAL_BENCHMARK,
) -> tuple[list[PaperRecord], list[RetrievalQuery]]:
    """Load the self-contained retrieval benchmark.

    Returns deduplicated-in-fixture-order ``(records, queries)``. Every
    query's ``relevant`` ids must resolve to benchmark papers; splits
    must be ``dev`` or ``test``.
    """
    data = _load_json(path)
    papers = data.get("papers")
    queries = data.get("queries")
    if not isinstance(papers, list) or not papers:
        raise ValueError(f"{path}: 'papers' must be a non-empty list")
    if not isinstance(queries, list) or not queries:
        raise ValueError(f"{path}: 'queries' must be a non-empty list")

    records: list[PaperRecord] = []
    seen_ids: set[str] = set()
    for item in papers:
        paper_id = item["id"]
        if paper_id in seen_ids:
            raise ValueError(f"{path}: duplicate paper id {paper_id!r}")
        seen_ids.add(paper_id)
        source, _, source_id = paper_id.partition(":")
        if not source or not source_id:
            raise ValueError(f"{path}: paper id must be '{{source}}:{{source_id}}': {paper_id!r}")
        records.append(
            BenchmarkPaper(
                paper_id=paper_id,
                source=source,
                source_id=source_id,
                title=item["title"],
                abstract=item["abstract"],
            ).to_record()
        )

    parsed: list[RetrievalQuery] = []
    seen_queries: set[str] = set()
    for item in queries:
        text = item["query"]
        if text in seen_queries:
            raise ValueError(f"{path}: duplicate query text {text!r}")
        seen_queries.add(text)
        relevant = item["relevant"]
        if not relevant:
            raise ValueError(f"{path}: query {text!r} has no relevant papers")
        unknown = set(relevant) - seen_ids
        if unknown:
            raise ValueError(f"{path}: query {text!r} references unknown papers {sorted(unknown)}")
        split = item["split"]
        if split not in VALID_SPLITS:
            raise ValueError(f"{path}: unknown split {split!r} for query {text!r}")
        parsed.append(
            RetrievalQuery(
                query=text,
                relevant=tuple(relevant),
                split=split,
                anchor=bool(item.get("anchor", False)),
            )
        )

    covered = {paper_id for q in parsed for paper_id in q.relevant}
    if covered != seen_ids:
        missing = sorted(seen_ids - covered)
        raise ValueError(f"{path}: papers never labeled relevant for any query: {missing}")

    return records, parsed


def _claim_from(data: dict, name: str, *, allow_empty: bool = False) -> CandidateClaim:
    sentence = data["sentence"]
    if not sentence or not sentence.strip():
        if not allow_empty:
            raise ValueError(f"{name}: empty sentence")
    return CandidateClaim(
        source=data["source"],
        source_id=data["source_id"],
        sentence=sentence,
        start=0,
        end=len(sentence),
        cue="",
        hedges=tuple(data.get("hedges", ())),
    )


def load_nli_pairs(path: Path = DEFAULT_NLI_PAIRS) -> list[LabeledPair]:
    """Load hand-labeled claim pairs with gold labels and splits.

    ``scripted_scores`` reproduce the deterministic fake classifier's
    outputs; they are plumbing-validation data, circular by
    construction, and must never be reported as model metrics.
    """
    data = _load_json(path)
    pairs = data.get("pairs")
    if not isinstance(pairs, list) or not pairs:
        raise ValueError(f"{path}: 'pairs' must be a non-empty list")
    from paperdna.nli.schema import RELATIONSHIPS

    parsed: list[LabeledPair] = []
    seen: set[str] = set()
    for item in pairs:
        name = item["name"]
        if name in seen:
            raise ValueError(f"{path}: duplicate pair name {name!r}")
        seen.add(name)
        label = item["expected_label"]
        if label not in RELATIONSHIPS:
            raise ValueError(f"{path}: pair {name!r} has unknown label {label!r}")
        split = item["split"]
        if split not in VALID_SPLITS:
            raise ValueError(f"{path}: unknown split {split!r} for pair {name!r}")
        scripted = item["scripted_scores"]
        if set(scripted) != set(RELATIONSHIPS):
            raise ValueError(f"{path}: pair {name!r} scripted_scores must cover {RELATIONSHIPS}")
        parsed.append(
            LabeledPair(
                name=name,
                premise=_claim_from(item["premise"], name),
                hypothesis=_claim_from(item["hypothesis"], name),
                expected_label=label,
                scripted_scores=dict(scripted),
                expected_uncertain=bool(item["expected_uncertain"]),
                split=split,
            )
        )
    return parsed


@dataclass(frozen=True)
class ConditionsPair:
    """One hand-labeled condition-comparison pair."""

    name: str
    premise: CandidateClaim
    hypothesis: CandidateClaim
    expected_statuses: dict[str, str]
    split: str


def load_conditions_pairs(path: Path = DEFAULT_CONDITIONS_PAIRS) -> list[ConditionsPair]:
    """Load hand-labeled condition-comparison pairs."""
    data = _load_json(path)
    pairs = data.get("pairs")
    if not isinstance(pairs, list) or not pairs:
        raise ValueError(f"{path}: 'pairs' must be a non-empty list")
    parsed: list[ConditionsPair] = []
    seen: set[str] = set()
    for item in pairs:
        name = item["name"]
        if name in seen:
            raise ValueError(f"{path}: duplicate pair name {name!r}")
        seen.add(name)
        # allow_empty: the fixture's "incomplete_claim_side" pair deliberately
        # has an empty premise — the rule baseline must treat it as "missing".
        parsed.append(
            ConditionsPair(
                name=name,
                premise=_claim_from(item["premise"], name, allow_empty=True),
                hypothesis=_claim_from(item["hypothesis"], name, allow_empty=True),
                expected_statuses=dict(item["expected_statuses"]),
                split=str(item.get("split", "dev")),
            )
        )
    return parsed

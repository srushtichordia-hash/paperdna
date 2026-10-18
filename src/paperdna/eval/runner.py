"""Command-line evaluation runner.

Usage (from a full checkout, after ``pip install -e ".[dev,api]"``)::

    python -m paperdna.eval retrieval --split test
    python -m paperdna.eval nli --split test --classifier overlap
    python -m paperdna.eval conditions
    python -m paperdna.eval evidence
    python -m paperdna.eval all --out docs/evaluation/reports

Output JSON is deterministic (sorted keys, no timestamps): rerunning on
the same checkout with the same flags reproduces the report byte for
byte. Real-model tiers (``--semantic``, ``--classifier model``) are
explicit opt-ins that may download assets; the offline tier never does.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from paperdna.eval.benchmark import (
    load_conditions_pairs,
    load_nli_pairs,
    load_retrieval_benchmark,
)
from paperdna.eval.baselines import OverlapNliClassifier, ScriptedNliClassifier
from paperdna.eval.conditions_report import evaluate_conditions
from paperdna.eval.evidence_audit import audit_attributions
from paperdna.eval.nli_report import evaluate_nli
from paperdna.eval.retrieval_report import DEFAULT_KS, evaluate_retrieval, lexical_rank


def _split_filter(items, split: str):
    if split == "all":
        return list(items)
    return [item for item in items if item.split == split]


def _run_retrieval(args: argparse.Namespace) -> dict:
    records, queries = load_retrieval_benchmark(Path(args.retrieval_benchmark))
    selected = _split_filter(queries, args.split)
    if not selected:
        raise SystemExit(f"no queries in split {args.split!r}")
    ks = tuple(args.k) or DEFAULT_KS

    if args.semantic:
        try:
            from paperdna.retrieval.embedder import SentenceTransformerEmbedder

            embedder = SentenceTransformerEmbedder()
        except Exception as exc:  # optional extra missing, download refused, ...
            raise SystemExit(
                f"semantic tier unavailable: {exc}\n"
                'Install the optional extra first: pip install -e ".[embed]" '
                "(this may download the embedding model)."
            )
        from paperdna.eval.retrieval_report import semantic_rank

        return evaluate_retrieval(
            records,
            selected,
            lambda q, r: semantic_rank(embedder, q, r),
            system=f"semantic:{embedder.name}",
            ranking_kind="semantic",
            ks=ks,
        )
    return evaluate_retrieval(
        records, selected, lexical_rank, system="lexical", ranking_kind="lexical", ks=ks
    )


def _make_nli_classifier(name: str, pairs_path: str):
    if name == "overlap":
        return OverlapNliClassifier(), False
    if name == "scripted":
        script = {
            (p.premise.sentence, p.hypothesis.sentence): p.scripted_scores
            for p in load_nli_pairs(Path(pairs_path))
        }
        return ScriptedNliClassifier(script), True
    if name == "model":
        try:
            from paperdna.nli.classifier import CrossEncoderNliClassifier

            return CrossEncoderNliClassifier(), False
        except Exception as exc:  # optional extra missing, download refused, ...
            raise SystemExit(
                f"NLI model unavailable: {exc}\n"
                'Install the optional extra first: pip install -e ".[nli]" '
                "(this may download the NLI model)."
            )
    raise SystemExit(f"unknown classifier: {name!r}")


def _run_nli(args: argparse.Namespace) -> dict:
    pairs = _split_filter(load_nli_pairs(Path(args.nli_pairs)), args.split)
    if not pairs:
        raise SystemExit(f"no pairs in split {args.split!r}")
    classifier, circular = _make_nli_classifier(args.classifier, args.nli_pairs)
    return evaluate_nli(pairs, classifier, circular=circular)


def _run_conditions(_args: argparse.Namespace) -> dict:
    return evaluate_conditions(load_conditions_pairs())


def _run_evidence(_args: argparse.Namespace) -> dict:
    from paperdna.api.pipeline import DEMO_CORPUS_PATH
    from paperdna.claims.extractor import extract_claims
    from paperdna.corpus.ingest import ingest_records, load_jsonl
    from paperdna.evidence.build import (
        attribute_from_classification,
        build_attribution,
    )
    from paperdna.nli.classify import classify_pair

    records = ingest_records(load_jsonl(str(DEMO_CORPUS_PATH))).papers
    attributions = []

    # Path A: extracted demo claims, both sides verified against abstracts.
    all_claims = []
    for record in records:
        for claim in extract_claims(record).claims:
            all_claims.append(claim)
            attributions.append(build_attribution(claim, claim, records=records))

    # Path B: real claims attributed WITHOUT their records -> the offsets
    # stay the extractor's own but are honestly marked unverified.
    unrecorded = 0
    for premise, hypothesis in list(zip(all_claims, all_claims[1:]))[:4]:
        attributions.append(build_attribution(premise, hypothesis, records=()))
        unrecorded += 1

    # Path C: classification-only attribution -> offsets inferred by search.
    classifier = OverlapNliClassifier()
    for premise, hypothesis in zip(all_claims, all_claims[1:]):
        classification = classify_pair(premise, hypothesis, classifier)
        attributions.append(
            attribute_from_classification(classification, records=records)
        )

    report = audit_attributions(attributions, records)
    report["sources"] = {
        "verified_self_pairs": len(all_claims),
        "unrecorded_pairs": unrecorded,
        "classification_only_pairs": max(0, len(all_claims) - 1),
    }
    return report


def _write(payload: dict, out: str | None) -> None:
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if out:
        path = Path(out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        print(f"wrote {path}", file=sys.stderr)
    else:
        print(text, end="")


def _run_all(args: argparse.Namespace) -> dict:
    retrieval_args = argparse.Namespace(
        retrieval_benchmark=args.retrieval_benchmark,
        split=args.split,
        k=[],
        semantic=False,
    )
    nli_overlap_args = argparse.Namespace(
        nli_pairs=args.nli_pairs, split=args.split, classifier="overlap"
    )
    nli_scripted_args = argparse.Namespace(
        nli_pairs=args.nli_pairs, split="all", classifier="scripted"
    )
    return {
        "retrieval_lexical": _run_retrieval(retrieval_args),
        "nli_overlap_baseline": _run_nli(nli_overlap_args),
        "nli_scripted_conformance": _run_nli(nli_scripted_args),
        "conditions_correctness": _run_conditions(args),
        "evidence_audit": _run_evidence(args),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m paperdna.eval",
        description="PaperDNA offline evaluation runner (deterministic, no downloads).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    fixtures_root = Path(__file__).resolve().parents[3] / "tests" / "fixtures"

    p_ret = sub.add_parser("retrieval", help="P@k / R@k over the labeled benchmark")
    p_ret.add_argument("--split", choices=["dev", "test", "all"], default="test")
    p_ret.add_argument("--k", nargs="*", type=int, default=[])
    p_ret.add_argument("--semantic", action="store_true",
                       help="use the optional sentence-transformers embedder (downloads)")
    p_ret.add_argument("--retrieval-benchmark",
                       default=str(fixtures_root / "retrieval_eval.json"))
    p_ret.set_defaults(func=lambda a: _write(_run_retrieval(a), a.out))

    p_nli = sub.add_parser("nli", help="macro-F1 / per-class metrics on labeled pairs")
    p_nli.add_argument("--split", choices=["dev", "test", "all"], default="test")
    p_nli.add_argument("--classifier", choices=["overlap", "scripted", "model"],
                       default="overlap")
    p_nli.add_argument("--nli-pairs", default=str(fixtures_root / "nli_pairs.json"))
    p_nli.set_defaults(func=lambda a: _write(_run_nli(a), a.out))

    p_cond = sub.add_parser("conditions", help="rule-baseline correctness on labeled pairs")
    p_cond.set_defaults(func=lambda a: _write(_run_conditions(a), a.out))

    p_evi = sub.add_parser("evidence", help="traceability audit over generated attributions")
    p_evi.set_defaults(func=lambda a: _write(_run_evidence(a), a.out))

    p_all = sub.add_parser("all", help="run the full offline tier as one report")
    p_all.add_argument("--split", choices=["dev", "test", "all"], default="test")
    p_all.add_argument("--retrieval-benchmark",
                       default=str(fixtures_root / "retrieval_eval.json"))
    p_all.add_argument("--nli-pairs", default=str(fixtures_root / "nli_pairs.json"))
    p_all.set_defaults(func=lambda a: _write(_run_all(a), a.out))

    for p in (p_ret, p_nli, p_cond, p_evi, p_all):
        p.add_argument("--out", default=None, help="write JSON to this path instead of stdout")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.func(args)
    return 0

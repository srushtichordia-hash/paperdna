"""Offline evaluation harness for PaperDNA.

Loads the hand-labeled synthetic benchmarks in ``tests/fixtures/``,
runs the deterministic baselines (and, when explicitly installed, the
real optional models), and reports metrics computed from ACTUAL
predictions versus labels: retrieval Precision@k / Recall@k,
classification macro-F1 with per-class metrics, condition-comparison
correctness, and an evidence-traceability audit.

Everything here is deterministic and offline by default; nothing in
this package downloads or loads an ML model implicitly. See
``docs/evaluation.md`` for methodology, splits, results, and the honest
limitations of these measurements.
"""

from paperdna.eval.baselines import OverlapNliClassifier, ScriptedNliClassifier
from paperdna.eval.benchmark import (
    ConditionsPair,
    LabeledPair,
    RetrievalQuery,
    load_conditions_pairs,
    load_nli_pairs,
    load_retrieval_benchmark,
)
from paperdna.eval.conditions_report import evaluate_conditions
from paperdna.eval.evidence_audit import audit_attribution, audit_attributions, audit_side
from paperdna.eval.nli_report import evaluate_nli
from paperdna.eval.retrieval_report import evaluate_retrieval, lexical_rank, semantic_rank

__all__ = [
    "ConditionsPair",
    "LabeledPair",
    "OverlapNliClassifier",
    "RetrievalQuery",
    "ScriptedNliClassifier",
    "audit_attribution",
    "audit_attributions",
    "audit_side",
    "evaluate_conditions",
    "evaluate_nli",
    "evaluate_retrieval",
    "lexical_rank",
    "load_conditions_pairs",
    "load_nli_pairs",
    "load_retrieval_benchmark",
    "semantic_rank",
]

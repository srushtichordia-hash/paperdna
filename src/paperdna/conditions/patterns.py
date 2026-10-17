"""Per-category extraction rules for experimental conditions.

Transparent, deterministic, stdlib-only regex rules. Each rule is
``(category, pattern, ambiguous)``:

- ``ambiguous=False`` rules capture specific, comparable mentions.
- ``ambiguous=True`` rules capture generic wording ("our dataset") that is
  extracted but flagged so comparisons report *uncertain* rather than
  matching/differing.

Matches are resolved deterministically: at the same start offset the
longest match wins, overlapping shorter matches are dropped, ambiguous
rules never shadow specific ones. No rule ever invents a value.
"""

from __future__ import annotations

import re

Rule = tuple[str, re.Pattern[str], bool]

_SPECIFIC_DATASET = (
    "dataset",
    re.compile(
        r"\b([A-Z][\w.\-]*(?:\s+[A-Z][\w.\-]*)*\s+(?:dataset|corpus|benchmark))\b"
    ),
    False,
)
_KNOWN_DATASETS = (
    "dataset",
    re.compile(
        r"\b(SQuAD|SuperGLUE|GLUE|ImageNet|MNIST|CIFAR-10|CIFAR-200|CIFAR-100|"
        r"MS-COCO|COCO|Wikipedia|PubMed|MIMIC-III|BLURB|SciFact|The Pile)\b"
    ),
    False,
)
_AMBIGUOUS_DATASET = (
    "dataset",
    re.compile(
        r"\b(our|a|an|the|this|these|several|various|multiple|public|internal|"
        r"standard)\s+(?:public\s+|standard\s+|internal\s+|new\s+|"
        r"benchmark\s+)?(datasets?|corpora|corpus|benchmarks?)\b",
        re.IGNORECASE,
    ),
    True,
)

_SPECIFIC_POPULATION = (
    "population",
    re.compile(r"\b(\d{1,4}(?:,\d{3})?\s+(?:patients|participants|subjects|"
               r"individuals|adults|children|volunteers|samples))\b",
               re.IGNORECASE),
    False,
)
_POPULATION_N = (
    "population",
    re.compile(r"\b(N\s*=\s*\d{1,4})\b"),
    False,
)
_AMBIGUOUS_POPULATION = (
    "population",
    re.compile(
        r"\b(our|a|an|the|this|these|several|various|multiple)\s+"
        r"(?:eligible\s+|small\s+|large\s+)?"
        r"(cohorts?|samples?|populations?|participants|patients|subjects)s?\b",
        re.IGNORECASE,
    ),
    True,
)

_KNOWN_MODELS = (
    "model",
    re.compile(
        r"\b(BERT|RoBERTa|GPT-3|GPT-4|DeBERTa|SciBERT|BioBERT|ViT|ResNet|"
        r"LSTM|T5|CLIP|U-Net)\b"
    ),
    False,
)
_SPECIFIC_MODEL = (
    "model",
    re.compile(
        r"\b(?:a|an|the|our|this)\s+([\w\-]+(?:\s+[\w\-]+)?)\s+"
        r"(model|models|network|classifier|architecture|transformer)\b",
        re.IGNORECASE,
    ),
    False,
)
_AMBIGUOUS_MODEL = (
    "model",
    re.compile(
        r"\b(our|a|an|the|this|their)\s+"
        r"(?:proposed\s+|novel\s+|baseline\s+|current\s+)?"
        r"(model|models|method|methods|approach|approaches|intervention|"
        r"treatment|architecture)\b",
        re.IGNORECASE,
    ),
    True,
)

_METRIC = (
    "metric",
    re.compile(
        r"\b(accuracy|f1(?:[\s\-]score)?|auc-roc|auc|bleu|rouge-l|rouge|precision|"
        r"recall|rmse|mae|perplexity|error[\s\-]?rate)\b",
        re.IGNORECASE,
    ),
    False,
)
_AMBIGUOUS_METRIC = (
    "metric",
    re.compile(
        r"\b(our|the|a|this|primary|main|standard|chosen)\s+"
        r"(?:evaluation\s+|performance\s+)?(metric|measure|scoring)\b",
        re.IGNORECASE,
    ),
    True,
)

_SETTING = (
    "setting",
    re.compile(
        r"\b((?:low-resource|high-resource|cross-domain|in-domain|real-world|"
        r"clinical|controlled|large-scale|small-scale|zero-shot|few-shot|"
        r"out-of-distribution|noisy)\s+"
        r"(?:settings?|conditions?|domains?|scenarios|environments))\b",
        re.IGNORECASE,
    ),
    False,
)

_DESIGN = (
    "design",
    re.compile(
        r"\b((?:randomi[sz]ed|observational|longitudinal|cross-sectional|"
        r"case-control|prospective|retrospective|double-blind|"
        r"quasi-experimental)(?:\s+controlled)?\s+"
        r"(?:study|trial|experiment|design|analysis|cohort)"
        r"|ablation stud(?:y|ies))\b",
        re.IGNORECASE,
    ),
    False,
)

# Application order: specific rules first (they win overlaps), then
# ambiguous rules (they never shadow specific ones).
SPECIFIC_RULES: tuple[Rule, ...] = (
    _SPECIFIC_DATASET,
    _KNOWN_DATASETS,
    _SPECIFIC_POPULATION,
    _POPULATION_N,
    _KNOWN_MODELS,
    _SPECIFIC_MODEL,
    _METRIC,
    _SETTING,
    _DESIGN,
)

AMBIGUOUS_RULES: tuple[Rule, ...] = (
    _AMBIGUOUS_DATASET,
    _AMBIGUOUS_POPULATION,
    _AMBIGUOUS_MODEL,
    _AMBIGUOUS_METRIC,
)


def normalize_value(text: str) -> str:
    """Comparison form of a mention value: casefolded, whitespace-
    collapsed, trailing punctuation stripped. The stored value itself is
    never altered — this is used only for equality checks."""
    return " ".join(text.split()).casefold().rstrip(" .,;:!?")

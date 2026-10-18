"""Tests for graph queries: relations by label, evidence, conditions,
components, and empty/disconnected behavior."""

from __future__ import annotations

import pytest

from paperdna.conditions.compare import compare_conditions
from paperdna.conditions.extract import extract_conditions
from paperdna.evidence import build_attribution
from paperdna.graph import (
    GraphBuilder,
    build_graph,
    claim_node_id,
    claims,
    condition_nodes,
    conditions_for_claim,
    conditions_for_relation,
    connected_components,
    contradicting_relations,
    evidence_for_claim,
    evidence_for_relation,
    evidence_nodes,
    neutral_relations,
    papers,
    paper_for_claim,
    relations,
    supporting_relations,
)
from tests.graph.conftest import (
    DEFAULT_SENTENCE,
    HYPOTHESIS_SENTENCE,
    make_claim,
    make_classification,
    make_hypothesis_claim,
    make_hypothesis_record,
    make_record,
)

THIRD_SENTENCE = "We find that robustness improves on the SQuAD dataset."


def _third_claim():
    abstract = "Intro text. " + THIRD_SENTENCE + " Outro."
    return make_claim(
        source="arxiv",
        source_id="2101.00002",
        sentence=THIRD_SENTENCE,
        start=abstract.index(THIRD_SENTENCE),
    ), abstract


def _three_relation_graph():
    """One premise claim related to three hypotheses (support, contradict,
    neutral) plus its own evidence.

    Only the premise's paper record is supplied: the synthetic hypothesis
    sentences do not appear in the hypothesis record's abstract, so their
    evidence stays ``unverified`` (which these queries do not depend on).
    """
    builder = GraphBuilder()
    premise = make_claim()
    pairs = [
        (make_hypothesis_claim(), "entailment", {"entailment": 0.9, "contradiction": 0.05, "neutral": 0.05}),
        (make_hypothesis_claim(source_id="f3", sentence="We find that accuracy drops on the MNIST dataset."), "contradiction", {"entailment": 0.1, "contradiction": 0.8, "neutral": 0.1}),
        (make_hypothesis_claim(source_id="f4", sentence="We find that latency increases on the CIFAR-10 dataset."), "neutral", {"entailment": 0.3, "contradiction": 0.2, "neutral": 0.5}),
    ]
    for hypothesis, label, scores in pairs:
        classification = make_classification(
            hypothesis_source_id=hypothesis.source_id,
            hypothesis_text=hypothesis.sentence,
            label=label,
            scores=scores,
            confidence=max(scores.values()),
        )
        builder.add_attribution(
            build_attribution(
                premise,
                hypothesis,
                records=[make_record()],
                classification=classification,
            )
        )
    return builder.build(), premise


class TestRelationQueries:
    def test_supporting_contradicting_neutral(self):
        graph, premise = _three_relation_graph()
        claim_id = claim_node_id(
            premise.source, premise.source_id, premise.start, premise.end
        )
        support = supporting_relations(graph, claim_id)
        contradict = contradicting_relations(graph, claim_id)
        neutral = neutral_relations(graph, claim_id)
        assert len(support) == 1 and support[0][0].label == "entailment"
        assert len(contradict) == 1 and contradict[0][0].label == "contradiction"
        assert len(neutral) == 1 and neutral[0][0].label == "neutral"
        # Each partner is the claim at the far end of the edge.
        assert support[0][1].source_id == "f2"
        assert contradict[0][1].source_id == "f3"
        assert neutral[0][1].source_id == "f4"

    def test_direction_in_out_both(self):
        graph, premise = _three_relation_graph()
        claim_id = claim_node_id(
            premise.source, premise.source_id, premise.start, premise.end
        )
        out = relations(graph, claim_id)
        assert len(out) == 3
        assert relations(graph, claim_id, direction="out") == out
        # From a hypothesis's perspective, the relation is incoming.
        hypothesis_id = out[0][1].id
        assert relations(graph, hypothesis_id, direction="out") == ()
        incoming = relations(graph, hypothesis_id, direction="in")
        assert len(incoming) == 1 and incoming[0][0].source == claim_id
        both = relations(graph, hypothesis_id, direction="both")
        assert both == incoming

    def test_invalid_direction(self):
        graph, premise = _three_relation_graph()
        claim_id = claim_node_id(
            premise.source, premise.source_id, premise.start, premise.end
        )
        with pytest.raises(ValueError):
            relations(graph, claim_id, direction="sideways")

    def test_unknown_claim_and_empty_graph(self):
        graph, _ = _three_relation_graph()
        assert relations(graph, "nope:1:unknown") == ()
        assert supporting_relations(graph, "nope:1:unknown") == ()
        empty = build_graph()
        assert relations(empty, "arxiv:1:0-4") == ()
        assert contradicting_relations(empty, "arxiv:1:0-4") == ()

    def test_unlabeled_relation_excluded_from_label_queries(self):
        attribution = build_attribution(
            make_claim(),
            make_hypothesis_claim(),
            records=[make_record(), make_hypothesis_record()],
        )
        builder = GraphBuilder()
        builder.add_attribution(attribution)
        graph = builder.build()
        claim_id = attribution.premise.claim.claim_id
        assert len(relations(graph, claim_id)) == 1
        assert supporting_relations(graph, claim_id) == ()
        assert contradicting_relations(graph, claim_id) == ()
        assert neutral_relations(graph, claim_id) == ()


class TestEvidenceAndPaperQueries:
    def test_paper_and_evidence_for_claim(self):
        graph, premise = _three_relation_graph()
        claim_id = claim_node_id(
            premise.source, premise.source_id, premise.start, premise.end
        )
        paper = paper_for_claim(graph, claim_id)
        assert paper.url == "https://arxiv.org/abs/2101.00001"
        evidence = evidence_for_claim(graph, claim_id)
        assert evidence.verification == "verified"
        assert evidence.sentence == DEFAULT_SENTENCE

    def test_unknown_claim_returns_none(self):
        graph, _ = _three_relation_graph()
        assert paper_for_claim(graph, "nope") is None
        assert evidence_for_claim(graph, "nope") is None
        assert conditions_for_claim(graph, "nope") == ()

    def test_evidence_for_relation_both_sides(self):
        graph, premise = _three_relation_graph()
        claim_id = claim_node_id(
            premise.source, premise.source_id, premise.start, premise.end
        )
        edge, _ = supporting_relations(graph, claim_id)[0]
        premise_ev, hypothesis_ev = evidence_for_relation(graph, edge)
        assert premise_ev.sentence == DEFAULT_SENTENCE
        assert hypothesis_ev.sentence == HYPOTHESIS_SENTENCE

    def test_evidence_for_relation_without_evidence_sides(self):
        builder = GraphBuilder()
        premise_id, hypothesis_id = builder.add_relation(make_classification())
        graph = builder.build()
        edge = graph.edges[-1]
        assert evidence_for_relation(graph, edge) == (None, None)

    def test_evidence_for_relation_type_check(self):
        graph, _ = _three_relation_graph()
        with pytest.raises(TypeError):
            evidence_for_relation(graph, object())


class TestConditionQueries:
    def test_conditions_for_claim_and_relation(self):
        claim = make_claim()
        hypothesis = make_hypothesis_claim()
        comparison = compare_conditions(
            extract_conditions(claim), extract_conditions(hypothesis)
        )
        builder = GraphBuilder()
        premise_id = builder.add_claim_conditions(claim, extract_conditions(claim))
        builder.add_relation(make_classification(), conditions=comparison)
        graph = builder.build()

        nodes = conditions_for_claim(graph, premise_id)
        assert nodes  # "GLUE benchmark" mention exists
        assert all(n.claim_id == premise_id for n in nodes)

        edge = next(e for e in graph.edges if e.type == "claim_relation")
        assert conditions_for_relation(graph, edge) == comparison

    def test_relation_without_conditions(self):
        builder = GraphBuilder()
        _, _ = builder.add_relation(make_classification())
        graph = builder.build()
        edge = graph.edges[-1]
        assert conditions_for_relation(graph, edge) is None

    def test_conditions_for_relation_type_check(self):
        graph, _ = _three_relation_graph()
        with pytest.raises(TypeError):
            conditions_for_relation(graph, object())


class TestListingsAndComponents:
    def test_typed_listings_sorted_by_id(self):
        graph, _ = _three_relation_graph()
        paper_list = papers(graph)
        assert all(p.type == "paper" for p in paper_list)
        assert [p.id for p in paper_list] == sorted(p.id for p in paper_list)
        claim_list = claims(graph)
        assert len(claim_list) == 4  # premise + 3 hypotheses
        ev_list = evidence_nodes(graph)
        assert len(ev_list) == 4
        assert condition_nodes(graph) == ()

    def test_single_component_when_connected(self):
        graph, _ = _three_relation_graph()
        components = connected_components(graph)
        assert len(components) == 1
        assert len(components[0]) == len(graph.nodes)

    def test_disconnected_and_isolated_nodes(self):
        # Two separate claim/paper islands plus a bare paper.
        graph = build_graph(
            claims=[make_claim(), make_hypothesis_claim()],
            papers=[make_record(source="arxiv", source_id="lonely")],
        )
        components = connected_components(graph)
        assert len(components) == 3
        sizes = sorted(len(c) for c in components)
        assert sizes == [1, 2, 2]
        lonely = [c for c in components if len(c) == 1][0]
        assert lonely == ("paper:arxiv:lonely",)

    def test_empty_graph_components(self):
        assert connected_components(build_graph()) == ()

    def test_disconnected_graph_is_valid_and_queries_work(self):
        third, _ = _third_claim()
        graph = build_graph(claims=[third])
        graph.validate()
        claim_id = claim_node_id(
            third.source, third.source_id, third.start, third.end
        )
        # Isolated claim: no relations, but its provenance edge works.
        assert relations(graph, claim_id) == ()
        assert paper_for_claim(graph, claim_id) is not None
        assert len(connected_components(graph)) == 1

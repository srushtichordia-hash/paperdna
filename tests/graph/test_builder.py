"""Tests for building graphs from upstream pipeline outputs."""

from __future__ import annotations

import pytest

from paperdna.claims.schema import CandidateClaim
from paperdna.conditions.compare import compare_conditions
from paperdna.conditions.extract import extract_conditions
from paperdna.evidence import (
    attribute_from_classification,
    build_attribution,
)
from paperdna.evidence.schema import (
    ClaimRef,
    PaperRef,
    claim_id_for,
)
from paperdna.graph import (
    ClaimNode,
    ClaimRelationEdge,
    GraphBuilder,
    GraphConflictError,
    KnowledgeGraph,
    PaperNode,
    build_graph,
    claim_node_id,
    paper_node_id,
)
from tests.graph.conftest import (
    DEFAULT_SENTENCE,
    make_claim,
    make_classification,
    make_hypothesis_claim,
    make_hypothesis_record,
    make_record,
)


def _attribution(**overrides):
    return build_attribution(
        make_claim(),
        make_hypothesis_claim(),
        records=[make_record(), make_hypothesis_record()],
        classification=make_classification(),
        **overrides,
    )


class TestAddPaper:
    def test_from_paper_record(self):
        builder = GraphBuilder()
        node_id = builder.add_paper(make_record())
        graph = builder.build()
        node = graph.node(node_id)
        assert isinstance(node, PaperNode)
        assert node.title == "Sparse Attention Studies"
        assert node.authors == ("A. Author",)
        assert node.url == "https://arxiv.org/abs/2101.00001"

    def test_from_paper_ref_without_metadata(self):
        builder = GraphBuilder()
        builder.add_paper(PaperRef(source="arxiv", source_id="x1"))
        node = builder.build().node(paper_node_id("arxiv", "x1"))
        assert isinstance(node, PaperNode)
        assert node.title is None and node.url is None and node.authors == ()

    def test_wrong_type(self):
        with pytest.raises(TypeError):
            GraphBuilder().add_paper("arxiv:1")


class TestAddClaim:
    def test_creates_claim_paper_and_edge(self):
        builder = GraphBuilder()
        claim_id = builder.add_claim(make_claim())
        graph = builder.build()
        claim = graph.node(claim_id)
        assert isinstance(claim, ClaimNode)
        assert claim.sentence == DEFAULT_SENTENCE
        assert claim.cue == "we find" and claim.hedges == ()
        assert paper_node_id("arxiv", "2101.00001") in graph
        edge = graph.edges[0]
        assert edge.type == "claim_paper"
        assert edge.source == claim_id
        assert edge.target == paper_node_id("arxiv", "2101.00001")

    def test_wrong_type(self):
        with pytest.raises(TypeError):
            GraphBuilder().add_claim(None)

    def test_add_claim_ref(self):
        ref = ClaimRef(
            claim_id=claim_id_for("fixture", "f9", None, None),
            source="fixture",
            source_id="f9",
            start=None,
            end=None,
        )
        builder = GraphBuilder()
        claim_id = builder.add_claim_ref(ref, "We find nothing here.")
        claim = builder.build().node(claim_id)
        assert claim.id.endswith(":unknown")
        assert claim.cue is None and claim.hedges is None


class TestAddEvidence:
    def test_creates_evidence_links_and_paper(self):
        attribution = _attribution()
        builder = GraphBuilder()
        claim_id = builder.add_evidence(attribution.premise)
        graph = builder.build()
        evidence = graph.node(f"evidence:{claim_id}")
        assert evidence is not None and evidence.verification == "verified"
        assert evidence.sentence == DEFAULT_SENTENCE
        assert evidence.start == make_claim().start
        edge_types = sorted(e.type for e in graph.edges)
        assert edge_types == ["claim_evidence", "claim_paper"]

    def test_wrong_type(self):
        with pytest.raises(TypeError):
            GraphBuilder().add_evidence(object())


class TestAddAttribution:
    def test_full_wiring(self):
        builder = GraphBuilder()
        premise_id, hypothesis_id = builder.add_attribution(_attribution())
        graph = builder.build()
        stats = graph.stats
        assert stats["paper"] == 2
        assert stats["claim"] == 2
        assert stats["evidence"] == 2
        assert stats["claim_relation"] == 1

        relation = graph.edges[-1]
        assert isinstance(relation, ClaimRelationEdge)
        assert relation.source == premise_id
        assert relation.target == hypothesis_id
        assert relation.label == "entailment"
        assert relation.role == "supports"
        assert relation.confidence == 0.8
        assert relation.model == "fake-nli"
        assert relation.scores == {
            "entailment": 0.8,
            "contradiction": 0.1,
            "neutral": 0.1,
        }

    def test_evidence_only_attribution_is_unlabeled(self):
        attribution = build_attribution(
            make_claim(),
            make_hypothesis_claim(),
            records=[make_record(), make_hypothesis_record()],
        )
        assert attribution.classification is None
        builder = GraphBuilder()
        builder.add_attribution(attribution)
        graph = builder.build()
        relation = next(
            e for e in graph.edges if isinstance(e, ClaimRelationEdge)
        )
        assert relation.label is None
        assert relation.role is None
        assert relation.confidence is None and relation.scores is None
        assert relation.uncertain is None

    def test_claim_ids_match_evidence_claim_references(self):
        attribution = _attribution()
        builder = GraphBuilder()
        premise_id, _ = builder.add_attribution(attribution)
        assert premise_id == attribution.premise.claim.claim_id
        assert premise_id == claim_node_id(
            "arxiv", "2101.00001", make_claim().start, make_claim().end
        )

    def test_wrong_type(self):
        with pytest.raises(TypeError):
            GraphBuilder().add_attribution(object())


class TestAddRelation:
    def test_classification_only_uses_unknown_offsets(self):
        builder = GraphBuilder()
        premise_id, hypothesis_id = builder.add_relation(make_classification())
        assert premise_id.endswith(":unknown")
        assert hypothesis_id.endswith(":unknown")
        graph = builder.build()
        assert graph.node(premise_id).sentence == DEFAULT_SENTENCE
        relation = next(
            e for e in graph.edges if isinstance(e, ClaimRelationEdge)
        )
        assert relation.hedged is False and relation.truncated is False

    def test_conditions_embedded_unchanged(self):
        comparison = compare_conditions(
            extract_conditions(make_claim()), extract_conditions(make_hypothesis_claim())
        )
        builder = GraphBuilder()
        builder.add_relation(make_classification(), conditions=comparison)
        relation = next(
            e for e in builder.build().edges if isinstance(e, ClaimRelationEdge)
        )
        assert relation.conditions == comparison

    def test_wrong_types(self):
        with pytest.raises(TypeError):
            GraphBuilder().add_relation(object())
        with pytest.raises(TypeError):
            GraphBuilder().add_relation(make_classification(), conditions=object())


class TestAddClaimConditions:
    def test_condition_nodes_and_global_offsets(self):
        claim = make_claim()
        conditions = extract_conditions(claim)
        builder = GraphBuilder()
        claim_id = builder.add_claim_conditions(claim, conditions)
        graph = builder.build()
        assert graph.stats["claim_condition"] == len(conditions.mentions)
        for node in graph.nodes:
            if node.type == "condition":
                assert node.claim_id == claim_id
                assert node.global_start == claim.start + node.start
                assert node.global_end == claim.start + node.end
                assert (
                    claim.sentence[node.start : node.end] == node.value
                )

    def test_identity_mismatch(self):
        with pytest.raises(ValueError):
            GraphBuilder().add_claim_conditions(
                make_claim(), extract_conditions(make_hypothesis_claim())
            )

    def test_wrong_types(self):
        with pytest.raises(TypeError):
            GraphBuilder().add_claim_conditions(object(), extract_conditions(make_claim()))
        with pytest.raises(TypeError):
            GraphBuilder().add_claim_conditions(make_claim(), object())


class TestDuplicatesAndConflicts:
    def test_identical_readd_is_idempotent(self):
        builder = GraphBuilder()
        builder.add_paper(make_record())
        builder.add_paper(make_record())
        builder.add_claim(make_claim())
        builder.add_claim(make_claim())
        graph = builder.build()
        assert graph.stats["paper"] == 1
        assert graph.stats["claim"] == 1
        # 1 paper re-add + 2 stub-vs-record enrichments (each add_claim
        # re-asserts the stub) + 1 claim re-add + 1 provenance edge re-add.
        assert builder.duplicates_ignored == 5

    def test_attribution_added_twice(self):
        builder = GraphBuilder()
        builder.add_attribution(_attribution())
        builder.add_attribution(_attribution())
        graph = builder.build()
        assert graph.stats["claim"] == 2
        assert graph.stats["claim_relation"] == 1

    def test_paper_enrichment_from_stub(self):
        builder = GraphBuilder()
        builder.add_claim(make_claim())  # creates a metadata-free stub
        builder.add_paper(make_record())  # enriches it
        node = builder.build().node(paper_node_id("arxiv", "2101.00001"))
        assert node.title == "Sparse Attention Studies"
        assert node.authors == ("A. Author",)

    def test_claim_enrichment_fills_cue_and_hedges(self):
        plain = CandidateClaim(
            source="arxiv",
            source_id="2101.00001",
            sentence=DEFAULT_SENTENCE,
            start=make_claim().start,
            end=make_claim().end,
            cue="",
            hedges=(),
        )
        rich = make_claim()
        builder = GraphBuilder()
        builder.add_claim(plain)
        builder.add_claim(rich)
        node = builder.build().node(
            claim_node_id("arxiv", "2101.00001", rich.start, rich.end)
        )
        assert node.cue == "we find"
        assert builder.claims_enriched == 1

    def test_conflicting_claim_sentence_rejected(self):
        builder = GraphBuilder()
        builder.add_claim(make_claim())
        conflicting = make_claim(sentence="A completely different sentence.")
        with pytest.raises(GraphConflictError):
            builder.add_claim(conflicting)

    def test_conflicting_paper_metadata_rejected(self):
        builder = GraphBuilder()
        builder.add_paper(make_record())
        with pytest.raises(GraphConflictError):
            builder.add_paper(make_record(title="A Different Title"))

    def test_conflicting_evidence_rejected(self):
        builder = GraphBuilder()
        attribution = _attribution()
        builder.add_attribution(attribution)
        other = build_attribution(
            make_claim(),
            make_hypothesis_claim(),
            classification=make_classification(),
        )  # same claims but no records -> unverified evidence, same ids
        with pytest.raises(GraphConflictError):
            builder.add_attribution(other)


class TestBuildGraph:
    def test_empty_inputs_produce_valid_empty_graph(self):
        graph = build_graph()
        assert isinstance(graph, KnowledgeGraph)
        assert len(graph) == 0
        assert graph.stats["edges"] == 0

    def test_mixed_inputs_and_order_independence(self):
        forward = build_graph(
            papers=[make_record(), make_hypothesis_record()],
            claims=[make_claim(), make_hypothesis_claim()],
            attributions=[_attribution()],
        )
        backward = build_graph(
            attributions=[_attribution()],
            claims=[make_hypothesis_claim(), make_claim()],
            papers=[make_hypothesis_record(), make_record()],
        )
        assert forward == backward
        assert graph_texts_equal(forward, backward)


def graph_texts_equal(a: KnowledgeGraph, b: KnowledgeGraph) -> bool:
    """Deterministic serialization is order-independent (used above)."""
    from paperdna.graph import graph_to_json

    return graph_to_json(a) == graph_to_json(b)


class TestFixturePipeline:
    """End-to-end: fixture case -> real pipeline -> graph facts."""

    def _build(self, case: dict) -> KnowledgeGraph:
        classification = make_classification(**case["classification"])
        if not case["claims"]:
            attribution = attribute_from_classification(classification, records=[])
            return build_graph(attributions=[attribution])

        abstracts = {
            (r["source"], r["source_id"]): r["abstract"] for r in case["records"]
        }
        records = [make_record(**r) for r in case["records"]]
        claims = []
        for c in case["claims"]:
            abstract = abstracts[(c["source"], c["source_id"])]
            start = abstract.index(c["sentence"])
            claims.append(
                CandidateClaim(
                    source=c["source"],
                    source_id=c["source_id"],
                    sentence=c["sentence"],
                    start=start,
                    end=start + len(c["sentence"]),
                    cue=c["cue"],
                    hedges=tuple(c["hedges"]),
                )
            )
        attribution = build_attribution(
            claims[0],
            claims[1],
            records=records,
            classification=classification,
            conditions=(
                compare_conditions(
                    extract_conditions(claims[0]), extract_conditions(claims[1])
                )
                if case.get("with_conditions")
                else None
            ),
        )
        builder = GraphBuilder()
        builder.add_attribution(attribution)
        if case.get("with_conditions"):
            for claim in claims:
                builder.add_claim_conditions(claim, extract_conditions(claim))
        return builder.build()

    def test_fixture_cases(self, fixture_cases):
        for case in fixture_cases:
            graph = self._build(case)
            expected = case["expected"]
            stats = graph.stats
            for node_type, count in expected.get("node_counts", {}).items():
                assert stats[node_type] == count, case["name"]
            for edge_type, count in expected.get("edge_counts", {}).items():
                assert stats[edge_type] == count, case["name"]
            relation = next(
                e for e in graph.edges if isinstance(e, ClaimRelationEdge)
            )
            assert relation.label == expected["label"], case["name"]
            assert relation.role == expected["role"], case["name"]
            assert relation.uncertain == expected["uncertain"], case["name"]
            if "hedged" in expected:
                assert relation.hedged == expected["hedged"], case["name"]
            if expected.get("claim_ids_with_unknown_offsets"):
                for claim in graph.nodes:
                    if claim.type == "claim":
                        assert claim.id.endswith(":unknown"), case["name"]
            if expected.get("has_condition_comparison"):
                assert relation.conditions is not None, case["name"]
            for claim_id_prefix, count in expected.get(
                "expected_condition_mentions", {}
            ).items():
                matching = [
                    n
                    for n in graph.nodes
                    if n.type == "condition"
                    and n.claim_id.startswith(claim_id_prefix + ":")
                ]
                assert len(matching) == count, case["name"]

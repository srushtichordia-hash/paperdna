"""Tests for deterministic graph serialization and exact round-trip."""

from __future__ import annotations

import json

import pytest

from paperdna.conditions.compare import compare_conditions
from paperdna.conditions.extract import extract_conditions
from paperdna.evidence import attribute_from_classification, build_attribution
from paperdna.graph import (
    GraphBuilder,
    GraphValidationError,
    build_graph,
    graph_from_dict,
    graph_from_json,
    graph_to_dict,
    graph_to_json,
)
from tests.graph.conftest import (
    make_claim,
    make_classification,
    make_hypothesis_claim,
    make_hypothesis_record,
    make_record,
)


def _rich_graph():
    """A graph with every node and edge type represented."""
    builder = GraphBuilder()
    attribution = build_attribution(
        make_claim(),
        make_hypothesis_claim(),
        records=[make_record(), make_hypothesis_record()],
        classification=make_classification(),
    )
    builder.add_attribution(attribution)
    builder.add_claim_conditions(make_claim(), extract_conditions(make_claim()))
    builder.add_claim_conditions(
        make_hypothesis_claim(), extract_conditions(make_hypothesis_claim())
    )
    # A classification-only relation (unknown offsets, unverified evidence).
    builder.add_relation(make_classification(premise_source_id="9999.1"))
    # An evidence-only attribution from a record-less path.
    builder.add_attribution(
        attribute_from_classification(
            make_classification(
                premise_source_id="9999.2",
                hypothesis_source_id="9999.3",
                premise_text="We find nothing here at all.",
                hypothesis_text="We find something else entirely.",
            ),
            records=[],
        )
    )
    return builder.build()


def _conditioned_graph():
    """Relation edge carrying a full ConditionComparison."""
    comparison = compare_conditions(
        extract_conditions(make_claim()), extract_conditions(make_hypothesis_claim())
    )
    builder = GraphBuilder()
    builder.add_relation(make_classification(), conditions=comparison)
    return builder.build()


class TestRoundTrip:
    def test_dict_round_trip_is_exact(self):
        graph = _rich_graph()
        assert graph_from_dict(graph_to_dict(graph)) == graph

    def test_json_round_trip_is_exact(self):
        graph = _rich_graph()
        assert graph_from_json(graph_to_json(graph)) == graph

    def test_dict_output_is_json_safe(self):
        graph = _rich_graph()
        json.dumps(graph_to_dict(graph))  # must not raise

    def test_conditions_survive_round_trip(self):
        graph = _conditioned_graph()
        restored = graph_from_dict(graph_to_dict(graph))
        original_edge = graph.edges[-1]
        restored_edge = restored.edges[-1]
        assert restored_edge.conditions == original_edge.conditions
        assert restored_edge.conditions is not None
        assert restored_edge.conditions.summary == original_edge.conditions.summary

    def test_empty_graph_round_trip(self):
        empty = build_graph()
        assert graph_from_dict(graph_to_dict(empty)) == empty
        assert graph_from_json(graph_to_json(empty)) == empty

    def test_optional_fields_survive(self):
        graph = _rich_graph()
        restored = graph_from_dict(graph_to_dict(graph))
        for node in restored.nodes:
            assert graph.node(node.id) == node
        for edge in restored.edges:
            original = next(e for e in graph.edges if e.id == edge.id)
            assert original == edge
        # Spot-check None-preservation on the unverified relation claims.
        unknown_claims = [n for n in restored.nodes if n.id.endswith(":unknown")]
        assert unknown_claims
        for claim in unknown_claims:
            assert claim.start is None and claim.end is None


class TestDeterminism:
    def test_byte_identical_json_for_equal_graphs(self):
        first = _rich_graph()
        second = _rich_graph()
        assert graph_to_json(first) == graph_to_json(second)

    def test_insertion_order_does_not_change_output(self):
        forward = build_graph(
            papers=[make_record(), make_hypothesis_record()],
            claims=[make_claim(), make_hypothesis_claim()],
            attributions=[
                build_attribution(
                    make_claim(),
                    make_hypothesis_claim(),
                    records=[make_record(), make_hypothesis_record()],
                    classification=make_classification(),
                )
            ],
        )
        backward = build_graph(
            attributions=[
                build_attribution(
                    make_claim(),
                    make_hypothesis_claim(),
                    records=[make_record(), make_hypothesis_record()],
                    classification=make_classification(),
                )
            ],
            claims=[make_hypothesis_claim(), make_claim()],
            papers=[make_hypothesis_record(), make_record()],
        )
        assert graph_to_json(forward) == graph_to_json(backward)

    def test_nodes_and_edges_sorted_canonically(self):
        data = graph_to_dict(_rich_graph())
        node_keys = [(n["type"], n["id"]) for n in data["nodes"]]
        assert node_keys == sorted(
            node_keys, key=lambda k: (["paper", "claim", "evidence", "condition"].index(k[0]), k[1])
        )
        edge_keys = [
            (e["type"], e["source"], e["target"], e["id"]) for e in data["edges"]
        ]
        type_rank = {
            t: i
            for i, t in enumerate(
                ["claim_paper", "claim_evidence", "claim_condition", "claim_relation"]
            )
        }
        assert edge_keys == sorted(edge_keys, key=lambda k: (type_rank[k[0]], k[1], k[2], k[3]))

    def test_json_has_trailing_newline_and_sorted_keys(self):
        text = graph_to_json(_rich_graph())
        assert text.endswith("\n")
        # sort_keys=True: top-level keys appear alphabetically — "edges"
        # (e) sorts before "format", "nodes", and "version".
        lines = text.splitlines()
        assert lines[1].strip() == '"edges": ['


class TestMalformedInput:
    def test_not_a_dict(self):
        with pytest.raises(ValueError):
            graph_from_dict([1, 2, 3])

    def test_wrong_format(self):
        data = graph_to_dict(build_graph())
        data["format"] = "someone.else"
        with pytest.raises(ValueError):
            graph_from_dict(data)

    def test_wrong_version(self):
        data = graph_to_dict(build_graph())
        data["version"] = 99
        with pytest.raises(ValueError):
            graph_from_dict(data)

    def test_missing_envelope_keys(self):
        with pytest.raises(ValueError):
            graph_from_dict({"format": "paperdna.graph", "version": 1, "nodes": []})
        with pytest.raises(ValueError):
            graph_from_dict({})

    def test_nodes_and_edges_must_be_lists(self):
        data = graph_to_dict(build_graph())
        data["nodes"] = {}
        with pytest.raises(ValueError):
            graph_from_dict(data)

    def test_item_missing_keys(self):
        with pytest.raises(ValueError):
            graph_from_dict(
                {
                    "format": "paperdna.graph",
                    "version": 1,
                    "nodes": [{"type": "paper", "attrs": {}}],  # no id
                    "edges": [],
                }
            )

    def test_unknown_node_and_edge_types(self):
        base = {
            "format": "paperdna.graph",
            "version": 1,
            "nodes": [],
            "edges": [],
        }
        node_data = dict(base, nodes=[{"id": "x", "type": "alien", "attrs": {}}])
        with pytest.raises(ValueError):
            graph_from_dict(node_data)
        edge_data = dict(
            base, edges=[{"id": "x", "type": "telepathy", "source": "a", "target": "b", "attrs": {}}]
        )
        with pytest.raises(ValueError):
            graph_from_dict(edge_data)

    def test_declared_id_must_match_payload(self):
        data = graph_to_dict(build_graph(papers=[make_record()]))
        data["nodes"][0]["id"] = "paper:wrong:id"
        with pytest.raises(ValueError):
            graph_from_dict(data)

    def test_dangling_edge_rejected(self):
        data = graph_to_dict(build_graph(papers=[make_record()]))
        data["edges"].append(
            {
                "id": "edge:claim_evidence:a:b:0-1->evidence:a:b:0-1",
                "type": "claim_evidence",
                "source": "a:b:0-1",
                "target": "evidence:a:b:0-1",
                "attrs": {},
            }
        )
        with pytest.raises(GraphValidationError):
            graph_from_dict(data)

    def test_bad_attr_shapes(self):
        base = graph_to_dict(build_graph(papers=[make_record()]))
        for key, value in [
            ("authors", "not-a-list"),
        ]:
            data = json.loads(json.dumps(base))
            data["nodes"][0]["attrs"][key] = value
            with pytest.raises(ValueError):
                graph_from_dict(data)

    def test_relation_attr_shapes(self):
        data = graph_to_dict(_conditioned_graph())
        edge = data["edges"][-1]
        assert edge["type"] == "claim_relation"
        for bad_scores in ("nope", ["a"]):
            broken = json.loads(json.dumps(data))
            broken["edges"][-1]["attrs"]["scores"] = bad_scores
            with pytest.raises(ValueError):
                graph_from_dict(broken)
        broken = json.loads(json.dumps(data))
        broken["edges"][-1]["attrs"]["notes"] = "not-a-list"
        with pytest.raises(ValueError):
            graph_from_dict(broken)

    def test_json_text_must_be_str(self):
        with pytest.raises(ValueError):
            graph_from_json(42)


class TestGraphToDictTypeCheck:
    def test_requires_graph(self):
        with pytest.raises(TypeError):
            graph_to_dict(object())

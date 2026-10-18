"""CLI-level tests for ``python -m paperdna.eval``.

Verifies the runner's contract: valid JSON on stdout or at ``--out``,
byte-identical output across runs (offline determinism), correct split
filtering, and honest flagging of the circular scripted system.
"""

from __future__ import annotations

import json

from paperdna.eval.runner import main


class TestRunnerCli:
    def test_retrieval_dev_split_json_on_stdout(self, capsys):
        assert main(["retrieval", "--split", "dev"]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["ranking_kind"] == "lexical"
        assert payload["n_queries"] == len(payload["per_query"])
        assert all(row["split"] == "dev" for row in payload["per_query"])
        assert "precision@1" in payload["aggregate_mean"]

    def test_retrieval_test_split_excludes_dev(self, capsys):
        main(["retrieval", "--split", "test"])
        payload = json.loads(capsys.readouterr().out)
        assert all(row["split"] == "test" for row in payload["per_query"])

    def test_nli_scripted_is_marked_circular(self, capsys):
        main(["nli", "--classifier", "scripted", "--split", "all"])
        payload = json.loads(capsys.readouterr().out)
        assert payload["circular"] is True
        assert payload["overall"]["accuracy"] == 1.0  # by construction

    def test_nli_overlap_is_not_circular(self, capsys):
        main(["nli", "--classifier", "overlap", "--split", "test"])
        payload = json.loads(capsys.readouterr().out)
        assert payload["circular"] is False
        assert payload["classifier"] == "lexical-overlap-baseline"

    def test_conditions_and_evidence_run(self, capsys):
        main(["conditions"])
        conditions = json.loads(capsys.readouterr().out)
        assert conditions["n_pairs_matching"] == conditions["n_pairs"]

        main(["evidence"])
        evidence = json.loads(capsys.readouterr().out)
        assert evidence["n_violations"] == 0

    def test_all_is_byte_deterministic(self, tmp_path, capsys):
        out1 = tmp_path / "run1.json"
        out2 = tmp_path / "run2.json"
        assert main(["all", "--out", str(out1)]) == 0
        assert main(["all", "--out", str(out2)]) == 0
        assert out1.read_bytes() == out2.read_bytes()

    def test_all_report_has_every_section(self, tmp_path):
        out = tmp_path / "report.json"
        main(["all", "--out", str(out)])
        payload = json.loads(out.read_text(encoding="utf-8"))
        assert set(payload) == {
            "retrieval_lexical",
            "nli_overlap_baseline",
            "nli_scripted_conformance",
            "conditions_correctness",
            "evidence_audit",
        }
        assert payload["evidence_audit"]["n_violations"] == 0

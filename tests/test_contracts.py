"""Cross-module contracts: privacy, retrieval, accounting and fallback explanations."""
from copy import deepcopy
import json
import numpy as np
from conftest import completion
from sallog import llm
from sallog.knowledge import Bank
from sallog.pipeline import make_case, api_usage
from sallog.storage import digest, file_hash, write_json, write_jsonl, read_jsonl
from sallog.explanation import explain


def api_case():
    refs = {key: {"reference_id": f"R000000{i}", "training_label": "normal",
                   "logs": [{"line_ref": "S0001", "content": "heartbeat"}]}
            for i, key in enumerate(("R1", "R2"), 1)}
    return {"dataset": "bgl", "case_id": "q", "target": {"logs": [{"line_ref": "L0001", "content": "failed"}]},
            "references": refs}


def test_target_answers_cannot_change_requests(config):
    row = {"sample_id": "q", "label": 0, "prediction": 0, "probability": .1,
           "logs": [{"event_id": "E1", "content": "operation failed", "anomalous": False,
                     "source_context": {"Host": "node-a", "Label": "normal", "Answer": 0}}]}
    other = deepcopy(row)
    other.update(label=1, prediction=1, probability=.9)
    other["logs"][0].update(anomalous=True)
    other["logs"][0]["source_context"].update(Label="abnormal", Answer=1)
    references = api_case()["references"]
    cases = [make_case(r, config, references) for r in (row, other)]
    assert llm.payload(config["llm"], cases[0], "card", "prompt") == llm.payload(config["llm"], cases[1], "card", "prompt")


def test_reading_top_two_is_global_with_source_index_ties():
    # Equal cosine ties are ordered by training source index, never by class.
    bank = Bank.__new__(Bank)
    bank.sources = np.array([9, 2, 1])
    bank.order = np.argsort(bank.sources, kind="stable")
    bank.references = [{"reference_id": "r9", "training_label": "abnormal"},
                       {"reference_id": "r2", "training_label": "normal"},
                       {"reference_id": "r1", "training_label": "normal"}]
    bank.unit_features = np.array([[1., 0.], [1., 0.], [0., 1.]])
    result = bank.top_two(np.array([[1., 0.]]))[0]
    assert [r["reference_id"] for r in result.values()] == ["r1", "r2"]
    assert all(r["training_label"] == "normal" for r in result.values())


def test_transport_uncertainty_needs_its_own_retry_flag(config, tmp_path, monkeypatch):
    monkeypatch.setenv(config["llm"]["api_key_env"], "unit-test")

    def timeout(*args):
        raise TimeoutError("simulated transport timeout")

    monkeypatch.setattr(llm, "transmit", timeout)
    call = lambda **kw: llm.execute(tmp_path, config["llm"], [api_case()], "card", "prompt", execute_api=True, **kw)
    assert call()["new_calls"] == 1
    assert llm.latest(tmp_path, "review")["q"]["status"] == "uncertain"
    assert call(retry_failed=True)["new_calls"] == 0
    monkeypatch.setattr(llm, "transmit", lambda cfg, body, key: completion(body))
    assert call(retry_uncertain=True)["new_calls"] == 1
    assert api_usage(tmp_path)["review_calls"] == 2
    assert api_usage(tmp_path)["missing_usage_calls"] == 1


def test_invalid_reply_keeps_usage_and_budget_prevents_send(config, tmp_path, monkeypatch):
    monkeypatch.setenv(config["llm"]["api_key_env"], "unit-test")

    def invalid(settings, body, key):
        envelope = json.loads(completion(body))
        envelope["choices"][0]["finish_reason"] = "length"
        return json.dumps(envelope)

    monkeypatch.setattr(llm, "transmit", invalid)
    call = lambda **kw: llm.execute(tmp_path, config["llm"], [api_case()], "card", "prompt", execute_api=True, **kw)
    assert call(token_budget=1)["new_calls"] == 0
    assert call()["new_calls"] == 1
    assert llm.latest(tmp_path, "review")["q"]["status"] == "invalid_output"
    assert api_usage(tmp_path)["total_tokens"] == 15
    assert api_usage(tmp_path)["missing_usage_calls"] == 0


def test_explanation_fallback_abstention_and_detection_immutability(config, tmp_path, monkeypatch):
    # A final CNN anomaly with an incompatible normal review needs a new
    # explanation. An abnormal suggestion rejected by the gate stays normal.
    target_file = config["dataset"]["test"]
    run = tmp_path / "explanation-run"
    run.mkdir()
    bank_dir = tmp_path / "bank"
    bank_dir.mkdir()
    write_json(bank_dir / "manifest.json", {"synthetic": True})
    case = api_case()
    write_jsonl(run / "cases.jsonl", [])
    (run / "knowledge_card.txt").write_text("card", encoding="utf8")
    (run / "explanation_prompt.txt").write_text("prompt", encoding="utf8")
    np.savez_compressed(run / "predictions.npz", features=np.ones((2, 2), dtype=np.float32))
    names = ("cases.jsonl", "knowledge_card.txt", "explanation_prompt.txt", "predictions.npz")
    manifest = {"config": config, "targets": 0, "bank": str(bank_dir), "checkpoint": "unused",
                "test_sha256": file_hash(target_file), "bank_manifest_sha256": file_hash(bank_dir / "manifest.json"),
                "files": {name: file_hash(run / name) for name in names}}
    manifest["signature"] = digest(manifest)
    write_json(run / "manifest.json", manifest)
    decisions = [
        {"sample_id": "test-0", "final_prediction": 1, "review_reply": {"verdict": "normal", "brief_analysis": "normal", "evidence": ["L0001"]}},
        {"sample_id": "test-1", "final_prediction": 0, "review_reply": {"verdict": "abnormal", "brief_analysis": "failed", "evidence": ["L0001"]}},
    ]
    write_jsonl(run / "decisions.jsonl", decisions)
    before = file_hash(run / "decisions.jsonl")

    class FakeBank:
        def __init__(self, *args):
            pass

        def top_two(self, queries):
            return [case["references"] for _ in queries]

    monkeypatch.setattr("sallog.explanation.Bank", FakeBank)
    monkeypatch.setattr("sallog.explanation.finalize", lambda *args: None)
    monkeypatch.setattr(llm, "transmit", lambda settings, body, key: completion(body, verdict="abstain"))
    monkeypatch.setenv(config["llm"]["api_key_env"], "unit-test")
    preview = explain(run)
    assert preview["review_reused"] == 0 and preview["final_anomalies"] == 1
    result = explain(run, execute_api=True)
    rows = list(read_jsonl(run / "explanations.jsonl"))
    assert len(rows) == 1 and rows[0]["sample_id"] == "test-0" and rows[0]["status"] == "abstained"
    assert result["api_usage"]["explanation_calls"] == 1
    assert file_hash(run / "decisions.jsonl") == before
    assert explain(run, execute_api=True)["new_calls"] == 0

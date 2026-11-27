import json
import sqlite3
import pytest
from conftest import completion
from sallog import llm
from sallog.small_model import train, load_detector
from sallog.knowledge import build_banks, Bank
from sallog.pipeline import prepare, review, finalize, load_run
from sallog.explanation import explain
from sallog.storage import read_json, read_jsonl


def test_training_to_review_gate_and_explanation(config, embeddings, tmp_path, monkeypatch):
    checkpoint = train(config, embeddings, tmp_path / "detector")
    _, _, state = load_detector(checkpoint, embeddings)
    assert state["training"]["learning_rate"] == .001
    assert state["training"]["batch_size"] == 4
    bank = build_banks(config, checkpoint, embeddings, tmp_path / "bank")
    meta = read_json(bank / "manifest.json")
    assert meta["fit_only"] is True and meta["spatial_size"] == 6
    assert meta["reading_size"] <= 6
    run = prepare(config, checkpoint, embeddings, bank, tmp_path / "review")
    directory, manifest, cases = load_run(run)
    assert manifest["population"] == 8 and manifest["targets"] == 4 and manifest["effective_neighbors"] == 6
    monkeypatch.delenv(config["llm"]["api_key_env"], raising=False)
    assert review(run)["dry_run"] is True
    assert not (run / "responses.sqlite").exists()
    calls = []

    def fake_transmit(settings, body, key):
        request = json.loads(body["messages"][1]["content"])
        target = request["target"]
        assert "label" not in target and "prediction" not in target and "probability" not in target
        assert all("anomalous" not in line for line in target["logs"])
        calls.append(body)
        return completion(body)

    monkeypatch.setenv(config["llm"]["api_key_env"], "unit-test")
    monkeypatch.setattr(llm, "transmit", fake_transmit)
    assert review(run, execute_api=True)["new_calls"] == 4
    assert review(run, execute_api=True)["new_calls"] == 0
    report = finalize(run)
    assert report["provisional"] is False and report["valid_replies"] == 4
    assert report["final"]["errors"] == report["small"]["errors"] - report["net_gain"]
    explanation = explain(run)
    assert len(calls) == 4
    records = list(read_jsonl(run / "explanations.jsonl"))
    assert explanation["final_anomalies"] == sum(r["final_prediction"] for r in read_jsonl(run / "decisions.jsonl"))
    assert all(r["semantic_support_verified"] is False for r in records)
    supplemented = explain(run, execute_api=True)
    assert supplemented["missing_or_abstained"] == 0
    before = len(calls)
    explain(run, execute_api=True)
    assert len(calls) == before
    assert report["api_usage"]["review_calls"] == 4
    (run / "review_prompt.txt").write_text("changed", encoding="utf8")
    with pytest.raises(ValueError, match="artifact changed"):
        load_run(run)


def test_api_failure_requires_explicit_retry(config, tmp_path, monkeypatch):
    run = tmp_path / "attempts"
    run.mkdir()
    reference = {"reference_id": "R0000001", "training_label": "normal", "logs": [{"line_ref": "S0001", "content": "heartbeat"}]}
    case = {"case_id": "q", "dataset": "bgl", "target": {"logs": [{"line_ref": "L0001", "content": "failed"}]},
            "references": {"R1": reference, "R2": {**reference, "reference_id": "R0000002"}}}
    monkeypatch.setenv(config["llm"]["api_key_env"], "unit-test")
    monkeypatch.setattr(llm, "transmit", lambda *args: "invalid-json")
    call = lambda **kw: llm.execute(run, config["llm"], [case], "card", "prompt", execute_api=True, **kw)
    assert call()["new_calls"] == 1
    assert call()["new_calls"] == 0
    monkeypatch.setattr(llm, "transmit", lambda settings, body, key: completion(body))
    assert call(retry_failed=True)["new_calls"] == 1
    assert llm.latest(run, "review")["q"]["status"] == "ok"
    with sqlite3.connect(run / "responses.sqlite") as con:
        assert con.execute("SELECT COUNT(*) FROM attempts").fetchone()[0] == 2


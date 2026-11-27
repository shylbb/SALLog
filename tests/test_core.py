from decimal import Decimal
import json
import numpy as np
import pytest
import torch
from sallog.dataset import ProcessedDataset, disjoint, public_logs
from sallog.gate import decide
from sallog.small_model import CNNDetector
from sallog.trigger import risk_scores, rank, budget_count, balanced_indices
from sallog.config import validate_config


def case():
    return {"case_id": "q", "target": {"logs": [{"line_ref": "L0001", "content": "failure"}]}}


def reply(verdict="abnormal"):
    return {"case_id": "q", "verdict": verdict, "evidence": ["L0001"], "brief_analysis": "failure"}


def test_gate_strict_boundary_and_no_gold_dependency():
    assert decide(case(), reply(), 0, [.9, .8])["final_prediction"] == 0
    accepted = decide(case(), reply(), 0, [.9, .8001])
    assert accepted["final_prediction"] == 1
    assert Decimal(accepted["margin"]) > Decimal("-.1")
    c = {**case(), "label": 1, "references": {"R1": {"training_label": "normal"}}}
    assert decide(c, reply(), 0, [.9, .8001]) == accepted
    assert decide(case(), {**reply(), "evidence": ["L0999"]}, 0, [.1, .9])["final_prediction"] == 0
    assert decide(case(), reply("abstain"), 0, [.1, .9])["final_prediction"] == 0
    assert accepted["semantic_support_verified"] is False


def test_risk_exact_formula_balancing_and_hash_order():
    labels = np.asarray([0, 0, 1, 1])
    refs = np.asarray([[1., 0], [1., .01], [0., 1], [.01, 1]])
    queries = np.asarray([[1., 0], [0., 1]])
    p = np.asarray([.4, .7], dtype=np.float32)
    scores, q, sims, k = risk_scores(p, queries, refs, labels, k=2)
    np.testing.assert_array_equal(q, [0, 1])
    np.testing.assert_allclose(scores, (1 - 2 * abs(p - .5)) * abs(p - q) ** 2)
    assert k == 2 and sims.shape == (2, 2)
    ids = ["a", "b", "c", "d"]
    chosen = [ids[i] for i in rank(np.ones(4), ids)]
    reverse = ids[::-1]
    assert chosen == [reverse[i] for i in rank(np.ones(4), reverse)]
    assert budget_count(47135, .005) == 236
    indices = balanced_indices(np.array([0] * 8 + [1] * 2), 10)
    assert len(indices) == 4
    assert set(indices) == set(balanced_indices(np.array([0] * 8 + [1] * 2), 10))


def test_cnn_shapes_mask_and_frozen_architecture():
    torch.manual_seed(1)
    model = CNNDetector().eval()
    x = torch.zeros(2, 5, 384)
    x[:, :2] = torch.randn(2, 2, 384)
    with torch.no_grad():
        logits, hidden, weights = model(x)
    assert logits.shape == (2,) and hidden.shape == (2, 512)
    assert torch.count_nonzero(weights[:, 2:]) == 0
    assert sum(p.numel() for p in model.parameters()) == 214149


def test_prepared_reader_and_public_projection(config, tmp_path):
    fit = ProcessedDataset(config["dataset"]["train"], True)
    assert len(fit) == 16
    projected = public_logs(fit[1])
    assert "anomalous" not in projected[0] and "Label" not in projected[0]["source_context"]
    assert set(projected[0]) <= {"line_ref", "event_id", "content", "level", "component", "source_context"}
    with pytest.raises(ValueError, match="overlaps"):
        disjoint(fit, fit)
    bad = tmp_path / "bad.jsonl"
    bad.write_text(json.dumps({"sample_id": "bad", "logs": []}), encoding="utf8")
    with pytest.raises(ValueError):
        ProcessedDataset(bad)
    unlabeled = {"sample_id": "unlabeled", "logs": [{"event_id": "E1", "content": "heartbeat"}]}
    bad.write_text(json.dumps(unlabeled), encoding="utf8")
    assert ProcessedDataset(bad).labels.tolist() == [-1]
    bad.write_text(json.dumps({**unlabeled, "label": None}), encoding="utf8")
    with pytest.raises(ValueError, match="binary sample label"):
        ProcessedDataset(bad)


@pytest.mark.parametrize("section,key,value", [
    ("training", "batch_size", 2.5),
    ("training", "min_delta", -1),
    ("small_model", "kernel_size", 2),
    ("calling", "reference_size", 5),
    ("llm", "interval_seconds", -1),
    ("explanation", "generate_missing", "false"),
])
def test_invalid_execution_configuration(config, section, key, value):
    config[section][key] = value
    with pytest.raises(ValueError):
        validate_config(config)


from pathlib import Path
import json
import numpy as np
import pytest
import yaml
from sallog.config import load_config
from sallog.storage import file_hash, write_json


@pytest.fixture
def config(tmp_path):
    root = Path(__file__).resolve().parents[1]
    cfg = yaml.safe_load((root / "configs/default.yaml").read_text(encoding="utf-8"))
    data = tmp_path / "inputs"
    data.mkdir()
    for split, n in [("train", 16), ("validation", 4), ("test", 8)]:
        rows = []
        for i in range(n):
            label = i % 2
            events = ["E3", "E4"] if label else ["E1", "E2"]
            rows.append({"sample_id": f"{split}-{i}", "label": label,
                         "logs": [{"event_id": event, "content": f"{'disk operation failed' if label else 'heartbeat complete'} counter={i} step={j}",
                                   "level": "ERROR" if label else "INFO", "anomalous": bool(label),
                                   "source_context": {"Host": "node-a", "Label": str(label)}} for j, event in enumerate(events)]})
        path = data / f"{split}.jsonl"
        path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
        cfg["dataset"][split] = str(path)
    templates = data / "templates.csv"
    templates.write_text("EventId,EventTemplate\nE1,heartbeat started\nE2,heartbeat complete\nE3,disk operation failed\nE4,disk failure reported\n", encoding="utf-8")
    cfg["dataset"].update(templates=str(templates), max_len=4)
    card = data / "card.txt"
    card.write_text("dataset=bgl\nversion=synthetic-test\nOnly the visible operation result supports the label.", encoding="utf-8")
    cfg["knowledge"].update(card_path=str(card), min_size=2, max_size=6, coarse_clusters=2, svd_components=3,
                             neighbor_count=3, local_neighbors=2, threads=1)
    cfg["calling"].update(reference_size=6, budget=.5, query_batch_size=2)
    cfg["training"].update(max_epochs=1, batch_size=4, threads=1, device="cpu")
    cfg["llm"].update(interval_seconds=0)
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return load_config(cfg_path)


@pytest.fixture
def embeddings(tmp_path, config):
    folder = tmp_path / "embeddings"
    folder.mkdir()
    array = np.zeros((4, 384), dtype=np.float32)
    array[:, :4] = np.eye(4, dtype=np.float32)
    np.savez_compressed(folder / "templates.npz", event_ids=np.asarray(["E1", "E2", "E3", "E4"]), embeddings=array)
    write_json(folder / "manifest.json", {"dimension": 384, "embedding_sha256": file_hash(folder / "templates.npz"),
               "templates_sha256": file_hash(config["dataset"]["templates"]), "encoder": {"kind": "synthetic-test"}})
    return folder


def completion(request_body, verdict=None):
    body = json.loads(request_body["messages"][1]["content"])
    group = "EXPLAIN" if "final_detection_label" in body else "C2"
    if verdict is None:
        verdict = "abnormal" if "failed" in body["target"]["logs"][0]["content"] else "normal"
        if group == "EXPLAIN" and verdict == "normal":
            verdict = "abstain"
    reply = {"case_id": body["case_id"], "group": group, "verdict": verdict,
             "evidence": [] if verdict == "abstain" else ["L0001"],
             "brief_analysis": "Visible operation outcome supports this explanation.",
             "reference_assessment": {key: f"条件支持；{r['reference_id']}:S0001；matching operation" for key, r in body["references"].items()}}
    return json.dumps({"model": "mock-model", "choices": [{"finish_reason": "stop", "message": {"content": json.dumps(reply)}}],
                       "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}})


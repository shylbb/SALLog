"""Compose the five modules using newly generated, version-bound run artifacts."""
from __future__ import annotations
from pathlib import Path
from collections import Counter
import json
import numpy as np
from .dataset import ProcessedDataset, disjoint, public_logs
from .small_model import load_detector, infer, device_for
from .knowledge import Bank, card_text
from .trigger import risk_scores, rank, budget_count
from .gate import decide
from .metrics import binary_metrics
from . import llm
from .storage import digest, file_hash, read_json, read_jsonl, write_json, write_jsonl, now


def make_case(sample, cfg, references):
    return {"case_id": sample["sample_id"], "dataset": cfg["dataset"]["name"],
            "target": {"label_scope": cfg["dataset"]["label_scope"],
                       "context_complete": sample.get("context_complete", True), "logs": public_logs(sample)},
            "references": references}


def prepare(cfg, checkpoint, embeddings_dir, bank_dir, output):
    checkpoint, embeddings_dir, bank_dir, output = map(lambda p: Path(p).resolve(), (checkpoint, embeddings_dir, bank_dir, output))
    fit = ProcessedDataset(cfg["dataset"]["train"], True)
    validation = ProcessedDataset(cfg["dataset"]["validation"], True)
    test = ProcessedDataset(cfg["dataset"]["test"])
    disjoint(fit, validation, test)
    bank = Bank(bank_dir, checkpoint)
    if (file_hash(fit.path) != bank.manifest["train_sha256"] or file_hash(validation.path) != bank.manifest["validation_sha256"]
            or bank.manifest["dataset"] != cfg["dataset"]["name"]):
        raise ValueError("Prepared run must use the bank's fitting/validation data and dataset name")
    device = device_for(cfg)
    model, embeddings, state = load_detector(checkpoint, embeddings_dir, device)
    probabilities, features = infer(model, test, embeddings, state["max_len"], cfg["training"]["batch_size"], device)
    cc = cfg["calling"]
    scores, support, similarities, effective_k = risk_scores(probabilities, features, bank.spatial_features, bank.spatial_labels,
                                                           cc["neighbors"], cc["exponent"], cc["query_batch_size"])
    count = budget_count(len(test), cc["budget"])
    chosen = rank(scores, test.ids, cc["tie_seed"])[:count]
    reading = bank.top_two(features[chosen])
    cases = [make_case(test[int(index)], cfg, refs) for index, refs in zip(chosen, reading)]
    output.mkdir(parents=True, exist_ok=False)
    mask = np.zeros(len(test), dtype=bool)
    mask[chosen] = True
    # Match the retained four-decimal class-similarity protocol; no class quota in reading Top-2.
    similarities = np.asarray([[float(f"{v:.4f}") for v in row] for row in similarities])
    np.savez_compressed(output / "predictions.npz", sample_ids=np.asarray(test.ids, dtype=str), probabilities=probabilities,
                        predictions=(probabilities >= state["threshold"]).astype(np.int64), labels=test.labels,
                        features=features, risk=scores, q=support, similarities=similarities, triggered=mask)
    write_jsonl(output / "cases.jsonl", cases)
    (output / "knowledge_card.txt").write_text(card_text(cfg), encoding="utf-8")
    for stage, name in (("review", "review_prompt.txt"), ("explanation", "explanation_prompt.txt")):
        (output / name).write_text(llm.prompt_text(stage), encoding="utf-8")
    names = ("predictions.npz", "cases.jsonl", "knowledge_card.txt", "review_prompt.txt", "explanation_prompt.txt")
    manifest = {"schema": 1, "created_at": now(), "config": cfg, "checkpoint": str(checkpoint),
                "embeddings": str(embeddings_dir), "bank": str(bank_dir), "checkpoint_sha256": file_hash(checkpoint),
                "bank_manifest_sha256": file_hash(bank_dir / "manifest.json"), "test_sha256": file_hash(test.path),
                "population": len(test), "targets": count, "effective_neighbors": effective_k,
                "calling_rule": f"(1-2*abs(p-0.5))*abs(p-q_k)**{cc['exponent']}", "classification_threshold": state["threshold"],
                "files": {name: file_hash(output / name) for name in names}}
    manifest["signature"] = digest(manifest)
    write_json(output / "manifest.json", manifest)
    return output


def load_run(directory):
    directory = Path(directory).resolve()
    manifest = read_json(directory / "manifest.json")
    identity = {k: v for k, v in manifest.items() if k != "signature"}
    if manifest["signature"] != digest(identity):
        raise ValueError("Prepared run configuration changed")
    for name, sha in manifest["files"].items():
        if Path(name).name != name or file_hash(directory / name) != sha:
            raise ValueError(f"Prepared run artifact changed: {name}")
    cases = list(read_jsonl(directory / "cases.jsonl"))
    if len(cases) != manifest["targets"] or len({v["case_id"] for v in cases}) != len(cases):
        raise ValueError("Prepared target identity mismatch")
    return directory, manifest, cases


def review(directory, **kwargs):
    directory, manifest, cases = load_run(directory)
    return llm.execute(directory, manifest["config"]["llm"], cases,
                       (directory / "knowledge_card.txt").read_text(encoding="utf-8"),
                       (directory / "review_prompt.txt").read_text(encoding="utf-8"), stage="review", **kwargs)


def finalize(directory):
    directory, manifest, cases = load_run(directory)
    replies = llm.latest(directory, "review")
    cases = {v["case_id"]: v for v in cases}
    with np.load(directory / "predictions.npz", allow_pickle=False) as data:
        ids, small, labels = data["sample_ids"], data["predictions"], data["labels"]
        triggered, similarities = data["triggered"], data["similarities"]
    final = small.copy()
    records = []
    for i, value in enumerate(ids):
        cid = str(value)
        decision, reply = None, None
        if triggered[i]:
            response = replies.get(cid)
            reply = json.loads(response["parsed_response"]) if response and response["status"] == "ok" else None
            decision = decide(cases[cid], reply, int(small[i]), similarities[i].tolist(), manifest["config"]["acceptance"]["threshold"])
            final[i] = decision["final_prediction"]
        records.append({"sample_id": cid, "small_prediction": int(small[i]), "triggered": bool(triggered[i]),
                        "final_prediction": int(final[i]), "gate": decision, "review_reply": reply})
    write_jsonl(directory / "decisions.jsonl", records)
    ok = sum(v["status"] == "ok" for v in replies.values())
    summary = {"population": len(ids), "triggered": manifest["targets"], "valid_replies": ok,
               "provisional": ok != manifest["targets"], "reply_statuses": dict(Counter(v["status"] for v in replies.values()))}
    labeled = labels >= 0
    if labeled.any():
        summary.update(evaluation_n=int(labeled.sum()), small=binary_metrics(labels[labeled], small[labeled]),
                       final=binary_metrics(labels[labeled], final[labeled]),
                       corrected=int(((small != labels) & (final == labels) & labeled).sum()),
                       corrupted=int(((small == labels) & (final != labels) & labeled).sum()))
        summary["net_gain"] = summary["corrected"] - summary["corrupted"]
    summary["api_usage"] = api_usage(directory)
    write_json(directory / "summary.json", summary)
    return summary


def api_usage(directory):
    import sqlite3
    path = Path(directory) / "responses.sqlite"
    if not path.exists():
        return {"calls": 0, "missing_usage_calls": 0}
    with sqlite3.connect(path) as con:
        rows = con.execute("SELECT stage,usage FROM attempts").fetchall()
    total = Counter(calls=len(rows), missing_usage_calls=0)
    for stage, raw in rows:
        total[stage + "_calls"] += 1
        usage = json.loads(raw) if raw else {}
        usage = usage if isinstance(usage, dict) else {}
        if not usage:
            total["missing_usage_calls"] += 1
        for key, value in usage.items():
            if key in ("prompt_tokens", "completion_tokens", "total_tokens") and type(value) is int:
                total[key] += value
    return dict(total)


"""Reuse label-consistent review explanations; optionally fill missing alerts."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from . import llm
from .dataset import ProcessedDataset
from .knowledge import Bank
from .pipeline import load_run, make_case, finalize, api_usage
from .storage import file_hash, read_jsonl, write_jsonl, write_json


def explain(directory, *, execute_api=False, limit=0, token_budget=2000000,
            retry_failed=False, retry_uncertain=False):
    directory, manifest, _ = load_run(directory)
    finalize(directory)
    decisions = list(read_jsonl(directory / "decisions.jsonl"))
    records, missing = [], []
    for index, row in enumerate(decisions):
        if row["final_prediction"] != 1:
            continue
        reply = row["review_reply"]
        if (reply and reply["verdict"] == "abnormal" and reply["brief_analysis"].strip() and reply["evidence"]):
            records.append({"sample_id": row["sample_id"], "status": "available", "source": "review_reuse",
                            "evidence": reply["evidence"], "brief_analysis": reply["brief_analysis"],
                            "reference_assessment": reply["reference_assessment"], "semantic_support_verified": False})
        else:
            missing.append(index)
    cases = []
    if missing:
        test = ProcessedDataset(manifest["config"]["dataset"]["test"])
        if file_hash(test.path) != manifest["test_sha256"]:
            raise ValueError("Explanation targets changed after preparation")
        bank = Bank(manifest["bank"], manifest["checkpoint"])
        if file_hash(Path(manifest["bank"]) / "manifest.json") != manifest["bank_manifest_sha256"]:
            raise ValueError("Explanation bank changed after preparation")
        with np.load(directory / "predictions.npz", allow_pickle=False) as data:
            features = data["features"][missing]
        references = bank.top_two(features)
        cases = [make_case(test[i], manifest["config"], refs) for i, refs in zip(missing, references)]
    result = llm.execute(directory, manifest["config"]["llm"], cases,
                         (directory / "knowledge_card.txt").read_text(encoding="utf-8"),
                         (directory / "explanation_prompt.txt").read_text(encoding="utf-8"), stage="explanation",
                         execute_api=execute_api, limit=limit, token_budget=token_budget,
                         retry_failed=retry_failed, retry_uncertain=retry_uncertain)
    prior = llm.latest(directory, "explanation")
    for index in missing:
        cid = decisions[index]["sample_id"]
        saved = prior.get(cid)
        reply = json.loads(saved["parsed_response"]) if saved and saved["status"] == "ok" else None
        available = reply and reply["verdict"] == "abnormal"
        records.append({"sample_id": cid, "status": "available" if available else "abstained" if reply else "missing",
                        "source": "supplemental_api" if reply else None,
                        "evidence": reply["evidence"] if reply else [],
                        "brief_analysis": reply["brief_analysis"] if reply else "",
                        "reference_assessment": reply["reference_assessment"] if reply else {},
                        "semantic_support_verified": False})
    order = {row["sample_id"]: i for i, row in enumerate(decisions)}
    records.sort(key=lambda row: order[row["sample_id"]])
    write_jsonl(directory / "explanations.jsonl", records)
    summary = {"final_anomalies": len(records), "review_reused": sum(r["source"] == "review_reuse" for r in records),
               "supplemental_available": sum(r["source"] == "supplemental_api" and r["status"] == "available" for r in records),
               "missing_or_abstained": sum(r["status"] != "available" for r in records),
               "api_usage": api_usage(directory), "semantic_accuracy_evaluated": False, **result}
    write_json(directory / "explanation_summary.json", summary)
    return summary


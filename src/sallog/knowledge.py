"""Fit-only spatial/reading banks and exact global Top-2 reading retrieval."""
from __future__ import annotations
from collections import Counter
from importlib.resources import files
from pathlib import Path
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD
from threadpoolctl import threadpool_limits
from ._bank_features import prepare_candidates, FineFeatureEncoder
from ._bank_selection import select_bank
from .dataset import ProcessedDataset, disjoint, public_logs
from .small_model import load_detector, infer, device_for
from .trigger import balanced_indices, normalized
from .storage import file_hash, read_json, read_jsonl, write_json, write_jsonl


def card_text(cfg):
    custom = cfg["knowledge"].get("card_path")
    name = cfg["dataset"]["name"]
    if custom:
        text = Path(custom).read_text(encoding="utf-8")
    else:
        if name not in ("hdfs", "bgl", "thunderbird"):
            raise ValueError("A custom dataset needs knowledge.card_path")
        text = files("sallog").joinpath(f"knowledge_cards/{name}.txt").read_text(encoding="utf-8")
    if not text.strip() or f"dataset={name}" not in text.splitlines():
        raise ValueError("Knowledge card is missing or belongs to another dataset")
    return text


def _candidates(samples, name):
    for i, sample in enumerate(samples):
        contexts = [{**line.get("source_context", {}), **{k: line[k] for k in ("level", "component") if k in line}}
                    for line in sample["logs"]]
        yield {"source_index": i, "sample_id": sample["sample_id"], "dataset": name, "label": sample["label"],
               "event_ids": [v["event_id"] for v in sample["logs"]],
               "raw_lines": [v["content"] for v in sample["logs"]], "raw_contexts": contexts}


def _reading_reference(sample, index, cfg):
    logs = public_logs(sample, "S")
    annotated = None
    if cfg["dataset"]["label_scope"] != "block_session" and all("anomalous" in v for v in sample["logs"]):
        annotated = [logs[i]["line_ref"] for i, v in enumerate(sample["logs"]) if v["anomalous"]]
    return {"reference_id": f"R{index:07d}", "training_label": "abnormal" if sample["label"] else "normal",
            "label_scope": cfg["dataset"]["label_scope"], "context_complete": sample.get("context_complete", True),
            "annotated_anomalous_lines": annotated, "logs": logs}


def build_banks(cfg, checkpoint, embeddings_dir, output):
    output = Path(output)
    train_data = ProcessedDataset(cfg["dataset"]["train"], True)
    validation = ProcessedDataset(cfg["dataset"]["validation"], True)
    disjoint(train_data, validation)
    device = device_for(cfg)
    model, embeddings, state = load_detector(checkpoint, embeddings_dir, device)
    if (file_hash(train_data.path) != state["train_sha256"] or file_hash(validation.path) != state["validation_sha256"]
            or cfg["dataset"]["name"] != state["dataset"]):
        raise ValueError("Bank training/validation sources must match the detector training run")
    output.mkdir(parents=True, exist_ok=False)
    calling, kc = cfg["calling"], cfg["knowledge"]
    spatial = balanced_indices(train_data.labels, int(calling["reference_size"]), int(calling["seed"]))
    _, spatial_features = infer(model, train_data, embeddings, state["max_len"], cfg["training"]["batch_size"], device, spatial)
    np.savez_compressed(output / "spatial.npz", features=spatial_features, labels=train_data.labels[spatial], source_indices=spatial)
    # Only the externally prepared fitting split enters TF-IDF, SVD and selection.
    candidates, preparation = prepare_candidates(_candidates(train_data, cfg["dataset"]["name"]), dataset=cfg["dataset"]["name"])
    if len(candidates) < 2:
        raise ValueError("Reading Top-2 needs at least two distinct training candidates")
    encoder = FineFeatureEncoder(int(kc["max_content_features"]))
    with threadpool_limits(limits=int(kc["threads"])):
        x = encoder.fit_transform(candidates)
        dim = min(int(kc["svd_components"]), x.shape[1] - 1, len(candidates) - 1)
        reduced = (TruncatedSVD(n_components=dim, random_state=kc["seed"], n_iter=5).fit_transform(x).astype(np.float32)
                   if dim >= 1 else np.asarray(x.sum(axis=1), dtype=np.float32))
        count = max(1, min(int(kc["coarse_clusters"]), len(candidates), len(np.unique(reduced, axis=0))))
        clusters = MiniBatchKMeans(n_clusters=count, random_state=kc["seed"], n_init=3, batch_size=2048,
                                  reassignment_ratio=0).fit_predict(reduced)
        labels = np.asarray([c["label"] for c in candidates])
        sources = np.asarray([c["source_index"] for c in candidates])
        selected = select_bank(x, reduced, clusters, labels, sources,
                               {key: kc[key] for key in ("min_size", "max_size", "seed", "cover_similarity", "neighbor_count", "local_neighbors")})
    positions = np.asarray(selected["selected_source_indices"], dtype=np.int64)
    if len(positions) < 2:
        raise ValueError("Reading bank construction selected fewer than two references")
    _, reading_features = infer(model, train_data, embeddings, state["max_len"], cfg["training"]["batch_size"], device, positions)
    np.savez_compressed(output / "reading.npz", features=reading_features, source_indices=positions)
    write_jsonl(output / "references.jsonl", (_reading_reference(train_data[int(i)], int(i), cfg) for i in positions))
    write_json(output / "selection.json", selected["report"])
    profile = {"fit_units": len(train_data), "normal": int((train_data.labels == 0).sum()),
               "abnormal": int((train_data.labels == 1).sum()), "rule_derivation": "Statistics support manual card review; they do not automatically prove individual rules"}
    write_json(output / "training_profile.json", profile)
    manifest = {"schema": 1, "dataset": cfg["dataset"]["name"], "fit_only": True,
                "checkpoint_sha256": file_hash(checkpoint), "embedding_sha256": state["embedding_sha256"],
                "train_sha256": state["train_sha256"], "validation_sha256": state["validation_sha256"],
                "calling": calling, "knowledge": kc, "candidate_preparation": preparation,
                "feature_metadata": encoder.metadata(), "spatial_size": len(spatial), "reading_size": len(positions),
                "reading_class_counts": dict(Counter(map(str, train_data.labels[positions]))),
                "files": {name: file_hash(output / name) for name in ("spatial.npz", "reading.npz", "references.jsonl", "selection.json", "training_profile.json")}}
    write_json(output / "manifest.json", manifest)
    return output


class Bank:
    def __init__(self, folder, checkpoint):
        self.folder = Path(folder)
        self.manifest = read_json(self.folder / "manifest.json")
        if self.manifest.get("fit_only") is not True or self.manifest["checkpoint_sha256"] != file_hash(checkpoint):
            raise ValueError("Bank/checkpoint identity mismatch")
        for name, sha in self.manifest["files"].items():
            if file_hash(self.folder / name) != sha:
                raise ValueError(f"Bank artifact changed: {name}")
        with np.load(self.folder / "spatial.npz", allow_pickle=False) as z:
            self.spatial_features, self.spatial_labels = z["features"], z["labels"]
        with np.load(self.folder / "reading.npz", allow_pickle=False) as z:
            self.reading_features, self.sources = z["features"], z["source_indices"]
        self.references = list(read_jsonl(self.folder / "references.jsonl"))
        if len(self.references) != len(self.sources) or len(set(self.sources.tolist())) != len(self.sources):
            raise ValueError("Reading reference identity mismatch")
        self.order = np.argsort(self.sources, kind="stable")
        self.unit_features = normalized(self.reading_features, np.float64)[self.order]

    def top_two(self, queries):
        queries = normalized(queries, np.float64)
        result = []
        for query in queries:
            scores = np.clip(query @ self.unit_features.T, -1, 1)
            rank = np.argsort(-scores, kind="stable")[:2]
            result.append({f"R{i+1}": self.references[int(self.order[j])] for i, j in enumerate(rank)})
        return result


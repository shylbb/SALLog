"""Frozen template encoding with masked mean pooling and L2 normalization."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
from .dataset import read_templates
from .storage import file_hash, read_json, write_json


def embed_templates(cfg, output):
    from transformers import AutoTokenizer, AutoModel
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    templates = read_templates(cfg["dataset"]["templates"])
    ec = cfg["encoder"]
    kwargs = {"local_files_only": ec.get("local_files_only", False)}
    if ec.get("revision") and not Path(ec["model"]).exists():
        kwargs["revision"] = ec["revision"]
    tokenizer = AutoTokenizer.from_pretrained(ec["model"], **kwargs)
    model = AutoModel.from_pretrained(ec["model"], **kwargs).eval()
    device = "cuda" if cfg["training"]["device"] == "auto" and torch.cuda.is_available() else cfg["training"]["device"]
    device = "cpu" if device == "auto" else device
    torch.set_num_threads(int(cfg["training"]["threads"]))
    model.to(device)
    texts, vectors = list(templates.values()), []
    for start in range(0, len(texts), int(ec["batch_size"])):
        tokens = tokenizer(texts[start:start + int(ec["batch_size"])], padding=True, truncation=True,
                           max_length=int(ec["max_tokens"]), return_tensors="pt")
        tokens = {k: v.to(device) for k, v in tokens.items()}
        with torch.no_grad():
            hidden = model(**tokens).last_hidden_state
            mask = tokens["attention_mask"].unsqueeze(-1).expand_as(hidden).float()
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            vectors.append(torch.nn.functional.normalize(pooled, p=2, dim=1).cpu().numpy())
    array = np.concatenate(vectors).astype(np.float32)
    np.savez_compressed(output / "templates.npz", event_ids=np.asarray(list(templates), dtype=str), embeddings=array)
    write_json(output / "manifest.json", {"encoder": ec, "templates_sha256": file_hash(cfg["dataset"]["templates"]),
               "embedding_sha256": file_hash(output / "templates.npz"), "dimension": int(array.shape[1]),
               "pooling": "attention-mask mean followed by L2", "frozen": True})
    return output


def load_embeddings(directory):
    path = Path(directory) / "templates.npz"
    meta = read_json(Path(directory) / "manifest.json")
    if file_hash(path) != meta["embedding_sha256"]:
        raise ValueError("Template embeddings changed after generation")
    with np.load(path, allow_pickle=False) as z:
        keys, values = z["event_ids"], z["embeddings"].astype(np.float32)
    if (keys.ndim != 1 or len(set(map(str, keys))) != len(keys) or not len(keys)
            or values.ndim != 2 or len(keys) != len(values) or values.shape[1] != meta["dimension"]
            or not np.isfinite(values).all() or np.any(np.linalg.norm(values, axis=1) <= 1e-12)):
        raise ValueError("Invalid template embeddings")
    return {str(k): values[i] for i, k in enumerate(keys)}, meta


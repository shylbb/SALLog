"""CNN detector, supervised training and batched hidden representation export."""
from __future__ import annotations
from pathlib import Path
import random
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from .dataset import ProcessedDataset, disjoint
from .encoder import load_embeddings
from .metrics import binary_metrics, best_threshold
from .storage import file_hash, now, write_json


class CNNDetector(nn.Module):
    def __init__(self, input_dim=384, channels=128, kernel_size=3, attention_heads=4, dropout=.2):
        super().__init__()
        if kernel_size < 1 or kernel_size % 2 == 0 or attention_heads < 1:
            raise ValueError("An odd positive convolution kernel and positive head count are required")
        self.conv = nn.Conv1d(input_dim, channels, kernel_size, padding=kernel_size // 2)
        self.norm = nn.LayerNorm(channels)
        self.activation = nn.GELU()
        self.attn = nn.Linear(channels, attention_heads)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(nn.Linear(channels * attention_heads, channels), nn.GELU(),
                                        nn.Dropout(dropout), nn.Linear(channels, 1))

    def forward(self, inputs):
        mask = inputs.abs().sum(-1).ne(0)
        if not mask.any(1).all():
            raise ValueError("A sequence must contain at least one non-padding event")
        hidden = self.activation(self.norm(self.conv(inputs.transpose(1, 2)).transpose(1, 2)))
        weights = torch.softmax(self.attn(hidden).masked_fill(~mask.unsqueeze(-1), -1e9), dim=1)
        pooled = torch.einsum("bth,btc->bhc", weights, hidden).flatten(1)
        return self.classifier(self.dropout(pooled)).squeeze(-1), pooled, weights


class TensorSamples(Dataset):
    def __init__(self, samples, embeddings, max_len, indices=None):
        self.samples, self.embeddings, self.max_len = samples, embeddings, int(max_len)
        self.indices = np.arange(len(samples)) if indices is None else np.asarray(indices)
        self.dim = len(next(iter(embeddings.values())))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        source = int(self.indices[index])
        sample = self.samples[source]
        values = np.zeros((self.max_len, self.dim), dtype=np.float32)
        for i, line in enumerate(sample["logs"][:self.max_len]):
            if line["event_id"] not in self.embeddings:
                raise ValueError(f"Missing template vector: {line['event_id']}")
            values[i] = self.embeddings[line["event_id"]]
        return torch.from_numpy(values), torch.tensor(sample.get("label", -1), dtype=torch.float32)


def device_for(config):
    requested = config["training"]["device"]
    return torch.device("cuda" if requested == "auto" and torch.cuda.is_available() else "cpu" if requested == "auto" else requested)


def load_detector(checkpoint, embeddings_dir, device="cpu"):
    # Public checkpoints use only tensors and primitive metadata.
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    embeddings, meta = load_embeddings(embeddings_dir)
    if state["embedding_sha256"] != meta["embedding_sha256"]:
        raise ValueError("Checkpoint and template embeddings do not match")
    model = CNNDetector(**state["model_args"])
    model.load_state_dict(state["model_state"], strict=True)
    return model.to(device).eval(), embeddings, state


@torch.no_grad()
def infer(model, samples, embeddings, max_len, batch_size=256, device="cpu", indices=None):
    data = TensorSamples(samples, embeddings, max_len, indices)
    model.eval()
    probabilities, features = [], []
    for x, _ in DataLoader(data, batch_size=int(batch_size), shuffle=False):
        logits, hidden, _ = model(x.to(device))
        probabilities.append(torch.sigmoid(logits).cpu().numpy())
        features.append(hidden.cpu().numpy())
    if not probabilities:
        raise ValueError("Cannot predict an empty set")
    return np.concatenate(probabilities).astype(np.float32), np.concatenate(features).astype(np.float32)


def train(cfg, embeddings_dir, output):
    output = Path(output)
    train_data = ProcessedDataset(cfg["dataset"]["train"], True)
    val_data = ProcessedDataset(cfg["dataset"]["validation"], True)
    disjoint(train_data, val_data)
    embeddings, embedding_meta = load_embeddings(embeddings_dir)
    args = cfg["training"]
    torch.set_num_threads(int(args["threads"]))
    torch.manual_seed(int(args["seed"]))
    np.random.seed(int(args["seed"]))
    random.seed(int(args["seed"]))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args["seed"]))
    model_args = {"input_dim": embedding_meta["dimension"], **cfg["small_model"]}
    device = device_for(cfg)
    model = CNNDetector(**model_args).to(device)
    positives = int((train_data.labels == 1).sum())
    negatives = len(train_data) - positives
    if positives == 0 or negatives == 0:
        raise ValueError("Supervised fitting needs both normal and abnormal training examples")
    pos_weight = max(1., negatives / positives)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args["learning_rate"]), weight_decay=float(args["weight_decay"]))
    max_len = int(cfg["dataset"]["max_len"])
    loader = DataLoader(TensorSamples(train_data, embeddings, max_len), batch_size=int(args["batch_size"]), shuffle=True)
    validation = DataLoader(TensorSamples(val_data, embeddings, max_len), batch_size=int(args["batch_size"]), shuffle=False)
    output.mkdir(parents=True, exist_ok=False)
    identity = {"dataset": cfg["dataset"]["name"], "train_sha256": file_hash(train_data.path),
                "validation_sha256": file_hash(val_data.path), "embedding_sha256": embedding_meta["embedding_sha256"]}
    manifest = {"status": "training", "started_at": now(), "config": cfg, **identity}
    write_json(output / "manifest.json", manifest)
    best_loss, stale, history = float("inf"), 0, []
    for epoch in range(1, int(args["max_epochs"]) + 1):
        model.train()
        loss_sum = 0.
        for x, y in loader:
            optimizer.zero_grad(set_to_none=True)
            logits, _, _ = model(x.to(device))
            loss = criterion(logits, y.to(device))
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.item()) * len(y)
        model.eval()
        val_loss, probs = 0., []
        with torch.no_grad():
            for x, y in validation:
                logits, _, _ = model(x.to(device))
                val_loss += float(criterion(logits, y.to(device)).item()) * len(y)
                probs.append(torch.sigmoid(logits).cpu().numpy())
        val_loss /= len(val_data)
        probabilities = np.concatenate(probs)
        threshold = best_threshold(probabilities, val_data.labels)
        row = {"epoch": epoch, "train_loss": loss_sum / len(train_data), "validation_loss": val_loss,
               "validation_threshold": threshold,
               "validation_metrics": binary_metrics(val_data.labels, (probabilities >= threshold).astype(int))}
        history.append(row)
        if val_loss < best_loss - float(args["min_delta"]):
            best_loss, stale = val_loss, 0
            state = {"model_state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                     "model_args": model_args, "max_len": max_len, "threshold": threshold,
                     "best_epoch": epoch, "validation_loss": val_loss, "training": dict(args),
                     "pos_weight": pos_weight, **identity}
            temp = output / "model.pt.tmp"
            torch.save(state, temp)
            temp.replace(output / "model.pt")
        else:
            stale += 1
        write_json(output / "training_history.json", history)
        print(f"epoch={epoch} train_loss={row['train_loss']:.6f} validation_loss={val_loss:.6f}", flush=True)
        if int(args["patience"]) > 0 and stale >= int(args["patience"]):
            break
    manifest.update(status="complete", completed_at=now(), epochs_run=len(history), checkpoint_sha256=file_hash(output / "model.pt"))
    write_json(output / "manifest.json", manifest)
    return output / "model.pt"


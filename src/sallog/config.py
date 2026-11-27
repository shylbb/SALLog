"""Configuration paths are portable and relative to the YAML file."""
from __future__ import annotations
from copy import deepcopy
from pathlib import Path
import math
import yaml


def load_config(path):
    path = Path(path).resolve()
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise ValueError("Configuration must be a mapping")
    for key in ("dataset", "encoder", "small_model", "training", "calling", "knowledge", "acceptance", "llm", "explanation"):
        if not isinstance(cfg.get(key), dict):
            raise ValueError(f"Missing configuration section: {key}")
    cfg = deepcopy(cfg)
    for key in ("train", "validation", "test", "templates"):
        value = Path(cfg["dataset"][key]).expanduser()
        cfg["dataset"][key] = str((path.parent / value).resolve() if not value.is_absolute() else value.resolve())
    card = cfg["knowledge"].get("card_path")
    if card:
        card = Path(card).expanduser()
        cfg["knowledge"]["card_path"] = str((path.parent / card).resolve() if not card.is_absolute() else card.resolve())
    model = cfg["encoder"]["model"]
    candidate = Path(model).expanduser()
    if candidate.is_absolute() or model.startswith((".", "~")):
        cfg["encoder"]["model"] = str((path.parent / candidate).resolve() if not candidate.is_absolute() else candidate.resolve())
    validate_config(cfg)
    return cfg


def validate_config(cfg):
    def integer(section, keys, minimum=1):
        for key in keys:
            value = cfg[section][key]
            if type(value) is not int or value < minimum:
                raise ValueError(f"{section}.{key} must be an integer >= {minimum}")

    def positive(section, keys):
        for key in keys:
            value = cfg[section][key]
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{section}.{key} must be positive and finite")
    positive("dataset", ["max_len"])
    positive("encoder", ["max_tokens", "batch_size"])
    positive("training", ["batch_size", "learning_rate", "max_epochs", "threads"])
    positive("calling", ["neighbors", "exponent", "reference_size", "query_batch_size"])
    positive("knowledge", ["max_size", "coarse_clusters", "svd_components", "threads"])
    positive("llm", ["max_tokens", "timeout_seconds", "max_input_bytes", "max_response_bytes"])
    integer("dataset", ["max_len"])
    integer("encoder", ["max_tokens", "batch_size"])
    integer("small_model", ["channels", "kernel_size", "attention_heads"])
    integer("training", ["batch_size", "max_epochs", "threads"])
    integer("training", ["patience", "seed"], 0)
    integer("calling", ["neighbors", "reference_size", "query_batch_size"])
    integer("calling", ["seed", "tie_seed"], 0)
    integer("knowledge", ["max_size", "coarse_clusters", "svd_components", "max_content_features", "neighbor_count", "local_neighbors", "threads"])
    integer("knowledge", ["min_size", "seed"], 0)
    integer("llm", ["max_tokens", "max_input_bytes", "max_response_bytes"])
    for section, key in (("training", "min_delta"), ("training", "weight_decay"), ("llm", "interval_seconds")):
        value = cfg[section][key]
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{section}.{key} must be nonnegative and finite")
    if cfg["small_model"]["kernel_size"] % 2 == 0:
        raise ValueError("small_model.kernel_size must be odd")
    dropout = cfg["small_model"]["dropout"]
    if type(dropout) not in (int, float) or not 0 <= dropout < 1:
        raise ValueError("small_model.dropout must be in [0, 1)")
    for section, key in (("encoder", "local_files_only"), ("explanation", "generate_missing")):
        if type(cfg[section][key]) is not bool:
            raise ValueError(f"{section}.{key} must be boolean")
    if not 0 < cfg["calling"]["budget"] <= 1:
        raise ValueError("calling.budget must be in (0, 1]")
    if cfg["calling"]["reference_size"] < 2 or cfg["calling"]["reference_size"] % 2:
        raise ValueError("calling.reference_size must be an even integer >= 2")
    if cfg["dataset"]["label_scope"] not in ("fixed_window", "block_session"):
        raise ValueError("label_scope must be fixed_window or block_session")
    if cfg["knowledge"]["min_size"] < 0 or cfg["knowledge"]["min_size"] > cfg["knowledge"]["max_size"]:
        raise ValueError("Invalid reading-bank capacity bounds")
    if not 0 < cfg["knowledge"]["cover_similarity"] <= 1:
        raise ValueError("Invalid coverage similarity")
    if not math.isfinite(float(cfg["acceptance"]["threshold"])):
        raise ValueError("Invalid acceptance threshold")
    extra = cfg["llm"].get("extra_body", {})
    if not isinstance(extra, dict) or set(extra) & {"messages", "model", "stream", "max_tokens", "response_format", "tools", "api_key"}:
        raise ValueError("llm.extra_body may only contain provider-specific generation settings")
    if any(key in cfg["llm"] for key in ("api_key", "token", "authorization")):
        raise ValueError("Use llm.api_key_env instead of storing a credential in YAML")


"""Read already prepared samples; no parsing, windowing or split generation."""
from __future__ import annotations
import csv
import json
from pathlib import Path
import numpy as np

SAFE_CONTEXT = {"level", "component", "type", "node", "noderepeat", "user", "location", "pid", "processid", "threadid", "host"}


def validate_sample(row, require_label=False):
    if not isinstance(row, dict) or not isinstance(row.get("sample_id"), str) or not row["sample_id"].strip():
        raise ValueError("Each sample needs a nonempty string sample_id")
    logs = row.get("logs")
    if not isinstance(logs, list) or not logs:
        raise ValueError(f"{row['sample_id']}: full ordered logs are required")
    for line in logs:
        if (not isinstance(line, dict) or not isinstance(line.get("event_id"), str) or not line["event_id"]
                or not isinstance(line.get("content"), str) or not line["content"].strip()):
            raise ValueError(f"{row['sample_id']}: each log needs event_id and Content without a label prefix")
        if "anomalous" in line and type(line["anomalous"]) is not bool:
            raise ValueError("Optional training point annotation must be boolean")
        if "source_context" in line and not isinstance(line["source_context"], dict):
            raise ValueError("source_context must be a mapping")
    if "events" in row and row["events"] != [line["event_id"] for line in logs]:
        raise ValueError("events and full logs must have identical order and length")
    label = row.get("label")
    if (require_label and "label" not in row) or ("label" in row and (type(label) is not int or label not in (0, 1))):
        raise ValueError(f"{row['sample_id']}: binary sample label required")
    if "context_complete" in row and type(row["context_complete"]) is not bool:
        raise ValueError("context_complete must be boolean")
    return row


class ProcessedDataset:
    """Index JSONL offsets and labels, then decode individual samples on demand."""
    def __init__(self, path, require_labels=False):
        self.path = Path(path).resolve()
        self.offsets, self.ids, labels = [], [], []
        seen = set()
        with self.path.open("rb") as stream:
            while True:
                offset = stream.tell()
                line = stream.readline()
                if not line:
                    break
                if not line.strip():
                    continue
                row = validate_sample(json.loads(line), require_labels)
                if row["sample_id"] in seen:
                    raise ValueError(f"Duplicate sample_id: {row['sample_id']}")
                seen.add(row["sample_id"])
                self.offsets.append(offset)
                self.ids.append(row["sample_id"])
                labels.append(row.get("label", -1))
        if not self.ids:
            raise ValueError(f"Empty dataset: {self.path}")
        self.labels = np.asarray(labels, dtype=np.int64)
        self._stream = None

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        if not 0 <= int(index) < len(self):
            raise IndexError(index)
        if self._stream is None:
            self._stream = self.path.open("rb")
        self._stream.seek(self.offsets[int(index)])
        return json.loads(self._stream.readline())

    def __iter__(self):
        with self.path.open("rb") as stream:
            for line in stream:
                if line.strip():
                    yield json.loads(line)

    def close(self):
        if getattr(self, "_stream", None) is not None:
            self._stream.close()
            self._stream = None

    def __getstate__(self):
        return {**self.__dict__, "_stream": None}

    def __del__(self):
        self.close()


def disjoint(*datasets):
    seen = set()
    for data in datasets:
        shared = seen.intersection(data.ids)
        if shared:
            raise ValueError(f"Sample identity overlaps across supplied splits: {next(iter(shared))}")
        seen.update(data.ids)


def read_templates(path):
    result = {}
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not {"EventId", "EventTemplate"}.issubset(reader.fieldnames or []):
            raise ValueError("Template CSV needs EventId and EventTemplate")
        for row in reader:
            key, text = row["EventId"], row["EventTemplate"]
            if not key or not text or key in result:
                raise ValueError("Template IDs must be unique and texts nonempty")
            result[key] = text
    if not result:
        raise ValueError("Template CSV is empty")
    return result


def public_logs(sample, prefix="L"):
    """Whitelist facts; target truth, point labels and arbitrary metadata stay local."""
    result = []
    for i, line in enumerate(sample["logs"], 1):
        context = {str(k): str(v) for k, v in line.get("source_context", {}).items()
                   if str(k).lower() in SAFE_CONTEXT and v is not None}
        item = {"line_ref": f"{prefix}{i:04d}", "event_id": line["event_id"], "content": line["content"]}
        for key in ("level", "component"):
            if key in line:
                item[key] = str(line[key])
        if context:
            item["source_context"] = context
        # Timestamps are deliberately not required; ordering is always preserved.
        result.append(item)
    return result


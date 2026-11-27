"""Auditable fit-training-only text/sequence features, never CNN query features."""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import re
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

NORMALIZATION_VERSION = "content-context-objects-v3.0"
_STAMP = re.compile(r"\b\d{4}[-/]\d{2}[-/]\d{2}(?:[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?)?\b|\b\d{2}:\d{2}:\d{2}(?:\.\d+)?\b")
_OBJECT = re.compile(r"(?P<block>\bblk_-?\d+\b)|(?P<uuid>\b[0-9a-fA-F]{8}-(?:[0-9a-fA-F]{4}-){3}[0-9a-fA-F]{12}\b)|(?P<ip>\b(?:\d{1,3}\.){3}\d{1,3}(?::\d+)?\b)|(?P<key>\b(?:pid|tid|process_id|thread_id|request_id|session_id|object_id|transaction_id|job_id|block_id)\s*[=:]\s*)(?P<value>[\w.-]+)", re.I)
_BRACKET_PID = re.compile(r"\b([A-Za-z][\w.-]*(?:\([\w.-]+\))?)\[(\d+)\]")
_MESSAGE_ID = re.compile(r"\b(msgid|message_id)\s*=\s*(<[^>]*>|[^\s,;]+)", re.I)
_MAIL_QUEUE = re.compile(r"\b((?:sendmail|postfix/[A-Za-z_-]+)(?:\[[^\]]+\])?:\s+)([A-Za-z0-9]{8,}):")
_TOKEN = re.compile(r"[\w.:-]+\s*=\s*[^\s,;()\[\]{}]+|[+-]?\d+(?:\.\d+)?|[\w]+(?:[.:-][\w]+)*", re.UNICODE)
_PARAM = re.compile(r"\b[\w.:-]+\s*=\s*[^\s,;()\[\]{}]+")
_NULL_IDS = {"none", "null", "unknown", "absent", "<>", "<none>", "<null>", "<unknown>", "<absent>"}
_CONTEXT_FIELDS = {"level", "component", "type", "node", "noderepeat", "user", "location", "pid", "processid", "threadid", "host"}


def _digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode()).hexdigest()


def normalize_window(raw_lines, event_ids, *, raw_has_label_prefix=False, dataset=None, raw_contexts=None):
    """Alias explicitly recognized object IDs by first occurrence inside a window.

    Only callers explicitly declaring a raw answer prefix request stripping it.
    Only whitelisted semantic and source-object context fields are projected.
    Labels, time fields and arbitrary caller metadata are always excluded.
    """
    aliases = {}
    counts = Counter()
    normalized, occurrences = [], []
    if raw_contexts is not None and len(raw_contexts) != len(raw_lines):
        raise ValueError("raw_contexts must align with complete raw lines")
    known_hosts = set()
    for position, raw in enumerate(raw_lines):
        line = str(raw)
        if raw_has_label_prefix:
            if dataset not in ("bgl", "thunderbird"):
                raise ValueError("Answer-prefix stripping requires BGL/Thunderbird provenance")
            line = re.sub(r"^\s*(?:-|[A-Z][A-Z0-9_]*)\s+", "", line, count=1)
        line = _STAMP.sub(" time_absolute ", line)
        objects = []
        context = {} if raw_contexts is None else {
            str(k).lower(): str(v).strip() for k, v in raw_contexts[position].items()
            if str(k).lower() in _CONTEXT_FIELDS and v is not None and str(v).strip()}
        source_scope = "unspecified_source_host"

        def alias(kind, value):
            if kind in ("pid", "process_id", "processid"):
                kind, value = "pid", (source_scope, value)
            elif kind in ("tid", "thread_id", "threadid"):
                kind, value = "tid", (source_scope, value)
            elif kind == "host":
                value = value.lower()
            key = (kind.lower(), value)
            if key not in aliases:
                aliases[key] = f"obj_{kind.lower()}_{counts[kind.lower()]}"
                counts[kind.lower()] += 1
            item = aliases[key]
            objects.append(item)
            return item

        def replace(match):
            if match.group("key"):
                key = re.sub(r"\s+", "", match.group("key")).rstrip("=:").lower()
                if match.group("value").lower() in _NULL_IDS:
                    return key + "=" + match.group("value")
                return key + "=" + alias(key, match.group("value"))
            kind = next(k for k in ("block", "uuid", "ip") if match.group(k))
            if kind == "ip" and match.group(kind).lower() in known_hosts:
                return alias("host", match.group(kind))
            return alias(kind, match.group(kind))

        def fragment(value):
            value = _STAMP.sub(" time_absolute ", value)
            value = _MESSAGE_ID.sub(lambda m: m.group(1).lower() + "=" + (
                m.group(2) if m.group(2).lower() in _NULL_IDS else alias("message", m.group(2))), value)
            value = _MAIL_QUEUE.sub(lambda m: m.group(1) + alias("mailqueue", m.group(2)) + ":", value)
            value = _OBJECT.sub(replace, value)
            return _BRACKET_PID.sub(lambda m: m.group(1) + "[" + alias("pid", m.group(2)) + "]", value)

        # Process source identities first so context PID, bracket PID and
        # explicit body pid= share a host-scoped alias. NodeRepeat shares Node.
        context_tokens, source_hosts = [], []
        for field in ("node", "noderepeat", "host", "user"):
            if field not in context:
                continue
            if field == "user" and dataset != "thunderbird":
                context_tokens.append("context_user=" + alias("actor", context[field]))
                continue
            known_hosts.add(context[field].lower())
            host = alias("host", context[field])
            source_hosts.append(host)
            context_tokens.append("context_" + field + "=" + host)
        if "location" in context:
            location = context["location"]
            if "@" in location:
                role, host_value = location.rsplit("@", 1)
                known_hosts.add(host_value.lower())
                host = alias("host", host_value)
                source_hosts.append(host)
                role = role.lower() if role.lower() in ("src", "local", "remote", "kernel", "daemon") else alias("actor", role)
                context_tokens.append("context_location=" + role + "@" + host)
            else:
                known_hosts.add(location.lower())
                host = alias("host", location)
                source_hosts.append(host)
                context_tokens.append("context_location=" + host)
        if source_hosts:
            source_scope = tuple(dict.fromkeys(source_hosts))
        for field, kind in (("pid", "pid"), ("processid", "pid"), ("threadid", "tid")):
            if field in context:
                value = context[field]
                context_tokens.append("context_" + field + "=" + (value if value.lower() in _NULL_IDS else alias(kind, value)))
        for field in ("level", "component", "type"):
            if field in context:
                context_tokens.append("context_" + field + "=" + fragment(context[field]))
        line = fragment(line)
        normalized.append(" ".join((" ".join(context_tokens) + " " + line).split()))
        occurrences.append(list(dict.fromkeys(objects)))
    relations, previous = [], {}
    for event, objects in zip(event_ids, occurrences):
        for obj in objects:
            relations.append("object_event:" + obj + ":" + event)
            if obj in previous:
                relations.append("same_object:" + obj + ":" + previous[obj] + ">" + event)
            previous[obj] = event
    for i in range(1, len(occurrences)):
        if occurrences[i - 1] and occurrences[i]:
            relation = "same" if set(occurrences[i - 1]) & set(occurrences[i]) else "different"
            relations.append("adjacent_objects:" + relation + ":" + event_ids[i - 1] + ">" + event_ids[i])
    # Intern only inside the window: common object/event relation strings are
    # reused without retaining a process-global pool of unique long messages.
    relation_pool = {}
    relations = [relation_pool.setdefault(item, item) for item in relations]
    parameters = [_PARAM.findall(line) for line in normalized]
    return normalized, occurrences, relations, parameters


def content_features(lines):
    for line in lines:
        tokens = [re.sub(r"\s*=\s*", "=", t).lower() for t in _TOKEN.findall(line)]
        for token in tokens:
            yield "word:" + token
        for left, right in zip(tokens, tokens[1:]):
            yield "bigram:" + left + "\x1f" + right


def event_features(sequence):
    return iter(sequence)


def relation_features(candidate):
    sequence = candidate["event_ids"]
    for left, right in zip(sequence, sequence[1:]):
        yield "event_pair:" + left + "\x1f" + right
    yield from candidate["object_relations"]


def prepare_candidates(records, *, dataset=None):
    """Stream raw candidates; merge only complete canonical matches of one label.

    Raw contents are discarded after canonicalization. Raw source positions remain
    losslessly recoverable in the indexed external fitting JSONL.
    """
    candidates, groups, contents = [], {}, defaultdict(list)
    seen_sources, count = set(), 0
    for record in records:
        count += 1
        source = int(record["source_index"])
        if source < 0 or source in seen_sources:
            raise ValueError("Candidate source indices must be unique and nonnegative")
        seen_sources.add(source)
        label = record["label"]
        if label not in (0, 1):
            raise ValueError("Candidates require a scalar binary original label")
        ds = record.get("dataset", dataset)
        if dataset is not None and ds != dataset:
            raise ValueError("Never combine datasets in one bank")
        seq = list(map(str, record["event_ids"]))
        raw = record.get("raw_lines", record.get("content_lines", record.get("contents")))
        if raw is None or not seq or len(raw) != len(seq) or any(not str(x).strip() for x in raw):
            raise ValueError(f"Incomplete raw window/session at source {source}")
        normalized, occurrences, relations, parameters = normalize_window(
            raw, seq, raw_has_label_prefix=record.get("raw_has_label_prefix", False), dataset=ds,
            raw_contexts=record.get("raw_contexts"))
        content_key = _digest({"normalized_lines": normalized, "event_ids": seq,
                               "object_occurrences": occurrences,
                               "object_relations": relations, "parameters": parameters})
        key = (ds, int(label), content_key)
        if key in groups:
            c = candidates[groups[key]]
            # A cryptographic digest is only an index; canonical equality is checked.
            if (c["normalized_lines"] != normalized or c["event_ids"] != seq
                    or c["object_occurrences"] != occurrences or c["object_relations"] != relations):
                raise ValueError("Canonical digest collision")
            c["sources"].append(source)
            c["occurrence_count"] += 1
            if source < c["source_index"]:
                c.update({k: v for k, v in record.items() if k not in ("raw_lines", "content_lines", "contents", "raw_contexts")})
                c["source_index"] = source
            continue
        c = {k: v for k, v in record.items() if k not in ("raw_lines", "content_lines", "contents", "raw_contexts")}
        c.update(dataset=ds, source_index=source, label=int(label), event_ids=seq,
                 normalized_lines=normalized, object_occurrences=occurrences,
                 object_relations=relations, retained_parameters=parameters,
                 content_key=content_key, canonical_key=_digest([ds, int(label), content_key]),
                 sources=[source], occurrence_count=1)
        groups[key] = len(candidates)
        contents[(ds, content_key)].append(len(candidates))
        candidates.append(c)
    conflicts = []
    for (ds, key), rows in contents.items():
        if len({candidates[i]["label"] for i in rows}) > 1:
            group = "conflict_" + key
            conflicts.append({"conflict_group": group, "dataset": ds, "content_key": key,
                              "members": [{"label": candidates[i]["label"],
                                           "source_index": candidates[i]["source_index"],
                                           "sources": sorted(candidates[i]["sources"]),
                                           "canonical_key": candidates[i]["canonical_key"]} for i in rows]})
            for i in rows:
                candidates[i]["conflict_group"] = group
    candidates.sort(key=lambda c: c["source_index"])
    for c in candidates:
        c["sources"].sort()
    return candidates, {"candidates_before_dedup": count, "candidates_after_dedup": len(candidates),
                        "duplicate_group_count": sum(c["occurrence_count"] > 1 for c in candidates),
                        "duplicate_occurrences_removed": count - len(candidates),
                        "conflict_group_count": len(conflicts), "conflict_groups": conflicts,
                        "deduplication": "full normalized ordered content + complete EventIds + retained parameters + object relations; same original label only",
                        "raw_contents": "lossless Content + safe raw contexts remain in indexed external fitting JSONL",
                        "normalization_version": NORMALIZATION_VERSION}


class FineFeatureEncoder:
    """Three sublinear-TF IDF blocks fit only on supplied fit_train candidates.

    Labels, source IDs, duplication frequency and model hidden features are never
    projected into documents. Empty blocks contribute zeros; the concatenation
    is L2 normalized so a row's nonempty blocks receive equal weight.
    """
    def __init__(self, max_content_features=200000):
        if max_content_features is not None and int(max_content_features) < 1:
            raise ValueError("max_content_features must be positive or None")
        self.max_content_features = max_content_features
        self.vectorizers = {}
        self.block_metadata = {}

    @staticmethod
    def _documents(candidates, name):
        for c in candidates:
            yield c["event_ids"] if name == "events" else c["normalized_lines"] if name == "content" else c

    def fit_transform(self, candidates):
        if not len(candidates):
            raise ValueError("Cannot fit an empty fit_train candidate pool")
        blocks = []
        for name, analyzer in (("events", event_features), ("content", content_features), ("relations", relation_features)):
            model = TfidfVectorizer(analyzer=analyzer, lowercase=False, sublinear_tf=True,
                                    norm="l2", dtype=np.float32,
                                    max_features=self.max_content_features if name == "content" else None)
            try:
                block = model.fit_transform(self._documents(candidates, name)).tocsr()
            except ValueError as exc:
                if "empty vocabulary" not in str(exc):
                    raise
                model = None
                block = sparse.csr_matrix((len(candidates), 0), dtype=np.float32)
            self.vectorizers[name] = model
            self.block_metadata[name] = {"features": block.shape[1], "empty_rows": int(np.sum(np.diff(block.indptr) == 0)),
                                         "nnz": block.nnz, "sublinear_tf": True, "normalization": "L2",
                                         "max_features": self.max_content_features if name == "content" else None,
                                         "vocabulary_cap_reached": bool(name == "content" and self.max_content_features is not None and block.shape[1] >= self.max_content_features)}
            blocks.append(block)
        self.fit_candidate_count = len(candidates)
        self.fit_content_keys_sha256 = _digest([c["content_key"] for c in candidates])
        return normalize(sparse.hstack(blocks, format="csr"), copy=False).astype(np.float32)

    def transform(self, candidates):
        if not self.vectorizers:
            raise ValueError("Encoder has not been fit on fit_train")
        blocks = []
        for name, model in self.vectorizers.items():
            blocks.append(sparse.csr_matrix((len(candidates), 0), dtype=np.float32) if model is None
                          else model.transform(self._documents(candidates, name)))
        return normalize(sparse.hstack(blocks, format="csr"), copy=False).astype(np.float32)

    def metadata(self):
        return {"normalization_version": NORMALIZATION_VERSION,
                "fit_scope": "fit_train distinct canonical candidates only; no labels in features",
                "fit_candidate_count": self.fit_candidate_count, "fit_content_keys_sha256": self.fit_content_keys_sha256,
                "blocks": self.block_metadata, "combination": "equal nonempty-block weight, each block L2 then concatenation L2",
                "empty_block_policy": "zero contribution; final global L2 distributes equal weight over nonempty blocks",
                "content_tokens": "words, all single-character numbers, literal key=value, within-line adjacent word bigrams; no stop words",
                "object_aliases": "window-local first-occurrence aliases per kind: block ID, UUID, IPv4 endpoint, explicit pid/tid/process_id/thread_id/request_id/session_id/object_id/transaction_id/job_id/block_id; process[number] including process(module)[number]; explicit msgid/message_id and sendmail/postfix queue IDs. Null/none/unknown/absent/empty message ID states remain literal. Source Node/NodeRepeat/Host share host aliases; Thunderbird User is source host, Location preserves known role@host; other User values are actor aliases. Context PID/ProcessId, bracket PID and explicit body PID share host-scoped aliases; unknown host scope is explicit.",
                "timestamp_rules": _STAMP.pattern, "object_rules": _OBJECT.pattern,
                "mail_message_rules": _MESSAGE_ID.pattern, "mail_queue_rules": _MAIL_QUEUE.pattern,
                "safe_context_fields": sorted(_CONTEXT_FIELDS),
                "safe_context_policy": "Level/Component/Type retain categorical state and normalize embedded objects. Node/NodeRepeat/Host, dataset-specific User/Location, Pid/PID/ProcessId/ThreadId retain stable object identities and relationships. Every other context field, including labels and date/time, is ignored. Original context remains in the external fitting JSONL.",
                "parameter_policy": "retain all other numeric and key=value states; no test-label-derived rules",
                "limitations": ["Adjacent event pairs do not encode all long-range order; complete sequence is retained.",
                                "Unrecognized objects remain literal; explicit object parsing cannot establish every semantic relationship.",
                                "Vocabulary limits and bag features can map different contents to the same vector; these rows are not merged by feature equality.",
                                "TF-IDF/SVD are selection features, never frozen CNN hidden_feature."]}

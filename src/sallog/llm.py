"""Independent JSON API calls with durable attempts and explicit retries."""
from __future__ import annotations
from contextlib import closing
from importlib.resources import files
import json
import os
from pathlib import Path
import re
import sqlite3
import time
import urllib.error
import urllib.request
from urllib.parse import urlparse
from .storage import digest, dumps, now, run_lock

FIELDS = {"case_id", "group", "verdict", "evidence", "brief_analysis", "reference_assessment"}


def prompt_text(stage):
    name = "review" if stage == "review" else "explanation"
    return files("sallog").joinpath(f"prompts/{name}.txt").read_text(encoding="utf-8")


def payload(config, case, card, prompt, stage="review"):
    body = {"dataset": case["dataset"], "case_id": case["case_id"], "target": case["target"],
            "references": case["references"], "knowledge_card": card}
    if stage == "explanation":
        body["final_detection_label"] = "abnormal"
    return {**config.get("extra_body", {}), "model": config["model"], "max_tokens": int(config["max_tokens"]),
            "response_format": {"type": "json_object"}, "stream": False,
            "messages": [{"role": "system", "content": prompt}, {"role": "user", "content": dumps(body)}]}


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON field: {key}")
        result[key] = value
    return result


def validate_reply(reply, case, stage="review"):
    group = "C2" if stage == "review" else "EXPLAIN"
    verdicts = {"normal", "abnormal", "abstain"} if stage == "review" else {"abnormal", "abstain"}
    if not isinstance(reply, dict) or set(reply) != FIELDS or reply.get("case_id") != case["case_id"] or reply.get("group") != group or reply.get("verdict") not in verdicts:
        raise ValueError("Invalid reply identity, group, fields or verdict")
    evidence = reply["evidence"]
    allowed = {v["line_ref"] for v in case["target"]["logs"]}
    if (not isinstance(evidence, list) or len(evidence) > 3 or any(not isinstance(v, str) or v not in allowed for v in evidence)
            or len(set(evidence)) != len(evidence) or (reply["verdict"] != "abstain" and not evidence)):
        raise ValueError("Invalid target evidence")
    if not isinstance(reply["brief_analysis"], str) or not reply["brief_analysis"].strip():
        raise ValueError("A nonempty brief analysis is required")
    assessment = reply["reference_assessment"]
    if not isinstance(assessment, dict) or set(assessment) != {"R1", "R2"}:
        raise ValueError("Both reference assessments are required")
    for key, reference in case["references"].items():
        text = assessment[key]
        if not isinstance(text, str) or not text.strip() or reference["reference_id"] not in text:
            raise ValueError("Reference assessment must cite its actual reference ID")
        citations = re.findall(r"\bS\d{4,}\b", text)
        available = {v["line_ref"] for v in reference["logs"]}
        if not citations or any(v not in available for v in citations):
            raise ValueError("Invalid training-reference line citation")
    return reply


def parse_completion(raw, case, stage="review"):
    response = json.loads(raw, object_pairs_hook=_unique)
    choice = response["choices"][0]
    if choice.get("finish_reason") != "stop":
        raise ValueError("Completion did not finish normally")
    content = choice["message"]["content"]
    if not isinstance(content, str):
        raise ValueError("Expected text JSON content")
    reply = json.loads(content, object_pairs_hook=_unique,
                       parse_constant=lambda value: (_ for _ in ()).throw(ValueError("Non-finite JSON")))
    return validate_reply(reply, case, stage), response.get("usage", {}), response.get("model")


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise urllib.error.HTTPError(req.full_url, code, "API redirect refused", headers, fp)


def transmit(config, request_body, key):
    endpoint = config["endpoint"]
    parsed = urlparse(endpoint)
    if parsed.scheme not in ("http", "https") or not parsed.netloc or parsed.username or parsed.password:
        raise ValueError("Expected an HTTP(S) API endpoint without embedded credentials")
    request = urllib.request.Request(endpoint, data=dumps(request_body).encode("utf-8"), method="POST",
                                     headers={"Content-Type": "application/json", "Authorization": "Bearer " + key})
    with urllib.request.build_opener(NoRedirect()).open(request, timeout=float(config["timeout_seconds"])) as response:
        raw = response.read(int(config["max_response_bytes"]) + 1)
        if len(raw) > int(config["max_response_bytes"]):
            raise ValueError("API response exceeds configured size limit")
        return raw.decode("utf-8").replace(key, "[REDACTED]")


def database(path):
    con = sqlite3.connect(path)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA synchronous=FULL")
    con.execute("""CREATE TABLE IF NOT EXISTS attempts (
        id INTEGER PRIMARY KEY, stage TEXT NOT NULL, case_id TEXT NOT NULL,
        request_hash TEXT NOT NULL, request_json TEXT NOT NULL, status TEXT NOT NULL,
        created_at TEXT NOT NULL, reserve INTEGER NOT NULL, elapsed_seconds REAL,
        raw_response TEXT, parsed_response TEXT, usage TEXT, returned_model TEXT, error TEXT)""")
    con.commit()
    return con


def latest(directory, stage):
    path = Path(directory) / "responses.sqlite"
    if not path.exists():
        return {}
    with closing(sqlite3.connect(path)) as con:
        con.row_factory = sqlite3.Row
        return {r["case_id"]: dict(r) for r in con.execute(
            "SELECT * FROM attempts WHERE id IN (SELECT MAX(id) FROM attempts WHERE stage=? GROUP BY case_id)", (stage,))}


def execute(directory, config, cases, card, prompt, *, stage="review", execute_api=False,
            limit=0, token_budget=2000000, retry_failed=False, retry_uncertain=False):
    if limit < 0 or token_budget <= 0:
        raise ValueError("Invalid request limit or reservation budget")
    previous = latest(directory, stage)
    plan = []
    for case in cases:
        body = payload(config, case, card, prompt, stage)
        encoded = dumps(body).encode("utf-8")
        if len(encoded) > int(config["max_input_bytes"]):
            raise ValueError(f"{case['case_id']}: input exceeds max_input_bytes; no logs were truncated")
        identity = digest([config["endpoint"], body])
        old = previous.get(case["case_id"])
        if old:
            if old["request_hash"] != identity:
                raise ValueError("Request identity changed; use a new run")
            if old["status"] == "ok":
                continue
            uncertain = old["status"] in ("uncertain", "in_flight")
            if (uncertain and not retry_uncertain) or (not uncertain and not retry_failed):
                continue
        plan.append((case, body, identity, len(encoded) + int(config["max_tokens"]) + 2048))
    if not execute_api:
        return {"stage": stage, "pending_eligible": len(plan), "new_calls": 0, "dry_run": True}
    sent = 0
    with run_lock(directory), closing(database(Path(directory) / "responses.sqlite")) as con:
        used = con.execute("SELECT COALESCE(SUM(reserve),0) FROM attempts WHERE stage=?", (stage,)).fetchone()[0]
        key = None
        for case, body, identity, reserve in plan:
            # Planning and acquiring the lock are separate operations. Check
            # committed state again to avoid resending a concurrently completed case.
            current = con.execute("SELECT status,request_hash FROM attempts WHERE stage=? AND case_id=? ORDER BY id DESC LIMIT 1",
                                  (stage, case["case_id"])).fetchone()
            if current:
                if current["request_hash"] != identity:
                    raise ValueError("Request identity changed; use a new run")
                if current["status"] == "ok":
                    continue
                uncertain = current["status"] in ("uncertain", "in_flight")
                if (uncertain and not retry_uncertain) or (not uncertain and not retry_failed):
                    continue
            if limit and sent >= limit:
                break
            if used + reserve > token_budget:
                break
            if key is None:
                key = os.environ.get(config["api_key_env"], "").strip()
                if not key:
                    raise ValueError(f"Set environment variable {config['api_key_env']}")
            cursor = con.execute("INSERT INTO attempts(stage,case_id,request_hash,request_json,status,created_at,reserve) VALUES(?,?,?,?,?,?,?)",
                                 (stage, case["case_id"], identity, dumps(body), "in_flight", now(), reserve))
            con.commit()
            used += reserve
            started = time.monotonic()
            status, raw, parsed, usage, returned, error = "uncertain", None, None, {}, None, None
            try:
                raw = transmit(config, body, key)
                envelope = json.loads(raw)
                if isinstance(envelope, dict):
                    usage, returned = envelope.get("usage", {}), envelope.get("model")
                parsed, usage, returned = parse_completion(raw, case, stage)
                status = "ok"
            except urllib.error.HTTPError as exc:
                status, error = "failed", f"HTTP {exc.code}"
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                error = f"{type(exc).__name__}: transport outcome uncertain"
            except (ValueError, KeyError, IndexError, TypeError) as exc:
                status, error = "invalid_output", str(exc).replace(key, "[REDACTED]")
            usage = usage if isinstance(usage, dict) else {}
            returned = returned if isinstance(returned, str) else None
            con.execute("UPDATE attempts SET status=?,elapsed_seconds=?,raw_response=?,parsed_response=?,usage=?,returned_model=?,error=? WHERE id=?",
                        (status, time.monotonic() - started, raw, None if parsed is None else dumps(parsed), dumps(usage), returned, error, cursor.lastrowid))
            con.commit()
            sent += 1
            print(f"{stage}/{case['case_id']}: {status}", flush=True)
            time.sleep(float(config["interval_seconds"]))
    return {"stage": stage, "new_calls": sent, "dry_run": False}


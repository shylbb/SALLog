"""Target-citation checks and a strict class-similarity acceptance margin."""
from decimal import Decimal
import math


def decide(case, reply, small_prediction, similarities, threshold=-.1):
    if type(small_prediction) is not int or small_prediction not in (0, 1):
        raise ValueError("Expected a binary detector prediction")
    cutoff = Decimal(str(threshold))
    if not cutoff.is_finite():
        raise ValueError("Acceptance threshold must be finite")
    faults, notes = [], []
    identity = isinstance(reply, dict) and reply.get("case_id") == case["case_id"]
    verdict = reply.get("verdict") if identity else None
    proposed = int(verdict == "abnormal") if verdict in ("normal", "abnormal") else None
    if not identity:
        faults.append("invalid_identity")
    if isinstance(reply, dict):
        evidence = reply.get("evidence")
        allowed = {v["line_ref"] for v in case["target"]["logs"]}
        if not isinstance(evidence, list) or any(not isinstance(v, str) or v not in allowed for v in evidence):
            faults.append("invalid_query_line_reference")
        elif proposed is not None and not evidence:
            faults.append("missing_query_evidence")
        else:
            if len(evidence) > 3:
                notes.append("more_than_three_evidence_references")
            if len(set(evidence)) != len(evidence):
                notes.append("duplicate_evidence_references")
        if not isinstance(reply.get("brief_analysis"), str):
            faults.append("invalid_brief_type")
        elif not reply["brief_analysis"].strip():
            notes.append("missing_explanation")
    valid_sims = (isinstance(similarities, (list, tuple)) and len(similarities) == 2
                  and all(not isinstance(v, bool) and isinstance(v, (float, int)) and math.isfinite(v) and -1 <= v <= 1 for v in similarities))
    margin = None
    if proposed is not None and valid_sims:
        margin = Decimal(str(similarities[proposed])) - Decimal(str(similarities[small_prediction]))
    final = small_prediction
    if proposed is None:
        reason = "abstain_fallback" if verdict == "abstain" else "missing_or_invalid_reply"
    elif proposed == small_prediction:
        reason = "agreement_keep_label"
    elif faults:
        reason = "target_check_failed"
    elif margin is None:
        reason = "missing_similarity"
    elif margin > cutoff:
        final, reason = proposed, "margin_passed_override"
    else:
        reason = "margin_rejected"
    return {"small_prediction": small_prediction, "llm_prediction": proposed, "final_prediction": final,
            "accepted_override": final != small_prediction, "reason": reason,
            "margin": None if margin is None else str(margin), "threshold": str(cutoff),
            "target_faults": faults, "explanation_notes": notes, "semantic_support_verified": False}


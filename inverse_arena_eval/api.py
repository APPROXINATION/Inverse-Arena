"""Framework-agnostic API helpers.

Two pieces:
  - ``sanitize_execution_report_payload(body)`` — validates + clamps an incoming
    JSON payload. Returns ``(sanitized_dict, error_code_or_None)``.
  - ``execution_report_handler(service, body, agent_id="", user_id="")`` —
    one-call convenience that validates then dispatches to
    ``EvaluationService.submit_execution_report``.

Plug these into any HTTP framework (stdlib, Flask, FastAPI, etc.) —
they have no framework dependencies.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional, Tuple


def _safe_text(value: Any, max_len: int = 160) -> str:
    s = re.sub(r"\s+", " ", str(value or "")).strip()
    return s[:max_len]


def _clamp_int(value: Any, low: int, high: int, default: int = 0) -> int:
    try:
        v = int(value)
    except (TypeError, ValueError):
        v = default
    return max(low, min(high, v))


ALLOWED_FIELDS = {
    "agent_id", "skill_id", "skill_name", "query",
    "steps_total", "steps_followed", "steps_improvised",
    "errors_from_skill", "task_completed", "output_produced",
    "latency_ms", "tokens_spent", "tool_calls_made", "retries",
    "replaceability", "uses_external_api", "domain_knowledge_density",
}


def sanitize_execution_report_payload(body: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Validate and normalize an incoming execution report.

    Returns ``(payload_or_None, error_code_or_None)``.

    Error codes:
      - ``"query_required"`` — empty or missing query
      - ``"skill_name_required"`` — empty or missing skill name
      - ``"no_steps_taken"`` — agent reported zero followed + zero improvised
    """
    if not isinstance(body, dict):
        return None, "invalid_payload"

    skill_id = _safe_text(body.get("skill_id", ""), 160)
    skill_name = _safe_text(body.get("skill_name", ""), 160)
    query = _safe_text(body.get("query", ""), 400)
    if not query:
        return None, "query_required"
    if not skill_name:
        return None, "skill_name_required"
    if not skill_id:
        skill_id = skill_name

    steps_total = _clamp_int(body.get("steps_total", 0), 1, 1000, 1)
    steps_followed = _clamp_int(body.get("steps_followed", 0), 0, 1000, 0)
    steps_improvised = _clamp_int(body.get("steps_improvised", 0), 0, 1000, 0)
    errors_from_skill = _clamp_int(body.get("errors_from_skill", 0), 0, 1000, 0)
    latency_ms = _clamp_int(body.get("latency_ms", 0), 0, 3_600_000, 0)
    tokens_spent = _clamp_int(body.get("tokens_spent", 0), 0, 10**9, 0)
    tool_calls_made = _clamp_int(body.get("tool_calls_made", 0), 0, 10_000, 0)
    retries = _clamp_int(body.get("retries", 0), 0, 1000, 0)

    # Invariants
    if steps_followed > steps_total:
        steps_followed = steps_total
    if errors_from_skill > steps_followed:
        errors_from_skill = steps_followed
    if steps_followed + steps_improvised < 1:
        return None, "no_steps_taken"

    replaceability = _clamp_int(body.get("replaceability", 1), 0, 3, 1)
    try:
        density_raw = float(body.get("domain_knowledge_density", 0.5))
    except (TypeError, ValueError):
        density_raw = 0.5
    domain_knowledge_density = max(0.0, min(1.0, density_raw))

    return {
        "skill_id": skill_id,
        "skill_name": skill_name,
        "query": query,
        "steps_total": steps_total,
        "steps_followed": steps_followed,
        "steps_improvised": steps_improvised,
        "errors_from_skill": errors_from_skill,
        "task_completed": bool(body.get("task_completed", False)),
        "output_produced": bool(body.get("output_produced", False)),
        "latency_ms": latency_ms,
        "tokens_spent": tokens_spent,
        "tool_calls_made": tool_calls_made,
        "retries": retries,
        "replaceability": replaceability,
        "uses_external_api": bool(body.get("uses_external_api", False)),
        "domain_knowledge_density": domain_knowledge_density,
    }, None


def execution_report_handler(
    service,
    body: Dict[str, Any],
    *,
    agent_id: str = "",
    user_id: str = "",
) -> Tuple[int, Dict[str, Any]]:
    """End-to-end handler: validate the payload and dispatch to the service.

    Returns ``(http_status, response_body)``.

    Example (stdlib http.server):
        >>> status, body = execution_report_handler(svc, request_json, agent_id=req_agent)
        >>> self.send_response(status)
        >>> self.wfile.write(json.dumps(body).encode())
    """
    payload, err = sanitize_execution_report_payload(body or {})
    if payload is None:
        return 400, {"error": err or "invalid_payload"}
    result = service.submit_execution_report(
        agent_id=agent_id,
        user_id=user_id,
        payload=payload,
    )
    return 200, result

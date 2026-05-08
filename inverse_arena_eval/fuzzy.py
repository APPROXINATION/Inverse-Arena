"""Fuzzy logic primitives for skill evaluation.

Pure functions, no external dependencies beyond Python stdlib. The module
provides membership functions, defuzzification, and a set of Mamdani-style
rule evaluators that convert noisy execution evidence into skill-quality
scores.

Two evaluation modes are supported:

1. **Adherence mode** (``compute_adherence_quality``) — the primary mode for
   organic agent usage. Inputs are adherence, self-reliance, error rate,
   outcome, token efficiency, retry rate. Output: skill quality 0-1.

2. **Pairwise mode** (``compute_tool_quality`` / ``score_evaluation_pair``) —
   classical A/B benchmark comparison. Takes an EvaluationJob and a result
   report and scores both tools.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from .models import EvaluationJob, EvaluationResult, ToolExecutionReport


# ═══════════════════════════════════════════════════════════════════════
# Primitives
# ═══════════════════════════════════════════════════════════════════════

def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    """Clamp ``value`` to [low, high]."""
    return max(low, min(high, float(value)))


def triangular(x: float, a: float, b: float, c: float) -> float:
    """Triangular membership — rises from ``a`` to peak at ``b`` and falls to ``c``."""
    x = float(x)
    if x <= a or x >= c:
        return 0.0
    if x == b:
        return 1.0
    if x < b:
        return (x - a) / max(1e-9, (b - a))
    return (c - x) / max(1e-9, (c - b))


def trapezoid(x: float, a: float, b: float, c: float, d: float) -> float:
    """Trapezoidal membership — plateau from ``b`` to ``c`` with linear edges."""
    x = float(x)
    if x < a or x > d:
        return 0.0
    if b <= x <= c:
        return 1.0
    if x < b:
        if b == a:
            return 1.0
        return (x - a) / max(1e-9, (b - a))
    if d == c:
        return 1.0
    return (d - x) / max(1e-9, (d - c))


def defuzzify(activations: Dict[str, float], weights: Dict[str, float]) -> tuple[float, str]:
    """Weighted-centroid defuzzification.

    ``activations`` maps label → rule strength. ``weights`` maps label → center
    of the output fuzzy set. Returns ``(score, dominant_label)``.
    """
    total = sum(max(0.0, float(v)) for v in activations.values())
    if total <= 1e-9:
        lowest = min(weights.items(), key=lambda item: item[1])[0]
        return 0.0, lowest
    score = sum(clamp(float(activations.get(k, 0.0))) * float(weights[k]) for k in weights) / total
    label = max(activations.items(), key=lambda item: (float(item[1]), weights.get(item[0], 0.0)))[0]
    return clamp(score), label


# ═══════════════════════════════════════════════════════════════════════
# Input fuzzy sets — classical pairwise mode
# ═══════════════════════════════════════════════════════════════════════

def _latency_sets(latency_ms: int, task_type: str) -> Dict[str, float]:
    """Fast/acceptable/slow classification, task-type aware."""
    x = max(0, int(latency_ms or 0))
    if str(task_type) == "search_answer":
        return {
            "fast": trapezoid(x, 0, 0, 1200, 2600),
            "acceptable": triangular(x, 1500, 3000, 4800),
            "slow": trapezoid(x, 3600, 5200, 9000, 9000),
        }
    return {
        "fast": trapezoid(x, 0, 0, 600, 1200),
        "acceptable": triangular(x, 700, 1400, 2500),
        "slow": trapezoid(x, 1800, 2800, 7000, 7000),
    }


def _quality_sets(value: float) -> Dict[str, float]:
    x = clamp(value)
    return {
        "low": trapezoid(x, 0.0, 0.0, 0.22, 0.45),
        "medium": triangular(x, 0.25, 0.55, 0.82),
        "high": trapezoid(x, 0.62, 0.8, 1.0, 1.0),
    }


def _json_sets(value: float) -> Dict[str, float]:
    x = clamp(value)
    return {
        "poor": trapezoid(x, 0.0, 0.0, 0.2, 0.45),
        "partial": triangular(x, 0.25, 0.55, 0.82),
        "strong": trapezoid(x, 0.65, 0.85, 1.0, 1.0),
    }


def _trust_sets(value: float) -> Dict[str, float]:
    x = clamp(value)
    return {
        "low": trapezoid(x, 0.0, 0.0, 0.25, 0.5),
        "medium": triangular(x, 0.25, 0.55, 0.82),
        "high": trapezoid(x, 0.62, 0.82, 1.0, 1.0),
    }


def _evidence_sets(value: float) -> Dict[str, float]:
    x = clamp(value)
    return {
        "weak": trapezoid(x, 0.0, 0.0, 0.25, 0.55),
        "medium": triangular(x, 0.3, 0.6, 0.85),
        "strong": trapezoid(x, 0.65, 0.85, 1.0, 1.0),
    }


# ═══════════════════════════════════════════════════════════════════════
# Derivation helpers — classical pairwise mode
# ═══════════════════════════════════════════════════════════════════════

def _derive_json_quality(report: ToolExecutionReport, task_type: str) -> float:
    if str(task_type) == "structured_json":
        if bool(report.schema_pass):
            return 1.0
        if bool(report.json_valid):
            return 0.68
        return 0.12 if bool(report.success) else 0.0
    if bool(report.success) and str(report.raw_output or "").strip():
        return 0.78
    return 0.18 if bool(report.success) else 0.0


def _derive_action_accuracy(report: ToolExecutionReport, *, winner: str, expected_tool: str) -> float:
    if not bool(report.success):
        return 0.08
    if winner == expected_tool:
        return 0.92
    if winner == "draw":
        return 0.72
    if winner == "neither":
        return 0.25
    return 0.56


def _derive_evidence_completeness(
    result: EvaluationResult,
    report: ToolExecutionReport,
    counterpart: ToolExecutionReport,
) -> float:
    checks = [
        1.0 if str(report.raw_output or "").strip() else 0.0,
        1.0 if str(counterpart.raw_output or "").strip() else 0.0,
        1.0 if int(report.latency_ms or 0) > 0 else 0.0,
        1.0 if int(counterpart.latency_ms or 0) > 0 else 0.0,
        1.0 if str(result.reason_code or "").strip() else 0.0,
        1.0 if str(result.summary or "").strip() else 0.65,
    ]
    return clamp(sum(checks) / max(1, len(checks)))


def derive_agent_trust(
    *,
    stats_successes: int = 0,
    stats_failures: int = 0,
    performance_score: float = 0.5,
    collaboration_score: float = 0.5,
) -> float:
    """Compute an agent's trust weight from its historical stats.

    Lightweight — no AgentProfile dependency. Pass whatever scoring data you
    have (successes/failures, per-domain performance, peer ratings).
    """
    total = max(1, int(stats_successes) + int(stats_failures))
    empirical = int(stats_successes) / total
    reliability = 0.35 * 0.65 + 0.65 * empirical
    trust = (
        0.45 * reliability
        + 0.3 * clamp(float(performance_score))
        + 0.25 * clamp(float(collaboration_score))
    )
    return clamp(trust)


# ═══════════════════════════════════════════════════════════════════════
# Classical pairwise evaluation (EvaluationJob + EvaluationResult)
# ═══════════════════════════════════════════════════════════════════════

def compute_tool_quality(
    *,
    job: EvaluationJob,
    report: ToolExecutionReport,
    winner: str,
    expected_tool: str,
    agent_trust: float,
) -> Dict[str, Any]:
    """Score a single tool's execution using the classical 4-input rule set."""
    latency = _latency_sets(int(report.latency_ms or 0), str(job.task_type))
    accuracy_value = _derive_action_accuracy(report, winner=winner, expected_tool=expected_tool)
    accuracy = _quality_sets(accuracy_value)
    json_quality_value = _derive_json_quality(report, str(job.task_type))
    json_quality = _json_sets(json_quality_value)
    trust = _trust_sets(agent_trust)

    activations = {
        "poor": max(
            json_quality["poor"],
            accuracy["low"],
            min(latency["slow"], accuracy["medium"]),
        ),
        "fair": max(
            min(accuracy["medium"], json_quality["partial"]),
            min(latency["slow"], accuracy["high"]),
            min(accuracy["high"], json_quality["partial"]),
        ),
        "good": max(
            min(accuracy["high"], json_quality["strong"], latency["acceptable"]),
            min(accuracy["medium"], json_quality["strong"]),
            min(accuracy["high"], trust["high"]),
        ),
        "excellent": max(
            min(accuracy["high"], json_quality["strong"], latency["fast"]),
            min(accuracy["high"], json_quality["strong"], trust["high"]),
        ),
    }
    score, label = defuzzify(
        activations,
        {"poor": 0.18, "fair": 0.45, "good": 0.72, "excellent": 0.92},
    )
    return {
        "score": score,
        "label": label,
        "inputs": {
            "latency_ms": int(report.latency_ms or 0),
            "action_accuracy": round(accuracy_value, 4),
            "json_quality": round(json_quality_value, 4),
            "agent_trust": round(agent_trust, 4),
        },
        "memberships": {
            "latency": latency,
            "accuracy": accuracy,
            "json_quality": json_quality,
            "trust": trust,
            "tool_quality": activations,
        },
    }


def compute_update_strength(
    *,
    agent_trust: float,
    confidence: float,
    evidence_completeness: float,
    outcome_clarity: float,
) -> Dict[str, Any]:
    """Score how much to trust this individual evaluation as a ranking signal."""
    trust_value = clamp(agent_trust)
    confidence_value = clamp(float(confidence or 0.0))
    clarity_value = clamp(float(outcome_clarity or 0.0))

    trust = _trust_sets(trust_value)
    conf = _quality_sets(confidence_value)
    evidence = _evidence_sets(evidence_completeness)
    clarity = _evidence_sets(clarity_value)

    activations = {
        "weak": max(
            trust["low"],
            conf["low"],
            evidence["weak"],
            clarity["weak"],
        ),
        "moderate": max(
            min(trust["medium"], conf["high"]),
            min(trust["high"], conf["medium"]),
            min(evidence["medium"], trust["medium"]),
        ),
        "strong": max(
            min(trust["high"], conf["high"], evidence["strong"]),
            min(trust["high"], clarity["strong"]),
        ),
    }
    score, label = defuzzify(
        activations,
        {"weak": 0.25, "moderate": 0.56, "strong": 0.84},
    )
    return {
        "score": score,
        "label": label,
        "inputs": {
            "agent_trust": round(trust_value, 4),
            "confidence": round(confidence_value, 4),
            "evidence_completeness": round(evidence_completeness, 4),
            "outcome_clarity": round(clarity_value, 4),
        },
        "memberships": {
            "trust": trust,
            "confidence": conf,
            "evidence": evidence,
            "clarity": clarity,
            "update_strength": activations,
        },
    }


def score_evaluation_pair(
    *,
    job: EvaluationJob,
    result: EvaluationResult,
    agent_trust: float,
) -> Dict[str, Any]:
    """Full pairwise scoring: both tools + update strength."""
    if result.tool_a_result is None or result.tool_b_result is None:
        return {
            "tool_a": {"score": 0.0, "label": "poor"},
            "tool_b": {"score": 0.0, "label": "poor"},
            "update_strength": {"score": 0.0, "label": "weak"},
        }

    tool_a = compute_tool_quality(
        job=job,
        report=result.tool_a_result,
        winner=result.winner,
        expected_tool=job.tool_a,
        agent_trust=agent_trust,
    )
    tool_b = compute_tool_quality(
        job=job,
        report=result.tool_b_result,
        winner=result.winner,
        expected_tool=job.tool_b,
        agent_trust=agent_trust,
    )
    evidence = _derive_evidence_completeness(result, result.tool_a_result, result.tool_b_result)
    clarity = abs(float(tool_a["score"]) - float(tool_b["score"])) * 1.6
    update_strength = compute_update_strength(
        agent_trust=agent_trust,
        confidence=float(result.confidence or 0.0),
        evidence_completeness=evidence,
        outcome_clarity=clarity,
    )
    return {
        "tool_a": tool_a,
        "tool_b": tool_b,
        "update_strength": update_strength,
    }


# ═══════════════════════════════════════════════════════════════════════
# Adherence-based evaluation (organic execution reports)
# ═══════════════════════════════════════════════════════════════════════

def _adherence_sets(x: float) -> Dict[str, float]:
    """How closely the agent followed the skill's prescribed steps."""
    x = clamp(x)
    return {
        "low": trapezoid(x, 0.0, 0.0, 0.30, 0.55),
        "medium": triangular(x, 0.35, 0.65, 0.88),
        "high": trapezoid(x, 0.72, 0.88, 1.0, 1.0),
    }


def _self_reliance_sets(x: float) -> Dict[str, float]:
    """How much the agent improvised beyond the skill."""
    x = clamp(x)
    return {
        "minimal": trapezoid(x, 0.0, 0.0, 0.12, 0.30),
        "moderate": triangular(x, 0.15, 0.38, 0.62),
        "heavy": trapezoid(x, 0.48, 0.68, 1.0, 1.0),
    }


def _error_rate_sets(x: float) -> Dict[str, float]:
    """Errors caused by the skill's instructions."""
    x = clamp(x)
    return {
        "clean": trapezoid(x, 0.0, 0.0, 0.05, 0.18),
        "some": triangular(x, 0.08, 0.25, 0.48),
        "buggy": trapezoid(x, 0.32, 0.52, 1.0, 1.0),
    }


def _outcome_sets(x: float) -> Dict[str, float]:
    """Task completion: failed / partial / success."""
    x = clamp(x)
    return {
        "failed": trapezoid(x, 0.0, 0.0, 0.15, 0.38),
        "partial": triangular(x, 0.22, 0.50, 0.78),
        "success": trapezoid(x, 0.62, 0.82, 1.0, 1.0),
    }


def _token_efficiency_sets(x: float) -> Dict[str, float]:
    """Relative token efficiency (median/actual, clamped 0-1)."""
    x = clamp(x)
    return {
        "wasteful": trapezoid(x, 0.0, 0.0, 0.20, 0.40),
        "normal": triangular(x, 0.30, 0.55, 0.80),
        "lean": trapezoid(x, 0.65, 0.85, 1.0, 1.0),
    }


def _retry_rate_sets(x: float) -> Dict[str, float]:
    """How often steps had to be retried — instruction clarity signal."""
    x = clamp(x)
    return {
        "smooth": trapezoid(x, 0.0, 0.0, 0.05, 0.18),
        "rough": triangular(x, 0.10, 0.28, 0.50),
        "broken": trapezoid(x, 0.38, 0.58, 1.0, 1.0),
    }


def compute_adherence_quality(
    *,
    adherence: float,
    self_reliance: float,
    error_rate: float,
    outcome: float,
    token_efficiency: float = 0.5,
    retry_rate: float = 0.0,
) -> Dict[str, Any]:
    """Score skill quality from a single agent's execution report.

    Key insight: outcome=success + self_reliance=heavy → only "fair". Task
    completed BUT the agent did most of the work themselves — the skill
    didn't really help, so we don't credit it.
    """
    adh = _adherence_sets(adherence)
    sr = _self_reliance_sets(self_reliance)
    err = _error_rate_sets(error_rate)
    out = _outcome_sets(outcome)
    tok = _token_efficiency_sets(token_efficiency)
    rtr = _retry_rate_sets(retry_rate)

    activations = {
        "poor": max(
            out["failed"],
            min(err["buggy"], adh["low"]),
            min(adh["low"], out["partial"]),
        ),
        "fair": max(
            min(out["partial"], adh["medium"]),
            min(adh["high"], err["some"]),
            min(out["success"], sr["heavy"]),
            min(out["success"], adh["medium"], rtr["rough"]),
            min(out["success"], sr["heavy"], tok["wasteful"]),
        ),
        "good": max(
            min(out["success"], adh["medium"], err["clean"]),
            min(adh["high"], sr["moderate"], out["success"]),
            min(out["success"], err["clean"]),
            min(out["success"], adh["high"], err["clean"], tok["wasteful"]),
        ),
        "excellent": max(
            min(out["success"], adh["high"], err["clean"], sr["minimal"]),
            min(out["success"], adh["high"], err["clean"], tok["lean"], rtr["smooth"]),
            min(out["success"], adh["high"], err["clean"], rtr["smooth"]),
        ),
    }

    score, label = defuzzify(
        activations,
        {"poor": 0.18, "fair": 0.45, "good": 0.72, "excellent": 0.92},
    )
    return {
        "score": round(score, 4),
        "label": label,
        "inputs": {
            "adherence": round(adherence, 4),
            "self_reliance": round(self_reliance, 4),
            "error_rate": round(error_rate, 4),
            "outcome": round(outcome, 4),
            "token_efficiency": round(token_efficiency, 4),
            "retry_rate": round(retry_rate, 4),
        },
        "activations": {k: round(v, 4) for k, v in activations.items()},
    }


def score_execution_report_pair(
    *,
    report_a: Dict[str, Any],
    report_b: Dict[str, Any],
    trust_a: float = 0.5,
    trust_b: float = 0.5,
) -> Dict[str, Any]:
    """Score a head-to-head comparison between two skills on the same task.

    Each ``report_*`` dict must contain the derived fields:
      ``adherence_score``, ``self_reliance``, ``error_rate``, ``outcome_score``,
      ``token_efficiency``, ``retry_rate``, ``steps_total``, ``latency_ms``.

    Returns the same shape as ``score_evaluation_pair`` so downstream ranking
    code can treat both modes uniformly.
    """
    quality_a = compute_adherence_quality(
        adherence=float(report_a.get("adherence_score", 0.0)),
        self_reliance=float(report_a.get("self_reliance", 0.0)),
        error_rate=float(report_a.get("error_rate", 0.0)),
        outcome=float(report_a.get("outcome_score", 0.0)),
        token_efficiency=float(report_a.get("token_efficiency", 0.5)),
        retry_rate=float(report_a.get("retry_rate", 0.0)),
    )
    quality_b = compute_adherence_quality(
        adherence=float(report_b.get("adherence_score", 0.0)),
        self_reliance=float(report_b.get("self_reliance", 0.0)),
        error_rate=float(report_b.get("error_rate", 0.0)),
        outcome=float(report_b.get("outcome_score", 0.0)),
        token_efficiency=float(report_b.get("token_efficiency", 0.5)),
        retry_rate=float(report_b.get("retry_rate", 0.0)),
    )

    score_a = float(quality_a["score"])
    score_b = float(quality_b["score"])

    if abs(score_a - score_b) < 0.05:
        winner = "draw"
    elif score_a > score_b:
        winner = "tool_a"
    else:
        winner = "tool_b"

    avg_trust = (float(trust_a) + float(trust_b)) / 2.0
    clarity = clamp(abs(score_a - score_b) * 1.6)

    # Evidence: both reports have real data?
    checks = [
        1.0 if int(report_a.get("steps_total", 0)) > 0 else 0.0,
        1.0 if int(report_b.get("steps_total", 0)) > 0 else 0.0,
        1.0 if int(report_a.get("latency_ms", 0)) > 0 else 0.0,
        1.0 if int(report_b.get("latency_ms", 0)) > 0 else 0.0,
        1.0, 1.0,  # task_completed + output_produced are always present booleans
    ]
    evidence = clamp(sum(checks) / max(1, len(checks)))

    update_strength = compute_update_strength(
        agent_trust=avg_trust,
        confidence=max(score_a, score_b),
        evidence_completeness=evidence,
        outcome_clarity=clarity,
    )

    return {
        "tool_a": quality_a,
        "tool_b": quality_b,
        "winner": winner,
        "update_strength": update_strength,
    }

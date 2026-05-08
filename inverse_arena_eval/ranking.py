"""Convert fuzzy scores into pairwise vote records and persist them.

This is the glue between ``fuzzy.py`` (scoring) and ``storage.py`` (persistence).
"""

from __future__ import annotations

from typing import Any, Dict, List

from .fuzzy import score_evaluation_pair
from .models import EvaluationJob, EvaluationResult


def build_rankings_from_evaluation(
    job: EvaluationJob,
    result: EvaluationResult,
    fuzzy_scores: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Convert a fuzzy scoring result into a ranking list suitable for
    ``EvaluationStorage.record_inverse_arena_votes``."""
    if result.tool_a_result is None or result.tool_b_result is None:
        return []

    update_strength = float((fuzzy_scores.get("update_strength") or {}).get("score", 0.0) or 0.0)
    tool_a_quality = fuzzy_scores.get("tool_a") or {}
    tool_b_quality = fuzzy_scores.get("tool_b") or {}

    return [
        {
            "tool": job.tool_a,
            "score": round(float(tool_a_quality.get("score", 0.0) or 0.0) * 10.0, 2),
            "reason": (
                f"{result.reason_code} | fuzzy_quality={float(tool_a_quality.get('score', 0.0) or 0.0):.2f}"
                f" ({str(tool_a_quality.get('label', 'unknown'))}) | update_strength={update_strength:.2f}"
            ),
        },
        {
            "tool": job.tool_b,
            "score": round(float(tool_b_quality.get("score", 0.0) or 0.0) * 10.0, 2),
            "reason": (
                f"{result.reason_code} | fuzzy_quality={float(tool_b_quality.get('score', 0.0) or 0.0):.2f}"
                f" ({str(tool_b_quality.get('label', 'unknown'))}) | update_strength={update_strength:.2f}"
            ),
        },
    ]


def apply_evaluation_result(
    *,
    storage,
    cycle_id: str,
    section: str,
    job: EvaluationJob,
    result: EvaluationResult,
    agent_trust: float,
) -> Dict[str, Any]:
    """Score a pairwise evaluation and persist the resulting votes.

    ``storage`` must expose ``record_inverse_arena_votes(...)`` — use
    ``EvaluationStorage`` from this package, or any compatible implementation.
    """
    fuzzy_scores = score_evaluation_pair(job=job, result=result, agent_trust=agent_trust)
    rankings = build_rankings_from_evaluation(job, result, fuzzy_scores)
    weighted_confidence = float(result.confidence or 0.0) * float(
        (fuzzy_scores.get("update_strength") or {}).get("score", 0.0) or 0.0
    )
    written = storage.record_inverse_arena_votes(
        cycle_id=cycle_id,
        agent_id=result.agent_id,
        section=section,
        question_number=job.question_number,
        question=job.prompt_or_input,
        rankings=rankings,
        confidence=weighted_confidence,
    )
    return {
        "rankings": rankings,
        "votes_written": int(written),
        "fuzzy_scores": fuzzy_scores,
        "weighted_confidence": round(weighted_confidence, 4),
    }

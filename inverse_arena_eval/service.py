"""High-level orchestration for the evaluation pipeline.

``EvaluationService`` is the main entry point. It accepts execution reports
from agents, stores them, pairs them with other reports on the same question,
runs fuzzy scoring, records votes, and triggers ELO rebuilds.

Usage:
    >>> from inverse_arena_eval import EvaluationStorage, EvaluationService
    >>> storage = EvaluationStorage("arena.db")
    >>> svc = EvaluationService(storage)
    >>> result = svc.submit_execution_report(agent_id="a1", payload={
    ...     "skill_id": "acme/security-auditor",
    ...     "skill_name": "security-auditor",
    ...     "query": "audit web app security",
    ...     "steps_total": 8,
    ...     "steps_followed": 7,
    ...     "steps_improvised": 1,
    ...     "errors_from_skill": 0,
    ...     "task_completed": True,
    ...     "output_produced": True,
    ...     "latency_ms": 4500,
    ...     "tokens_spent": 3200,
    ...     "tool_calls_made": 12,
    ...     "retries": 0,
    ... })
"""

from __future__ import annotations

import secrets
from typing import Any, Dict, Optional

from .fuzzy import clamp, score_execution_report_pair
from .storage import EvaluationStorage


DEFAULT_REBUILD_INTERVAL = 10  # rebuild ELO every N micro-cycles
# Reference defaults for the agent-trust heuristic. Production deployments
# typically override these with tuned values.
DEFAULT_TRUST_DEFAULT = 0.85
DEFAULT_TRUST_NEUTRAL = 0.5
DEFAULT_TRUST_TIERS = (
    (0.9, 0.3),
    (0.7, 0.6),
)
DEFAULT_TRUST_MIN_HISTORY = 5
DEFAULT_TRUST_HISTORY_WINDOW = 20


class EvaluationService:
    """Orchestrates the execution-report → pair → score → vote → ELO pipeline.

    Anti-gaming knobs (``trust_*``) are illustrative defaults. Deployed arenas
    are expected to override them with values tuned to their own traffic.
    """

    def __init__(
        self,
        storage: EvaluationStorage,
        *,
        rebuild_interval: int = DEFAULT_REBUILD_INTERVAL,
        trust_default: float = DEFAULT_TRUST_DEFAULT,
        trust_neutral: float = DEFAULT_TRUST_NEUTRAL,
        trust_tiers: tuple[tuple[float, float], ...] = DEFAULT_TRUST_TIERS,
        trust_min_history: int = DEFAULT_TRUST_MIN_HISTORY,
        trust_history_window: int = DEFAULT_TRUST_HISTORY_WINDOW,
    ) -> None:
        self.storage = storage
        self.rebuild_interval = max(1, int(rebuild_interval))
        self.trust_default = float(trust_default)
        self.trust_neutral = float(trust_neutral)
        self.trust_tiers = tuple(sorted(trust_tiers, key=lambda t: -t[0]))
        self.trust_min_history = max(1, int(trust_min_history))
        self.trust_history_window = max(1, int(trust_history_window))
        self._micro_cycle_count = 0

    # ── main entry point ─────────────────────────────────────────────

    def submit_execution_report(
        self,
        *,
        agent_id: str = "",
        user_id: str = "",
        payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Process one execution report. Stores it, tries to pair, runs
        fuzzy scoring, records votes, and optionally triggers ELO rebuild.

        Returns one of:
          - ``{status: "invalid", error: ...}`` — payload missing fields
          - ``{status: "stored", pair_formed: False, ...}`` — stored, no partner yet
          - ``{status: "paired", pair_formed: True, micro_cycle: {...}, ...}`` — pair found + scored
        """
        skill_id = str(payload.get("skill_id", "")).strip()
        skill_name = str(payload.get("skill_name", "")).strip()
        query = str(payload.get("query", "")).strip()
        steps_total = int(payload.get("steps_total", 0))
        steps_followed = int(payload.get("steps_followed", 0))
        steps_improvised = int(payload.get("steps_improvised", 0))
        errors_from_skill = int(payload.get("errors_from_skill", 0))
        task_completed = bool(payload.get("task_completed", False))
        output_produced = bool(payload.get("output_produced", False))
        latency_ms = int(payload.get("latency_ms", 0))
        tokens_spent = int(payload.get("tokens_spent", 0))
        tool_calls_made = int(payload.get("tool_calls_made", 0))
        retries = int(payload.get("retries", 0))
        replaceability = int(payload.get("replaceability", 1))
        uses_external_api = bool(payload.get("uses_external_api", False))
        try:
            domain_knowledge_density = float(payload.get("domain_knowledge_density", 0.5))
        except (TypeError, ValueError):
            domain_knowledge_density = 0.5

        if not query or not skill_name or steps_total < 1:
            return {"status": "invalid", "error": "missing_required_fields"}

        stored = self.storage.store_execution_report(
            agent_id=agent_id,
            user_id=user_id,
            skill_id=skill_id,
            skill_name=skill_name,
            query=query,
            steps_total=steps_total,
            steps_followed=steps_followed,
            steps_improvised=steps_improvised,
            errors_from_skill=errors_from_skill,
            task_completed=task_completed,
            output_produced=output_produced,
            latency_ms=latency_ms,
            tokens_spent=tokens_spent,
            tool_calls_made=tool_calls_made,
            retries=retries,
            replaceability=replaceability,
            uses_external_api=uses_external_api,
            domain_knowledge_density=domain_knowledge_density,
        )
        report_id = stored["report_id"]
        question_key = stored["question_key"]

        self_reliance = float(steps_improvised) / max(1, steps_followed + steps_improvised)
        error_rate = float(errors_from_skill) / max(1, steps_followed) if steps_followed > 0 else 0.0

        partner = self.storage.find_unpaired_partner(question_key, skill_name)
        if partner is None:
            return {
                "status": "stored",
                "report_id": report_id,
                "question_key": question_key,
                "adherence_score": stored["adherence_score"],
                "outcome_score": stored["outcome_score"],
                "pair_formed": False,
                "micro_cycle": None,
            }

        # ── Pair found → micro-cycle ──────────────────────────────
        micro_cycle_id = f"micro_{secrets.token_hex(6)}"
        median_tokens = self.storage.get_question_token_median(question_key)

        token_eff_a = (
            clamp(float(median_tokens) / max(1, tokens_spent))
            if median_tokens > 0 and tokens_spent > 0 else 0.5
        )
        report_a = {
            "adherence_score": float(stored["adherence_score"]),
            "self_reliance": self_reliance,
            "error_rate": error_rate,
            "outcome_score": float(stored["outcome_score"]),
            "token_efficiency": token_eff_a,
            "retry_rate": float(stored["retry_rate"]),
            "steps_total": steps_total,
            "latency_ms": latency_ms,
        }

        p_followed = int(partner.get("steps_followed", 0))
        p_improvised = int(partner.get("steps_improvised", 0))
        p_tokens = int(partner.get("tokens_spent", 0))
        token_eff_b = (
            clamp(float(median_tokens) / max(1, p_tokens))
            if median_tokens > 0 and p_tokens > 0 else 0.5
        )
        report_b = {
            "adherence_score": float(partner.get("adherence_score", 0.0)),
            "self_reliance": float(p_improvised) / max(1, p_followed + p_improvised),
            "error_rate": float(partner.get("errors_from_skill", 0)) / max(1, p_followed) if p_followed > 0 else 0.0,
            "outcome_score": float(partner.get("outcome_score", 0.0)),
            "token_efficiency": token_eff_b,
            "retry_rate": float(partner.get("retry_rate", 0.0)),
            "steps_total": int(partner.get("steps_total", 0)),
            "latency_ms": int(partner.get("latency_ms", 0)),
        }

        trust_a = self._agent_trust(agent_id)
        trust_b = self._agent_trust(str(partner.get("agent_id", "")))

        fuzzy_result = score_execution_report_pair(
            report_a=report_a,
            report_b=report_b,
            trust_a=trust_a,
            trust_b=trust_b,
        )

        score_a = float(fuzzy_result["tool_a"]["score"])
        score_b = float(fuzzy_result["tool_b"]["score"])
        update_strength = float(fuzzy_result["update_strength"]["score"])
        weighted_confidence = clamp(update_strength * max(trust_a, trust_b))

        skill_b_name = str(partner.get("skill_name", "")).strip()
        rankings = [
            {
                "tool": skill_name,
                "score": round(score_a * 10.0, 2),
                "reason": (
                    f"adherence={stored['adherence_score']:.2f} outcome={stored['outcome_score']} | "
                    f"fuzzy={score_a:.2f} ({fuzzy_result['tool_a']['label']})"
                ),
            },
            {
                "tool": skill_b_name,
                "score": round(score_b * 10.0, 2),
                "reason": (
                    f"adherence={partner.get('adherence_score', 0):.2f} outcome={partner.get('outcome_score', 0)} | "
                    f"fuzzy={score_b:.2f} ({fuzzy_result['tool_b']['label']})"
                ),
            },
        ]

        q_meta = self.storage.get_question_metadata(question_key) or {
            "section": "Organic Execution Reports",
            "question_number": 1,
            "question": query,
        }
        votes_written = self.storage.record_inverse_arena_votes(
            cycle_id=micro_cycle_id,
            agent_id=agent_id or user_id or "organic",
            section=q_meta["section"],
            question_number=int(q_meta["question_number"]),
            question=q_meta["question"],
            rankings=rankings,
            confidence=weighted_confidence,
        )

        # Also sync into agent_tools_table so ELO rebuild can consume
        sorted_rankings = sorted(rankings, key=lambda r: float(r.get("score", 0)), reverse=True)
        self.storage.upsert_ranking(
            section=q_meta["section"],
            question_number=int(q_meta["question_number"]),
            question=q_meta["question"],
            rankings=[
                {
                    "tool": r["tool"],
                    "tool_rank": idx + 1,
                    "score": float(r["score"]),
                    "installs": 0,
                    "url": "",
                }
                for idx, r in enumerate(sorted_rankings)
            ],
        )

        self.storage.mark_reports_paired(report_id, str(partner.get("report_id", "")), micro_cycle_id)

        # Periodically rebuild ELO
        self._micro_cycle_count += 1
        if self._micro_cycle_count >= self.rebuild_interval:
            self._micro_cycle_count = 0
            try:
                self.storage.rebuild_tool_elo_ratings(cycle_id=micro_cycle_id)
            except Exception:
                pass

        return {
            "status": "paired",
            "report_id": report_id,
            "question_key": question_key,
            "adherence_score": stored["adherence_score"],
            "outcome_score": stored["outcome_score"],
            "pair_formed": True,
            "micro_cycle": {
                "cycle_id": micro_cycle_id,
                "skill_a": skill_name,
                "skill_b": skill_b_name,
                "quality_a": score_a,
                "quality_b": score_b,
                "winner": fuzzy_result["winner"],
                "update_strength": update_strength,
                "votes_written": votes_written,
            },
        }

    # ── agent trust / anti-gaming ─────────────────────────────────────

    def _agent_trust(self, agent_id: str) -> float:
        """Discount agents who consistently submit suspiciously perfect reports.

        Thresholds are configured via ``trust_*`` constructor arguments and
        intentionally not hard-coded here.
        """
        if not agent_id:
            return self.trust_neutral
        recent = self.storage.get_recent_execution_reports(
            agent_id, limit=self.trust_history_window)
        if len(recent) < self.trust_min_history:
            return self.trust_neutral
        perfect = sum(
            1 for r in recent
            if float(r.get("adherence_score", 0)) >= 0.99
            and int(r.get("errors_from_skill", 0)) == 0
            and float(r.get("outcome_score", 0)) >= 0.99
        )
        ratio = float(perfect) / len(recent)
        for threshold, trust in self.trust_tiers:
            if ratio > threshold:
                return trust
        return self.trust_default

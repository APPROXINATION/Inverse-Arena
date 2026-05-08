"""Inverse Arena Evaluation — open-source skill evaluation pipeline.

A fuzzy-logic framework for aggregating agent feedback on AI skills into
community-driven ELO rankings. Agents submit execution reports (how well a
skill guided them through a real task); the pipeline pairs reports, runs
fuzzy scoring on adherence/outcome/efficiency, records votes, and updates
per-skill ELO.

No opinions, no star ratings — just objective execution evidence.

Core API:
    >>> from inverse_arena_eval import EvaluationStorage, EvaluationService
    >>> storage = EvaluationStorage("evaluation.db")
    >>> service = EvaluationService(storage)
    >>> result = service.submit_execution_report(agent_id="a1", payload={...})
"""

from .fuzzy import (
    # Core primitives
    triangular,
    trapezoid,
    clamp,
    defuzzify,
    # Adherence-based evaluation (new)
    compute_adherence_quality,
    score_execution_report_pair,
    # Classical pairwise evaluation
    compute_tool_quality,
    compute_update_strength,
    score_evaluation_pair,
)

from .models import (
    EvaluationJob,
    ToolExecutionReport,
    EvaluationResult,
    ExecutionReport,
)

from .ranking import (
    build_rankings_from_evaluation,
    apply_evaluation_result,
)

from .storage import EvaluationStorage

from .service import EvaluationService

from .api import execution_report_handler, sanitize_execution_report_payload

__version__ = "0.1.0"

__all__ = [
    # Fuzzy primitives
    "triangular",
    "trapezoid",
    "clamp",
    "defuzzify",
    # Adherence evaluation
    "compute_adherence_quality",
    "score_execution_report_pair",
    # Classical pairwise
    "compute_tool_quality",
    "compute_update_strength",
    "score_evaluation_pair",
    # Models
    "EvaluationJob",
    "ToolExecutionReport",
    "EvaluationResult",
    "ExecutionReport",
    # Ranking update
    "build_rankings_from_evaluation",
    "apply_evaluation_result",
    # Storage + service
    "EvaluationStorage",
    "EvaluationService",
    # API helpers
    "execution_report_handler",
    "sanitize_execution_report_payload",
]

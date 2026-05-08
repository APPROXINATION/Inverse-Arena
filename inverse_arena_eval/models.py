"""Data models for the Inverse Arena evaluation pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class EvaluationJob:
    """A benchmark job: compare two tools on the same task.

    The job describes WHICH tools to run, WHAT input to run them on, and
    HOW to report the result. Agents pick up jobs, run the tools, and return
    an ``EvaluationResult``.
    """
    evaluation_id: str
    bucket_id: str
    bucket_label: str
    tenant_id: str
    task_type: str                      # "search_answer" | "structured_json" | "action_task"
    prompt_or_input: str
    tool_a: str
    tool_b: str
    question_number: int = 1
    instructions: List[str] = field(default_factory=list)
    expected_schema: Dict[str, Any] = field(default_factory=dict)
    timeout_sec: int = 45
    vote_type: str = "execution"
    pair_strategy: str = "top_vs_runner_up"
    candidate_tools: List[str] = field(default_factory=list)
    enqueue_reason: str = "scheduled_benchmark"
    status: str = "pending"
    schema_version: str = "1.0"

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ToolExecutionReport:
    """The raw execution outcome for a single tool invocation."""
    tool: str
    success: bool
    latency_ms: int
    raw_output: str = ""
    json_valid: Optional[bool] = None
    schema_pass: Optional[bool] = None
    error: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EvaluationResult:
    """An agent's verdict after running both tools in an ``EvaluationJob``."""
    evaluation_id: str
    agent_id: str
    winner: str                         # "tool_a" | "tool_b" | "draw" | "neither"
    confidence: float
    reason_code: str
    summary: str = ""
    tool_a_result: Optional[ToolExecutionReport] = None
    tool_b_result: Optional[ToolExecutionReport] = None
    schema_version: str = "1.0"

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ExecutionReport:
    """An organic execution report — what actually happened when an agent used
    a skill on a real task. This is the primary input to the Inverse Arena.

    Ten objective fields, no opinions:
      - Adherence: how closely the agent followed the skill's steps
      - Outcome: did the task complete and produce output
      - Efficiency: how many tokens/tool-calls/retries were needed

    Plus three optional uniqueness signals (orthogonal to the 10 above):
      - ``replaceability`` (0..3): 0 = trivially reproducible locally,
        3 = irreplaceable external intelligence (live API, proprietary data)
      - ``uses_external_api``: did execution hit a real external endpoint?
      - ``domain_knowledge_density`` (0..1): subjective — how much the skill
        taught the agent something it didn't already know
    """
    skill_id: str
    skill_name: str
    query: str
    steps_total: int
    steps_followed: int
    steps_improvised: int
    errors_from_skill: int
    task_completed: bool
    output_produced: bool
    latency_ms: int
    tokens_spent: int = 0
    tool_calls_made: int = 0
    retries: int = 0
    replaceability: int = 1
    uses_external_api: bool = False
    domain_knowledge_density: float = 0.5
    agent_id: str = ""
    user_id: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)

"""SQLite persistence for the evaluation pipeline.

Self-contained — creates its own schema, manages its own connection. The
tables here hold execution reports, pairwise comparisons, ELO ratings, and
question catalogs. Everything needed to run a community-driven skill arena.

Schema:
  - ``skill_execution_reports``: raw agent reports (adherence, outcome, tokens)
  - ``inverse_arena_questions`` / ``_tokens``: benchmark questions + search index
  - ``inverse_arena_cycles``: evaluation batch metadata
  - ``inverse_arena_votes``: pairwise scoring rows
  - ``agent_tools_table``: per-question rankings (consumed by ELO rebuild)
  - ``tool_elo_ratings`` / ``_history``: current + historical ELO per skill
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import secrets
import sqlite3
import threading
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# Reference defaults. Production deployments are expected to override these
# (stop-word list, ELO seed/K-schedule, uniqueness weights, question-match
# thresholds) with values tuned to their own corpus and traffic.
DEFAULT_STOP_WORDS = frozenset({
    "a", "an", "the", "to", "and", "or", "for", "of", "in", "on", "at", "by",
    "from", "with", "as", "is", "are", "be", "that", "this", "it", "my", "your",
    "best", "skill", "skills", "tool", "tools",
})

# Uniqueness blend: (replaceability_weight, external_api_weight, density_weight)
DEFAULT_UNIQUENESS_WEIGHTS = (0.55, 0.25, 0.20)

# ELO calibration: (seed_min, seed_span, default_rating)
DEFAULT_ELO_SEED = (1200.0, 600.0, 1500.0)

# K-factor schedule by games played: list of (games_threshold, k_value).
# First entry whose threshold games < entry applies; final entry is the floor.
DEFAULT_ELO_K_SCHEDULE = ((30, 40), (200, 20), (10**9, 10))

# Question-matching thresholds: minimum token-coverage fraction and minimum
# absolute number of shared tokens to merge a free-form query into an existing
# benchmark question.
DEFAULT_QUESTION_MATCH = (0.6, 3)


def _tokenize(text: str, stop_words: frozenset = DEFAULT_STOP_WORDS) -> List[str]:
    """Basic tokenization: lowercase, split on non-word, strip stop words."""
    if not text:
        return []
    tokens = re.findall(r"[a-z0-9]+", str(text).lower())
    return [t for t in tokens if len(t) >= 2 and t not in stop_words]


class EvaluationStorage:
    """SQLite-backed persistence for the evaluation pipeline.

    Usage:
        >>> storage = EvaluationStorage("arena.db")
        >>> storage.store_execution_report(...)
        >>> storage.record_inverse_arena_votes(...)
        >>> storage.rebuild_tool_elo_ratings()
    """

    def __init__(
        self,
        db_path: str = "arena.db",
        *,
        stop_words: Optional[frozenset] = None,
        uniqueness_weights: tuple = DEFAULT_UNIQUENESS_WEIGHTS,
        elo_seed: tuple = DEFAULT_ELO_SEED,
        elo_k_schedule: tuple = DEFAULT_ELO_K_SCHEDULE,
        question_match: tuple = DEFAULT_QUESTION_MATCH,
    ) -> None:
        self.db_path = db_path
        self.stop_words = stop_words if stop_words is not None else DEFAULT_STOP_WORDS
        self.uniqueness_weights = tuple(uniqueness_weights)
        self.elo_seed = tuple(elo_seed)
        self.elo_k_schedule = tuple(sorted(elo_k_schedule, key=lambda t: t[0]))
        self.question_match = tuple(question_match)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._init_schema()

    def _k_factor(self, games: int) -> int:
        for threshold, k in self.elo_k_schedule:
            if games < threshold:
                return int(k)
        return int(self.elo_k_schedule[-1][1])

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # ── schema ─────────────────────────────────────────────────────────

    def _init_schema(self) -> None:
        with self._lock:
            cur = self._conn.cursor()
            cur.execute("""
                CREATE TABLE IF NOT EXISTS skill_execution_reports (
                    report_id TEXT PRIMARY KEY,
                    agent_id TEXT NOT NULL,
                    user_id TEXT NOT NULL DEFAULT '',
                    skill_id TEXT NOT NULL,
                    skill_name TEXT NOT NULL DEFAULT '',
                    query TEXT NOT NULL,
                    query_norm TEXT NOT NULL DEFAULT '',
                    question_key TEXT NOT NULL DEFAULT '',
                    steps_total INTEGER NOT NULL DEFAULT 0,
                    steps_followed INTEGER NOT NULL DEFAULT 0,
                    steps_improvised INTEGER NOT NULL DEFAULT 0,
                    errors_from_skill INTEGER NOT NULL DEFAULT 0,
                    task_completed INTEGER NOT NULL DEFAULT 0,
                    output_produced INTEGER NOT NULL DEFAULT 0,
                    latency_ms INTEGER NOT NULL DEFAULT 0,
                    tokens_spent INTEGER NOT NULL DEFAULT 0,
                    tool_calls_made INTEGER NOT NULL DEFAULT 0,
                    retries INTEGER NOT NULL DEFAULT 0,
                    adherence_score REAL NOT NULL DEFAULT 0.0,
                    outcome_score REAL NOT NULL DEFAULT 0.0,
                    token_efficiency REAL NOT NULL DEFAULT 0.5,
                    retry_rate REAL NOT NULL DEFAULT 0.0,
                    replaceability INTEGER NOT NULL DEFAULT 1,
                    uses_external_api INTEGER NOT NULL DEFAULT 0,
                    domain_knowledge_density REAL NOT NULL DEFAULT 0.5,
                    uniqueness_score REAL NOT NULL DEFAULT 0.5,
                    paired INTEGER NOT NULL DEFAULT 0,
                    pair_cycle_id TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL
                )
            """)
            # Idempotent migration — add uniqueness columns to existing DBs
            existing_cols = {r[1] for r in cur.execute("PRAGMA table_info(skill_execution_reports)").fetchall()}
            for coldef in [
                ("replaceability", "INTEGER NOT NULL DEFAULT 1"),
                ("uses_external_api", "INTEGER NOT NULL DEFAULT 0"),
                ("domain_knowledge_density", "REAL NOT NULL DEFAULT 0.5"),
                ("uniqueness_score", "REAL NOT NULL DEFAULT 0.5"),
            ]:
                if coldef[0] not in existing_cols:
                    cur.execute(f"ALTER TABLE skill_execution_reports ADD COLUMN {coldef[0]} {coldef[1]}")
            cur.execute("""
                CREATE TABLE IF NOT EXISTS inverse_arena_questions (
                    question_key TEXT PRIMARY KEY,
                    section TEXT NOT NULL,
                    question_number INTEGER NOT NULL,
                    question TEXT NOT NULL,
                    question_norm TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS inverse_arena_question_tokens (
                    question_key TEXT NOT NULL,
                    token TEXT NOT NULL,
                    PRIMARY KEY (question_key, token)
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS inverse_arena_cycles (
                    cycle_id TEXT PRIMARY KEY,
                    started_at TEXT NOT NULL,
                    finished_at TEXT,
                    status TEXT NOT NULL,
                    template_source TEXT NOT NULL DEFAULT '',
                    agents_considered INTEGER NOT NULL DEFAULT 0,
                    comparisons_completed INTEGER NOT NULL DEFAULT 0,
                    comparisons_failed INTEGER NOT NULL DEFAULT 0,
                    credits_awarded INTEGER NOT NULL DEFAULT 0,
                    rows_written INTEGER NOT NULL DEFAULT 0,
                    duration_ms INTEGER NOT NULL DEFAULT 0,
                    error TEXT NOT NULL DEFAULT ''
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS inverse_arena_votes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    cycle_id TEXT NOT NULL,
                    agent_id TEXT NOT NULL,
                    section TEXT NOT NULL,
                    question_number INTEGER NOT NULL,
                    question TEXT NOT NULL,
                    tool TEXT NOT NULL,
                    score REAL NOT NULL,
                    reason TEXT NOT NULL DEFAULT '',
                    confidence REAL NOT NULL DEFAULT 0.5,
                    created_at TEXT NOT NULL
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS agent_tools_table (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    section TEXT NOT NULL,
                    question_number INTEGER NOT NULL,
                    question TEXT NOT NULL,
                    tool_rank INTEGER NOT NULL,
                    tool TEXT NOT NULL,
                    score REAL NOT NULL,
                    source_installs INTEGER NOT NULL DEFAULT 0,
                    url TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS tool_elo_ratings (
                    tool TEXT PRIMARY KEY,
                    elo REAL NOT NULL DEFAULT 1500,
                    games INTEGER NOT NULL DEFAULT 0,
                    wins INTEGER NOT NULL DEFAULT 0,
                    losses INTEGER NOT NULL DEFAULT 0,
                    draws INTEGER NOT NULL DEFAULT 0,
                    last_updated_at TEXT NOT NULL
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS tool_elo_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    cycle_id TEXT NOT NULL,
                    tool TEXT NOT NULL,
                    elo_before REAL NOT NULL,
                    elo_after REAL NOT NULL,
                    delta REAL NOT NULL,
                    games_before INTEGER NOT NULL DEFAULT 0,
                    games_after INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL
                )
            """)
            # Indexes
            cur.execute("CREATE INDEX IF NOT EXISTS idx_exec_reports_question_key ON skill_execution_reports(question_key)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_exec_reports_skill ON skill_execution_reports(skill_name)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_exec_reports_unpaired ON skill_execution_reports(paired, question_key)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_exec_reports_agent ON skill_execution_reports(agent_id, created_at DESC)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_q_tokens_token ON inverse_arena_question_tokens(token)")
            cur.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_votes_unique ON inverse_arena_votes(cycle_id, agent_id, section, question_number, question, tool)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_agent_tools_section_q ON agent_tools_table(section, question_number)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_agent_tools_tool ON agent_tools_table(tool)")
            self._conn.commit()

    # ── question catalog ──────────────────────────────────────────────

    def _normalize_question(self, text: str) -> str:
        return " ".join(_tokenize(text, self.stop_words))

    @staticmethod
    def _question_key(*, section: str, question_number: int, question: str) -> str:
        base = f"{str(section).lower().strip()}|{int(question_number)}|{str(question).lower().strip()}"
        digest = hashlib.sha256(base.encode("utf-8")).hexdigest()[:24]
        return f"iaq_{digest}"

    def match_or_create_question(self, query: str, *, section: str = "Organic Execution Reports") -> str:
        """Match a free-form query to an existing benchmark question, or create a new one.

        Uses token-overlap matching (≥60% coverage and ≥3 shared tokens). If
        no match, registers a new question under the given section.
        """
        text = str(query or "").strip()
        if not text:
            return ""
        q_norm = self._normalize_question(text)
        q_tokens = _tokenize(text, self.stop_words)
        if not q_tokens:
            return ""

        with self._lock:
            exact = self._conn.execute(
                "SELECT question_key FROM inverse_arena_questions WHERE question_norm = ? LIMIT 1",
                (q_norm,),
            ).fetchone()
            if exact:
                return str(exact["question_key"])

            if len(q_tokens) >= 2:
                placeholders = ",".join("?" for _ in q_tokens)
                rows = self._conn.execute(
                    f"""SELECT question_key, COUNT(*) AS overlap
                        FROM inverse_arena_question_tokens
                        WHERE token IN ({placeholders})
                        GROUP BY question_key
                        ORDER BY overlap DESC
                        LIMIT 5""",
                    q_tokens,
                ).fetchall()
                min_coverage, min_shared = self.question_match
                for r in rows:
                    coverage = float(r["overlap"]) / len(q_tokens)
                    if coverage >= min_coverage and int(r["overlap"]) >= min_shared:
                        return str(r["question_key"])

            # Create new
            max_row = self._conn.execute(
                "SELECT MAX(question_number) AS mx FROM inverse_arena_questions WHERE section = ?",
                (section,),
            ).fetchone()
            q_num = (int(max_row["mx"] or 0) if max_row else 0) + 1
            q_key = self._question_key(section=section, question_number=q_num, question=text)
            ts = _now_iso()
            self._conn.execute(
                """INSERT OR IGNORE INTO inverse_arena_questions
                   (question_key, section, question_number, question, question_norm, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (q_key, section, q_num, text[:400], q_norm, ts),
            )
            seen = set()
            for tok in q_tokens:
                if tok and tok not in seen:
                    seen.add(tok)
                    self._conn.execute(
                        "INSERT OR IGNORE INTO inverse_arena_question_tokens (question_key, token) VALUES (?, ?)",
                        (q_key, tok),
                    )
            self._conn.commit()
            return q_key

    def get_question_metadata(self, question_key: str) -> Optional[Dict[str, Any]]:
        """Return section / number / question text for a stored question_key."""
        with self._lock:
            row = self._conn.execute(
                "SELECT section, question_number, question FROM inverse_arena_questions WHERE question_key = ?",
                (str(question_key),),
            ).fetchone()
        return dict(row) if row else None

    # ── execution reports ─────────────────────────────────────────────

    def store_execution_report(
        self,
        *,
        agent_id: str,
        user_id: str = "",
        skill_id: str,
        skill_name: str,
        query: str,
        steps_total: int,
        steps_followed: int,
        steps_improvised: int,
        errors_from_skill: int,
        task_completed: bool,
        output_produced: bool,
        latency_ms: int,
        tokens_spent: int = 0,
        tool_calls_made: int = 0,
        retries: int = 0,
        replaceability: int = 1,
        uses_external_api: bool = False,
        domain_knowledge_density: float = 0.5,
        section: str = "Organic Execution Reports",
    ) -> Dict[str, Any]:
        """Store a single agent's execution report and map it to an arena question.

        Supersedes any previous report from the same (agent, skill, question).
        Returns ``{report_id, question_key, adherence_score, outcome_score, retry_rate, uniqueness_score}``.

        Uniqueness axis (orthogonal to adherence/outcome):
          - ``replaceability`` (0..3): 0 = trivially reproducible locally,
            1 = local with effort, 2 = needs specific knowledge,
            3 = irreplaceable (live API, proprietary data, niche domain).
          - ``uses_external_api``: did execution hit a real external endpoint?
          - ``domain_knowledge_density`` (0..1): subjective — how much did the
            skill teach that the agent didn't already know?
        """
        ts = _now_iso()
        query_norm = self._normalize_question(query)
        question_key = self.match_or_create_question(query, section=section)

        adherence = float(steps_followed) / max(1, steps_total)
        outcome = 1.0 if (task_completed and output_produced) else 0.5 if task_completed else 0.0
        retry_rate = float(retries) / max(1, steps_followed) if steps_followed > 0 else 0.0

        rep = max(0, min(3, int(replaceability)))
        ext = 1 if uses_external_api else 0
        density = max(0.0, min(1.0, float(domain_knowledge_density)))
        w_rep, w_ext, w_density = self.uniqueness_weights
        uniqueness = (rep / 3.0) * w_rep + ext * w_ext + density * w_density
        uniqueness = max(0.0, min(1.0, uniqueness))

        report_id = f"ser_{secrets.token_hex(8)}"

        with self._lock:
            self._conn.execute(
                """UPDATE skill_execution_reports SET paired = -1
                   WHERE agent_id = ? AND skill_name = ? AND question_key = ? AND paired = 0""",
                (str(agent_id), str(skill_name), question_key),
            )
            self._conn.execute(
                """INSERT INTO skill_execution_reports (
                    report_id, agent_id, user_id, skill_id, skill_name,
                    query, query_norm, question_key,
                    steps_total, steps_followed, steps_improvised, errors_from_skill,
                    task_completed, output_produced, latency_ms,
                    tokens_spent, tool_calls_made, retries,
                    adherence_score, outcome_score, token_efficiency, retry_rate,
                    replaceability, uses_external_api, domain_knowledge_density, uniqueness_score,
                    paired, pair_cycle_id, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, '', ?)""",
                (
                    report_id, str(agent_id), str(user_id), str(skill_id), str(skill_name),
                    str(query)[:400], query_norm, question_key,
                    int(steps_total), int(steps_followed), int(steps_improvised), int(errors_from_skill),
                    1 if task_completed else 0, 1 if output_produced else 0, int(latency_ms),
                    int(tokens_spent), int(tool_calls_made), int(retries),
                    round(adherence, 4), round(outcome, 4), 0.5, round(retry_rate, 4),
                    rep, ext, round(density, 4), round(uniqueness, 4),
                    ts,
                ),
            )
            self._conn.commit()

        return {
            "report_id": report_id,
            "question_key": question_key,
            "adherence_score": round(adherence, 4),
            "outcome_score": round(outcome, 4),
            "retry_rate": round(retry_rate, 4),
            "uniqueness_score": round(uniqueness, 4),
        }

    def find_unpaired_partner(self, question_key: str, skill_name: str) -> Optional[Dict[str, Any]]:
        """Find an unpaired report for the same question but a different skill."""
        with self._lock:
            row = self._conn.execute(
                """SELECT * FROM skill_execution_reports
                   WHERE question_key = ? AND skill_name != ? AND paired = 0
                   ORDER BY created_at DESC LIMIT 1""",
                (str(question_key), str(skill_name)),
            ).fetchone()
        return dict(row) if row else None

    def mark_reports_paired(self, report_a_id: str, report_b_id: str, cycle_id: str) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE skill_execution_reports SET paired = 1, pair_cycle_id = ? WHERE report_id IN (?, ?)",
                (str(cycle_id), str(report_a_id), str(report_b_id)),
            )
            self._conn.commit()

    def get_recent_execution_reports(self, agent_id: str, *, limit: int = 20) -> List[Dict[str, Any]]:
        """Recent reports by an agent — used for anti-gaming trust scoring."""
        with self._lock:
            rows = self._conn.execute(
                """SELECT adherence_score, outcome_score, errors_from_skill, retry_rate
                   FROM skill_execution_reports
                   WHERE agent_id = ? AND paired >= 0
                   ORDER BY created_at DESC LIMIT ?""",
                (str(agent_id), max(1, min(limit, 100))),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_question_token_median(self, question_key: str) -> int:
        """Median tokens_spent across all paired reports for a question — used for
        relative token efficiency scoring."""
        with self._lock:
            rows = self._conn.execute(
                """SELECT tokens_spent FROM skill_execution_reports
                   WHERE question_key = ? AND tokens_spent > 0 AND paired >= 0
                   ORDER BY tokens_spent""",
                (str(question_key),),
            ).fetchall()
        if not rows:
            return 0
        values = [int(r["tokens_spent"]) for r in rows]
        mid = len(values) // 2
        return values[mid] if len(values) % 2 == 1 else (values[mid - 1] + values[mid]) // 2

    # ── cycles + votes ────────────────────────────────────────────────

    def begin_cycle(self, template_source: str = "") -> str:
        """Start a new evaluation cycle. Returns ``cycle_id``."""
        cycle_id = f"iac_{secrets.token_hex(8)}"
        ts = _now_iso()
        with self._lock:
            self._conn.execute(
                """INSERT INTO inverse_arena_cycles
                   (cycle_id, started_at, status, template_source)
                   VALUES (?, ?, 'running', ?)""",
                (cycle_id, ts, str(template_source)),
            )
            self._conn.commit()
        return cycle_id

    def finalize_cycle(
        self,
        cycle_id: str,
        *,
        status: str = "completed",
        comparisons_completed: int = 0,
        rows_written: int = 0,
        error: str = "",
    ) -> bool:
        with self._lock:
            cur = self._conn.execute(
                """UPDATE inverse_arena_cycles
                   SET finished_at = ?, status = ?, comparisons_completed = ?,
                       rows_written = ?, error = ?
                   WHERE cycle_id = ?""",
                (_now_iso(), status, int(comparisons_completed), int(rows_written), str(error), cycle_id),
            )
            self._conn.commit()
        return cur.rowcount > 0

    def record_inverse_arena_votes(
        self,
        *,
        cycle_id: str,
        agent_id: str,
        section: str,
        question_number: int,
        question: str,
        rankings: List[Dict[str, Any]],
        confidence: float = 0.5,
    ) -> int:
        """Record pairwise scoring votes into ``inverse_arena_votes``.

        Each ``rankings`` entry: ``{tool, score (0-10), reason}``.
        Returns the number of rows inserted.
        """
        if not rankings:
            return 0
        ts = _now_iso()
        rows = []
        for r in rankings[:12]:
            tool = str(r.get("tool", "")).strip()
            if not tool:
                continue
            rows.append((
                cycle_id, str(agent_id), str(section), int(question_number), str(question),
                tool, float(r.get("score", 0.0)), str(r.get("reason", ""))[:240],
                float(confidence), ts,
            ))
        if not rows:
            return 0
        with self._lock:
            cur = self._conn.executemany(
                """INSERT INTO inverse_arena_votes
                   (cycle_id, agent_id, section, question_number, question, tool, score, reason, confidence, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(cycle_id, agent_id, section, question_number, question, tool) DO UPDATE SET
                     score=excluded.score, reason=excluded.reason, confidence=excluded.confidence""",
                rows,
            )
            self._conn.commit()
        return cur.rowcount

    def upsert_ranking(
        self,
        *,
        section: str,
        question_number: int,
        question: str,
        rankings: List[Dict[str, Any]],
    ) -> int:
        """Write per-question rankings into ``agent_tools_table`` — consumed by
        ``rebuild_tool_elo_ratings`` to update ELO. Clears prior rows for the
        same (section, question) before inserting."""
        ts = _now_iso()
        with self._lock:
            self._conn.execute(
                "DELETE FROM agent_tools_table WHERE section = ? AND question_number = ? AND question = ?",
                (str(section), int(question_number), str(question)),
            )
            rows = []
            for idx, r in enumerate(rankings[:20], start=1):
                tool = str(r.get("tool", "")).strip()
                if not tool:
                    continue
                rows.append((
                    str(section), int(question_number), str(question),
                    int(r.get("tool_rank", idx)),
                    tool,
                    float(r.get("score", 0.0)),
                    int(r.get("installs", 0)),
                    str(r.get("url", ""))[:500],
                    ts,
                ))
            if rows:
                self._conn.executemany(
                    """INSERT INTO agent_tools_table
                       (section, question_number, question, tool_rank, tool, score, source_installs, url, created_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    rows,
                )
            self._conn.commit()
        return len(rows)

    # ── ELO rebuild ───────────────────────────────────────────────────

    def rebuild_tool_elo_ratings(self, *, cycle_id: str = "") -> Dict[str, Any]:
        """Run pairwise ELO updates across every question in ``agent_tools_table``.

        For each question, tools are compared pairwise (winner = higher score);
        ELO is updated with a standard Elo formula, K-factor decaying with games
        played. Snapshots go to ``tool_elo_history``.
        """
        with self._lock:
            question_rows = self._conn.execute(
                "SELECT DISTINCT section, question_number, question FROM agent_tools_table"
            ).fetchall()
            tool_rows = self._conn.execute(
                "SELECT tool, source_installs FROM agent_tools_table"
            ).fetchall()
            prior_rows = self._conn.execute(
                "SELECT tool, elo, games FROM tool_elo_ratings"
            ).fetchall()

        seed_min, seed_span, default_elo = self.elo_seed
        ratings: Dict[str, float] = {}
        games: Dict[str, int] = {}
        for r in prior_rows:
            ratings[str(r["tool"])] = float(r["elo"] or default_elo)
            games[str(r["tool"])] = int(r["games"] or 0)

        # Seed from installs for fresh tools
        max_installs = 1
        installs: Dict[str, int] = {}
        for r in tool_rows:
            tool = str(r["tool"])
            i = int(r["source_installs"] or 0)
            if i > installs.get(tool, 0):
                installs[tool] = i
            if i > max_installs:
                max_installs = i
        for tool, inst in installs.items():
            if tool not in ratings:
                norm = math.log1p(inst) / max(1e-9, math.log1p(max_installs))
                ratings[tool] = seed_min + norm * seed_span
                games[tool] = games.get(tool, 0)

        prior_snapshot = {t: {"elo": ratings.get(t, default_elo), "games": games.get(t, 0)} for t in ratings}
        questions_processed = 0

        for q in question_rows:
            section = str(q["section"])
            q_num = int(q["question_number"])
            q_text = str(q["question"])
            with self._lock:
                rows = self._conn.execute(
                    """SELECT tool, score FROM agent_tools_table
                       WHERE section=? AND question_number=? AND question=?
                       ORDER BY score DESC""",
                    (section, q_num, q_text),
                ).fetchall()
            tools_here = [(str(r["tool"]), float(r["score"] or 0.0)) for r in rows]
            if len(tools_here) < 2:
                continue
            questions_processed += 1

            for i in range(len(tools_here)):
                for j in range(i + 1, len(tools_here)):
                    tool_a, score_a = tools_here[i]
                    tool_b, score_b = tools_here[j]
                    if tool_a not in ratings:
                        ratings[tool_a] = default_elo
                        games[tool_a] = 0
                    if tool_b not in ratings:
                        ratings[tool_b] = default_elo
                        games[tool_b] = 0
                    ra, rb = ratings[tool_a], ratings[tool_b]
                    ea = 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))
                    eb = 1.0 - ea
                    if abs(score_a - score_b) < 0.05:
                        sa, sb = 0.5, 0.5
                    elif score_a > score_b:
                        sa, sb = 1.0, 0.0
                    else:
                        sa, sb = 0.0, 1.0
                    k_a = self._k_factor(games[tool_a])
                    k_b = self._k_factor(games[tool_b])
                    ratings[tool_a] = ra + k_a * (sa - ea)
                    ratings[tool_b] = rb + k_b * (sb - eb)
                    games[tool_a] += 1
                    games[tool_b] += 1

        ts = _now_iso()
        with self._lock:
            self._conn.execute("DELETE FROM tool_elo_ratings")
            self._conn.executemany(
                """INSERT INTO tool_elo_ratings (tool, elo, games, last_updated_at)
                   VALUES (?, ?, ?, ?)""",
                [(t, ratings[t], games[t], ts) for t in ratings],
            )
            if cycle_id:
                history_rows = []
                for t in ratings:
                    before = prior_snapshot.get(t, {"elo": default_elo, "games": 0})
                    history_rows.append((
                        cycle_id, t, float(before["elo"]), float(ratings[t]),
                        float(ratings[t]) - float(before["elo"]),
                        int(before["games"]), int(games[t]), ts,
                    ))
                self._conn.executemany(
                    """INSERT INTO tool_elo_history
                       (cycle_id, tool, elo_before, elo_after, delta, games_before, games_after, created_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    history_rows,
                )
            self._conn.commit()

        return {
            "ok": True,
            "rated_tools": len(ratings),
            "questions_processed": questions_processed,
            "updated_at": ts,
        }

    # ── read helpers ──────────────────────────────────────────────────

    def get_tool_elo(self, tool: str) -> Dict[str, Any]:
        with self._lock:
            row = self._conn.execute(
                "SELECT tool, elo, games FROM tool_elo_ratings WHERE tool = ?",
                (str(tool),),
            ).fetchone()
        return dict(row) if row else {"tool": tool, "elo": self.elo_seed[2], "games": 0}

    def leaderboard(self, limit: int = 50) -> List[Dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT tool, elo, games FROM tool_elo_ratings ORDER BY elo DESC LIMIT ?",
                (max(1, min(int(limit), 500)),),
            ).fetchall()
        return [dict(r) for r in rows]

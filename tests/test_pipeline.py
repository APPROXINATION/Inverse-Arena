"""End-to-end test: submit two reports, verify pairing + ELO update."""

import os
import tempfile
import sys

# Allow running `python tests/test_pipeline.py` from the package root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from inverse_arena_eval import EvaluationStorage, EvaluationService, compute_adherence_quality


def test_adherence_quality_sanity():
    """High adherence + clean + success → excellent; heavy self_reliance → fair."""
    excellent = compute_adherence_quality(
        adherence=0.95, self_reliance=0.05, error_rate=0.0,
        outcome=1.0, token_efficiency=0.9, retry_rate=0.0,
    )
    assert excellent["label"] in ("good", "excellent"), f"got {excellent['label']}"
    assert excellent["score"] > 0.65

    # Task done but agent did everything themselves — not the skill's credit
    fair = compute_adherence_quality(
        adherence=0.3, self_reliance=0.8, error_rate=0.0,
        outcome=1.0, token_efficiency=0.5, retry_rate=0.0,
    )
    assert fair["score"] < 0.65, f"self_reliance=heavy should not be rewarded, got {fair['score']}"
    print(f"  [PASS] excellent={excellent['score']:.2f} ({excellent['label']})")
    print(f"  [PASS] fair={fair['score']:.2f} ({fair['label']})")


def test_end_to_end_pairing():
    db = tempfile.NamedTemporaryFile(suffix=".db", delete=False).name
    try:
        storage = EvaluationStorage(db)
        svc = EvaluationService(storage, rebuild_interval=1)

        r_a = svc.submit_execution_report(
            agent_id="alice",
            payload={
                "skill_id": "acme/security-auditor",
                "skill_name": "security-auditor",
                "query": "audit my web app for security vulnerabilities",
                "steps_total": 8, "steps_followed": 7, "steps_improvised": 1,
                "errors_from_skill": 0, "task_completed": True, "output_produced": True,
                "latency_ms": 4500, "tokens_spent": 3200, "tool_calls_made": 12, "retries": 0,
            },
        )
        assert r_a["status"] == "stored"
        assert r_a["pair_formed"] is False
        print(f"  [PASS] first report stored (no pair yet)")

        r_b = svc.submit_execution_report(
            agent_id="bob",
            payload={
                "skill_id": "acme/007",
                "skill_name": "007",
                "query": "security audit for web application",
                "steps_total": 12, "steps_followed": 8, "steps_improvised": 4,
                "errors_from_skill": 2, "task_completed": True, "output_produced": True,
                "latency_ms": 8200, "tokens_spent": 7500, "tool_calls_made": 22, "retries": 3,
            },
        )
        assert r_b["status"] == "paired", f"expected paired, got {r_b['status']}"
        assert r_b["pair_formed"] is True
        mc = r_b["micro_cycle"]
        assert mc["winner"] in ("tool_a", "tool_b", "draw")
        assert mc["votes_written"] == 2
        # In the micro-cycle, tool_a is the newly submitted report (007) and
        # tool_b is the earlier partner (security-auditor). Security-auditor
        # had better adherence/no errors, so tool_b should win.
        assert mc["quality_b"] > mc["quality_a"], "security-auditor (partner) should score higher"
        assert mc["winner"] == "tool_b"
        print(f"  [PASS] pair formed: {mc['skill_a']}({mc['quality_a']:.2f}) vs {mc['skill_b']}({mc['quality_b']:.2f}) → {mc['winner']}")

        # ELO should have been updated
        elo_a = storage.get_tool_elo("007")
        elo_b = storage.get_tool_elo("security-auditor")
        # security-auditor was tool_a (winner) → its ELO should be higher
        assert elo_b["elo"] > elo_a["elo"], f"winner {elo_b} should have higher ELO than {elo_a}"
        print(f"  [PASS] ELO updated: security-auditor={elo_b['elo']:.0f} > 007={elo_a['elo']:.0f}")
    finally:
        if os.path.exists(db):
            os.unlink(db)


def test_anti_gaming_trust_penalty():
    """An agent submitting only perfect reports should get trust discounted."""
    db = tempfile.NamedTemporaryFile(suffix=".db", delete=False).name
    try:
        storage = EvaluationStorage(db)
        svc = EvaluationService(storage, rebuild_interval=100)

        # Submit 10 perfect reports from the same agent
        for i in range(10):
            svc.submit_execution_report(
                agent_id="gamer",
                payload={
                    "skill_id": f"fake/skill-{i}",
                    "skill_name": f"skill-{i}",
                    "query": f"distinct query number {i} that does not match others",
                    "steps_total": 5, "steps_followed": 5, "steps_improvised": 0,
                    "errors_from_skill": 0, "task_completed": True, "output_produced": True,
                    "latency_ms": 1000, "tokens_spent": 500, "tool_calls_made": 3, "retries": 0,
                },
            )
        trust = svc._agent_trust("gamer")
        assert trust <= 0.3, f"expected heavy penalty, got {trust}"
        print(f"  [PASS] gaming agent trust discounted to {trust}")
    finally:
        if os.path.exists(db):
            os.unlink(db)


if __name__ == "__main__":
    print("=== test_adherence_quality_sanity ===")
    test_adherence_quality_sanity()
    print("\n=== test_end_to_end_pairing ===")
    test_end_to_end_pairing()
    print("\n=== test_anti_gaming_trust_penalty ===")
    test_anti_gaming_trust_penalty()
    print("\nAll tests passed.")

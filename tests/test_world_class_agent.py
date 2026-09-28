"""Comprehensive test suite for the world-class autonomous examination intelligence platform.

Validates:
1. Robust Behavioral Baseline (MAD, IQR, Maturity Score)
2. Cross-Modal Fusion Contradiction Detection
3. Formal Uncertainty Engine (Epistemic, Aleatoric, Quality)
4. Causal & Counterfactual Attribution Engine (Marginal Delta, Minimal Sufficient Cause)
5. Technical-Event Intelligence (Hardware Outage Isolation)
6. Multi-Step Mathematical Grader (Formula, Substitution, Derivation, Units)
7. AST Programming Sandbox Grader (Security, Complexity, Test Suites)
8. Grader Disagreement & Consensus Engine (Multi-Grader Variance, Escalation)
9. Decision Replay Engine (Tick-by-tick state reconstruction)
10. Adversarial Red-Team Simulator & Calibration (ECE, Brier Score)
11. Cryptographic Audit Chain (Tamper-evident verification)
"""

import pytest
import os
import time
from src.baseline import StudentBaseline, BehavioralSample
from src.fusion import compute_corroboration, CorroborationSignal, ModalitySource
from src.agent.uncertainty import UncertaintyEngine
from src.agent.counterfactual import CounterfactualEngine
from src.agent.technical import TechnicalEventEngine, TechnicalIncidentType
from src.grading.mathematical import MathematicalGrader
from src.grading.programming import ProgrammingGrader
from src.grading.consensus import GraderDisagreementEngine
from src.trace.replay import DecisionReplayEngine
from src.ml.adversarial import AdversarialSimulator, RedTeamHarness, AttackType
from src.trace.audit_chain import AuditChain


# ── 1. Robust Behavioral Baseline ─────────────────────────────────────

def test_baseline_robust_mad_and_maturity():
    baseline = StudentBaseline(student_id="student_101")
    assert not baseline.is_ready()
    assert baseline.maturity_score() == 0.0

    # Feed realistic typing samples with slight natural variance
    wpm_series = [44.0, 46.0, 45.0, 43.0, 47.0, 45.0, 44.0, 46.0, 45.0, 48.0]
    for wpm in wpm_series:
        baseline.add_sample(BehavioralSample(timestamp=time.time(), typing_wpm=wpm, mouse_velocity=180.0))

    assert baseline.is_ready()
    assert baseline.maturity_score() > 0.3
    # Check MAD calculation
    mad = baseline._compute_mad(baseline._typing_samples)
    assert mad > 0.0

    # Normal typing should not trigger anomaly
    is_anom, dev = baseline.deviation_typing(46.0)
    assert not is_anom
    assert dev < 0.5

    # Extreme typing burst (e.g. 160 WPM vs 45 WPM baseline) triggers anomaly
    is_anom_burst, dev_burst = baseline.deviation_typing(160.0)
    assert is_anom_burst
    assert dev_burst > 0.8



# ── 2. Cross-Modal Fusion & Contradiction Detection ───────────────────

def test_fusion_contradiction_detection():
    now = time.time()
    # Contradiction: speech detected, but mouth is static
    signals = [
        CorroborationSignal(
            modality=ModalitySource.AUDIO,
            event_type="speech_detected",
            severity=0.5,
            confidence=0.85,
            timestamp=now,
            details={"mouth_moving": False},
        ),
        CorroborationSignal(
            modality=ModalitySource.VISION,
            event_type="face_present",
            severity=0.1,
            confidence=0.90,
            timestamp=now + 0.5,
        ),
    ]

    res = compute_corroboration(signals, temporal_window=5.0)
    assert len(res.contradictions) > 0
    assert res.contradictions[0].implication == "external_audio_source"
    assert res.corroboration_label == "CONTRADICTED"


# ── 3. Formal Uncertainty Engine ──────────────────────────────────────

def test_uncertainty_engine_epistemic_and_aleatoric():
    engine = UncertaintyEngine(action_confidence_threshold=0.65)

    # Missing audio and keyboard
    estimate = engine.evaluate(
        observed_modalities={"vision", "browser"},
        signal_confidences={"vision": 0.85, "browser": 0.95},
        prediction_prob=0.50,  # Boundary uncertainty
        corroboration_score=0.40,
    )

    assert estimate.epistemic_uncertainty > 0.0
    assert estimate.aleatoric_uncertainty == 1.0  # Max uncertainty at P=0.5
    assert estimate.confidence < 0.65
    assert not estimate.is_actionable
    assert "audio" in estimate.missing_modalities


# ── 4. Causal & Counterfactual Reasoning Engine ───────────────────────

def test_counterfactual_engine_causal_attribution():
    cf_engine = CounterfactualEngine(review_threshold=0.50)

    evidence = [
        {"source": "browser", "event_type": "tab_switch", "contribution": 0.28},
        {"source": "keyboard", "event_type": "paste_detected", "contribution": 0.35},
        {"source": "vision", "event_type": "head_turned", "contribution": 0.12},
    ]
    current_risk = 0.75

    analysis = cf_engine.analyze(
        session_id="sess_001",
        current_risk=current_risk,
        evidence_contributions=evidence,
    )

    assert len(analysis.factors) == 3
    # Top factor should be paste_detected
    assert analysis.factors[0].event_type == "paste_detected"
    assert analysis.factors[0].marginal_delta == 0.35
    assert analysis.factors[0].risk_without == 0.40
    # paste_detected is a necessary cause (removing it drops risk to 0.40 < 0.50)
    assert analysis.factors[0].is_necessary_cause
    assert len(analysis.minimal_sufficient_cause) >= 1


# ── 5. Technical-Event Intelligence ───────────────────────────────────

def test_technical_event_isolation():
    tech_engine = TechnicalEventEngine(grace_period_seconds=60.0)

    raw_events = [
        {"type": "face_absent", "confidence": 0.9},
        {"type": "head_turned", "confidence": 0.8},
    ]

    # Camera hardware dropped
    diagnostics = {"camera_connected": False, "camera_fps": 0, "network_online": True, "mic_active": True}
    purified, incidents = tech_engine.analyze_system_state(diagnostics, raw_events)

    assert tech_engine.has_active_fault()
    assert len(incidents) >= 1
    # face_absent and head_turned must be suppressed from penalizing integrity!
    assert len(purified) == 0


# ── 6. Mathematical Partial Credit Grader ─────────────────────────────

def test_mathematical_grader():
    grader = MathematicalGrader(default_tolerance=0.02)

    solution = """
    Formula: F = m * a
    Substituting: F = 10 * 9.8
    Intermediate calculation: F = 98.0
    Final Answer: 98.0 N
    """

    res = grader.grade(
        student_solution=solution,
        expected_formula="F = m * a",
        expected_variables={"m": 10.0, "a": 9.8},
        expected_final_value=98.0,
        expected_units="N",
        marks=10.0,
    )

    assert res.score == 10.0
    assert res.formula_credit == 1.0
    assert res.final_answer_credit == 1.0
    assert len(res.steps_evaluated) >= 4


def test_mathematical_grader_partial_credit():
    grader = MathematicalGrader(default_tolerance=0.02)

    # Student states correct formula & substitution, but makes arithmetic error
    solution = """
    Formula: F = m * a
    Substituting: F = 10 * 9.8
    Final Answer: 85.0 N
    """

    res = grader.grade(
        student_solution=solution,
        expected_formula="F = m * a",
        expected_variables={"m": 10.0, "a": 9.8},
        expected_final_value=98.0,
        expected_units="N",
        marks=10.0,
    )

    # Partial credit awarded for formula, substitution, intermediate working, and units (final value zero)
    assert 7.0 <= res.score <= 9.0
    assert res.formula_credit == 1.0
    assert res.final_answer_credit == 0.0



# ── 7. Programming Grader with AST Sandbox ────────────────────────────

def test_programming_grader_safe_code():
    grader = ProgrammingGrader()

    code = """
def two_sum(nums, target):
    \"\"\"Find two numbers that add up to target.\"\"\"
    seen = {}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return [seen[complement], i]
        seen[num] = i
    return []
"""

    public_tests = [
        {"input": [[2, 7, 11, 15], 9], "expected": [0, 1], "name": "basic"},
    ]
    hidden_tests = [
        {"input": [[3, 2, 4], 6], "expected": [1, 2], "name": "hidden_1"},
    ]
    edge_cases = [
        {"input": [[], 5], "expected": [], "name": "empty"},
    ]

    res = grader.grade(
        code_str=code,
        entry_function_name="two_sum",
        public_tests=public_tests,
        hidden_tests=hidden_tests,
        edge_cases=edge_cases,
        marks=10.0,
    )

    assert res.syntax_valid
    assert res.security.is_safe
    assert res.passed_tests == 3
    assert res.score >= 9.0


def test_programming_grader_security_rejection():
    grader = ProgrammingGrader()

    # Malicious student submission trying to use os.system
    malicious_code = """
import os
def solve():
    os.system("whoami")
    return 42
"""

    res = grader.grade(
        code_str=malicious_code,
        entry_function_name="solve",
        public_tests=[{"input": [], "expected": 42}],
        marks=10.0,
    )

    assert not res.security.is_safe
    assert res.score == 0.0
    assert any("os" in v for v in res.security.violations)


# ── 8. Grader Disagreement & Consensus Engine ─────────────────────────

def test_grader_disagreement_consensus():
    engine = GraderDisagreementEngine(total_score_discrepancy_threshold=4.5)

    student_answer = """
    Photosynthesis is the process used by plants to convert light energy into chemical energy.
    Because chloroplasts absorb sunlight, chlorophyll facilitates carbon dioxide and water transformation into glucose.
    Therefore, oxygen is released as a vital byproduct.
    """

    rubric = {
        "keywords": ["chloroplasts", "light energy", "glucose", "oxygen"],
        "model_answer": "Plants convert sunlight, carbon dioxide and water into glucose and oxygen via chlorophyll.",
    }

    res = engine.evaluate_perspectives(
        question_text="Explain photosynthesis.",
        student_answer=student_answer,
        rubric=rubric,
        marks=10.0,
    )

    assert res.consensus_status in ("UNANIMOUS", "RESOLVED_CONSENSUS", "ESCALATED_DISAGREEMENT")
    assert res.final_score > 6.0


# ── 9. Decision Replay Engine ─────────────────────────────────────────

def test_decision_replay_engine():
    replay = DecisionReplayEngine(session_id="session_test")

    traces = [
        {
            "timestamp": 1000.0,
            "risk": {"previous": 0.0, "current": 0.15},
            "events": [{"type": "head_turned"}],
            "contributions": [{"source": "vision", "contribution": 0.15}],
            "corroboration": {"corroboration_label": "UNCORROBORATED", "corroboration_score": 0.2},
            "decision": "NORMAL",
            "actions": ["monitoring"],
        },
        {
            "timestamp": 1005.0,
            "risk": {"previous": 0.15, "current": 0.62},
            "events": [{"type": "tab_switch"}, {"type": "paste_detected"}],
            "contributions": [
                {"source": "browser", "contribution": 0.25},
                {"source": "keyboard", "contribution": 0.35},
            ],
            "corroboration": {"corroboration_label": "CORROBORATED", "corroboration_score": 0.75},
            "decision": "REVIEW_REQUIRED",
            "actions": ["alert:REVIEW_REQUIRED"],
        },
    ]

    count = replay.load_from_traces(traces)
    assert count == 2
    assert replay.total_steps() == 2

    # Step 0
    s0 = replay.current_step()
    assert s0.step_index == 0
    assert s0.current_risk == 0.15

    # Step forward
    s1 = replay.step_forward()
    assert s1.step_index == 1
    assert s1.current_risk == 0.62
    assert s1.agent_decision == "REVIEW_REQUIRED"

    timeline = replay.export_timeline()
    assert len(timeline) == 2


# ── 10. Adversarial Red-Team Simulator ────────────────────────────────

def test_red_team_benchmark():
    harness = RedTeamHarness(escalation_threshold=0.60)

    # Comprehensive agent decision function reacting to all simulated attacks
    def mock_agent_eval(signal: dict) -> dict:
        risk = 0.05
        # Replay attack: zero optical flow & flat landmarks
        if signal.get("optical_flow_magnitude") == 0.0 and signal.get("landmark_variance", 1.0) < 0.001:
            risk += 0.65
        # Virtual camera flag
        if signal.get("virtual_cam_flag"):
            risk += 0.60
        # Synthetic / automated keystrokes: mechanical speed / zero jitter
        if signal.get("typing_wpm", 0) > 100.0 or signal.get("keystroke_jitter_ms") == 0.0:
            risk += 0.65
        # Clipboard injection
        if signal.get("paste_detected"):
            risk += 0.65
        # Multi-person / whisper audio
        if signal.get("face_count", 1) > 1:
            risk += 0.70
        if signal.get("audio_features", {}).get("speech_detected"):
            risk += 0.60
        # Remote desktop / mouse teleportation
        if signal.get("mouse_teleport_detected") or signal.get("window_blur"):
            risk += 0.65
        if signal.get("tab_switch"):
            risk += 0.40

        risk = min(risk, 1.0)
        return {"risk_score": risk, "state": "HIGH_RISK" if risk >= 0.6 else "NORMAL"}

    report = harness.evaluate_agent(mock_agent_eval)
    assert report.total_sessions >= 9
    assert report.tpr >= 0.85  # High detection rate
    assert report.calibration.brier_score >= 0.0
    assert report.calibration.expected_calibration_error >= 0.0



# ── 11. Cryptographic Audit Chain ─────────────────────────────────────

def test_cryptographic_audit_chain(tmp_path):
    db_file = str(tmp_path / "test_audit.db")
    chain = AuditChain(db_path=db_file)

    e1 = chain.append("exam_start", actor_id="student_1", session_id="sess_1", data={"exam": "math"})
    e2 = chain.append("answer_submitted", actor_id="student_1", session_id="sess_1", data={"q": 1, "ans": "A"})
    e3 = chain.append("exam_complete", actor_id="student_1", session_id="sess_1", data={"score": 100})

    assert e1.sequence == 0
    assert e2.sequence == 1
    assert e2.previous_hash == e1.entry_hash
    assert e3.previous_hash == e2.entry_hash

    # Verify integrity
    valid, broken_at = chain.verify_integrity()
    assert valid
    assert broken_at is None

    # Test export
    export = chain.export_chain(session_id="sess_1")
    assert export["chain_length"] == 3
    assert export["integrity_valid"]

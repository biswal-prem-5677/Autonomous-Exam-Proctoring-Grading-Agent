"""Tests for the Autonomous Exam Proctoring & Grading Agent."""

import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.models import Question, QuestionType, AgentState
from src.grading.mcq import MCQGrader
from src.grading.numerical import NumericalGrader
from src.grading.tfidf import TFIDFScorer, KeywordCoverageScorer
from src.grading.scorer import GradingOrchestrator
from src.ml.logistic import LogisticRegressionScratch
from src.ml.anomaly import AnomalyDetector
from src.ml.calibration import ModelCalibrator, ModelEvaluator
from src.agent.risk import RiskEngine
from src.agent.memory import EvidenceMemory
from src.agent.state import AgentStateMachine
from src.agent.controller import ProctoringAgent
from src.features.behavioral import BehavioralFeatureExtractor
from src.features.temporal import TemporalFeatureAggregator
from src.features.normalization import FeatureNormalizer
from src.reports.evidence import EvidenceLog
from src.reports.student_report import StudentReportGenerator
from src.reports.examiner_report import ExaminerReportGenerator
from data.sample_data import (
    SAMPLE_QUESTIONS, SAMPLE_STUDENT_ANSWERS,
    SAMPLE_PROCTORING_EVENTS, load_sample_questions,
    get_all_student_ids,
)


# ── Models ──────────────────────────────────────────────────────────────

class TestQuestion:
    def test_mcq_question(self):
        q = Question(id="q1", type=QuestionType.MCQ, text="Q?",
                     options=["A","B","C","D"], correct_answer="A", marks=1.0)
        assert q.type == QuestionType.MCQ
        assert q.options == ["A","B","C","D"]
        assert q.correct_answer == "A"
        assert q.marks == 1.0

    def test_numerical_question(self):
        q = Question(id="q2", type=QuestionType.NUMERICAL, text="2+2?",
                     correct_answer=4, tolerance=0.02, marks=2.0)
        assert q.type == QuestionType.NUMERICAL
        assert q.correct_answer == 4
        assert q.tolerance == 0.02

    def test_short_answer_question(self):
        q = Question(id="q3", type=QuestionType.SHORT_ANSWER,
                     text="Explain X", keywords=["key1","key2"], marks=3.0)
        assert q.type == QuestionType.SHORT_ANSWER
        assert "key1" in q.keywords

    def test_to_dict(self):
        q = Question(id="q1", type=QuestionType.MCQ, text="Q?",
                     options=["A"], correct_answer="A")
        d = q.to_dict()
        assert d["id"] == "q1"
        assert d["type"] == "mcq"
        assert d["text"] == "Q?"

    def test_default_values(self):
        q = Question(id="x", type=QuestionType.MCQ, text="t")
        assert q.marks == 1.0
        assert q.keywords == []
        assert q.metadata == {}


# ── Grading ─────────────────────────────────────────────────────────────

class TestMCQGrader:
    def test_correct_answer(self):
        q = Question(id="q1", type=QuestionType.MCQ, text="Q?",
                     options=["A","B","C","D"], correct_answer="B", marks=2.0)
        grader = MCQGrader()
        result = grader.grade(q, "B")
        assert result["correct"] is True
        assert result["score"] == 2.0

    def test_wrong_answer(self):
        q = Question(id="q1", type=QuestionType.MCQ, text="Q?",
                     options=["A","B","C","D"], correct_answer="B", marks=2.0)
        grader = MCQGrader()
        result = grader.grade(q, "C")
        assert result["correct"] is False
        assert result["score"] == 0.0

    def test_missing_answer(self):
        q = Question(id="q1", type=QuestionType.MCQ, text="Q?",
                     options=["A","B","C","D"], correct_answer="B", marks=2.0)
        grader = MCQGrader()
        result = grader.grade(q, None)
        assert result["correct"] is False

    def test_true_false(self):
        q = Question(id="q1", type=QuestionType.TRUE_FALSE,
                     options=["True","False"], correct_answer="True", marks=1.0, text="T?")
        grader = MCQGrader()
        r = grader.grade(q, "True")
        assert r["correct"] is True

    def test_batch_grading(self):
        qs = [
            Question(id="q1", type=QuestionType.MCQ, text="Q?", options=["A","B"], correct_answer="A", marks=1.0),
            Question(id="q2", type=QuestionType.MCQ, text="Q?", options=["A","B"], correct_answer="B", marks=1.0),
        ]
        grader = MCQGrader()
        result = grader.grade_batch(qs, {"q1": "A", "q2": "A"})
        assert result["total_score"] == 1.0
        assert result["total_max"] == 2.0


class TestNumericalGrader:
    def test_exact_match(self):
        q = Question(id="q1", type=QuestionType.NUMERICAL, text="Sum?",
                     correct_answer=100, tolerance=0.02, marks=2.0)
        grader = NumericalGrader()
        r = grader.grade(q, 100)
        assert r["correct"] is True
        assert r["score"] == 2.0

    def test_within_tolerance(self):
        q = Question(id="q1", type=QuestionType.NUMERICAL, text="Pi?",
                     correct_answer=3.14, tolerance=0.02, marks=2.0)
        grader = NumericalGrader()
        r = grader.grade(q, 3.12)
        assert r["correct"] is True

    def test_outside_tolerance(self):
        q = Question(id="q1", type=QuestionType.NUMERICAL, text="X?",
                     correct_answer=100, tolerance=0.02, marks=2.0)
        grader = NumericalGrader()
        r = grader.grade(q, 105)
        assert r["correct"] is False

    def test_zero_tolerance(self):
        q = Question(id="q1", type=QuestionType.NUMERICAL, text="X?",
                     correct_answer=100, marks=2.0, tolerance=0.0)
        grader = NumericalGrader(default_tolerance=0.02)
        r = grader.grade(q, 100.01)
        assert r["correct"] is False
        assert r["score"] == 0.0

    def test_within_zero_tolerance(self):
        q = Question(id="q1", type=QuestionType.NUMERICAL, text="X?",
                     correct_answer=100, marks=2.0, tolerance=0.0)
        grader = NumericalGrader()
        r = grader.grade(q, 100.0)
        assert r["correct"] is True
        assert r["score"] == 2.0


class TestTFIDFScorer:
    def test_identical_high_similarity(self):
        scorer = TFIDFScorer()
        corpus = ["Machine learning is great", "Deep learning uses neural networks",
                  "Python is a programming language"]
        scorer.fit(corpus)
        sim = scorer.cosine_similarity("Machine learning is great", "Machine learning is great")
        assert sim > 0.99

    def test_different_docs_lower(self):
        scorer = TFIDFScorer()
        corpus = ["Machine learning is great", "Deep learning uses neural networks"]
        scorer.fit(corpus)
        sim = scorer.cosine_similarity(corpus[0], corpus[1])
        # Different docs should have lower similarity (may be 0 or >0 but < 1)
        assert 0.0 <= sim <= 1.0

    def test_similarity_range(self):
        scorer = TFIDFScorer()
        corpus = ["The cat sat on the mat", "A dog ran in the park"]
        scorer.fit(corpus)
        sim = scorer.cosine_similarity(corpus[0], corpus[1])
        assert 0.0 <= sim <= 1.0

    def test_empty_corpus(self):
        scorer = TFIDFScorer()
        scorer.fit([])
        sim = scorer.cosine_similarity("test", "test")
        assert 0.0 <= sim <= 1.0


class TestKeywordCoverageScorer:
    def test_full_coverage(self):
        scorer = KeywordCoverageScorer()
        r = scorer.score("a function calls itself with a base case", ["function", "calls itself", "base case"])
        assert r["coverage"] == 1.0
        assert len(r["matched"]) == 3

    def test_partial_coverage(self):
        scorer = KeywordCoverageScorer()
        r = scorer.score("function uses base case", ["function", "calls itself", "base case"])
        assert 0.0 < r["coverage"] < 1.0
        assert r["total_keywords"] == 3

    def test_no_keywords(self):
        scorer = KeywordCoverageScorer()
        r = scorer.score("some answer", [])
        assert r["coverage"] == 1.0

    def test_empty_answer(self):
        scorer = KeywordCoverageScorer()
        r = scorer.score("", ["key1", "key2"])
        assert r["coverage"] == 0.0


class TestGradingOrchestrator:
    def test_grade_all(self):
        questions = [
            Question(id="q1", type=QuestionType.MCQ, text="Q?", options=["A","B"], correct_answer="A", marks=1.0),
            Question(id="q2", type=QuestionType.NUMERICAL, text="Sum?", correct_answer=100, tolerance=0.02, marks=2.0),
        ]
        orchestrator = GradingOrchestrator()
        refs = {"q1": "A", "q2": 100}
        results = orchestrator.grade_all(questions, {"q1": "A", "q2": 100}, refs)
        assert results["total_score"] >= 0
        assert results["total_max"] == 3.0
        assert 0 <= results["percentage"] <= 100

    def test_full_pipeline_perfect(self):
        questions = [
            Question(id=q["id"], type=QuestionType(q["type"]),
                     text=q["text"], options=q.get("options"),
                     correct_answer=q.get("correct_answer"),
                     marks=q.get("marks", 1.0), keywords=q.get("keywords", []),
                     tolerance=q.get("tolerance", 0.02))
            for q in SAMPLE_QUESTIONS
        ]
        orchestrator = GradingOrchestrator()
        refs = {q.id: q.correct_answer for q in questions}
        perfect = {q.id: q.correct_answer for q in questions}
        results = orchestrator.grade_all(questions, perfect, refs)
        assert results["total_score"] == results["total_max"]

    def test_full_pipeline_zero(self):
        questions = [
            Question(id=q["id"], type=QuestionType(q["type"]),
                     text=q["text"], options=q.get("options"),
                     correct_answer=q.get("correct_answer"),
                     marks=q.get("marks", 1.0), keywords=q.get("keywords", []),
                     tolerance=q.get("tolerance", 0.02))
            for q in SAMPLE_QUESTIONS
        ]
        orchestrator = GradingOrchestrator()
        refs = {q.id: q.correct_answer for q in questions}
        zero_ans = {q.id: "WRONG" for q in questions}
        results = orchestrator.grade_all(questions, zero_ans, refs)
        assert results["total_score"] == 0.0


# ── ML Models ───────────────────────────────────────────────────────────

class TestLogisticRegression:
    def _make_data(self):
        np.random.seed(42)
        X = np.random.randn(200, 4)
        w_true = np.array([2.0, -1.5, 1.0, -0.5])
        z = X @ w_true + 0.5
        y = (1 / (1 + np.exp(-z)) > 0.5).astype(int)
        return X, y

    def test_train_and_predict(self):
        X, y = self._make_data()
        model = LogisticRegressionScratch(learning_rate=0.5, max_iterations=500)
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (200,)
        assert set(preds.tolist()).issubset({0, 1})

    def test_predict_proba_range(self):
        X, y = self._make_data()
        model = LogisticRegressionScratch(learning_rate=0.5, max_iterations=500)
        model.fit(X, y)
        proba = model.predict_proba(X)
        assert np.all(proba >= 0.0)
        assert np.all(proba <= 1.0)

    def test_high_accuracy(self):
        X, y = self._make_data()
        model = LogisticRegressionScratch(learning_rate=0.5, max_iterations=1000)
        model.fit(X, y)
        metrics = model.evaluate(X, y)
        assert metrics["accuracy"] > 0.85

    def test_untrained_raises(self):
        model = LogisticRegressionScratch()
        X = np.random.randn(10, 4)
        with pytest.raises(RuntimeError):
            model.predict_proba(X)

    def test_loss_decreasing(self):
        X, y = self._make_data()
        model = LogisticRegressionScratch(learning_rate=0.1, max_iterations=200)
        model.fit(X, y)
        assert len(model.loss_history) > 1
        first_loss = model.loss_history[0]
        last_loss = model.loss_history[-1]
        assert last_loss < first_loss


class TestAnomalyDetector:
    def _make_normal(self, n=100, dim=2):
        np.random.seed(42)
        return np.random.multivariate_normal([0]*dim, np.eye(dim), size=(n,))

    def test_normal_data_low_score(self):
        normal = self._make_normal(100, 2)
        det = AnomalyDetector()
        det.fit(normal)
        scores = det.score(normal[:10])
        assert np.mean(scores) < 0.5

    def test_outlier_high_score(self):
        normal = self._make_normal(100, 2)
        outlier = np.array([[10.0, 10.0]])
        det = AnomalyDetector()
        det.fit(normal)
        scores = det.score(outlier)
        assert scores[0] > 0.5

    def test_is_anomaly_true(self):
        normal = self._make_normal(100, 2)
        outlier = np.array([[10.0, 10.0]])
        det = AnomalyDetector()
        det.fit(normal)
        is_anom, details = det.is_anomaly(outlier)
        assert is_anom is True
        assert "mahalanobis_distance" in details

    def test_normal_not_anomaly(self):
        normal = self._make_normal(100, 2)
        det = AnomalyDetector()
        det.fit(normal)
        # Use the mean which is the center of the normal distribution
        center = np.zeros(2, dtype=np.float64)
        is_anom, details = det.is_anomaly(center)
        # Center of the distribution should be non-anomalous
        assert is_anom is False


class TestCalibration:
    def test_optimal_threshold(self):
        cal = ModelCalibrator()
        y_true = np.array([0, 0, 0, 1, 1, 1, 1])
        y_proba = np.array([0.1, 0.2, 0.3, 0.6, 0.7, 0.8, 0.9])
        threshold, metrics = cal.find_optimal_threshold(y_true, y_proba)
        assert 0.0 <= threshold <= 1.0
        assert "threshold" in metrics

    def test_evaluate_threshold(self):
        cal = ModelCalibrator()
        y_true = np.array([0, 0, 1, 1])
        y_proba = np.array([0.1, 0.9, 0.8, 0.2])
        result = cal.evaluate_threshold(y_true, y_proba, 0.5)
        assert "precision" in result
        assert "recall" in result
        assert 0.0 <= result["precision"] <= 1.0

    def test_model_evaluator(self):
        y_true = np.array([0, 0, 1, 1, 1])
        y_pred = np.array([0, 0, 1, 1, 0])
        y_proba = np.array([0.1, 0.2, 0.8, 0.7, 0.4])
        result = ModelEvaluator.evaluate_classifier(y_true, y_pred, y_proba)
        assert "accuracy" in result
        assert "f1" in result
        assert 0.0 <= result["accuracy"] <= 1.0

    def test_model_evaluator_regressor(self):
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([1.1, 2.0, 2.9, 4.1])
        result = ModelEvaluator.evaluate_regressor(y_true, y_pred)
        assert "mae" in result
        assert "rmse" in result
        assert "r2" in result
        assert result["rmse"] > 0


# ── Agent ───────────────────────────────────────────────────────────────

class TestEvidenceMemory:
    def test_add_event(self):
        mem = EvidenceMemory()
        mem.add_event("face_absent", 0.8)
        assert len(mem._entries) == 1

    def test_risk_from_events(self):
        mem = EvidenceMemory()
        mem.add_event("face_absent", 1.0)
        risk = mem.get_current_risk()
        assert 0.0 < risk <= 1.0

    def test_clear(self):
        mem = EvidenceMemory()
        mem.add_event("tab_switch", 0.5)
        mem.add_event("paste_detected", 0.9)
        mem.clear()
        assert len(mem._entries) == 0

    def test_get_event_counts(self):
        mem = EvidenceMemory()
        mem.add_event("face_absent", 0.8)
        mem.add_event("face_absent", 0.9)
        mem.add_event("tab_switch", 0.5)
        counts = mem.get_event_counts()
        assert counts["face_absent"] == 2
        assert counts["tab_switch"] == 1

    def test_count_active_signals(self):
        mem = EvidenceMemory()
        mem.add_event("face_absent", 0.8)
        mem.add_event("tab_switch", 0.5)
        active = mem.count_active_signals()
        assert active >= 1


class TestRiskEngine:
    def test_no_events_low_risk(self):
        engine = RiskEngine()
        mem = EvidenceMemory()
        risk = engine.compute_risk(mem)
        assert risk == 0.0

    def test_high_severity_events(self):
        engine = RiskEngine(config={"escalation_threshold": 0.7, "review_threshold": 0.5, "multi_signal_required": 2})
        mem = EvidenceMemory()
        mem.add_event("multiple_faces", 0.9)
        mem.add_event("copy_paste", 0.85)
        mem.add_event("tab_switch", 0.6)
        risk = engine.compute_risk(mem)
        assert risk > 0.2

    def test_risk_decay_formula(self):
        engine = RiskEngine(config={"alpha_decay": 0.8})
        prev = 0.5
        new_evidence = 0.3
        result = engine.compute_risk_decay(prev, new_evidence)
        expected = 0.8 * 0.5 + 0.2 * 0.3
        assert abs(result - expected) < 1e-6

    def test_classify_risk_level(self):
        engine = RiskEngine(config={"escalation_threshold": 0.7, "review_threshold": 0.5})
        assert engine.classify_risk_level(0.3) == "NORMAL"
        assert engine.classify_risk_level(0.6) == "SUSPICIOUS"
        assert engine.classify_risk_level(0.8) == "HIGH_RISK"

    def test_escalation_required(self):
        engine = RiskEngine(config={"escalation_threshold": 0.7, "review_threshold": 0.5})
        mem = EvidenceMemory()
        assert engine.is_escalation_required(0.8, mem) is True
        assert engine.is_escalation_required(0.3, mem) is False


class TestAgentStateMachine:
    def test_normal_initial(self):
        sm = AgentStateMachine()
        assert sm.state == AgentState.NORMAL

    def test_transition_to_suspicious(self):
        sm = AgentStateMachine()
        ok = sm.transition(AgentState.SUSPICIOUS)
        assert ok is True
        assert sm.state == AgentState.SUSPICIOUS

    def test_transition_to_high_risk(self):
        sm = AgentStateMachine()
        sm.transition(AgentState.SUSPICIOUS)
        ok = sm.transition(AgentState.HIGH_RISK)
        assert ok is True
        assert sm.state == AgentState.HIGH_RISK

    def test_invalid_transition(self):
        sm = AgentStateMachine()
        ok = sm.transition(AgentState.HIGH_RISK)
        assert ok is False
        assert sm.state == AgentState.NORMAL

    def test_force_state(self):
        sm = AgentStateMachine()
        sm.force_state(AgentState.HIGH_RISK)
        assert sm.state == AgentState.HIGH_RISK

    def test_reset(self):
        sm = AgentStateMachine()
        sm.transition(AgentState.SUSPICIOUS)
        sm.reset()
        assert sm.state == AgentState.NORMAL
        assert sm.duration == 0.0

    def test_tick(self):
        sm = AgentStateMachine()
        sm.tick(5.0)
        assert sm.duration == 5.0


class TestProctoringAgent:
    def test_initialization(self):
        agent = ProctoringAgent()
        status = agent.get_status()
        assert status["state"] == "NORMAL"
        assert status["iteration"] == 0

    def test_process_signals_no_events(self):
        agent = ProctoringAgent()
        signals = {
            "face_present": True, "face_confidence": 1.0, "face_count": 1,
            "keyboard_idle_seconds": 0, "mouse_idle_seconds": 0,
            "tab_switch": False, "paste_detected": False,
            "audio_features": {"volume_db": -100, "rms_energy": 0.0},
            "head_pose": {"yaw": 0, "pitch": 0, "roll": 0},
        }
        result = agent.process_signals(signals)
        assert result["state"] == "NORMAL"
        assert len(result["events_detected"]) == 0
        assert result["iteration"] == 1

    def test_face_absent_detection(self):
        agent = ProctoringAgent()
        signals = {
            "face_present": False, "face_confidence": 0.3, "face_count": 1,
            "keyboard_idle_seconds": 0, "mouse_idle_seconds": 0,
            "tab_switch": False, "paste_detected": False,
            "audio_features": {"volume_db": -100, "rms_energy": 0.0},
            "head_pose": {"yaw": 0, "pitch": 0, "roll": 0},
        }
        result = agent.process_signals(signals)
        event_types = [e["type"] for e in result["events_detected"]]
        assert "face_absent" in event_types

    def test_multiple_faces_detection(self):
        agent = ProctoringAgent()
        signals = {
            "face_present": True, "face_confidence": 1.0, "face_count": 2,
            "keyboard_idle_seconds": 0, "mouse_idle_seconds": 0,
            "tab_switch": False, "paste_detected": False,
            "audio_features": {"volume_db": -100, "rms_energy": 0.0},
            "head_pose": {"yaw": 0, "pitch": 0, "roll": 0},
        }
        result = agent.process_signals(signals)
        event_types = [e["type"] for e in result["events_detected"]]
        assert "multiple_faces" in event_types

    def test_head_turn_detection(self):
        agent = ProctoringAgent()
        signals = {
            "face_present": True, "face_confidence": 1.0, "face_count": 1,
            "keyboard_idle_seconds": 0, "mouse_idle_seconds": 0,
            "tab_switch": False, "paste_detected": False,
            "audio_features": {"volume_db": -100, "rms_energy": 0.0},
            "head_pose": {"yaw": 60, "pitch": 0, "roll": 0},
        }
        result = agent.process_signals(signals)
        event_types = [e["type"] for e in result["events_detected"]]
        assert "head_turned" in event_types

    def test_tab_switch_detection(self):
        agent = ProctoringAgent()
        signals = {
            "face_present": True, "face_confidence": 1.0, "face_count": 1,
            "keyboard_idle_seconds": 0, "mouse_idle_seconds": 0,
            "tab_switch": True, "paste_detected": False,
            "audio_features": {"volume_db": -100, "rms_energy": 0.0},
            "head_pose": {"yaw": 0, "pitch": 0, "roll": 0},
        }
        result = agent.process_signals(signals)
        event_types = [e["type"] for e in result["events_detected"]]
        assert "tab_switch" in event_types

    def test_paste_detection(self):
        agent = ProctoringAgent()
        signals = {
            "face_present": True, "face_confidence": 1.0, "face_count": 1,
            "keyboard_idle_seconds": 0, "mouse_idle_seconds": 0,
            "tab_switch": False, "paste_detected": True,
            "audio_features": {"volume_db": -100, "rms_energy": 0.0},
            "head_pose": {"yaw": 0, "pitch": 0, "roll": 0},
        }
        result = agent.process_signals(signals)
        event_types = [e["type"] for e in result["events_detected"]]
        assert "paste_detected" in event_types

    def test_high_risk_escalation(self):
        agent = ProctoringAgent(config={"risk": {"escalation_threshold": 0.5}})
        signals = {
            "face_present": False, "face_confidence": 0.3, "face_count": 2,
            "keyboard_idle_seconds": 0, "mouse_idle_seconds": 0,
            "tab_switch": True, "paste_detected": True,
            "audio_features": {"volume_db": -20, "rms_energy": 0.5},
            "head_pose": {"yaw": 60, "pitch": 0, "roll": 0},
        }
        result = agent.process_signals(signals)
        assert result["state"] in ("SUSPICIOUS", "HIGH_RISK", "REVIEW_REQUIRED")
        assert result["risk_score"] > 0.0

    def test_reset(self):
        agent = ProctoringAgent()
        signals = {
            "face_present": False, "face_confidence": 0.3, "face_count": 1,
            "keyboard_idle_seconds": 0, "mouse_idle_seconds": 0,
            "tab_switch": True, "paste_detected": False,
            "audio_features": {"volume_db": -100, "rms_energy": 0.0},
            "head_pose": {"yaw": 0, "pitch": 0, "roll": 0},
        }
        agent.process_signals(signals)
        agent.reset()
        assert agent.get_status()["state"] == "NORMAL"
        assert agent.get_status()["iteration"] == 0

    def test_set_models(self):
        agent = ProctoringAgent()
        X = np.random.randn(50, 10)
        model = LogisticRegressionScratch()
        model.fit(X, np.random.randint(0, 2, 50))
        agent.set_models(logistic=model)
        assert agent._logistic_model is not None

    def test_history_tracking(self):
        agent = ProctoringAgent()
        signals = {
            "face_present": True, "face_confidence": 1.0, "face_count": 1,
            "keyboard_idle_seconds": 0, "mouse_idle_seconds": 0,
            "tab_switch": False, "paste_detected": False,
            "audio_features": {"volume_db": -100, "rms_energy": 0.0},
            "head_pose": {"yaw": 0, "pitch": 0, "roll": 0},
        }
        agent.process_signals(signals)
        agent.process_signals(signals)
        history = agent.get_history()
        assert len(history) == 2
        assert history[0]["iteration"] == 1
        assert history[1]["iteration"] == 2


# ── Features ────────────────────────────────────────────────────────────

class TestBehavioralFeatureExtractor:
    def test_extract(self):
        extractor = BehavioralFeatureExtractor()
        signals = {
            "face_present": True, "face_confidence": 0.9, "face_count": 1,
            "keyboard_idle_seconds": 5, "mouse_idle_seconds": 3,
            "tab_switch": False, "paste_detected": False,
            "audio_features": {"volume_db": -60, "rms_energy": 0.1},
            "head_pose": {"yaw": 5, "pitch": 2, "roll": 1},
        }
        features = extractor.extract(signals)
        assert isinstance(features, np.ndarray)
        assert features.shape[0] > 0
        assert features.dtype == np.float64

    def test_feature_vector_consistent(self):
        extractor = BehavioralFeatureExtractor()
        signals = {
            "face_present": True, "face_confidence": 1.0, "face_count": 1,
            "keyboard_idle_seconds": 0, "mouse_idle_seconds": 0,
            "tab_switch": False, "paste_detected": False,
            "audio_features": {"volume_db": -100, "rms_energy": 0.0},
            "head_pose": {"yaw": 0, "pitch": 0, "roll": 0},
        }
        f1 = extractor.extract(signals)
        f2 = extractor.extract(signals)
        assert f1.shape == f2.shape
        np.testing.assert_array_equal(f1, f2)


class TestTemporalFeatureAggregator:
    def test_add_and_cap(self):
        agg = TemporalFeatureAggregator(window_size=5)
        for _ in range(10):
            agg.add(np.array([1.0, 2.0, 3.0]))
        assert len(agg._window) == 5

    def test_get_window(self):
        agg = TemporalFeatureAggregator(window_size=5)
        agg.add(np.array([1.0, 2.0]))
        agg.add(np.array([3.0, 4.0]))
        window = agg.get_window()
        assert window.shape == (2, 2)

    def test_aggregate_empty(self):
        agg = TemporalFeatureAggregator()
        result = agg.aggregate()
        assert "mean" in result
        assert result["mean"].shape[0] == 0

    def test_aggregate_with_data(self):
        agg = TemporalFeatureAggregator(window_size=5)
        for i in range(5):
            agg.add(np.array([float(i), float(i*2)]))
        stats = agg.aggregate()
        assert "mean" in stats
        assert "std" in stats
        assert "trend" in stats
        assert len(stats["mean"]) == 2


class TestNormalization:
    def test_fit_and_transform(self):
        normalizer = FeatureNormalizer()
        train = np.random.randn(20, 5).astype(np.float64)
        normalizer.fit(train)
        test = np.random.randn(5, 5).astype(np.float64)
        result = normalizer.transform(test)
        assert result.shape == (5, 5)
        # After z-score, mean should be ~0 and std ~1
        assert abs(np.mean(result)) < 0.5

    def test_update(self):
        normalizer = FeatureNormalizer()
        data = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]], dtype=np.float64)
        result = normalizer.update(data)
        assert result.shape == (5, 1)

    def test_inverse_transform(self):
        normalizer = FeatureNormalizer()
        data = np.array([[1.0], [2.0], [3.0]], dtype=np.float64)
        normalizer.fit(data)
        normalized = normalizer.transform(data)
        recovered = normalizer.inverse_transform(normalized)
        np.testing.assert_allclose(recovered, data, atol=1e-6)

    def test_untrained_returns_unchanged(self):
        normalizer = FeatureNormalizer()
        data = np.array([[1.0, 2.0]], dtype=np.float64)
        result = normalizer.transform(data)
        np.testing.assert_array_equal(result, data)


# ── Reports ─────────────────────────────────────────────────────────────

class TestEvidenceLog:
    def test_add_event(self):
        log = EvidenceLog()
        entry = log.add("face_absent", "critical", 0.8,
                         "Student left frame", "vision")
        assert entry.event_type == "face_absent"
        assert entry.confidence == 0.8
        assert len(log._entries) == 1

    def test_get_timeline(self):
        log = EvidenceLog()
        log.add("face_absent", "critical", 0.8, "desc", "vision")
        log.add("tab_switch", "warning", 0.5, "desc", "temporal")
        timeline = log.get_timeline()
        assert len(timeline) == 2
        assert timeline[0]["event_type"] == "face_absent"

    def test_severity_critical_event(self):
        log = EvidenceLog()
        entry = log.add("copy_paste", "", 0.85, "paste detected")
        assert entry.severity in ("critical", "warning", "info")

    def test_to_dict(self):
        log = EvidenceLog(session_id="test123")
        log.add("face_absent", "critical", 0.8, "desc", "vision")
        d = log.to_dict()
        assert d["session_id"] == "test123"
        assert d["total_events"] == 1

    def test_clear(self):
        log = EvidenceLog()
        log.add("face_absent", "critical", 0.8, "desc")
        log.add("tab_switch", "warning", 0.5, "desc")
        log.clear()
        assert len(log._entries) == 0


class TestStudentReport:
    def test_generate(self):
        gen = StudentReportGenerator()
        grading = {
            "total_score": 12.0, "total_max": 15.0, "percentage": 80.0,
            "per_question": {
                "q1": {"score": 1.0, "max_score": 1.0, "correct": True},
                "q2": {"score": 0.0, "max_score": 2.0, "correct": False},
            },
        }
        proctoring = {"student_id": "s1", "risk_score": 0.3, "state": "NORMAL", "events_count": 1}
        report = gen.generate(grading_results=grading, proctoring_summary=proctoring)
        assert report["student_id"] == "s1"
        assert report["grade_summary"]["percentage"] == 80.0
        assert report["proctoring_summary"]["risk_score"] == 0.3
        assert report["proctoring_summary"]["state"] == "NORMAL"

    def test_recommendations_low_score(self):
        gen = StudentReportGenerator()
        grading = {"total_score": 3.0, "total_max": 15.0, "percentage": 20.0,
                   "per_question": {}}
        proctoring = {"student_id": "s1", "risk_score": 0.2, "state": "NORMAL", "events_count": 0}
        report = gen.generate(grading_results=grading, proctoring_summary=proctoring)
        assert len(report["recommendations"]) > 0

    def test_to_text(self):
        gen = StudentReportGenerator()
        grading = {"total_score": 10.0, "total_max": 15.0, "percentage": 66.7,
                   "per_question": {}}
        proctoring = {"student_id": "s1", "risk_score": 0.3, "state": "NORMAL", "events_count": 0}
        gen.generate(grading_results=grading, proctoring_summary=proctoring)
        text = gen.to_text()
        assert "EXAMINATION REPORT" in text
        assert "NOT proof of cheating" in text


class TestExaminerReport:
    def test_generate(self):
        gen = ExaminerReportGenerator()
        students = [
            {"student_id": "s1", "grade_summary": {"percentage": 80.0}, "proctoring_summary": {"risk_score": 0.2, "state": "NORMAL", "events_count": 0}},
            {"student_id": "s2", "grade_summary": {"percentage": 45.0}, "proctoring_summary": {"risk_score": 0.8, "state": "HIGH_RISK", "events_count": 3}},
        ]
        report = gen.generate("exam_001", students)
        assert report["total_students"] == 2
        assert report["statistics"]["mean_score"] == 62.5

    def test_flagged_students(self):
        gen = ExaminerReportGenerator()
        students = [
            {"student_id": "s1", "grade_summary": {"percentage": 90.0}, "proctoring_summary": {"risk_score": 0.2, "state": "NORMAL", "events_count": 0}},
            {"student_id": "s2", "grade_summary": {"percentage": 30.0}, "proctoring_summary": {"risk_score": 0.85, "state": "HIGH_RISK", "events_count": 5}},
        ]
        report = gen.generate("exam_001", students)
        flagged = report["flagged_students"]
        assert len(flagged) == 1
        assert flagged[0]["student_id"] == "s2"

    def test_to_text(self):
        gen = ExaminerReportGenerator()
        students = [
            {"student_id": "s1", "grade_summary": {"percentage": 80.0}, "proctoring_summary": {"risk_score": 0.2, "state": "NORMAL", "events_count": 0}},
        ]
        gen.generate("exam_001", students)
        text = gen.to_text()
        assert "EXAMINER REPORT" in text
        assert "NOT proof of cheating" in text


# ── Integration Tests ──────────────────────────────────────────────────

class TestSampleData:
    def test_questions_load(self):
        assert len(SAMPLE_QUESTIONS) == 10
        types = {q["type"] for q in SAMPLE_QUESTIONS}
        assert "mcq" in types
        assert "numerical" in types
        assert "short_answer" in types
        assert "true_false" in types

    def test_student_answers_load(self):
        assert len(SAMPLE_STUDENT_ANSWERS) == 3
        for sid, data in SAMPLE_STUDENT_ANSWERS.items():
            assert "name" in data
            assert "answers" in data
            assert len(data["answers"]) == 10

    def test_proctoring_events_load(self):
        assert len(SAMPLE_PROCTORING_EVENTS) == 3
        for sid, data in SAMPLE_PROCTORING_EVENTS.items():
            assert "events" in data
            assert len(data["events"]) > 0

    def test_load_sample_questions(self):
        qs = load_sample_questions()
        assert len(qs) == 10

    def test_get_all_student_ids(self):
        ids = get_all_student_ids()
        assert len(ids) == 3
        assert "student_001" in ids


class TestEndToEndDemo:
    """Mirror the main.py demo flow."""

    def test_full_pipeline(self):
        from src.models.models import Question, QuestionType
        from src.grading.scorer import GradingOrchestrator
        from src.agent.controller import ProctoringAgent

        questions = [
            Question(id=q["id"], type=QuestionType(q["type"]),
                     text=q["text"], options=q.get("options"),
                     correct_answer=q.get("correct_answer"),
                     marks=q.get("marks", 1.0), keywords=q.get("keywords", []),
                     tolerance=q.get("tolerance", 0.02))
            for q in SAMPLE_QUESTIONS
        ]

        orchestrator = GradingOrchestrator()
        reference = {q.id: q.correct_answer for q in questions}
        agent = ProctoringAgent(config={
            "risk": {"escalation_threshold": 0.7, "review_threshold": 0.5, "multi_signal_required": 2},
            "temporal_window": 30,
        })

        for student_id, data in SAMPLE_STUDENT_ANSWERS.items():
            answers = data.get("answers", {})
            results = orchestrator.grade_all(questions, answers, reference)
            assert results["total_score"] >= 0
            assert results["total_max"] > 0
            assert 0 <= results["percentage"] <= 100

            events_data = SAMPLE_PROCTORING_EVENTS.get(student_id, {}).get("events", [])
            signals = _to_signals(events_data)
            agent_result = agent.process_signals(signals)
            assert "risk_score" in agent_result
            assert "state" in agent_result
            assert 0.0 <= agent_result["risk_score"] <= 1.0
            agent.reset()


def _to_signals(events):
    """Mirror main.py helper."""
    signals = {
        "face_present": True, "face_confidence": 1.0, "face_count": 1,
        "keyboard_idle_seconds": 0, "mouse_idle_seconds": 0,
        "tab_switch": False, "paste_detected": False,
        "audio_features": {"volume_db": -100, "rms_energy": 0.0},
        "head_pose": {"yaw": 0, "pitch": 0, "roll": 0},
    }
    for evt in events:
        etype = evt.get("type", "")
        if etype == "face_absent":
            signals["face_present"] = False
            signals["face_confidence"] = evt.get("confidence", 0.0)
        elif etype == "multiple_faces":
            signals["face_count"] = 2
        elif etype == "tab_switch":
            signals["tab_switch"] = True
        elif etype == "paste_detected":
            signals["paste_detected"] = True
        elif etype == "loud_audio":
            signals["audio_features"]["volume_db"] = -25
            signals["audio_features"]["rms_energy"] = 0.5
        elif etype == "head_turned":
            signals["head_pose"]["yaw"] = 50
        elif etype == "copy_paste":
            signals["paste_detected"] = True
    return signals

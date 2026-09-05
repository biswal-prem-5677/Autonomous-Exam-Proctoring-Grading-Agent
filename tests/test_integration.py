"""End-to-end integration tests matching the actual API signatures."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import numpy as np


# ─── Grading Pipeline ──────────────────────────────────────────────────────────

class TestGradingPipeline:
    """End-to-end grading: MCQ + Numerical + Short Answer + Long Answer."""

    def test_mcq_correct(self):
        from src.grading.mcq import MCQGrader
        from src.models.models import Question, QuestionType
        grader = MCQGrader()
        q = Question(id="q1", type=QuestionType.MCQ, text="2+2?",
                    options=["3", "4", "5"], correct_answer="4", marks=1.0)
        result = grader.grade(q, "4")
        assert result["score"] == 1.0

    def test_mcq_wrong(self):
        from src.grading.mcq import MCQGrader
        from src.models.models import Question, QuestionType
        grader = MCQGrader()
        q = Question(id="q1", type=QuestionType.MCQ, text="2+2?",
                    options=["3", "4", "5"], correct_answer="4", marks=1.0)
        result = grader.grade(q, "3")
        assert result["score"] == 0.0

    def test_numerical_correct(self):
        from src.grading.numerical import NumericalGrader
        from src.models.models import Question, QuestionType
        grader = NumericalGrader()
        q = Question(id="q1", type=QuestionType.NUMERICAL, text="2+2",
                    correct_answer="4", marks=1.0, tolerance=0.01)
        result = grader.grade(q, "4")
        assert result["score"] == 1.0

    def test_numerical_within_tolerance(self):
        from src.grading.numerical import NumericalGrader
        from src.models.models import Question, QuestionType
        grader = NumericalGrader(default_tolerance=0.1)
        q = Question(id="q1", type=QuestionType.NUMERICAL, text="sqrt(16)",
                    correct_answer="4", marks=1.0, tolerance=0.1)
        result = grader.grade(q, "4.05")
        assert result["score"] == 1.0

    def test_short_answer_tfidf(self):
        from src.grading.tfidf import TFIDFScorer
        ref = "Newton's first law states that an object at rest stays at rest"
        ans = "An object at rest will stay at rest according to Newton's first law"
        tfidf = TFIDFScorer()
        tfidf.fit([ref])
        sim = tfidf.cosine_similarity(ans, ref)
        assert sim > 0.5

    def test_long_answer(self):
        from src.grading.long_answer import LongAnswerGrader
        grader = LongAnswerGrader()
        ref = "Photosynthesis converts light energy into chemical energy"
        ans = "Photosynthesis is the process by which plants convert light energy into chemical energy stored in glucose"
        result = grader.grade(ans, ref, keywords=["photosynthesis", "energy", "glucose"], marks=10)
        assert result.final_score > 0


# ─── ML Pipeline ──────────────────────────────────────────────────────────────

class TestMLPipeline:
    """ML training and prediction."""

    def test_logistic_regression(self):
        from src.ml.logistic import LogisticRegressionScratch
        X = np.random.randn(100, 5)
        y = (X[:, 0] + X[:, 1] > 0).astype(float)
        model = LogisticRegressionScratch(learning_rate=0.1, max_iterations=200)
        model.fit(X, y)
        proba = model.predict_proba(X[:5])
        assert proba.shape == (5,)
        assert all(0 <= p <= 1 for p in proba)

    def test_anomaly_zscore(self):
        from src.ml.anomaly import AnomalyDetector
        normal = np.random.randn(50, 3)
        detector = AnomalyDetector(zscore_threshold=2.5)
        detector.fit(normal)
        scores = detector.score(normal[:5])
        assert all(s < 0.5 for s in scores)

    def test_anomaly_mahalanobis(self):
        from src.ml.anomaly import AnomalyDetector
        normal = np.random.randn(50, 3) * 0.5
        detector = AnomalyDetector(mahalanobis_confidence=0.95)
        detector.fit(normal)
        scores = detector.score(normal[:5])
        assert all(s < 1.0 for s in scores)

    def test_training_data_generator(self):
        from src.ml.training_data import TrainingDataGenerator
        gen = TrainingDataGenerator(seed=42)
        X, y = gen.generate_dataset(n_normal=30, n_suspicious=20)
        assert X.shape[0] == 50
        assert set(np.unique(y)).issubset({0.0, 1.0})

    def test_performance_predictor(self):
        from src.prediction.performance import PerformancePredictor
        np.random.seed(42)
        X = np.random.randn(50, 4)
        y = 50 + 20 * X[:, 0] + np.random.randn(50) * 5
        predictor = PerformancePredictor()
        predictor.fit(X, y)
        pred = predictor.predict(X[:3])
        assert pred.shape == (3,)

    def test_performance_predictor_with_confidence(self):
        from src.prediction.performance import PerformancePredictor
        np.random.seed(42)
        X = np.random.randn(50, 4)
        y = 50 + 20 * X[:, 0] + np.random.randn(50) * 5
        predictor = PerformancePredictor()
        predictor.fit(X, y)
        result = predictor.predict_with_confidence(X[:3])
        assert "predicted_score" in result
        assert "confidence" in result
        assert isinstance(result["predicted_score"], float)


# ─── Risk Engine ──────────────────────────────────────────────────────────────

class TestRiskEngine:
    """Risk computation."""

    def test_risk_decay_formula(self):
        from src.agent.risk import RiskEngine
        engine = RiskEngine({"alpha_decay": 0.8})
        r = engine.compute_risk_decay(0.5, 0.3)
        assert 0 <= r <= 1

    def test_classify_risk_normal(self):
        from src.agent.risk import RiskEngine
        engine = RiskEngine({"escalation_threshold": 0.7, "review_threshold": 0.5})
        assert engine.classify_risk_level(0.3) == "NORMAL"

    def test_classify_risk_suspicious(self):
        from src.agent.risk import RiskEngine
        engine = RiskEngine({"escalation_threshold": 0.7, "review_threshold": 0.5})
        assert engine.classify_risk_level(0.6) == "SUSPICIOUS"

    def test_classify_risk_high(self):
        from src.agent.risk import RiskEngine
        engine = RiskEngine({"escalation_threshold": 0.7, "review_threshold": 0.5})
        assert engine.classify_risk_level(0.8) == "HIGH_RISK"

    def test_compute_risk_with_memory(self):
        from src.agent.risk import RiskEngine
        from src.agent.memory import EvidenceMemory
        engine = RiskEngine({"alpha_decay": 0.8})
        memory = EvidenceMemory(window_seconds=300, alpha=0.8)
        memory.add_event("face_absent", 0.9)
        memory.add_event("head_turned", 0.7)
        risk = engine.compute_risk(memory)
        assert 0 <= risk <= 1


# ─── Evidence Memory ──────────────────────────────────────────────────────────

class TestEvidenceMemory:
    """Evidence memory operations."""

    def test_add_event(self):
        from src.agent.memory import EvidenceMemory
        mem = EvidenceMemory(window_seconds=300, alpha=0.8)
        mem.add_event("face_absent", 0.9)
        assert mem.get_current_risk() > 0

    def test_event_counts(self):
        from src.agent.memory import EvidenceMemory
        mem = EvidenceMemory(window_seconds=300, alpha=0.8)
        mem.add_event("face_absent", 0.9)
        mem.add_event("face_absent", 0.8)
        mem.add_event("head_turned", 0.7)
        counts = mem.get_event_counts()
        assert counts.get("face_absent") == 2
        assert counts.get("head_turned") == 1

    def test_signal_summary(self):
        from src.agent.memory import EvidenceMemory
        mem = EvidenceMemory(window_seconds=300, alpha=0.8)
        mem.add_event("face_absent", 0.9)
        mem.add_event("tab_switch", 0.6)
        summary = mem.get_signal_summary()
        assert "vision" in summary
        assert "temporal" in summary

    def test_count_active_signals(self):
        from src.agent.memory import EvidenceMemory
        mem = EvidenceMemory(window_seconds=300, alpha=0.8)
        mem.add_event("face_absent", 0.9)
        mem.add_event("multiple_faces", 0.8)
        active = mem.count_active_signals(threshold=1)
        assert active >= 1

    def test_decay_over_time(self):
        from src.agent.memory import EvidenceMemory
        import time
        mem = EvidenceMemory(window_seconds=300, alpha=0.8)
        mem.add_event("face_absent", 0.9)
        risk_before = mem.get_current_risk()
        time.sleep(0.05)
        mem.add_event("tab_switch", 0.3)
        # After adding a low-confidence event, risk should be bounded
        risk_after = mem.get_current_risk()
        assert 0 <= risk_after <= 1

    def test_clear(self):
        from src.agent.memory import EvidenceMemory
        mem = EvidenceMemory(window_seconds=300, alpha=0.8)
        mem.add_event("face_absent", 0.9)
        mem.clear()
        assert mem.get_current_risk() == 0.0
        assert len(mem.get_recent_events()) == 0


# ─── Agent Controller ─────────────────────────────────────────────────────────

class TestAgent:
    """Agent controller."""

    def test_agent_initialization(self):
        from src.agent.controller import ProctoringAgent
        agent = ProctoringAgent({"risk": {"escalation_threshold": 0.7}})
        status = agent.get_status()
        assert "state" in status
        assert status["state"] == "NORMAL"

    def test_agent_process_signals_normal(self):
        from src.agent.controller import ProctoringAgent
        agent = ProctoringAgent({"risk": {"escalation_threshold": 0.7}})
        signals = {
            "face_present": True,
            "face_count": 1,
            "head_pose": {"yaw": 0, "pitch": 0},
            "audio_features": {"volume_db": -50},
            "keyboard_idle_seconds": 0,
            "paste_detected": False,
            "mouse_idle_seconds": 0,
            "tab_switch": False,
        }
        result = agent.process_signals(signals)
        assert "risk_score" in result
        assert "state" in result
        assert "events_detected" in result
        assert "iteration" in result

    def test_agent_detects_face_absent(self):
        from src.agent.controller import ProctoringAgent
        agent = ProctoringAgent({"risk": {"escalation_threshold": 0.7}})
        signals = {
            "face_present": False,
            "face_confidence": 0.3,
            "face_count": 0,
            "head_pose": {},
            "audio_features": {"volume_db": -60},
            "keyboard_idle_seconds": 0,
            "paste_detected": False,
            "mouse_idle_seconds": 0,
            "tab_switch": False,
        }
        result = agent.process_signals(signals)
        event_types = [e["type"] for e in result["events_detected"]]
        assert "face_absent" in event_types

    def test_agent_detects_multiple_faces(self):
        from src.agent.controller import ProctoringAgent
        agent = ProctoringAgent({"risk": {"escalation_threshold": 0.7}})
        signals = {
            "face_present": True,
            "face_count": 2,
            "face_confidence": 0.9,
            "head_pose": {},
            "audio_features": {"volume_db": -60},
            "keyboard_idle_seconds": 0,
            "paste_detected": False,
            "mouse_idle_seconds": 0,
            "tab_switch": False,
        }
        result = agent.process_signals(signals)
        event_types = [e["type"] for e in result["events_detected"]]
        assert "multiple_faces" in event_types

    def test_agent_detects_paste(self):
        from src.agent.controller import ProctoringAgent
        agent = ProctoringAgent({"risk": {"escalation_threshold": 0.7}})
        signals = {
            "face_present": True,
            "face_count": 1,
            "head_pose": {},
            "audio_features": {"volume_db": -60},
            "keyboard_idle_seconds": 0,
            "paste_detected": True,
            "mouse_idle_seconds": 0,
            "tab_switch": False,
        }
        result = agent.process_signals(signals)
        event_types = [e["type"] for e in result["events_detected"]]
        assert "paste_detected" in event_types

    def test_agent_history(self):
        from src.agent.controller import ProctoringAgent
        agent = ProctoringAgent({"risk": {"escalation_threshold": 0.7}})
        signals = {
            "face_present": True, "face_count": 1,
            "head_pose": {}, "audio_features": {"volume_db": -60},
            "keyboard_idle_seconds": 0, "paste_detected": False,
            "mouse_idle_seconds": 0, "tab_switch": False,
        }
        agent.process_signals(signals)
        agent.process_signals(signals)
        history = agent.get_history()
        assert len(history) == 2

    def test_agent_reset(self):
        from src.agent.controller import ProctoringAgent
        agent = ProctoringAgent({"risk": {"escalation_threshold": 0.7}})
        signals = {
            "face_present": True, "face_count": 1,
            "head_pose": {}, "audio_features": {"volume_db": -60},
            "keyboard_idle_seconds": 0, "paste_detected": False,
            "mouse_idle_seconds": 0, "tab_switch": False,
        }
        agent.process_signals(signals)
        agent.reset()
        assert len(agent.get_history()) == 0


# ─── Proctoring Vision & Features ─────────────────────────────────────────────

class TestProctoring:
    """Vision and behavioral features."""

    def test_face_tracker(self):
        from src.vision.face import FaceTracker
        tracker = FaceTracker()
        assert tracker is not None

    def test_landmark_analyzer(self):
        from src.vision.landmarks import LandmarkAnalyzer
        analyzer = LandmarkAnalyzer()
        pts = [(0.5, 0.5)] * 468
        result = analyzer.analyze(pts)
        assert result is not None

    def test_movement_detector(self):
        from src.vision.movement import MovementDetector
        detector = MovementDetector()
        pts = [
            (0.50, 0.47), (0.40, 0.45), (0.60, 0.45),
            (0.42, 0.55), (0.58, 0.55),
        ]
        result = detector.process_landmarks(pts)
        assert "posture" in result

    def test_behavioral_features(self):
        from src.features.behavioral import BehavioralFeatureExtractor
        extractor = BehavioralFeatureExtractor()
        features = extractor.extract({})
        assert features is not None


# ─── Knowledge Tracing ─────────────────────────────────────────────────────────

class TestKnowledgeTracing:
    """Knowledge tracing."""

    def test_knowledge_tracer_update_correct(self):
        from src.prediction.performance import KnowledgeTracer
        tracer = KnowledgeTracer()
        tracer.initialize_topic("algebra")
        m = tracer.update("algebra", correct=True)
        assert 0 <= m <= 1
        assert m > 0.5  # Should increase from initial 0.5

    def test_knowledge_tracer_update_wrong(self):
        from src.prediction.performance import KnowledgeTracer
        tracer = KnowledgeTracer()
        tracer.initialize_topic("algebra")
        m = tracer.update("algebra", correct=False)
        assert 0 <= m <= 1
        assert m < 0.5  # Should decrease from initial 0.5

    def test_knowledge_tracer_get_mastery(self):
        from src.prediction.performance import KnowledgeTracer
        tracer = KnowledgeTracer()
        tracer.initialize_topic("algebra")
        tracer.update("algebra", correct=True)
        tracer.update("algebra", correct=True)
        m = tracer.get_mastery("algebra")
        assert 0 <= m <= 1
        assert m > 0.5

    def test_knowledge_tracer_weak_topics(self):
        from src.prediction.performance import KnowledgeTracer
        tracer = KnowledgeTracer()
        tracer.initialize_topic("algebra")
        tracer.initialize_topic("calculus")
        # Several correct for algebra
        for _ in range(5):
            tracer.update("algebra", correct=True)
        # Several wrong for calculus
        for _ in range(5):
            tracer.update("calculus", correct=False)
        weak = tracer.get_weak_topics(threshold=0.4)
        assert "calculus" in weak
        assert "algebra" not in weak

    def test_difficulty_estimator(self):
        from src.prediction.performance import DifficultyEstimator
        est = DifficultyEstimator()
        # 20 correct out of 50 → difficulty = 1 - 20/50 = 0.6
        diff = est.estimate_from_responses("q1", correct_count=20, total_count=50)
        assert 0 <= diff <= 1
        assert abs(diff - 0.6) < 0.01


# ─── Database ─────────────────────────────────────────────────────────────────

class TestDatabase:
    """Database operations."""

    def test_db_init(self, tmp_path):
        from src.database.db import Database
        db = Database(tmp_path / "test.db")
        sid = db.create_session("stu1", "exam1")
        assert sid is not None
        session = db.get_session(sid)
        assert session["student_id"] == "stu1"
        db.close()

    def test_db_questions(self, tmp_path):
        from src.database.db import Database
        db = Database(tmp_path / "test2.db")
        sid = db.create_session("stu1", "exam1")
        db.add_question(sid, "q1", "mcq", "2+2?", marks=1.0)
        qs = db.get_questions(sid)
        assert len(qs) == 1
        db.close()

    def test_db_events(self, tmp_path):
        from src.database.db import Database
        db = Database(tmp_path / "test3.db")
        sid = db.create_session("stu1", "exam1")
        db.add_event(sid, "face_detected", 0.9)
        events = db.get_events(sid)
        assert len(events) == 1
        db.close()

    def test_db_answers(self, tmp_path):
        from src.database.db import Database
        db = Database(tmp_path / "test4.db")
        sid = db.create_session("stu1", "exam1")
        db.add_question(sid, "q1", "mcq", "2+2?", marks=1.0)
        db.record_answer(sid, "q1", "4", score=1.0, max_score=1.0)
        results = db.get_session_results(sid)
        assert results["total_score"] == 1.0
        db.close()

    def test_db_event_counts(self, tmp_path):
        from src.database.db import Database
        db = Database(tmp_path / "test5.db")
        sid = db.create_session("stu1", "exam1")
        db.add_event(sid, "face_detected", 0.9)
        db.add_event(sid, "face_detected", 0.8)
        db.add_event(sid, "tab_switch", 0.6)
        counts = db.get_event_counts(sid)
        assert counts["face_detected"] == 2
        assert counts["tab_switch"] == 1
        db.close()

    def test_db_predictions(self, tmp_path):
        from src.database.db import Database
        db = Database(tmp_path / "test6.db")
        sid = db.create_session("stu1", "exam1")
        db.save_prediction(sid, predicted_score=85.0, confidence=0.9,
                          method="logistic", question_id="q1")
        # Verify via session
        session = db.get_session(sid)
        assert session is not None
        db.close()


# ─── Reports ──────────────────────────────────────────────────────────────────

class TestReports:
    """Report generation."""

    def test_student_report(self):
        from src.reports.student_report import StudentReportGenerator
        gen = StudentReportGenerator()
        report = gen.generate(
            grading_results={"total_score": 80, "total_max": 100, "percentage": 80},
            proctoring_summary={"risk_score": 0.3}
        )
        assert report is not None
        assert report["grade_summary"]["percentage"] == 80

    def test_examiner_report(self):
        from src.reports.examiner_report import ExaminerReportGenerator
        gen = ExaminerReportGenerator()
        report = gen.generate("exam1", [
            {"student_id": "s1", "grade_summary": {"percentage": 80},
             "proctoring_summary": {"risk_score": 0.3, "state": "normal"}}
        ])
        assert report is not None
        assert report["total_students"] == 1

    def test_student_report_risk_grade(self):
        from src.reports.student_report import StudentReportGenerator
        gen = StudentReportGenerator()
        report = gen.generate(
            grading_results={"total_score": 40, "total_max": 100, "percentage": 40},
            proctoring_summary={"risk_score": 0.85, "state": "high"}
        )
        assert report is not None

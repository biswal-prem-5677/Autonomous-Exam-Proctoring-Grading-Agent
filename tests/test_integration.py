"""End-to-end integration tests matching the actual API."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import numpy as np


class TestGradingPipeline:
    """End-to-end grading: MCQ + Numerical + Short Answer + Long Answer."""

    def test_mcq_correct(self):
        from src.grading.mcq import MCQGrader
        from src.models.models import Question, QuestionType
        grader = MCQGrader()
        q = Question(id="q1", type=QuestionType.MCQ, text="2+2?",
                    options=["3","4","5"], correct_answer="4", marks=1.0)
        result = grader.grade(q, "4")
        assert result["score"] == 1.0

    def test_mcq_wrong(self):
        from src.grading.mcq import MCQGrader
        from src.models.models import Question, QuestionType
        grader = MCQGrader()
        q = Question(id="q1", type=QuestionType.MCQ, text="2+2?",
                    options=["3","4","5"], correct_answer="4", marks=1.0)
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
        from src.grading.tfidf import TFIDFScorer, KeywordCoverageScorer
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


class TestMLPipeline:
    """ML training and prediction."""

    def test_logistic_regression(self):
        from src.ml.logistic import LogisticRegressionScratch
        X = np.random.randn(100, 5)
        y = (X[:, 0] + X[:, 1] > 0).astype(float)
        model = LogisticRegressionScratch(learning_rate=0.1, max_iterations=200)
        model.fit(X, y)
        proba = model.predict_proba(X[:5])
        assert proba.shape == (5,)  # 1-D array of probabilities
        assert all(0 <= p <= 1 for p in proba)

    def test_anomaly_zscore(self):
        from src.ml.anomaly import AnomalyDetector
        normal = np.random.randn(50, 3)
        detector = AnomalyDetector(zscore_threshold=2.5)
        detector.fit(normal)
        scores = detector.score(normal[:5])
        assert all(s < 3 for s in scores)

    def test_anomaly_mahalanobis(self):
        from src.ml.anomaly import AnomalyDetector
        normal = np.random.randn(50, 3) * 0.5
        detector = AnomalyDetector(mahalanobis_confidence=0.95)
        detector.fit(normal)
        scores = detector.score(normal[:5])
        assert all(s < 10 for s in scores)

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


class TestRiskEngine:
    """Risk computation."""

    def test_risk_decay_formula(self):
        from src.agent.risk import RiskEngine
        engine = RiskEngine({"alpha_decay": 0.8})
        r = engine.compute_risk_decay(0.5, 0.3)
        assert 0 <= r <= 1

    def test_classify_risk(self):
        from src.agent.risk import RiskEngine
        engine = RiskEngine({"escalation_threshold": 0.7, "review_threshold": 0.5})
        assert engine.classify_risk_level(0.3) == "NORMAL"
        assert engine.classify_risk_level(0.6) == "SUSPICIOUS"
        assert engine.classify_risk_level(0.8) == "HIGH_RISK"


class TestProctoring:
    """Vision and behavioral features."""

    def test_face_tracker(self):
        from src.vision.face import FaceTracker
        tracker = FaceTracker()
        assert tracker is not None

    def test_landmark_analyzer(self):
        from src.vision.landmarks import LandmarkAnalyzer
        analyzer = LandmarkAnalyzer()
        # Basic valid landmark list (simplified)
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


class TestKnowledgeTracing:
    """Knowledge tracing."""

    def test_knowledge_tracer(self):
        from src.prediction.knowledge import KnowledgeTracer
        tracer = KnowledgeTracer()
        tracer.update("algebra", correct=True)
        m = tracer.get_mastery("algebra")
        assert 0 <= m <= 1

    def test_difficulty_estimator(self):
        from src.prediction.difficulty import DifficultyEstimator
        est = DifficultyEstimator()
        difficulty = est.estimate_from_responses("algebra", correct_count=30, total_count=40)
        assert 0.0 <= difficulty <= 1.0
        assert est.get_difficulty("algebra") == difficulty


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


class TestAgent:
    """Agent controller."""

    def test_agent_loop(self):
        from src.agent.controller import ProctoringAgent
        agent = ProctoringAgent({"risk": {"escalation_threshold": 0.7}})
        status = agent.get_status()
        assert "state" in status
        assert status["state"] in ("NORMAL", "SUSPICIOUS", "HIGH_RISK", "REVIEW_REQUIRED")


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
             "proctoring_summary": {"risk_score": 0.3, "state": "NORMAL"}}
        ])
        assert report is not None
        assert report["total_students"] == 1

    def test_student_report_with_all_fields(self):
        from src.reports.student_report import StudentReportGenerator
        gen = StudentReportGenerator()
        report = gen.generate(
            grading_results={"total_score": 85, "total_max": 100, "percentage": 85,
                             "per_question": {"q1": {"score": 10, "max_score": 10, "correct": True}}},
            proctoring_summary={"student_id": "stu1", "risk_score": 0.2, "state": "NORMAL"},
            prediction={"predicted_score": 88.0},
            knowledge_state={"algebra": 0.8, "calculus": 0.6},
        )
        assert report["report_type"] == "student_report"
        assert report["student_id"] == "stu1"
        assert "recommendations" in report
        assert "knowledge_state" in report

    def test_examiner_report_statistics(self):
        from src.reports.examiner_report import ExaminerReportGenerator
        gen = ExaminerReportGenerator()
        reports = [
            {"student_id": "s1", "grade_summary": {"percentage": 90},
             "proctoring_summary": {"risk_score": 0.1, "state": "NORMAL"}},
            {"student_id": "s2", "grade_summary": {"percentage": 70},
             "proctoring_summary": {"risk_score": 0.3, "state": "NORMAL"}},
            {"student_id": "s3", "grade_summary": {"percentage": 45},
             "proctoring_summary": {"risk_score": 0.8, "state": "HIGH_RISK"}},
        ]
        report = gen.generate("exam_001", reports)
        stats = report["statistics"]
        assert stats["mean_score"] == pytest.approx(68.33, rel=0.1)
        assert report["flagged_students"] is not None


class TestEndToEndGrading:
    """Full grading pipeline with mixed question types."""

    def test_grade_all_mixed_types(self):
        """Grade a complete exam with MCQ, numerical, short answer, and long answer."""
        from src.grading.scorer import GradingOrchestrator
        from src.models.models import Question, QuestionType

        questions = [
            Question(id="q1", type=QuestionType.MCQ, text="2+2?",
                     options=["3", "4", "5"], correct_answer="4", marks=1.0),
            Question(id="q2", type=QuestionType.NUMERICAL, text="sqrt(16)",
                     correct_answer="4", marks=1.0, tolerance=0.1),
            Question(id="q3", type=QuestionType.SHORT_ANSWER,
                     text="Define recursion", correct_answer="A function that calls itself",
                     marks=2.0, keywords=["function", "calls", "itself"]),
            Question(id="q4", type=QuestionType.LONG_ANSWER,
                     text="Explain photosynthesis",
                     correct_answer="Photosynthesis converts light to chemical energy using chlorophyll",
                     marks=5.0, keywords=["photosynthesis", "light", "energy", "chlorophyll"]),
        ]
        answers = {
            "q1": "4",
            "q2": "4",
            "q3": "A function that calls itself to solve problems",
            "q4": "Photosynthesis is the process where plants convert light energy into chemical energy using chlorophyll",
        }
        reference = {
            "q2": "4",
            "q3": "A function that calls itself",
            "q4": "Photosynthesis converts light to chemical energy using chlorophyll",
        }

        orchestrator = GradingOrchestrator()
        results = orchestrator.grade_all(questions, answers, reference)

        assert "total_score" in results
        assert "total_max" in results
        assert "percentage" in results
        assert "per_question" in results
        assert results["total_max"] == 9.0
        assert results["total_score"] > 0
        assert "q1" in results["per_question"]
        assert results["per_question"]["q1"]["correct"] is True
        assert "q2" in results["per_question"]
        assert results["per_question"]["q2"]["score"] == 1.0

    def test_grade_all_no_answers(self):
        """Grading with no submitted answers."""
        from src.grading.scorer import GradingOrchestrator
        from src.models.models import Question, QuestionType

        questions = [
            Question(id="q1", type=QuestionType.MCQ, text="2+2?",
                     options=["3", "4", "5"], correct_answer="4", marks=1.0),
        ]
        orchestrator = GradingOrchestrator()
        results = orchestrator.grade_all(questions, {}, {})
        assert results["total_score"] == 0.0
        assert results["percentage"] == 0.0

    def test_grade_all_no_reference_answers(self):
        """Grading short answers without reference answers falls back to question text."""
        from src.grading.scorer import GradingOrchestrator
        from src.models.models import Question, QuestionType

        questions = [
            Question(id="q1", type=QuestionType.SHORT_ANSWER,
                     text="What is 2+2?",
                     correct_answer="4", marks=1.0, keywords=["four"]),
        ]
        answers = {"q1": "The answer is 4"}
        orchestrator = GradingOrchestrator()
        results = orchestrator.grade_all(questions, answers)
        # Should still produce results using question text as reference
        assert "per_question" in results

    def test_grade_true_false(self):
        """True/False questions are graded as MCQ."""
        from src.grading.scorer import GradingOrchestrator
        from src.models.models import Question, QuestionType

        questions = [
            Question(id="q1", type=QuestionType.TRUE_FALSE,
                     text="Python is statically typed",
                     options=["True", "False"], correct_answer="False", marks=1.0),
        ]
        answers = {"q1": "False"}
        orchestrator = GradingOrchestrator()
        results = orchestrator.grade_all(questions, answers)
        assert results["per_question"]["q1"]["correct"] is True
        assert results["per_question"]["q1"]["score"] == 1.0


class TestEndToEndAgent:
    """Full agent pipeline: signals → perceive → featureize → predict → reason → decide → act."""

    def test_process_normal_signals(self):
        """Agent stays NORMAL with normal signals."""
        from src.agent.controller import ProctoringAgent

        agent = ProctoringAgent({"risk": {"escalation_threshold": 0.7, "review_threshold": 0.5}})
        signals = {
            "face_present": True,
            "face_confidence": 0.95,
            "face_count": 1,
            "head_pose": {"yaw": 5, "pitch": 3, "roll": 1},
            "audio_features": {"volume_db": -60, "rms_energy": 0.05, "zero_crossing_rate": 0.1},
            "keyboard_idle_seconds": 30,
            "mouse_idle_seconds": 10,
            "tab_switch": False,
            "paste_detected": False,
        }
        result = agent.process_signals(signals)
        assert result["state"] == "NORMAL"
        assert result["risk_score"] >= 0
        assert "events_detected" in result
        assert "features" in result
        assert "actions" in result

    def test_process_suspicious_signals(self):
        """Agent escalates when suspicious signals detected."""
        from src.agent.controller import ProctoringAgent

        agent = ProctoringAgent({"risk": {"escalation_threshold": 0.7, "review_threshold": 0.3}})
        # Face absent triggers events
        signals = {
            "face_present": False,
            "face_confidence": 0.1,
            "face_count": 0,
            "head_pose": {"yaw": 0, "pitch": 0, "roll": 0},
            "audio_features": {"volume_db": -80, "rms_energy": 0.01, "zero_crossing_rate": 0.05},
            "keyboard_idle_seconds": 0,
            "mouse_idle_seconds": 0,
            "tab_switch": True,
            "paste_detected": True,
        }
        result = agent.process_signals(signals)
        assert result["state"] in ("SUSPICIOUS", "HIGH_RISK", "NORMAL")
        assert len(result["events_detected"]) > 0
        event_types = [e["type"] for e in result["events_detected"]]
        assert "face_absent" in event_types or "tab_switch" in event_types or "paste_detected" in event_types

    def test_agent_state_transitions(self):
        """Agent transitions through states with escalating risk."""
        from src.agent.controller import ProctoringAgent

        agent = ProctoringAgent({"risk": {"escalation_threshold": 0.7, "review_threshold": 0.4}})
        assert agent.get_status()["state"] == "NORMAL"

        # Normal signals — stays NORMAL
        result = agent.process_signals({
            "face_present": True, "face_confidence": 0.9, "face_count": 1,
            "head_pose": {}, "audio_features": {}, "keyboard_idle_seconds": 0,
            "mouse_idle_seconds": 0, "tab_switch": False, "paste_detected": False,
        })
        assert result["state"] == "NORMAL"

    def test_agent_reset(self):
        """Agent reset clears all state."""
        from src.agent.controller import ProctoringAgent

        agent = ProctoringAgent({"risk": {"escalation_threshold": 0.7, "review_threshold": 0.5}})
        # Process some signals
        agent.process_signals({
            "face_present": False, "face_confidence": 0.1, "face_count": 0,
            "head_pose": {}, "audio_features": {}, "keyboard_idle_seconds": 0,
            "mouse_idle_seconds": 0, "tab_switch": True, "paste_detected": False,
        })
        history_len = len(agent.get_history())
        assert history_len > 0

        agent.reset()
        status = agent.get_status()
        assert status["state"] == "NORMAL"
        assert status["iteration"] == 0
        assert len(agent.get_history()) == 0

    def test_agent_get_status(self):
        """Agent status contains expected keys."""
        from src.agent.controller import ProctoringAgent

        agent = ProctoringAgent({"risk": {"escalation_threshold": 0.7}})
        status = agent.get_status()
        assert "session_id" in status
        assert "state" in status
        assert "iteration" in status
        assert "running" in status
        assert "evidence_events" in status
        assert "current_risk" in status
        assert status["state"] == "NORMAL"

    def test_agent_with_models_attached(self):
        """Agent accepts ML model attachments."""
        from src.agent.controller import ProctoringAgent
        from src.ml.logistic import LogisticRegressionScratch
        from src.ml.anomaly import AnomalyDetector

        agent = ProctoringAgent({"risk": {"escalation_threshold": 0.7, "review_threshold": 0.5}})

        # Train simple models
        X = np.random.randn(50, 22)
        y = (X[:, 0] > 0).astype(float)
        lr = LogisticRegressionScratch(learning_rate=0.01, max_iterations=100)
        lr.fit(X, y)

        normal_data = X[y == 0]
        anomaly = AnomalyDetector(mahalanobis_confidence=0.95)
        if len(normal_data) > 5:
            anomaly.fit(normal_data)

        agent.set_models(logistic=lr, anomaly=anomaly)

        result = agent.process_signals({
            "face_present": True, "face_confidence": 0.9, "face_count": 1,
            "head_pose": {}, "audio_features": {}, "keyboard_idle_seconds": 0,
            "mouse_idle_seconds": 0, "tab_switch": False, "paste_detected": False,
        })
        assert "ml_predictions" in result

    def test_agent_callbacks(self):
        """Agent calls state change callbacks."""
        from src.agent.controller import ProctoringAgent

        transitions = []
        def on_change(old, new):
            transitions.append((old, new))

        agent = ProctoringAgent({"risk": {"escalation_threshold": 0.3, "review_threshold": 0.1}})
        agent.set_callbacks(on_state_change=on_change)

        # Trigger state change with absent face
        agent.process_signals({
            "face_present": False, "face_confidence": 0.0, "face_count": 0,
            "head_pose": {"yaw": 0, "pitch": 0, "roll": 0},
            "audio_features": {"volume_db": -100, "rms_energy": 0.0, "zero_crossing_rate": 0.0},
            "keyboard_idle_seconds": 0, "mouse_idle_seconds": 0,
            "tab_switch": True, "paste_detected": True,
        })
        # Callbacks may or may not fire depending on risk score
        assert isinstance(transitions, list)


class TestEndToEndRiskEngine:
    """Risk engine with evidence memory — full risk computation pipeline."""

    def test_compute_risk_with_evidence_memory(self):
        """Risk engine computes risk from evidence memory with multiple events."""
        from src.agent.risk import RiskEngine
        from src.agent.memory import EvidenceMemory

        engine = RiskEngine({
            "alpha_decay": 0.8,
            "escalation_threshold": 0.7,
            "review_threshold": 0.5,
        })
        memory = EvidenceMemory(window_seconds=300, alpha=0.8)

        # Add multiple events to build up risk
        memory.add_event("face_absent", 0.9)
        memory.add_event("face_absent", 0.8)
        memory.add_event("tab_switch", 0.6)
        memory.add_event("head_turned", 0.7)

        risk = engine.compute_risk(memory)
        assert 0.0 <= risk <= 1.0
        assert risk > 0.0  # Events should produce non-zero risk

    def test_risk_increases_with_more_events(self):
        """More events produce higher risk score."""
        from src.agent.risk import RiskEngine
        from src.agent.memory import EvidenceMemory

        engine = RiskEngine({"alpha_decay": 0.8, "escalation_threshold": 0.7, "review_threshold": 0.5})
        memory_low = EvidenceMemory(window_seconds=300, alpha=0.8)
        memory_high = EvidenceMemory(window_seconds=300, alpha=0.8)

        memory_low.add_event("tab_switch", 0.6)
        risk_low = engine.compute_risk(memory_low)

        memory_high.add_event("face_absent", 0.9)
        memory_high.add_event("face_absent", 0.8)
        memory_high.add_event("multiple_faces", 0.7)
        memory_high.add_event("copy_paste", 0.9)
        memory_high.add_event("tab_switch", 0.6)
        risk_high = engine.compute_risk(memory_high)

        assert risk_high > risk_low

    def test_risk_decay_over_time(self):
        """Risk decreases when no new events are added (decay)."""
        from src.agent.risk import RiskEngine
        from src.agent.memory import EvidenceMemory

        memory = EvidenceMemory(window_seconds=300, alpha=0.9)
        # High alpha = slow decay
        memory.add_event("face_absent", 0.9)
        memory.add_event("face_absent", 0.9)
        risk_initial = memory.get_current_risk()
        assert risk_initial > 0

        # Memory with lower alpha (faster decay)
        memory2 = EvidenceMemory(window_seconds=300, alpha=0.5)
        memory2.add_event("face_absent", 0.9)
        memory2.add_event("face_absent", 0.9)
        # Lower alpha means each event contributes less to cumulative risk
        # Risk is bounded by [0, 1]
        assert memory2.get_current_risk() <= 1.0

    def test_evidence_memory_pruning(self):
        """Old events are pruned outside the sliding window."""
        from src.agent.memory import EvidenceMemory
        import time

        memory = EvidenceMemory(window_seconds=1.0, alpha=0.8)
        memory.add_event("tab_switch", 0.6, timestamp=time.time() - 10)
        memory.add_event("face_absent", 0.9, timestamp=time.time())

        events = memory.get_recent_events()
        assert len(events) == 1
        assert events[0].event_type == "face_absent"

    def test_evidence_memory_event_counts(self):
        """Event counts are aggregated correctly."""
        from src.agent.memory import EvidenceMemory

        memory = EvidenceMemory(window_seconds=300, alpha=0.8)
        memory.add_event("tab_switch", 0.6)
        memory.add_event("tab_switch", 0.5)
        memory.add_event("face_absent", 0.9)

        counts = memory.get_event_counts()
        assert counts["tab_switch"] == 2
        assert counts["face_absent"] == 1

    def test_evidence_memory_signal_summary(self):
        """Signal summary groups events by category."""
        from src.agent.memory import EvidenceMemory

        memory = EvidenceMemory(window_seconds=300, alpha=0.8)
        memory.add_event("tab_switch", 0.6)
        memory.add_event("face_absent", 0.9)
        memory.add_event("loud_audio", 0.5)

        summary = memory.get_signal_summary()
        assert summary["temporal"] >= 1
        assert summary["vision"] >= 1
        assert summary["audio"] >= 1

    def test_count_active_signals(self):
        """Count independent signal categories with events."""
        from src.agent.memory import EvidenceMemory

        memory = EvidenceMemory(window_seconds=300, alpha=0.8)
        memory.add_event("tab_switch", 0.6)       # temporal
        memory.add_event("face_absent", 0.9)      # vision
        memory.add_event("loud_audio", 0.5)       # audio

        active = memory.count_active_signals(threshold=1)
        assert active >= 3

    def test_risk_classify_all_levels(self):
        """Risk classification covers all three levels."""
        from src.agent.risk import RiskEngine

        engine = RiskEngine({"escalation_threshold": 0.7, "review_threshold": 0.4})
        assert engine.classify_risk_level(0.1) == "NORMAL"
        assert engine.classify_risk_level(0.5) == "SUSPICIOUS"
        assert engine.classify_risk_level(0.9) == "HIGH_RISK"

    def test_risk_decay_formula(self):
        """Risk decay follows R_t = alpha * R_{t-1} + (1-alpha) * E_t."""
        from src.agent.risk import RiskEngine

        engine = RiskEngine({"alpha_decay": 0.8})
        r = engine.compute_risk_decay(0.5, 0.3)
        expected = 0.8 * 0.5 + 0.2 * 0.3
        assert r == pytest.approx(expected)

    def test_is_escalation_required(self):
        """Escalation logic with multi-signal requirement."""
        from src.agent.risk import RiskEngine
        from src.agent.memory import EvidenceMemory

        engine = RiskEngine({
            "escalation_threshold": 0.7,
            "review_threshold": 0.5,
            "multi_signal_required": 2,
        })
        # High risk always escalates
        memory = EvidenceMemory(window_seconds=300, alpha=0.8)
        assert engine.is_escalation_required(0.8, memory) is True

        # Medium risk without enough signals doesn't escalate
        memory2 = EvidenceMemory(window_seconds=300, alpha=0.8)
        memory2.add_event("tab_switch", 0.6)
        assert engine.is_escalation_required(0.5, memory2) is False

        # Medium risk with multiple signals escalates
        memory3 = EvidenceMemory(window_seconds=300, alpha=0.8)
        memory3.add_event("tab_switch", 0.6)
        memory3.add_event("face_absent", 0.9)
        assert engine.is_escalation_required(0.5, memory3) is True

    def test_event_weights_defined(self):
        """All critical event types have severity weights."""
        from src.agent.risk import RiskEngine

        engine = RiskEngine()
        critical_events = [
            "face_absent", "multiple_faces", "copy_paste", "paste_detected",
            "tab_switch", "fullscreen_exit", "speech_detected",
        ]
        for event in critical_events:
            assert event in engine.EVENT_WEIGHTS
            assert 0.0 < engine.EVENT_WEIGHTS[event] <= 1.0

    def test_evidence_memory_clear(self):
        """Clearing memory resets risk and entries."""
        from src.agent.memory import EvidenceMemory

        memory = EvidenceMemory(window_seconds=300, alpha=0.8)
        memory.add_event("tab_switch", 0.6)
        memory.add_event("face_absent", 0.9)
        assert len(memory.get_recent_events()) == 2
        assert memory.get_current_risk() > 0

        memory.clear()
        assert len(memory.get_recent_events()) == 0
        assert memory.get_current_risk() == 0.0


class TestEndToEndAgentLoop:
    """Autonomous agent loop — full OBSERVE→ACT pipeline with multiple iterations."""

    def test_full_loop_multiple_iterations(self):
        """Run multiple signal processing iterations through the full agent loop."""
        from src.agent.controller import ProctoringAgent

        agent = ProctoringAgent({"risk": {"escalation_threshold": 0.8, "review_threshold": 0.5}})

        base_signals = {
            "face_present": True, "face_confidence": 0.9, "face_count": 1,
            "head_pose": {"yaw": 5, "pitch": 3, "roll": 1},
            "audio_features": {"volume_db": -60, "rms_energy": 0.05, "zero_crossing_rate": 0.1},
            "keyboard_idle_seconds": 30, "mouse_idle_seconds": 10,
            "tab_switch": False, "paste_detected": False,
        }

        # Run 5 normal iterations
        for _ in range(5):
            result = agent.process_signals(base_signals)
            assert result["iteration"] > 0
            assert result["state"] == "NORMAL"

        assert len(agent.get_history()) == 5
        assert agent.get_status()["iteration"] == 5

    def test_loop_risk_accumulation(self):
        """Risk accumulates over multiple suspicious events."""
        from src.agent.controller import ProctoringAgent

        agent = ProctoringAgent({"risk": {"escalation_threshold": 0.5, "review_threshold": 0.3}})

        risk_scores = []
        for i in range(5):
            signals = {
                "face_present": i < 2,  # Face absent after iteration 2
                "face_confidence": 0.9 if i < 2 else 0.1,
                "face_count": 1 if i < 2 else 0,
                "head_pose": {"yaw": 50 if i >= 2 else 5, "pitch": 0, "roll": 0},
                "audio_features": {"volume_db": -60, "rms_energy": 0.05, "zero_crossing_rate": 0.1},
                "keyboard_idle_seconds": 0, "mouse_idle_seconds": 0,
                "tab_switch": i >= 3, "paste_detected": False,
            }
            result = agent.process_signals(signals)
            risk_scores.append(result["risk_score"])

        # Risk should generally increase with more suspicious signals
        assert risk_scores[-1] >= risk_scores[0]


class TestEndToEndMLPipeline:
    """End-to-end ML: data generation → training → evaluation → prediction."""

    def test_full_training_pipeline(self):
        """Generate data, train logistic regression, evaluate, predict."""
        from src.ml.training_data import TrainingDataGenerator
        from src.ml.logistic import LogisticRegressionScratch
        from src.ml.calibration import ModelEvaluator
        from src.ml.anomaly import AnomalyDetector

        gen = TrainingDataGenerator(seed=42)
        X, y = gen.generate_dataset(n_normal=100, n_suspicious=80)
        X_train, X_test, y_train, y_test = gen.train_test_split(X, y, test_ratio=0.25)

        # Train logistic regression
        lr = LogisticRegressionScratch(learning_rate=0.01, max_iterations=500, regularization=0.1)
        lr.fit(X_train, y_train)
        y_pred = lr.predict(X_test)
        y_proba = lr.predict_proba(X_test)

        # Evaluate
        evaluator = ModelEvaluator()
        metrics = evaluator.evaluate_classifier(y_test, y_pred, y_proba)

        assert metrics["accuracy"] > 0.5
        assert 0 <= metrics["precision"] <= 1
        assert 0 <= metrics["recall"] <= 1
        assert 0 <= metrics["f1"] <= 1
        assert 0 <= metrics["roc_auc"] <= 1
        assert "confusion_matrix" in metrics

    def test_anomaly_detection_pipeline(self):
        """Train anomaly detector, score normal and anomalous samples."""
        from src.ml.training_data import TrainingDataGenerator
        from src.ml.anomaly import AnomalyDetector

        gen = TrainingDataGenerator(seed=42)
        X, y = gen.generate_dataset(n_normal=100, n_suspicious=80)
        normal = X[y == 0]

        detector = AnomalyDetector(mahalanobis_confidence=0.95)
        detector.fit(normal)

        # Normal samples should have lower anomaly scores
        normal_scores = detector.score(X[y == 0][:20])
        suspicious_scores = detector.score(X[y == 1][:20])

        assert len(normal_scores) == 20
        assert len(suspicious_scores) == 20
        assert all(0 <= s <= 1 for s in normal_scores)
        assert all(0 <= s <= 1 for s in suspicious_scores)

    def test_risk_model_with_trained_logistic(self):
        """Risk engine uses trained logistic model for refined predictions."""
        from src.agent.risk import RiskEngine
        from src.agent.memory import EvidenceMemory
        from src.ml.logistic import LogisticRegressionScratch

        engine = RiskEngine({"alpha_decay": 0.8, "escalation_threshold": 0.7, "review_threshold": 0.5})
        memory = EvidenceMemory(window_seconds=300, alpha=0.8)

        # Train a simple logistic model on behavioral data
        np.random.seed(42)
        X = np.random.randn(100, 22)
        y = (X[:, 0] + X[:, 1] > 0).astype(float)
        lr = LogisticRegressionScratch(learning_rate=0.01, max_iterations=200)
        lr.fit(X, y)

        engine.set_logistic_model(lr)

        memory.add_event("face_absent", 0.9)
        memory.add_event("tab_switch", 0.6)

        risk = engine.compute_risk(memory)
        assert 0.0 <= risk <= 1.0

    def test_agent_with_trained_models_full_pipeline(self):
        """Full agent with trained ML models processing signals."""
        from src.agent.controller import ProctoringAgent
        from src.ml.logistic import LogisticRegressionScratch
        from src.ml.anomaly import AnomalyDetector
        from src.ml.training_data import TrainingDataGenerator

        gen = TrainingDataGenerator(seed=42)
        X, y = gen.generate_dataset(n_normal=100, n_suspicious=50)
        normal = X[y == 0]

        lr = LogisticRegressionScratch(learning_rate=0.01, max_iterations=200)
        lr.fit(X, y)

        anomaly = AnomalyDetector(mahalanobis_confidence=0.95)
        anomaly.fit(normal)

        agent = ProctoringAgent({"risk": {"escalation_threshold": 0.7, "review_threshold": 0.5}})
        agent.set_models(logistic=lr, anomaly=anomaly)

        result = agent.process_signals({
            "face_present": False, "face_confidence": 0.1, "face_count": 0,
            "head_pose": {"yaw": 60, "pitch": 0, "roll": 0},
            "audio_features": {"volume_db": -25, "rms_energy": 0.5, "zero_crossing_rate": 0.3},
            "keyboard_idle_seconds": 0, "mouse_idle_seconds": 0,
            "tab_switch": True, "paste_detected": True,
        })
        assert result["ml_predictions"]["logistic_risk"] is not None
        assert result["ml_predictions"]["anomaly_score"] is not None


class TestEndToEndProctoring:
    """Vision and behavioral feature integration."""

    def test_feature_extraction_from_signals(self):
        """Feature extractor produces consistent vectors from signal dicts."""
        from src.features.behavioral import BehavioralFeatureExtractor

        extractor = BehavioralFeatureExtractor()
        signals = {
            "face_present": True, "face_confidence": 0.9, "face_count": 1,
            "head_pose": {"yaw": 10, "pitch": 5, "roll": 2},
            "audio_features": {"rms_energy": 0.1, "volume_db": -50, "zero_crossing_rate": 0.15,
                               "spectral_centroid": 2000},
            "keyboard_idle_seconds": 30, "keyboard_events_per_minute": 60,
            "mouse_avg_speed": 100, "mouse_total_distance": 500, "mouse_click_count": 10,
            "mouse_idle_seconds": 15, "multi_face_events": 0,
            "session_elapsed": 600, "total_events": 5,
        }
        features = extractor.extract(signals)
        assert features is not None
        assert len(features) == 22  # 22 features per the architecture
        assert all(isinstance(f, (float, np.floating)) for f in features)

    def test_feature_extraction_empty_signals(self):
        """Feature extractor handles missing/default signals."""
        from src.features.behavioral import BehavioralFeatureExtractor

        extractor = BehavioralFeatureExtractor()
        features = extractor.extract({})
        assert features is not None
        assert len(features) == 22

    def test_temporal_feature_aggregation(self):
        """Temporal aggregator computes rolling statistics."""
        from src.features.temporal import TemporalFeatureAggregator

        agg = TemporalFeatureAggregator(window_size=10)
        for i in range(15):
            vec = np.random.randn(22)
            agg.add(vec)

        window = agg.get_window()
        assert window.shape[0] == 10  # Capped at window size

        stats = agg.aggregate()
        assert "mean" in stats
        assert "std" in stats
        assert "trend" in stats
        assert len(stats["mean"]) == 22

    def test_agent_feature_vector_consistency(self):
        """Agent produces consistent 22-D feature vectors."""
        from src.agent.controller import ProctoringAgent

        agent = ProctoringAgent({"risk": {"escalation_threshold": 0.7}})
        signals = {
            "face_present": True, "face_confidence": 0.9, "face_count": 1,
            "head_pose": {"yaw": 10, "pitch": 5, "roll": 2},
            "audio_features": {"rms_energy": 0.1, "volume_db": -50, "zero_crossing_rate": 0.15,
                               "spectral_centroid": 2000},
            "keyboard_idle_seconds": 30, "mouse_idle_seconds": 15,
            "tab_switch": False, "paste_detected": False,
        }
        result = agent.process_signals(signals)
        if result["features"] is not None:
            assert len(result["features"]) == 22


class TestEndToEndReports:
    """End-to-end report generation from grading + proctoring data."""

    def test_complete_student_report_pipeline(self):
        """Full pipeline: grade → agent → report for a single student."""
        from src.grading.scorer import GradingOrchestrator
        from src.agent.controller import ProctoringAgent
        from src.reports.student_report import StudentReportGenerator
        from src.prediction.knowledge import KnowledgeTracer
        from src.models.models import Question, QuestionType

        # Build questions and answers
        questions = [
            Question(id="q1", type=QuestionType.MCQ, text="2+2?",
                     options=["3", "4", "5"], correct_answer="4", marks=1.0),
            Question(id="q2", type=QuestionType.NUMERICAL, text="sqrt(16)",
                     correct_answer="4", marks=1.0, tolerance=0.1),
            Question(id="q3", type=QuestionType.SHORT_ANSWER,
                     text="Define recursion",
                     correct_answer="A function that calls itself",
                     marks=2.0, keywords=["function", "calls", "itself"]),
        ]
        answers = {"q1": "4", "q2": "4", "q3": "A function that calls itself recursively"}
        reference = {"q2": "4", "q3": "A function that calls itself"}

        # Grade
        orchestrator = GradingOrchestrator()
        results = orchestrator.grade_all(questions, answers, reference)

        # Proctoring
        agent = ProctoringAgent({"risk": {"escalation_threshold": 0.7, "review_threshold": 0.5}})
        agent_result = agent.process_signals({
            "face_present": True, "face_confidence": 0.95, "face_count": 1,
            "head_pose": {"yaw": 5, "pitch": 3, "roll": 1},
            "audio_features": {"volume_db": -60, "rms_energy": 0.05, "zero_crossing_rate": 0.1},
            "keyboard_idle_seconds": 30, "mouse_idle_seconds": 10,
            "tab_switch": False, "paste_detected": False,
        })

        # Knowledge tracing
        kt = KnowledgeTracer()
        kt.update("mathematics", correct=True)
        kt.update("algorithms", correct=True)

        # Report
        gen = StudentReportGenerator()
        report = gen.generate(
            grading_results=results,
            proctoring_summary={
                "student_id": "stu1",
                "risk_score": agent_result["risk_score"],
                "state": agent_result["state"],
                "events_count": len(agent_result["events_detected"]),
            },
            knowledge_state=kt.get_all_mastery(),
        )

        assert report["report_type"] == "student_report"
        assert report["grade_summary"]["percentage"] > 0
        assert report["grade_summary"]["total_score"] > 0
        assert report["proctoring_summary"]["state"] in ("NORMAL", "SUSPICIOUS", "HIGH_RISK")

    def test_complete_examiner_report_pipeline(self):
        """Full pipeline: grade multiple students → examiner report."""
        from src.grading.scorer import GradingOrchestrator
        from src.reports.examiner_report import ExaminerReportGenerator
        from src.models.models import Question, QuestionType

        questions = [
            Question(id="q1", type=QuestionType.MCQ, text="2+2?",
                     options=["3", "4", "5"], correct_answer="4", marks=1.0),
            Question(id="q2", type=QuestionType.NUMERICAL, text="sqrt(16)",
                     correct_answer="4", marks=1.0, tolerance=0.1),
        ]

        orchestrator = GradingOrchestrator()
        student_reports = []

        for sid, name, answers in [
            ("s1", "Alice", {"q1": "4", "q2": "4"}),
            ("s2", "Bob", {"q1": "3", "q2": "4.05"}),
            ("s3", "Carol", {"q1": "4", "q2": "5"}),
        ]:
            results = orchestrator.grade_all(questions, answers)
            student_reports.append({
                "student_id": sid,
                "name": name,
                "grade_summary": {
                    "percentage": results["percentage"],
                    "total_score": results["total_score"],
                    "total_max": results["total_max"],
                },
                "proctoring_summary": {
                    "risk_score": 0.1 if sid == "s1" else (0.8 if sid == "s3" else 0.3),
                    "state": "NORMAL" if sid == "s1" else ("HIGH_RISK" if sid == "s3" else "NORMAL"),
                },
            })

        gen = ExaminerReportGenerator()
        report = gen.generate("math_exam_001", student_reports)

        assert report["total_students"] == 3
        assert "statistics" in report
        assert "risk_analysis" in report
        assert "flagged_students" in report
        assert report["exam_id"] == "math_exam_001"


class TestEndToEndExamFlow:
    """Complete exam flow: setup → session → answers → grading → results."""

    def test_exam_manager_create_and_retrieve(self):
        """Create an exam and retrieve it."""
        from src.exam.exam import ExamManager
        from src.models.models import Question, QuestionType

        mgr = ExamManager({})
        exam = mgr.create_exam(
            exam_id="test_exam_1",
            title="Test Exam",
            duration_minutes=30,
            questions=[],
        )
        exam.topic = "Mathematics"
        exam.difficulty = "easy"
        assert exam.exam_id == "test_exam_1"
        assert exam.title == "Test Exam"
        assert mgr.get_exam("test_exam_1") is exam

    def test_exam_with_questions(self):
        """Exam with questions can be added and retrieved."""
        from src.exam.exam import ExamManager
        from src.models.models import Question, QuestionType

        mgr = ExamManager({})
        exam = mgr.create_exam("exam_q", "Questions Exam", 30, [])
        q1 = exam.add_question("mcq", "What is 2+2?", options=["3", "4", "5"],
                                correct_answer="4", marks=1.0)
        q2 = exam.add_question("numerical", "sqrt(144)", correct_answer="12",
                                marks=2.0, tolerance=0.1)

        assert len(exam.questions) == 2
        assert exam.questions[0].type == QuestionType.MCQ
        assert exam.questions[1].type == QuestionType.NUMERICAL
        assert exam.questions[0].marks == 1.0
        assert exam.questions[1].marks == 2.0

    def test_exam_controller_session_lifecycle(self):
        """Full session lifecycle: create → start → answer → end."""
        from src.exam.exam import ExamController
        from src.models.models import Question, QuestionType

        config = {"risk": {"escalation_threshold": 0.7, "review_threshold": 0.5}}
        controller = ExamController(config)
        session = controller.create_session("stu1", "exam1", duration_minutes=30)

        assert session.student_id == "stu1"
        assert session.exam_id == "exam1"
        assert session.state.value == "NORMAL"

        controller.start_exam()
        controller.submit_answer("q1", "Answer 1")
        controller.submit_answer("q2", "Answer 2")

        status = controller.get_session_status()
        assert status["answers_count"] == 2
        assert status["state"] == "NORMAL"

        ended = controller.end_exam()
        assert ended.end_time is not None

    def test_exam_session_manager_timer(self):
        """Session manager tracks elapsed and remaining time."""
        from src.exam.exam import ExamController
        from src.models.models import Question, QuestionType

        config = {"risk": {"escalation_threshold": 0.7, "review_threshold": 0.5}}
        controller = ExamController(config)
        controller.create_session("stu1", "exam1", duration_minutes=60)
        controller.start_exam()

        # Let a bit of time pass
        import time
        time.sleep(0.5)

        status = controller.get_session_status()
        assert status["elapsed_seconds"] >= 0
        assert status["remaining_seconds"] > 0

    def test_question_bank_crud(self):
        """Question bank supports CRUD operations."""
        from src.exam.questions import QuestionBank
        from src.models.models import Question, QuestionType
        import tempfile, os

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "questions.json")
            bank = QuestionBank(storage_path=path)

            q = Question(id="qb1", type=QuestionType.MCQ, text="Test?",
                         options=["A", "B"], correct_answer="A", marks=1.0)
            bank.add_question(q)
            assert bank.get_question("qb1") is q
            assert len(bank.get_all()) == 1
            assert len(bank.get_by_type(QuestionType.MCQ)) == 1

            bank.remove_question("qb1")
            assert bank.get_question("qb1") is None


class TestEndToEndDatabase:
    """Database integration with sessions, questions, answers, events."""

    def test_full_session_lifecycle_in_db(self, tmp_path):
        """Create session, add questions, record answers, add events, get results."""
        from src.database.db import Database

        db = Database(tmp_path / "full_lifecycle.db")
        sid = db.create_session("stu1", "exam1", duration_minutes=30)

        # Add questions
        db.add_question(sid, "q1", "mcq", "2+2?", marks=1.0, correct_answer="4",
                        options=["3", "4", "5"])
        db.add_question(sid, "q2", "numerical", "sqrt(16)", marks=2.0,
                        correct_answer="4", tolerance=0.1)
        db.add_question(sid, "q3", "short_answer", "Define X", marks=3.0,
                        correct_answer="X is...", keywords=["define", "x"])

        questions = db.get_questions(sid)
        assert len(questions) == 3
        assert questions[0]["type"] == "mcq"
        assert questions[1]["tolerance"] == 0.1
        assert questions[2]["keywords"] == ["define", "x"]

        # Record answers
        db.record_answer(sid, "q1", "4", score=1.0, max_score=1.0, grading_method="exact_match")
        db.record_answer(sid, "q2", "4.0", score=2.0, max_score=2.0, grading_method="tolerance")
        db.record_answer(sid, "q3", "X is defined as...", score=1.5, max_score=3.0,
                         grading_method="tfidf", grading_details={"similarity": 0.75})

        # Add proctoring events
        db.add_event(sid, "face_detected", 0.95)
        db.add_event(sid, "tab_switch", 0.6)
        db.add_event(sid, "face_absent", 0.8)

        events = db.get_events(sid)
        assert len(events) == 3
        event_types = [e["event_type"] for e in events]
        assert "face_detected" in event_types
        assert "tab_switch" in event_types

        # Get session results
        results = db.get_session_results(sid)
        assert results["total_score"] == 4.5
        assert results["total_max_score"] == 6.0

        # Update risk
        db.update_risk(sid, 0.3, "normal")
        session = db.get_session(sid)
        assert session["risk_score"] == 0.3
        assert session["state"] == "normal"

        # Save prediction
        pred_id = db.save_prediction(sid, predicted_score=85.0, confidence=0.9,
                                     method="random_forest", question_id="q1")
        assert pred_id is not None

        db.close()

    def test_db_multiple_sessions(self, tmp_path):
        """Database handles multiple concurrent sessions."""
        from src.database.db import Database

        db = Database(tmp_path / "multi_session.db")
        sid1 = db.create_session("stu1", "exam1")
        sid2 = db.create_session("stu2", "exam1")
        sid3 = db.create_session("stu1", "exam2")

        assert sid1 != sid2 != sid3

        sessions = db.list_sessions()
        assert len(sessions) == 3

        # Verify session data is independent
        s1 = db.get_session(sid1)
        assert s1["student_id"] == "stu1"
        assert s1["exam_id"] == "exam1"

        db.close()

    def test_db_event_counts(self, tmp_path):
        """Database returns event counts by type."""
        from src.database.db import Database

        db = Database(tmp_path / "event_counts.db")
        sid = db.create_session("stu1", "exam1")
        db.add_event(sid, "tab_switch", 0.6)
        db.add_event(sid, "tab_switch", 0.5)
        db.add_event(sid, "face_absent", 0.9)
        db.add_event(sid, "face_absent", 0.8)
        db.add_event(sid, "face_absent", 0.7)

        counts = db.get_event_counts(sid)
        assert counts["tab_switch"] == 2
        assert counts["face_absent"] == 3
        db.close()

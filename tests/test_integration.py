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
             "proctoring_summary": {"risk_score": 0.3, "state": "normal"}}
        ])
        assert report is not None
        assert report["total_students"] == 1

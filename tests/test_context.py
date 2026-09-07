"""Tests for context-aware risk engine (src/context)."""

import pytest
from src.context import (
    ExamPolicy,
    ExamContext,
    QuestionContext,
    StudentBaseline,
    get_default_policy,
    compute_context_weight,
)
from src.baseline import BaselineManager, BehavioralSample


class TestExamPolicy:
    def test_default_policy(self):
        policy = ExamPolicy()
        assert policy.camera_required is True
        assert policy.microphone_required is True
        assert policy.multiple_face_enabled is True
        assert policy.escalation_threshold == 0.7
        assert policy.review_threshold == 0.5
        assert policy.multi_signal_required == 2

    def test_programming_policy(self):
        policy = get_default_policy("programming")
        assert policy.allow_ide is True
        assert policy.allow_documentation is True
        assert policy.weight_mouse == 0.8

    def test_open_book_policy(self):
        policy = get_default_policy("open_book")
        assert policy.allow_external_sites is True
        assert policy.allow_documentation is True
        assert policy.weight_browser == 0.3  # browser less suspicious

    def test_essay_policy(self):
        policy = get_default_policy("essay")
        assert policy.weight_keyboard == 0.95
        assert policy.weight_interaction == 0.95

    def test_mathematics_policy(self):
        policy = get_default_policy("mathematics")
        assert policy.allow_calculator is True
        assert policy.weight_keyboard == 0.9

    def test_unknown_type_returns_standard(self):
        policy = get_default_policy("unknown")
        assert policy.camera_required is True

    def test_policy_as_dict(self):
        policy = ExamPolicy()
        d = policy.as_dict()
        assert isinstance(d, dict)
        assert "camera_required" in d
        assert "escalation_threshold" in d
        assert d["camera_required"] is True

    def test_custom_policy(self):
        policy = ExamPolicy(
            camera_required=False,
            allow_external_sites=True,
            escalation_threshold=0.9,
            weight_audio=1.5,
        )
        assert policy.camera_required is False
        assert policy.allow_external_sites is True
        assert policy.escalation_threshold == 0.9
        assert policy.weight_audio == 1.5


class TestExamContext:
    def test_exam_progress(self):
        policy = ExamPolicy()
        ctx = ExamContext(
            policy=policy,
            exam_duration_seconds=3600.0,
            exam_elapsed_seconds=1800.0,
        )
        assert ctx.exam_progress() == pytest.approx(0.5)

    def test_exam_progress_not_started(self):
        policy = ExamPolicy()
        ctx = ExamContext(
            policy=policy,
            exam_duration_seconds=3600.0,
            exam_elapsed_seconds=0.0,
        )
        assert ctx.exam_progress() == 0.0

    def test_exam_progress_complete(self):
        policy = ExamPolicy()
        ctx = ExamContext(
            policy=policy,
            exam_duration_seconds=3600.0,
            exam_elapsed_seconds=7200.0,  # over duration
        )
        assert ctx.exam_progress() == 1.0

    def test_time_remaining(self):
        policy = ExamPolicy()
        ctx = ExamContext(
            policy=policy,
            exam_duration_seconds=3600.0,
            exam_elapsed_seconds=1800.0,
        )
        assert ctx.time_remaining() == 1800.0

    def test_time_remaining_negative(self):
        policy = ExamPolicy()
        ctx = ExamContext(
            policy=policy,
            exam_duration_seconds=3600.0,
            exam_elapsed_seconds=4000.0,
        )
        assert ctx.time_remaining() == 0.0  # clamped

    def test_with_question_context(self):
        policy = ExamPolicy()
        qctx = QuestionContext(
            question_id="q1",
            question_type="mcq",
            question_text="What is 2+2?",
            expected_interaction="selecting",
            expects_typing=False,
            expects_mouse=True,
        )
        ctx = ExamContext(policy=policy, current_question=qctx)
        assert ctx.current_question.expects_mouse is True
        assert ctx.current_question.expects_typing is False


class TestContextWeight:
    def test_no_context_returns_one(self):
        weight = compute_context_weight("tab_switch", context=None)
        assert weight == 1.0

    def test_no_question_returns_one(self):
        policy = ExamPolicy()
        ctx = ExamContext(policy=policy)
        weight = compute_context_weight("tab_switch", context=ctx)
        assert weight == 1.0

    def test_typing_event_reduced_during_typing_question(self):
        policy = ExamPolicy()
        qctx = QuestionContext(
            question_id="q1",
            question_type="long_answer",
            question_text="Write an essay.",
            expected_interaction="typing",
            expects_typing=True,
        )
        ctx = ExamContext(policy=policy, current_question=qctx)
        weight = compute_context_weight("rapid_typing", context=ctx)
        assert weight == 0.3  # typing expected, reduced

    def test_typing_event_increased_during_mcq(self):
        policy = ExamPolicy()
        qctx = QuestionContext(
            question_id="q1",
            question_type="mcq",
            question_text="Pick one.",
            expected_interaction="selecting",
            expects_typing=False,
        )
        ctx = ExamContext(policy=policy, current_question=qctx)
        weight = compute_context_weight("rapid_typing", context=ctx)
        assert weight == 0.9  # unexpected typing

    def test_tab_switch_reduced_when_sites_allowed(self):
        policy = ExamPolicy(allow_external_sites=True)
        qctx = QuestionContext(
            question_id="q1",
            question_type="mcq",
            question_text="Pick one.",
            expected_interaction="selecting",
        )
        ctx = ExamContext(policy=policy, current_question=qctx)
        weight = compute_context_weight("tab_switch", context=ctx)
        assert weight == 0.2

    def test_tab_switch_full_when_sites_blocked(self):
        policy = ExamPolicy(allow_external_sites=False)
        qctx = QuestionContext(
            question_id="q1",
            question_type="mcq",
            question_text="Pick one.",
            expected_interaction="selecting",
        )
        ctx = ExamContext(policy=policy, current_question=qctx)
        weight = compute_context_weight("tab_switch", context=ctx)
        assert weight == 1.0


class TestStudentBaseline:
    def test_not_ready_initially(self):
        baseline = StudentBaseline(student_id="s1")
        assert baseline.is_ready() is False

    def test_becomes_ready_after_samples(self):
        baseline = StudentBaseline(student_id="s1")
        for _ in range(5):
            baseline.add_sample(BehavioralSample(
                timestamp=0.0, typing_wpm=50.0,
                mouse_velocity=200.0, face_present=0.98,
            ))
        assert baseline.is_ready() is True

    def test_typing_deviation_ready(self):
        baseline = StudentBaseline(student_id="s1")
        for wpm in [45, 48, 52, 50, 47, 53, 49, 51, 46, 54]:
            baseline.add_sample(BehavioralSample(
                timestamp=0.0, typing_wpm=float(wpm),
                mouse_velocity=200.0, face_present=0.98,
            ))
        anomalous, deviation = baseline.deviation_typing(120.0)
        assert anomalous is True
        assert deviation > 0.0

    def test_normal_typing_not_anomalous(self):
        baseline = StudentBaseline(student_id="s1")
        for wpm in [45, 48, 52, 50, 47, 53, 49, 51, 46, 54]:
            baseline.add_sample(BehavioralSample(
                timestamp=0.0, typing_wpm=float(wpm),
                mouse_velocity=200.0, face_present=0.98,
            ))
        anomalous, _ = baseline.deviation_typing(52.0)
        assert anomalous is False

    def test_face_presence_deviation(self):
        baseline = StudentBaseline(student_id="s1")
        for fp in [0.97, 0.98, 0.99, 0.98, 0.97, 0.99, 0.98, 0.97, 0.98, 0.99]:
            baseline.add_sample(BehavioralSample(
                timestamp=0.0, face_present=fp,
            ))
        anomalous, deviation = baseline.deviation_face_presence(0.5)
        assert anomalous is True
        assert deviation > 0.0

    def test_mouse_deviation(self):
        baseline = StudentBaseline(student_id="s1")
        for vel in [180, 200, 220, 190, 210, 195, 205, 185, 215, 200]:
            baseline.add_sample(BehavioralSample(
                timestamp=0.0, mouse_velocity=float(vel),
            ))
        anomalous, _ = baseline.deviation_mouse(800.0)
        assert anomalous is True

    def test_not_ready_returns_zero(self):
        baseline = StudentBaseline(student_id="s1")
        anomalous, deviation = baseline.deviation_typing(100.0)
        assert anomalous is False
        assert deviation == 0.0

    def test_compute_all_deviations(self):
        baseline = StudentBaseline(student_id="s1")
        for _ in range(10):
            baseline.add_sample(BehavioralSample(
                timestamp=0.0, typing_wpm=50.0, mouse_velocity=200.0,
                face_present=0.98,
            ))
        devs = baseline.compute_all_deviations({
            "typing_wpm": 100.0,
            "mouse_velocity": 500.0,
            "face_present_rate": 0.7,
            "gaze_off_screen": 5.0,
        })
        assert "typing_wpm" in devs
        assert "mouse_velocity" in devs
        assert "face_presence" in devs
        assert "gaze_off_screen" in devs
        assert all(isinstance(d, tuple) and len(d) == 2 for d in devs.values())

    def test_to_dict(self):
        baseline = StudentBaseline(student_id="s1")
        d = baseline.to_dict()
        assert d["student_id"] == "s1"
        assert "typing_wpm" in d
        assert "ready" in d

    def test_update_updates_baseline(self):
        baseline = StudentBaseline(student_id="s1")
        baseline.add_sample(BehavioralSample(
            timestamp=0.0, typing_wpm=40.0, mouse_velocity=150.0,
            face_present=0.95,
        ))
        baseline.add_sample(BehavioralSample(
            timestamp=1.0, typing_wpm=60.0, mouse_velocity=250.0,
            face_present=1.0,
        ))
        # Mean should be between the two values
        assert 40.0 < baseline.typing_wpm_mean < 60.0
        assert 150.0 < baseline.mouse_velocity_mean < 250.0


class TestBaselineManager:
    def test_create_new_baseline(self):
        mgr = BaselineManager()
        baseline = mgr.get_or_create("s1")
        assert baseline.student_id == "s1"
        assert baseline.samples == 0

    def test_retrieve_existing_baseline(self):
        mgr = BaselineManager()
        mgr.get_or_create("s1")
        baseline = mgr.get("s1")
        assert baseline is not None
        assert baseline.student_id == "s1"

    def test_record_observation(self):
        mgr = BaselineManager()
        sample = BehavioralSample(
            timestamp=0.0, typing_wpm=50.0, mouse_velocity=200.0,
            face_present=0.98,
        )
        baseline = mgr.record_observation("s1", sample)
        assert baseline.samples == 1

    def test_remove_baseline(self):
        mgr = BaselineManager()
        mgr.get_or_create("s1")
        mgr.remove("s1")
        assert mgr.get("s1") is None

    def test_get_missing_returns_none(self):
        mgr = BaselineManager()
        assert mgr.get("nonexistent") is None

    def test_all_baselines(self):
        mgr = BaselineManager()
        mgr.get_or_create("s1")
        mgr.get_or_create("s2")
        all_b = mgr.all_baselines()
        assert len(all_b) == 2
        assert "s1" in all_b
        assert "s2" in all_b

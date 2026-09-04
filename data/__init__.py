"""Data package — sample data and persistent storage for the exam proctoring agent."""

from data.sample_data import (
    SAMPLE_QUESTIONS,
    SAMPLE_STUDENT_ANSWERS,
    SAMPLE_PROCTORING_EVENTS,
    load_sample_questions,
    load_sample_answers,
    load_sample_events,
    get_all_student_ids,
    save_questions_to_file,
)

__all__ = [
    "SAMPLE_QUESTIONS",
    "SAMPLE_STUDENT_ANSWERS",
    "SAMPLE_PROCTORING_EVENTS",
    "load_sample_questions",
    "load_sample_answers",
    "load_sample_events",
    "get_all_student_ids",
    "save_questions_to_file",
]

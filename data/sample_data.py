"""Sample exam data — questions, reference answers, and student responses for demo/testing."""

import json
from pathlib import Path

SAMPLE_QUESTIONS = [
    {
        "id": "q1",
        "type": "mcq",
        "text": "What is the time complexity of binary search?",
        "options": ["O(1)", "O(log n)", "O(n)", "O(n^2)"],
        "correct_answer": "O(log n)",
        "marks": 1.0,
        "keywords": [],
        "metadata": {"topic": "algorithms"}
    },
    {
        "id": "q2",
        "type": "mcq",
        "text": "Which data structure uses LIFO ordering?",
        "options": ["Queue", "Stack", "Linked List", "Array"],
        "correct_answer": "Stack",
        "marks": 1.0,
        "keywords": [],
        "metadata": {"topic": "data_structures"}
    },
    {
        "id": "q3",
        "type": "mcq",
        "text": "In Python, what does 'self' refer to in a class method?",
        "options": ["The class itself", "The instance of the class", "A global variable", "The parent class"],
        "correct_answer": "The instance of the class",
        "marks": 1.0,
        "keywords": [],
        "metadata": {"topic": "python"}
    },
    {
        "id": "q4",
        "type": "true_false",
        "text": "A linked list provides O(1) random access to elements.",
        "options": ["True", "False"],
        "correct_answer": "False",
        "marks": 1.0,
        "keywords": [],
        "metadata": {"topic": "data_structures"}
    },
    {
        "id": "q5",
        "type": "true_false",
        "text": "Python is a statically typed language.",
        "options": ["True", "False"],
        "correct_answer": "False",
        "marks": 1.0,
        "keywords": [],
        "metadata": {"topic": "python"}
    },
    {
        "id": "q6",
        "type": "numerical",
        "text": "Calculate the sum of the first 10 natural numbers (1+2+...+10).",
        "options": None,
        "correct_answer": 55,
        "marks": 2.0,
        "keywords": [],
        "tolerance": 0.02,
        "metadata": {"topic": "mathematics"}
    },
    {
        "id": "q7",
        "type": "numerical",
        "text": "What is 15% of 200?",
        "options": None,
        "correct_answer": 30,
        "marks": 1.5,
        "keywords": [],
        "tolerance": 0.02,
        "metadata": {"topic": "mathematics"}
    },
    {
        "id": "q8",
        "type": "short_answer",
        "text": "Explain the concept of recursion in programming.",
        "options": None,
        "correct_answer": "A function that calls itself to solve smaller subproblems until a base case is reached",
        "marks": 3.0,
        "keywords": ["function", "calls itself", "base case", "subproblem"],
        "tolerance": 0.02,
        "metadata": {"topic": "algorithms"}
    },
    {
        "id": "q9",
        "type": "short_answer",
        "text": "What is the purpose of a constructor in OOP?",
        "options": None,
        "correct_answer": "A constructor initializes object attributes and allocates memory when an object is created",
        "marks": 2.5,
        "keywords": ["initialize", "object", "attributes", "create"],
        "tolerance": 0.02,
        "metadata": {"topic": "oop"}
    },
    {
        "id": "q10",
        "type": "mcq",
        "text": "Which sorting algorithm has the best average-case time complexity?",
        "options": ["Bubble Sort", "Merge Sort", "Selection Sort", "Insertion Sort"],
        "correct_answer": "Merge Sort",
        "marks": 1.0,
        "keywords": [],
        "metadata": {"topic": "algorithms"}
    },
]

SAMPLE_STUDENT_ANSWERS = {
    "student_001": {
        "student_id": "student_001",
        "name": "Priya Sharma",
        "answers": {
            "q1": "O(log n)",
            "q2": "Stack",
            "q3": "The instance of the class",
            "q4": "False",
            "q5": "False",
            "q6": 55,
            "q7": 30,
            "q8": "Recursion is when a function calls itself to break down a problem into smaller parts until reaching a base case.",
            "q9": "The constructor initializes an object's attributes when it is created.",
            "q10": "Merge Sort",
        }
    },
    "student_002": {
        "student_id": "student_002",
        "name": "Rahul Verma",
        "answers": {
            "q1": "O(n)",
            "q2": "Queue",
            "q3": "The class itself",
            "q4": "True",
            "q5": "True",
            "q6": 50,
            "q7": 30,
            "q8": "Recursion is a technique where a function calls another function to solve the problem.",
            "q9": "A constructor is used to destroy objects in memory.",
            "q10": "Merge Sort",
        }
    },
    "student_003": {
        "student_id": "student_003",
        "name": "Ananya Iyer",
        "answers": {
            "q1": "O(log n)",
            "q2": "Stack",
            "q3": "The instance of the class",
            "q4": "False",
            "q5": "False",
            "q6": 55,
            "q7": 30,
            "q8": "Recursion is when a function calls itself to solve smaller subproblems until a base case is reached.",
            "q9": "The constructor initializes object attributes when an object is instantiated.",
            "q10": "Merge Sort",
        }
    },
}

SAMPLE_PROCTORING_EVENTS = {
    "student_001": {
        "student_id": "student_001",
        "events": [
            {"type": "head_turned", "confidence": 0.3, "timestamp": "2026-01-01T10:05:00"},
            {"type": "tab_switch", "confidence": 0.5, "timestamp": "2026-01-01T10:12:00"},
        ]
    },
    "student_002": {
        "student_id": "student_002",
        "events": [
            {"type": "face_absent", "confidence": 0.8, "timestamp": "2026-01-01T10:03:00"},
            {"type": "face_absent", "confidence": 0.9, "timestamp": "2026-01-01T10:04:00"},
            {"type": "multiple_faces", "confidence": 0.7, "timestamp": "2026-01-01T10:08:00"},
            {"type": "copy_paste", "confidence": 0.85, "timestamp": "2026-01-01T10:15:00"},
        ]
    },
    "student_003": {
        "student_id": "student_003",
        "events": [
            {"type": "loud_audio", "confidence": 0.4, "timestamp": "2026-01-01T10:06:00"},
            {"type": "head_turned", "confidence": 0.5, "timestamp": "2026-01-01T10:10:00"},
        ]
    },
}


def load_sample_questions() -> list:
    """Return sample questions as dicts."""
    return SAMPLE_QUESTIONS


def load_sample_answers(student_id: str) -> dict:
    """Return answers for a given student."""
    return SAMPLE_STUDENT_ANSWERS.get(student_id, {})


def load_sample_events(student_id: str) -> dict:
    """Return proctoring events for a given student."""
    return SAMPLE_PROCTORING_EVENTS.get(student_id, {"student_id": student_id, "events": []})


def get_all_student_ids() -> list:
    """Return all sample student IDs."""
    return list(SAMPLE_STUDENT_ANSWERS.keys())


def save_questions_to_file(path: str = "data/questions.json") -> None:
    """Write sample questions to JSON file for QuestionBank to load."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(SAMPLE_QUESTIONS, f, indent=2)

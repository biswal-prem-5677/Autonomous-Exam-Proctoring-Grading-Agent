"""Question bank — CRUD operations for exam questions."""

import json
import uuid
from typing import List, Optional, Dict, Any
from pathlib import Path

from src.models.models import Question, QuestionType


class QuestionBank:
    """Manages a collection of exam questions with persistence."""

    def __init__(self, storage_path: str = "data/questions.json"):
        self.storage_path = Path(storage_path)
        self.questions: Dict[str, Question] = {}
        self._load()

    def _load(self) -> None:
        """Load questions from disk."""
        if self.storage_path.exists():
            with open(self.storage_path, "r") as f:
                data = json.load(f)
                for q_data in data:
                    q = Question(
                        id=q_data["id"],
                        type=QuestionType(q_data["type"]),
                        text=q_data["text"],
                        options=q_data.get("options"),
                        correct_answer=q_data.get("correct_answer"),
                        marks=q_data.get("marks", 1.0),
                        keywords=q_data.get("keywords", []),
                        tolerance=q_data.get("tolerance", 0.02),
                        metadata=q_data.get("metadata", {}),
                    )
                    self.questions[q.id] = q

    def _save(self) -> None:
        """Persist questions to disk."""
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        data = [q.to_dict() for q in self.questions.values()]
        with open(self.storage_path, "w") as f:
            json.dump(data, f, indent=2)

    def add_question(self, question: Question) -> Question:
        """Add a question to the bank."""
        if question.id is None:
            question.id = str(uuid.uuid4())[:8]
        self.questions[question.id] = question
        self._save()
        return question

    def get_question(self, question_id: str) -> Optional[Question]:
        """Retrieve a question by ID."""
        return self.questions.get(question_id)

    def get_all(self) -> List[Question]:
        """Return all questions."""
        return list(self.questions.values())

    def get_by_type(self, q_type: QuestionType) -> List[Question]:
        """Filter questions by type."""
        return [q for q in self.questions.values() if q.type == q_type]

    def remove_question(self, question_id: str) -> bool:
        """Remove a question by ID."""
        if question_id in self.questions:
            del self.questions[question_id]
            self._save()
            return True
        return False

    def get_random(self, n: int, q_type: Optional[QuestionType] = None) -> List[Question]:
        """Get n random questions, optionally filtered by type."""
        import random
        pool = self.get_by_type(q_type) if q_type else self.get_all()
        return random.sample(pool, min(n, len(pool)))

    def bulk_add(self, questions: List[Question]) -> List[Question]:
        """Add multiple questions at once."""
        for q in questions:
            self.add_question(q)
        return questions

    def to_dict(self) -> Dict[str, Any]:
        """Serialize all questions to a dictionary."""
        return {q.id: q.to_dict() for q in self.questions.values()}

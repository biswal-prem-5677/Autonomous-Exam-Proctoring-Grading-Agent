"""SQLite database for persistent exam session and event storage."""

import json
import sqlite3
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional


DB_PATH = Path(__file__).parent.parent.parent / "data" / "exam_agent.db"


def _ensure_parent() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)


SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id     TEXT PRIMARY KEY,
    student_id     TEXT NOT NULL,
    exam_id        TEXT NOT NULL,
    state          TEXT DEFAULT 'normal',
    risk_score     REAL DEFAULT 0.0,
    duration_minutes INTEGER DEFAULT 60,
    started_at     TEXT,
    ended_at       TEXT,
    created_at     TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS questions (
    question_id    TEXT PRIMARY KEY,
    session_id     TEXT NOT NULL,
    type           TEXT NOT NULL,
    text           TEXT NOT NULL,
    options        TEXT,           -- JSON array for MCQ
    correct_answer TEXT,
    marks          REAL DEFAULT 1.0,
    tolerance      REAL DEFAULT 0.01,
    keywords       TEXT,           -- JSON list
    reference_answer TEXT,
    position       INTEGER DEFAULT 0,
    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
);

CREATE TABLE IF NOT EXISTS answers (
    answer_id      TEXT PRIMARY KEY,
    session_id     TEXT NOT NULL,
    question_id    TEXT NOT NULL,
    student_answer TEXT,
    score          REAL,
    max_score      REAL,
    grading_method TEXT,
    grading_details TEXT,          -- JSON
    submitted_at   TEXT DEFAULT (datetime('now')),
    FOREIGN KEY (session_id) REFERENCES sessions(session_id),
    FOREIGN KEY (question_id) REFERENCES questions(question_id)
);

CREATE TABLE IF NOT EXISTS events (
    event_id       TEXT PRIMARY KEY,
    session_id     TEXT NOT NULL,
    event_type     TEXT NOT NULL,
    confidence     REAL,
    details        TEXT,           -- JSON
    timestamp      TEXT DEFAULT (datetime('now')),
    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
);

CREATE TABLE IF NOT EXISTS predictions (
    prediction_id  TEXT PRIMARY KEY,
    session_id     TEXT NOT NULL,
    question_id    TEXT,
    predicted_score REAL,
    confidence     REAL,
    method         TEXT,
    created_at     TEXT DEFAULT (datetime('now')),
    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
);

CREATE INDEX IF NOT EXISTS idx_events_session   ON events(session_id);
CREATE INDEX IF NOT EXISTS idx_answers_session  ON answers(session_id);
CREATE INDEX IF NOT EXISTS idx_questions_session ON questions(session_id);
"""


def init_db(db_path: Optional[Path] = None) -> None:
    """Create tables if they don't exist."""
    global DB_PATH
    target = db_path or DB_PATH
    if db_path:
        DB_PATH = db_path
    _ensure_parent()
    conn = sqlite3.connect(str(DB_PATH))
    conn.executescript(SCHEMA)
    conn.commit()
    conn.close()


class Database:
    """High-level database interface for the exam agent."""

    def __init__(self, db_path: Optional[Path] = None):
        if db_path:
            global DB_PATH
            DB_PATH = db_path
        _ensure_parent()
        self._conn = sqlite3.connect(str(DB_PATH))
        self._conn.row_factory = sqlite3.Row
        init_db(DB_PATH)

    def close(self) -> None:
        self._conn.close()

    # ─── Sessions ────────────────────────────────────────────────────────────

    def create_session(self, student_id: str, exam_id: str,
                       duration_minutes: int = 60) -> str:
        session_id = str(uuid.uuid4())[:12]
        self._conn.execute(
            """INSERT INTO sessions (session_id, student_id, exam_id, duration_minutes)
               VALUES (?, ?, ?, ?)""",
            (session_id, student_id, exam_id, duration_minutes),
        )
        self._conn.commit()
        return session_id

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        row = self._conn.execute(
            "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
        ).fetchone()
        return dict(row) if row else None

    def end_session(self, session_id: str) -> None:
        self._conn.execute(
            "UPDATE sessions SET ended_at = ?, state = 'ended' WHERE session_id = ?",
            (datetime.utcnow().isoformat(), session_id),
        )
        self._conn.commit()

    def update_risk(self, session_id: str, risk_score: float, state: str) -> None:
        self._conn.execute(
            "UPDATE sessions SET risk_score = ?, state = ? WHERE session_id = ?",
            (risk_score, state, session_id),
        )
        self._conn.commit()

    def list_sessions(self, limit: int = 50) -> List[Dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM sessions ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]

    # ─── Questions ───────────────────────────────────────────────────────────

    def add_question(self, session_id: str, question_id: str, qtype: str,
                     text: str, marks: float = 1.0,
                     correct_answer: str = None, tolerance: float = 0.01,
                     keywords: List[str] = None, reference_answer: str = None,
                     options: List[str] = None, position: int = 0) -> None:
        self._conn.execute(
            """INSERT INTO questions
               (question_id, session_id, type, text, options, correct_answer,
                marks, tolerance, keywords, reference_answer, position)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                question_id, session_id, qtype, text,
                json.dumps(options) if options else None,
                correct_answer, marks, tolerance,
                json.dumps(keywords) if keywords else None,
                reference_answer, position,
            ),
        )
        self._conn.commit()

    def get_questions(self, session_id: str) -> List[Dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM questions WHERE session_id = ? ORDER BY position",
            (session_id,),
        ).fetchall()
        results = []
        for r in rows:
            d = dict(r)
            if d.get("options"):
                d["options"] = json.loads(d["options"])
            if d.get("keywords"):
                d["keywords"] = json.loads(d["keywords"])
            results.append(d)
        return results

    # ─── Answers ─────────────────────────────────────────────────────────────

    def record_answer(self, session_id: str, question_id: str,
                      student_answer: str, score: float = None,
                      max_score: float = None, grading_method: str = None,
                      grading_details: Dict = None) -> str:
        answer_id = str(uuid.uuid4())[:12]
        self._conn.execute(
            """INSERT INTO answers
               (answer_id, session_id, question_id, student_answer,
                score, max_score, grading_method, grading_details)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                answer_id, session_id, question_id, student_answer,
                score, max_score, grading_method,
                json.dumps(grading_details) if grading_details else None,
            ),
        )
        self._conn.commit()
        return answer_id

    def get_answers(self, session_id: str) -> List[Dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM answers WHERE session_id = ?", (session_id,)
        ).fetchall()
        results = []
        for r in rows:
            d = dict(r)
            if d.get("grading_details"):
                d["grading_details"] = json.loads(d["grading_details"])
            results.append(d)
        return results

    def get_session_results(self, session_id: str) -> Dict[str, Any]:
        answers = self.get_answers(session_id)
        session = self.get_session(session_id)
        total_score = sum(a["score"] or 0.0 for a in answers)
        total_max = sum(a["max_score"] or 0.0 for a in answers)
        return {
            "session_id": session_id,
            "student_id": session["student_id"] if session else None,
            "exam_id": session["exam_id"] if session else None,
            "total_score": total_score,
            "total_max_score": total_max,
            "percentage": round(total_score / total_max * 100, 2) if total_max > 0 else 0.0,
            "answers_count": len(answers),
            "answers": answers,
        }

    # ─── Events ──────────────────────────────────────────────────────────────

    def add_event(self, session_id: str, event_type: str,
                  confidence: float = None, details: Dict = None) -> str:
        event_id = str(uuid.uuid4())[:12]
        self._conn.execute(
            """INSERT INTO events (event_id, session_id, event_type, confidence, details)
               VALUES (?, ?, ?, ?, ?)""",
            (
                event_id, session_id, event_type, confidence,
                json.dumps(details) if details else None,
            ),
        )
        self._conn.commit()
        return event_id

    def get_events(self, session_id: str,
                   since: Optional[datetime] = None) -> List[Dict[str, Any]]:
        query = "SELECT * FROM events WHERE session_id = ?"
        params: list = [session_id]
        if since:
            query += " AND timestamp >= ?"
            params.append(since.isoformat())
        query += " ORDER BY timestamp"
        rows = self._conn.execute(query, params).fetchall()
        results = []
        for r in rows:
            d = dict(r)
            if d.get("details"):
                d["details"] = json.loads(d["details"])
            results.append(d)
        return results

    def get_event_counts(self, session_id: str) -> Dict[str, int]:
        rows = self._conn.execute(
            """SELECT event_type, COUNT(*) as cnt
               FROM events WHERE session_id = ?
               GROUP BY event_type""",
            (session_id,),
        ).fetchall()
        return {r["event_type"]: r["cnt"] for r in rows}

    # ─── Predictions ─────────────────────────────────────────────────────────

    def save_prediction(self, session_id: str, predicted_score: float,
                        confidence: float, method: str,
                        question_id: str = None) -> str:
        pred_id = str(uuid.uuid4())[:12]
        self._conn.execute(
            """INSERT INTO predictions
               (prediction_id, session_id, question_id, predicted_score,
                confidence, method)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (pred_id, session_id, question_id, predicted_score, confidence, method),
        )
        self._conn.commit()
        return pred_id

"""SQLite database for persistent exam session and event storage."""

import json
import sqlite3
import threading
import uuid
from datetime import datetime, timedelta, timezone
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

CREATE TABLE IF NOT EXISTS session_meta (
    session_key  TEXT PRIMARY KEY,
    session_id   TEXT NOT NULL,
    student_id   TEXT NOT NULL,
    exam_id      TEXT NOT NULL,
    meta_json    TEXT NOT NULL,          -- JSON blob with arbitrary session fields
    is_active    INTEGER DEFAULT 1,      -- 1 = live, 0 = ended
    created_at   TEXT DEFAULT (datetime('now')),
    updated_at   TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_session_meta_active ON session_meta(is_active);

CREATE TABLE IF NOT EXISTS exams (
    exam_id          TEXT PRIMARY KEY,
    title            TEXT NOT NULL,
    duration_minutes INTEGER DEFAULT 60,
    topic            TEXT DEFAULT '',
    difficulty       TEXT DEFAULT 'medium',
    instructions     TEXT DEFAULT '',
    passing_score    REAL DEFAULT 40.0,
    is_released      INTEGER DEFAULT 0,
    settings         TEXT DEFAULT '{}',
    created_by       TEXT,
    created_at       TEXT DEFAULT (datetime('now')),
    updated_at       TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS exam_questions (
    question_id      TEXT NOT NULL,
    exam_id          TEXT NOT NULL,
    type             TEXT NOT NULL,
    text             TEXT NOT NULL,
    options          TEXT,           -- JSON array for MCQ
    correct_answer   TEXT,
    marks            REAL DEFAULT 1.0,
    tolerance        REAL,
    keywords         TEXT,           -- JSON list
    reference_answer TEXT,
    rubric           TEXT,           -- JSON
    metadata         TEXT,           -- JSON
    position         INTEGER DEFAULT 0,
    PRIMARY KEY (exam_id, question_id),
    FOREIGN KEY (exam_id) REFERENCES exams(exam_id)
);
CREATE INDEX IF NOT EXISTS idx_exam_questions_exam ON exam_questions(exam_id);

CREATE TABLE IF NOT EXISTS exam_roster (
    roster_id        TEXT PRIMARY KEY,
    exam_id          TEXT NOT NULL,
    student_id       TEXT NOT NULL,
    student_email    TEXT DEFAULT '',
    extra_time_mins  INTEGER DEFAULT 0,
    start_window     TEXT,
    end_window       TEXT,
    enrolled_at      TEXT DEFAULT (datetime('now')),
    status           TEXT DEFAULT 'enrolled',
    UNIQUE(exam_id, student_id)
);
CREATE INDEX IF NOT EXISTS idx_roster_exam_student ON exam_roster(exam_id, student_id);

CREATE TABLE IF NOT EXISTS grade_overrides (
    override_id      TEXT PRIMARY KEY,
    session_id       TEXT NOT NULL,
    question_id      TEXT NOT NULL,
    previous_score   REAL,
    new_score        REAL NOT NULL,
    reason           TEXT NOT NULL,
    examiner_id      TEXT NOT NULL,
    timestamp        TEXT DEFAULT (datetime('now')),
    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
);
CREATE INDEX IF NOT EXISTS idx_overrides_session ON grade_overrides(session_id);

CREATE TABLE IF NOT EXISTS examiner_actions (
    action_id        TEXT PRIMARY KEY,
    session_id       TEXT NOT NULL,
    action_type      TEXT NOT NULL, -- warn, flag, pause, terminate, note
    message          TEXT,
    examiner_id      TEXT NOT NULL,
    timestamp        TEXT DEFAULT (datetime('now')),
    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
);
CREATE INDEX IF NOT EXISTS idx_actions_session ON examiner_actions(session_id);

CREATE TABLE IF NOT EXISTS agent_states (
    session_id       TEXT PRIMARY KEY,
    agent_session_id TEXT,
    student_id       TEXT NOT NULL,
    exam_id          TEXT NOT NULL,
    current_state    TEXT,
    current_risk     REAL,
    risk_history     TEXT,          -- JSON list
    evidence_log     TEXT,          -- JSON list
    trace_data       TEXT,          -- JSON dict
    updated_at       TEXT DEFAULT (datetime('now'))
);
"""


def init_db(db_path: Optional[Path] = None) -> None:
    """Create tables if they don't exist."""
    global DB_PATH
    if db_path:
        DB_PATH = db_path
    _ensure_parent()
    conn = sqlite3.connect(str(DB_PATH))
    conn.executescript(SCHEMA)
    # Migrate exam_questions to composite primary key if needed
    try:
        pk_info = conn.execute("PRAGMA table_info(exam_questions)").fetchall()
        pk_cols = [row[1] for row in pk_info if row[5] > 0]
        if pk_cols == ["question_id"]:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS exam_questions_new (
                    question_id      TEXT NOT NULL,
                    exam_id          TEXT NOT NULL,
                    type             TEXT NOT NULL,
                    text             TEXT NOT NULL,
                    options          TEXT,
                    correct_answer   TEXT,
                    marks            REAL DEFAULT 1.0,
                    tolerance        REAL,
                    keywords         TEXT,
                    reference_answer TEXT,
                    rubric           TEXT,
                    metadata         TEXT,
                    position         INTEGER DEFAULT 0,
                    PRIMARY KEY (exam_id, question_id),
                    FOREIGN KEY (exam_id) REFERENCES exams(exam_id)
                );
                INSERT OR IGNORE INTO exam_questions_new SELECT * FROM exam_questions;
                DROP TABLE exam_questions;
                ALTER TABLE exam_questions_new RENAME TO exam_questions;
                CREATE INDEX IF NOT EXISTS idx_exam_questions_exam ON exam_questions(exam_id);
            """)
    except Exception:
        pass
    conn.commit()
    conn.close()


class Database:
    """High-level database interface for the exam agent."""

    def __init__(self, db_path: Optional[Path] = None):
        global DB_PATH
        if db_path:
            DB_PATH = db_path
        _ensure_parent()
        # check_same_thread=False is safe here because we serialise writes via
        # _write_lock; reads in Flask worker threads are naturally serialised by
        # SQLite's reader-writer locking.
        self._conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._write_lock = threading.Lock()
        init_db(DB_PATH)

    def close(self) -> None:
        self._conn.close()

    # ─── Sessions ────────────────────────────────────────────────────────────

    def create_session(self, student_id: str, exam_id: str,
                       duration_minutes: int = 60) -> str:
        session_id = str(uuid.uuid4())[:12]
        with self._write_lock:
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
        with self._write_lock:
            self._conn.execute(
                "UPDATE sessions SET ended_at = ?, state = 'ended' WHERE session_id = ?",
                (datetime.now(timezone.utc).isoformat(), session_id),
            )
            self._conn.commit()

    def update_risk(self, session_id: str, risk_score: float, state: str) -> None:
        with self._write_lock:
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
                     correct_answer: Optional[str] = None, tolerance: float = 0.01,
                     keywords: Optional[List[str]] = None, reference_answer: Optional[str] = None,
                     options: Optional[List[str]] = None, position: int = 0) -> None:
        with self._write_lock:
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
                      student_answer: str, score: Optional[float] = None,
                      max_score: Optional[float] = None, grading_method: Optional[str] = None,
                      grading_details: Optional[Dict] = None) -> str:
        answer_id = str(uuid.uuid4())[:12]
        with self._write_lock:
            existing = self._conn.execute(
                "SELECT answer_id FROM answers WHERE session_id = ? AND question_id = ?",
                (session_id, question_id)
            ).fetchone()
            if existing:
                self._conn.execute(
                    """UPDATE answers SET student_answer = ?, score = coalesce(?, score),
                       max_score = coalesce(?, max_score), grading_method = coalesce(?, grading_method),
                       grading_details = coalesce(?, grading_details), submitted_at = datetime('now')
                       WHERE session_id = ? AND question_id = ?""",
                    (
                        student_answer, score, max_score, grading_method,
                        json.dumps(grading_details) if grading_details else None,
                        session_id, question_id
                    )
                )
                self._conn.commit()
                return existing["answer_id"]
            else:
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
                  confidence: Optional[float] = None, details: Optional[Dict] = None) -> str:
        event_id = str(uuid.uuid4())[:12]
        with self._write_lock:
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
                        question_id: Optional[str] = None) -> str:
        pred_id = str(uuid.uuid4())[:12]
        with self._write_lock:
            self._conn.execute(
                """INSERT INTO predictions
                   (prediction_id, session_id, question_id, predicted_score,
                    confidence, method)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (pred_id, session_id, question_id, predicted_score, confidence, method),
            )
            self._conn.commit()
        return pred_id

    # ─── Session Metadata (server-restart recovery) ───────────────────────────

    def upsert_session_meta(
        self,
        session_key: str,
        session_id: str,
        student_id: str,
        exam_id: str,
        meta: Dict[str, Any],
        is_active: bool = True,
    ) -> None:
        """Persist lightweight session metadata so the server can recover state
        after a restart without re-running the full exam-creation flow."""
        with self._write_lock:
            self._conn.execute(
                """INSERT INTO session_meta
                       (session_key, session_id, student_id, exam_id, meta_json, is_active, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, datetime('now'))
                   ON CONFLICT(session_key) DO UPDATE SET
                       meta_json  = excluded.meta_json,
                       is_active  = excluded.is_active,
                       updated_at = excluded.updated_at""",
                (
                    session_key, session_id, student_id, exam_id,
                    json.dumps(meta), 1 if is_active else 0,
                ),
            )
            self._conn.commit()

    def deactivate_session_meta(self, session_key: str) -> None:
        """Mark a session as no longer active (ended/submitted)."""
        with self._write_lock:
            self._conn.execute(
                """UPDATE session_meta
                   SET is_active = 0, updated_at = datetime('now')
                   WHERE session_key = ?""",
                (session_key,),
            )
            self._conn.commit()

    def get_active_sessions_meta(self) -> List[Dict[str, Any]]:
        """Return all sessions that were active when the server last shut down.
        Used on startup to repopulate the in-memory session dict."""
        rows = self._conn.execute(
            "SELECT * FROM session_meta WHERE is_active = 1 ORDER BY created_at"
        ).fetchall()
        results = []
        for r in rows:
            d = dict(r)
            try:
                d["meta"] = json.loads(d.pop("meta_json", "{}"))
            except (json.JSONDecodeError, TypeError):
                d["meta"] = {}
            results.append(d)
        return results

    # ─── Exams Persistence ───────────────────────────────────────────────────

    def save_exam(self, exam_data: Dict[str, Any], created_by: Optional[str] = None) -> str:
        """Persist an exam and all its questions into SQLite."""
        exam_id = exam_data.get("exam_id") or exam_data.get("id") or str(uuid.uuid4())[:8]
        title = exam_data.get("title", "Untitled Exam")
        duration = int(exam_data.get("duration_minutes", 60))
        topic = exam_data.get("topic", "")
        difficulty = exam_data.get("difficulty", "medium")
        instructions = exam_data.get("instructions", "")
        passing_score = float(exam_data.get("passing_score", 40.0))
        is_released = 1 if exam_data.get("is_released") else 0
        settings = exam_data.get("settings", {})
        
        with self._write_lock:
            self._conn.execute(
                """INSERT INTO exams
                   (exam_id, title, duration_minutes, topic, difficulty, instructions,
                    passing_score, is_released, settings, created_by, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
                   ON CONFLICT(exam_id) DO UPDATE SET
                       title            = excluded.title,
                       duration_minutes = excluded.duration_minutes,
                       topic            = excluded.topic,
                       difficulty       = excluded.difficulty,
                       instructions     = excluded.instructions,
                       passing_score    = excluded.passing_score,
                       is_released      = excluded.is_released,
                       settings         = excluded.settings,
                       updated_at       = datetime('now')""",
                (
                    exam_id, title, duration, topic, difficulty, instructions,
                    passing_score, is_released, json.dumps(settings), created_by
                ),
            )
            # Delete existing questions and re-insert
            self._conn.execute("DELETE FROM exam_questions WHERE exam_id = ?", (exam_id,))
            questions = exam_data.get("questions", [])
            for idx, q in enumerate(questions):
                qid = q.get("id") or str(uuid.uuid4())[:8]
                qtype = q.get("type", "mcq")
                if hasattr(qtype, "value"):
                    qtype = qtype.value
                self._conn.execute(
                    """INSERT INTO exam_questions
                       (question_id, exam_id, type, text, options, correct_answer,
                        marks, tolerance, keywords, reference_answer, rubric, metadata, position)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        qid, exam_id, str(qtype), q.get("text", ""),
                        json.dumps(q.get("options")) if q.get("options") is not None else None,
                        str(q.get("correct_answer")) if q.get("correct_answer") is not None else None,
                        float(q.get("marks", 1.0)),
                        float(q.get("tolerance")) if q.get("tolerance") is not None else None,
                        json.dumps(q.get("keywords")) if q.get("keywords") is not None else None,
                        q.get("reference_answer"),
                        json.dumps(q.get("rubric")) if q.get("rubric") is not None else None,
                        json.dumps(q.get("metadata")) if q.get("metadata") is not None else None,
                        idx
                    )
                )
            self._conn.commit()
        return exam_id

    def get_exam(self, exam_id: str) -> Optional[Dict[str, Any]]:
        row = self._conn.execute(
            "SELECT * FROM exams WHERE exam_id = ?", (exam_id,)
        ).fetchone()
        if not row:
            return None
        d = dict(row)
        d["is_released"] = bool(d.get("is_released", 0))
        try:
            d["settings"] = json.loads(d.get("settings") or "{}")
        except Exception:
            d["settings"] = {}

        qrows = self._conn.execute(
            "SELECT * FROM exam_questions WHERE exam_id = ? ORDER BY position", (exam_id,)
        ).fetchall()
        questions = []
        for qr in qrows:
            qd = dict(qr)
            qd["id"] = qd.pop("question_id")
            for k in ["options", "keywords", "rubric", "metadata"]:
                if qd.get(k):
                    try:
                        qd[k] = json.loads(qd[k])
                    except Exception:
                        pass
                else:
                    qd[k] = [] if k in ["options", "keywords"] else {}
            questions.append(qd)
        d["questions"] = questions
        return d

    def list_exams(self) -> List[Dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM exams ORDER BY created_at DESC"
        ).fetchall()
        result = []
        for r in rows:
            exam = self.get_exam(r["exam_id"])
            if exam:
                result.append(exam)
        return result

    def set_exam_released(self, exam_id: str, is_released: bool = True) -> bool:
        with self._write_lock:
            cur = self._conn.execute(
                "UPDATE exams SET is_released = ?, updated_at = datetime('now') WHERE exam_id = ?",
                (1 if is_released else 0, exam_id)
            )
            self._conn.commit()
            return cur.rowcount > 0

    # ─── Roster & Enrollment ─────────────────────────────────────────────────

    def enroll_student(self, exam_id: str, student_id: str, student_email: str = "",
                       extra_time_mins: int = 0, start_window: Optional[str] = None,
                       end_window: Optional[str] = None) -> str:
        roster_id = str(uuid.uuid4())[:12]
        with self._write_lock:
            self._conn.execute(
                """INSERT INTO exam_roster
                   (roster_id, exam_id, student_id, student_email, extra_time_mins,
                    start_window, end_window, enrolled_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, datetime('now'))
                   ON CONFLICT(exam_id, student_id) DO UPDATE SET
                       student_email   = excluded.student_email,
                       extra_time_mins = excluded.extra_time_mins,
                       start_window    = excluded.start_window,
                       end_window      = excluded.end_window""",
                (roster_id, exam_id, student_id, student_email, extra_time_mins,
                 start_window, end_window)
            )
            self._conn.commit()
        return roster_id

    def remove_student_from_roster(self, exam_id: str, student_id: str) -> bool:
        with self._write_lock:
            cur = self._conn.execute(
                "DELETE FROM exam_roster WHERE exam_id = ? AND student_id = ?",
                (exam_id, student_id)
            )
            self._conn.commit()
            return cur.rowcount > 0

    def get_roster(self, exam_id: str) -> List[Dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM exam_roster WHERE exam_id = ? ORDER BY enrolled_at", (exam_id,)
        ).fetchall()
        return [dict(r) for r in rows]

    def is_student_enrolled(self, exam_id: str, student_id: str) -> bool:
        # If no students are in the roster, the exam is open to any enrolled student.
        # Otherwise, student_id must be in exam_roster.
        count_all = self._conn.execute(
            "SELECT COUNT(*) as cnt FROM exam_roster WHERE exam_id = ?", (exam_id,)
        ).fetchone()
        if not count_all or count_all["cnt"] == 0:
            return True
        row = self._conn.execute(
            "SELECT roster_id FROM exam_roster WHERE exam_id = ? AND student_id = ?",
            (exam_id, student_id)
        ).fetchone()
        return row is not None

    def get_student_enrollment(self, exam_id: str, student_id: str) -> Optional[Dict[str, Any]]:
        row = self._conn.execute(
            "SELECT * FROM exam_roster WHERE exam_id = ? AND student_id = ?",
            (exam_id, student_id)
        ).fetchone()
        return dict(row) if row else None

    # ─── Grade Overrides ─────────────────────────────────────────────────────

    def record_grade_override(self, session_id: str, question_id: str,
                              previous_score: Optional[float], new_score: float,
                              reason: str, examiner_id: str) -> str:
        override_id = str(uuid.uuid4())[:12]
        with self._write_lock:
            self._conn.execute(
                """INSERT INTO grade_overrides
                   (override_id, session_id, question_id, previous_score, new_score, reason, examiner_id)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (override_id, session_id, question_id, previous_score, new_score, reason, examiner_id)
            )
            # Update the answer score in answers table if present
            self._conn.execute(
                """UPDATE answers SET score = ?, grading_method = 'manual_override'
                   WHERE session_id = ? AND question_id = ?""",
                (new_score, session_id, question_id)
            )
            self._conn.commit()
        return override_id

    def get_grade_overrides(self, session_id: Optional[str] = None) -> List[Dict[str, Any]]:
        if session_id:
            rows = self._conn.execute(
                "SELECT * FROM grade_overrides WHERE session_id = ? ORDER BY timestamp DESC",
                (session_id,)
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM grade_overrides ORDER BY timestamp DESC"
            ).fetchall()
        result = []
        for r in rows:
            d = dict(r)
            d["override_score"] = d.get("new_score")
            result.append(d)
        return result

    # ─── Examiner Actions ────────────────────────────────────────────────────

    def record_examiner_action(self, session_id: str, action_type: str,
                               message: str, examiner_id: str) -> str:
        action_id = str(uuid.uuid4())[:12]
        with self._write_lock:
            self._conn.execute(
                """INSERT INTO examiner_actions
                   (action_id, session_id, action_type, message, examiner_id)
                   VALUES (?, ?, ?, ?, ?)""",
                (action_id, session_id, action_type, message, examiner_id)
            )
            self._conn.commit()
        return action_id

    def get_examiner_actions(self, session_id: str) -> List[Dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM examiner_actions WHERE session_id = ? ORDER BY timestamp ASC",
            (session_id,)
        ).fetchall()
        return [dict(r) for r in rows]

    # ─── Agent State Persistence ─────────────────────────────────────────────

    def save_agent_state(self, session_id: str, agent_session_id: str,
                         student_id: str, exam_id: str, current_state: str,
                         current_risk: float, risk_history: List[Any],
                         evidence_log: List[Any], trace_data: Dict[str, Any]) -> None:
        with self._write_lock:
            self._conn.execute(
                """INSERT INTO agent_states
                   (session_id, agent_session_id, student_id, exam_id, current_state,
                    current_risk, risk_history, evidence_log, trace_data, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
                   ON CONFLICT(session_id) DO UPDATE SET
                       agent_session_id = excluded.agent_session_id,
                       current_state    = excluded.current_state,
                       current_risk     = excluded.current_risk,
                       risk_history     = excluded.risk_history,
                       evidence_log     = excluded.evidence_log,
                       trace_data       = excluded.trace_data,
                       updated_at       = datetime('now')""",
                (
                    session_id, agent_session_id, student_id, exam_id,
                    current_state, current_risk, json.dumps(risk_history),
                    json.dumps(evidence_log), json.dumps(trace_data)
                )
            )
            self._conn.commit()

    def get_agent_state(self, session_id: str) -> Optional[Dict[str, Any]]:
        row = self._conn.execute(
            "SELECT * FROM agent_states WHERE session_id = ?", (session_id,)
        ).fetchone()
        if not row:
            return None
        d = dict(row)
        for k in ["risk_history", "evidence_log", "trace_data"]:
            if d.get(k):
                try:
                    d[k] = json.loads(d[k])
                except Exception:
                    pass
        return d

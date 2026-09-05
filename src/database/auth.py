"""Authentication database — users, organizations, sessions."""

import json
import uuid
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, List, Any
import sqlite3

from src.models.auth_models import User, UserRole, Organization, OrganizationType

DB_PATH = Path(__file__).parent.parent.parent / "data" / "auth.db"


def _ensure_parent() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)


AUTH_SCHEMA = """
CREATE TABLE IF NOT EXISTS organizations (
    org_id       TEXT PRIMARY KEY,
    name         TEXT NOT NULL,
    org_type     TEXT NOT NULL,
    admin_id     TEXT NOT NULL,
    settings     TEXT DEFAULT '{}',
    created_at   TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS users (
    user_id      TEXT PRIMARY KEY,
    username     TEXT UNIQUE NOT NULL,
    email        TEXT UNIQUE NOT NULL,
    full_name    TEXT NOT NULL,
    password_hash TEXT NOT NULL,
    role         TEXT NOT NULL DEFAULT 'student',
    org_id       TEXT,
    avatar_url   TEXT DEFAULT '',
    preferences  TEXT DEFAULT '{}',
    created_at   TEXT DEFAULT (datetime('now')),
    last_login   TEXT,
    is_active    INTEGER DEFAULT 1,
    FOREIGN KEY (org_id) REFERENCES organizations(org_id)
);

CREATE TABLE IF NOT EXISTS auth_sessions (
    session_id   TEXT PRIMARY KEY,
    user_id      TEXT NOT NULL,
    token        TEXT NOT NULL,
    expires_at   TEXT NOT NULL,
    created_at   TEXT DEFAULT (datetime('now')),
    FOREIGN KEY (user_id) REFERENCES users(user_id)
);

CREATE TABLE IF NOT EXISTS exam_permissions (
    perm_id      TEXT PRIMARY KEY,
    user_id      TEXT NOT NULL,
    exam_id      TEXT NOT NULL,
    can_view     INTEGER DEFAULT 1,
    can_edit     INTEGER DEFAULT 0,
    can_grade    INTEGER DEFAULT 0,
    can_monitor  INTEGER DEFAULT 0,
    created_at   TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS exam_customization (
    exam_id      TEXT PRIMARY KEY,
    settings     TEXT DEFAULT '{}',
    instructions TEXT DEFAULT '',
    passing_score REAL DEFAULT 40.0,
    created_at   TEXT DEFAULT (datetime('now')),
    updated_at   TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_users_org ON users(org_id);
CREATE INDEX IF NOT EXISTS idx_users_role ON users(role);
CREATE INDEX IF NOT EXISTS idx_exam_perms_exam ON exam_permissions(exam_id);
"""


def hash_password(password: str) -> str:
    """Hash a password with SHA-256 (demo only — use bcrypt in production)."""
    return hashlib.sha256(password.encode()).hexdigest()


class AuthDatabase:
    """User and organization database."""

    def __init__(self, db_path: Optional[Path] = None):
        global DB_PATH
        if db_path:
            DB_PATH = db_path
        _ensure_parent()
        self._conn = sqlite3.connect(str(DB_PATH))
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(AUTH_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    # ── Organizations ──────────────────────────────────────────────────────────

    def create_org(self, name: str, admin_id: str, org_type: str = "university",
                   settings: Dict = None) -> str:
        org_id = f"org_{uuid.uuid4().hex[:8]}"
        self._conn.execute(
            "INSERT INTO organizations (org_id, name, org_type, admin_id, settings) VALUES (?, ?, ?, ?, ?)",
            (org_id, name, org_type, admin_id, json.dumps(settings or {})))
        self._conn.commit()
        return org_id

    def get_org(self, org_id: str) -> Optional[Dict]:
        row = self._conn.execute(
            "SELECT * FROM organizations WHERE org_id = ?", (org_id,)).fetchone()
        if not row:
            return None
        d = dict(row)
        d["settings"] = json.loads(d.get("settings") or "{}")
        return d

    def list_orgs(self) -> List[Dict]:
        rows = self._conn.execute("SELECT * FROM organizations ORDER BY created_at DESC").fetchall()
        result = []
        for r in rows:
            d = dict(r)
            d["settings"] = json.loads(d.get("settings") or "{}")
            result.append(d)
        return result

    # ── Users ──────────────────────────────────────────────────────────────────

    def create_user(self, username: str, email: str, full_name: str,
                    password: str, role: str = "student",
                    org_id: Optional[str] = None,
                    preferences: Dict = None) -> str:
        user_id = f"user_{uuid.uuid4().hex[:8]}"
        self._conn.execute(
            "INSERT INTO users (user_id, username, email, full_name, password_hash, role, org_id, preferences) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (user_id, username, email, full_name, hash_password(password),
             role, org_id, json.dumps(preferences or {})))
        self._conn.commit()
        return user_id

    def get_user(self, user_id: str) -> Optional[Dict]:
        row = self._conn.execute(
            "SELECT * FROM users WHERE user_id = ?", (user_id,)).fetchone()
        if not row:
            return None
        d = dict(row)
        d["preferences"] = json.loads(d.get("preferences") or "{}")
        return d

    def get_user_by_username(self, username: str) -> Optional[Dict]:
        row = self._conn.execute(
            "SELECT * FROM users WHERE username = ?", (username,)).fetchone()
        if not row:
            return None
        d = dict(row)
        d["preferences"] = json.loads(d.get("preferences") or "{}")
        return d

    def get_user_by_email(self, email: str) -> Optional[Dict]:
        row = self._conn.execute(
            "SELECT * FROM users WHERE email = ?", (email,)).fetchone()
        if not row:
            return None
        d = dict(row)
        d["preferences"] = json.loads(d.get("preferences") or "{}")
        return d

    def list_users(self, org_id: Optional[str] = None,
                   role: Optional[str] = None) -> List[Dict]:
        query = "SELECT * FROM users WHERE 1=1"
        params = []
        if org_id:
            query += " AND org_id = ?"
            params.append(org_id)
        if role:
            query += " AND role = ?"
            params.append(role)
        query += " ORDER BY created_at DESC"
        rows = self._conn.execute(query, params).fetchall()
        result = []
        for r in rows:
            d = dict(r)
            d.pop("password_hash", None)
            d["preferences"] = json.loads(d.get("preferences") or "{}")
            result.append(d)
        return result

    def update_user(self, user_id: str, **fields) -> bool:
        if not fields:
            return False
        set_clause = ", ".join(f"{k} = ?" for k in fields)
        values = list(fields.values()) + [user_id]
        self._conn.execute(f"UPDATE users SET {set_clause} WHERE user_id = ?", values)
        self._conn.commit()
        return True

    def verify_password(self, username_or_email: str, password: str) -> Optional[Dict]:
        """Verify credentials and return user dict if valid."""
        user = (self.get_user_by_username(username_or_email) or
                self.get_user_by_email(username_or_email))
        if not user:
            return None
        if user.get("password_hash") != hash_password(password):
            return None
        if not user.get("is_active", 1):
            return None
        # Update last login
        now = datetime.utcnow().isoformat()
        self._conn.execute(
            "UPDATE users SET last_login = ? WHERE user_id = ?", (now, user["user_id"]))
        self._conn.commit()
        user.pop("password_hash", None)
        return user

    def delete_user(self, user_id: str) -> bool:
        self._conn.execute("DELETE FROM users WHERE user_id = ?", (user_id,))
        self._conn.commit()
        return True

    # ── Auth Sessions ──────────────────────────────────────────────────────────

    def create_session(self, user_id: str, expires_hours: int = 24) -> str:
        token = uuid.uuid4().hex
        expires = (datetime.utcnow() + timedelta(hours=expires_hours)).isoformat()
        session_id = f"sess_{uuid.uuid4().hex[:12]}"
        self._conn.execute(
            "INSERT INTO auth_sessions (session_id, user_id, token, expires_at) VALUES (?, ?, ?, ?)",
            (session_id, user_id, token, expires))
        self._conn.commit()
        return token

    def get_session_user(self, token: str) -> Optional[Dict]:
        row = self._conn.execute(
            """SELECT u.* FROM auth_sessions a
               JOIN users u ON a.user_id = u.user_id
               WHERE a.token = ? AND a.expires_at > ?""",
            (token, datetime.utcnow().isoformat())).fetchone()
        if not row:
            return None
        d = dict(row)
        d.pop("password_hash", None)
        d["preferences"] = json.loads(d.get("preferences") or "{}")
        return d

    def delete_session(self, token: str) -> bool:
        self._conn.execute("DELETE FROM auth_sessions WHERE token = ?", (token,))
        self._conn.commit()
        return True

    # ── Exam Permissions ────────────────────────────────────────────────────────

    def grant_permission(self, user_id: str, exam_id: str,
                         can_edit: bool = False, can_grade: bool = False,
                         can_monitor: bool = False) -> str:
        perm_id = f"perm_{uuid.uuid4().hex[:8]}"
        self._conn.execute(
            "INSERT INTO exam_permissions (perm_id, user_id, exam_id, can_edit, can_grade, can_monitor) VALUES (?, ?, ?, ?, ?, ?)",
            (perm_id, user_id, exam_id, int(can_edit), int(can_grade), int(can_monitor)))
        self._conn.commit()
        return perm_id

    def get_exam_permissions(self, exam_id: str) -> List[Dict]:
        rows = self._conn.execute(
            "SELECT * FROM exam_permissions WHERE exam_id = ?", (exam_id,)).fetchall()
        return [dict(r) for r in rows]

    def get_user_exams(self, user_id: str) -> List[Dict]:
        rows = self._conn.execute(
            "SELECT * FROM exam_permissions WHERE user_id = ?", (user_id,)).fetchall()
        return [dict(r) for r in rows]

    # ── Exam Customization ──────────────────────────────────────────────────────

    def save_exam_settings(self, exam_id: str, settings: Dict,
                           instructions: str = "", passing_score: float = 40.0) -> None:
        self._conn.execute(
            """INSERT OR REPLACE INTO exam_customization
               (exam_id, settings, instructions, passing_score, updated_at)
               VALUES (?, ?, ?, ?, ?)""",
            (exam_id, json.dumps(settings), instructions, passing_score,
             datetime.utcnow().isoformat()))
        self._conn.commit()

    def get_exam_settings(self, exam_id: str) -> Optional[Dict]:
        row = self._conn.execute(
            "SELECT * FROM exam_customization WHERE exam_id = ?", (exam_id,)).fetchone()
        if not row:
            return None
        d = dict(row)
        d["settings"] = json.loads(d.get("settings") or "{}")
        return d


# Singleton
_db: Optional[AuthDatabase] = None


def get_auth_db() -> AuthDatabase:
    global _db
    if _db is None:
        _db = AuthDatabase()
    return _db

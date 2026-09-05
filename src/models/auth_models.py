"""Extended data models — users, organizations, roles, permissions."""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum

from src.models.models import (
    QuestionType, AgentState, Question, ProctoringEvent, Exam, ExamSession,
)


# ── Enums ──────────────────────────────────────────────────────────────────────

class UserRole(Enum):
    """System-wide user roles."""
    SUPER_ADMIN = "super_admin"
    ADMIN = "admin"
    EXAMINER = "examiner"
    STUDENT = "student"
    VIEWER = "viewer"


class OrganizationType(Enum):
    UNIVERSITY = "university"
    SCHOOL = "school"
    COACHING = "coaching"
    CORPORATE = "corporate"
    CERTIFICATION = "certification"


class ExamStatus(Enum):
    DRAFT = "draft"
    PUBLISHED = "published"
    ACTIVE = "active"
    COMPLETED = "completed"
    ARCHIVED = "archived"


class Permission(Enum):
    # Exam management
    CREATE_EXAM = "create_exam"
    EDIT_EXAM = "edit_exam"
    DELETE_EXAM = "delete_exam"
    PUBLISH_EXAM = "publish_exam"
    VIEW_EXAM = "view_exam"

    # Session management
    START_SESSION = "start_session"
    END_SESSION = "end_session"
    VIEW_SESSION = "view_session"
    MONITOR_SESSION = "monitor_session"

    # Proctoring
    VIEW_PROCTORING = "view_proctoring"
    MANAGE_PROCTORING = "manage_proctoring"
    REVIEW_EVIDENCE = "review_evidence"
    DISMISS_EVIDENCE = "dismiss_evidence"

    # Grading
    GRADE_ANSWERS = "grade_answers"
    VIEW_GRADES = "view_grades"
    OVERRIDE_GRADES = "override_grades"

    # Analytics
    VIEW_ANALYTICS = "view_analytics"
    EXPORT_DATA = "export_data"

    # ML / Training
    TRAIN_MODELS = "train_models"
    VIEW_MODELS = "view_models"

    # User management
    MANAGE_USERS = "manage_users"
    MANAGE_ROLES = "manage_roles"

    # Organization
    MANAGE_ORG = "manage_org"
    VIEW_ORG = "view_org"


# ── Role definitions ──────────────────────────────────────────────────────────

ROLE_PERMISSIONS: Dict[UserRole, List[Permission]] = {
    UserRole.SUPER_ADMIN: list(Permission),
    UserRole.ADMIN: [
        Permission.CREATE_EXAM, Permission.EDIT_EXAM, Permission.DELETE_EXAM,
        Permission.PUBLISH_EXAM, Permission.VIEW_EXAM,
        Permission.START_SESSION, Permission.END_SESSION, Permission.VIEW_SESSION,
        Permission.MONITOR_SESSION,
        Permission.VIEW_PROCTORING, Permission.MANAGE_PROCTORING,
        Permission.REVIEW_EVIDENCE, Permission.DISMISS_EVIDENCE,
        Permission.GRADE_ANSWERS, Permission.VIEW_GRADES, Permission.OVERRIDE_GRADES,
        Permission.VIEW_ANALYTICS, Permission.EXPORT_DATA,
        Permission.TRAIN_MODELS, Permission.VIEW_MODELS,
        Permission.MANAGE_USERS, Permission.MANAGE_ROLES,
        Permission.MANAGE_ORG, Permission.VIEW_ORG,
    ],
    UserRole.EXAMINER: [
        Permission.CREATE_EXAM, Permission.EDIT_EXAM, Permission.VIEW_EXAM,
        Permission.PUBLISH_EXAM,
        Permission.VIEW_SESSION, Permission.MONITOR_SESSION,
        Permission.VIEW_PROCTORING, Permission.REVIEW_EVIDENCE, Permission.DISMISS_EVIDENCE,
        Permission.GRADE_ANSWERS, Permission.VIEW_GRADES, Permission.OVERRIDE_GRADES,
        Permission.VIEW_ANALYTICS, Permission.EXPORT_DATA,
        Permission.VIEW_MODELS,
    ],
    UserRole.STUDENT: [
        Permission.VIEW_EXAM,
        Permission.START_SESSION, Permission.END_SESSION, Permission.VIEW_SESSION,
        Permission.VIEW_GRADES,
    ],
    UserRole.VIEWER: [
        Permission.VIEW_EXAM, Permission.VIEW_SESSION,
        Permission.VIEW_ANALYTICS, Permission.VIEW_ORG,
    ],
}


def has_permission(role: UserRole, permission: Permission) -> bool:
    """Check if a role has a specific permission."""
    return permission in ROLE_PERMISSIONS.get(role, [])


# ── Data models ───────────────────────────────────────────────────────────────

@dataclass
class Organization:
    """An organization (university, school, etc.) that uses the platform."""
    org_id: str
    name: str
    org_type: OrganizationType
    admin_id: str
    settings: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        return {
            "org_id": self.org_id,
            "name": self.name,
            "org_type": self.org_type.value,
            "admin_id": self.admin_id,
            "settings": self.settings,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class User:
    """A platform user with role-based access."""
    user_id: str
    username: str
    email: str
    full_name: str
    role: UserRole
    org_id: Optional[str] = None
    password_hash: str = ""
    avatar_url: str = ""
    preferences: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    last_login: Optional[datetime] = None
    is_active: bool = True

    def to_dict(self, include_sensitive: bool = False) -> dict:
        d = {
            "user_id": self.user_id,
            "username": self.username,
            "email": self.email,
            "full_name": self.full_name,
            "role": self.role.value,
            "org_id": self.org_id,
            "avatar_url": self.avatar_url,
            "preferences": self.preferences,
            "created_at": self.created_at.isoformat(),
            "last_login": self.last_login.isoformat() if self.last_login else None,
            "is_active": self.is_active,
        }
        if include_sensitive:
            d["password_hash"] = self.password_hash
        return d


@dataclass
class ExamSettings:
    """Customizable exam configuration per organization."""
    randomize_questions: bool = False
    randomize_options: bool = False
    show_results_immediately: bool = False
    allow_review_after_submit: bool = True
    require_webcam: bool = True
    require_microphone: bool = False
    enable_proctoring: bool = True
    auto_submit_on_timeout: bool = True
    warn_before_submit: bool = True
    max_tab_switches: int = 5
    lock_fullscreen: bool = False
    prevent_copy_paste: bool = True
    max_risk_score_before_flag: float = 0.7
    grading_release_delay_minutes: int = 0
    allow_student_notes: bool = True
    calculator_allowed: bool = False
    scratch_pad_allowed: bool = True
    custom_instructions: str = ""
    passing_score_percent: float = 40.0
    time_warning_minutes: int = 5


@dataclass
class EvidenceEntry:
    """A single proctoring evidence entry with review state."""
    entry_id: str
    session_id: str
    timestamp: datetime
    event_type: str
    source: str
    confidence: float
    risk_contribution: float
    details: Dict[str, Any] = field(default_factory=dict)
    review_state: str = "pending"  # pending | reviewed | dismissed | flagged
    review_note: str = ""
    reviewed_by: Optional[str] = None
    reviewed_at: Optional[datetime] = None

    def to_dict(self) -> dict:
        return {
            "entry_id": self.entry_id,
            "session_id": self.session_id,
            "timestamp": self.timestamp.isoformat(),
            "event_type": self.event_type,
            "source": self.source,
            "confidence": self.confidence,
            "risk_contribution": self.risk_contribution,
            "details": self.details,
            "review_state": self.review_state,
            "review_note": self.review_note,
            "reviewed_by": self.reviewed_by,
            "reviewed_at": self.reviewed_at.isoformat() if self.reviewed_at else None,
        }

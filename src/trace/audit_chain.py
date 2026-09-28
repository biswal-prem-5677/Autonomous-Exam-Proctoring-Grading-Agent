"""Cryptographic audit trail with SHA-256 hash chaining.

Each audit entry includes a hash that chains to the previous entry:
    hash_n = SHA256(hash_{n-1} ‖ timestamp ‖ event_type ‖ actor ‖ data)

This creates a tamper-evident, append-only log. If any historical entry
is modified, all subsequent hashes become invalid and the chain breaks.

Mathematical foundation:
    - SHA-256 (Secure Hash Algorithm 256-bit) — collision-resistant,
      preimage-resistant hash function
    - Hash chaining — each block references the previous block's hash,
      forming a linked integrity structure (blockchain-like)
"""

import hashlib
import json
import time
import uuid
import sqlite3
import threading
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path


GENESIS_HASH = "0" * 64  # The "genesis" block has no predecessor


@dataclass
class AuditEntry:
    """A single entry in the cryptographic audit chain."""
    entry_id: str
    sequence: int               # Position in the chain
    timestamp: float            # Unix timestamp
    event_type: str             # e.g., "exam_started", "answer_submitted", "flag_raised"
    actor_id: str               # Who triggered the event (student_id, examiner_id, "system")
    session_id: str             # Exam session this belongs to
    data: Dict[str, Any]        # Event-specific payload
    previous_hash: str          # Hash of the previous entry
    entry_hash: str             # SHA-256 hash of this entry

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def compute_entry_hash(
    previous_hash: str,
    timestamp: float,
    event_type: str,
    actor_id: str,
    session_id: str,
    data: Dict[str, Any],
    sequence: int,
) -> str:
    """Compute SHA-256 hash for an audit entry.

    Hash input: previous_hash ‖ sequence ‖ timestamp ‖ event_type ‖ actor_id ‖ session_id ‖ json(data)

    The hash function is deterministic — the same inputs always produce
    the same hash, enabling independent verification.
    """
    payload = (
        f"{previous_hash}|"
        f"{sequence}|"
        f"{timestamp:.6f}|"
        f"{event_type}|"
        f"{actor_id}|"
        f"{session_id}|"
        f"{json.dumps(data, sort_keys=True, default=str)}"
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class AuditChain:
    """Append-only, tamper-evident audit trail with SHA-256 hash chaining.

    Thread-safe implementation with SQLite persistence.

    Usage:
        chain = AuditChain("data/audit.db")
        chain.append("exam_started", actor_id="student_001",
                      session_id="sess_abc", data={"exam_id": "exam_1"})
        chain.append("answer_submitted", actor_id="student_001",
                      session_id="sess_abc", data={"question_id": "q1", "answer": "B"})

        # Verify integrity
        valid, broken_at = chain.verify_integrity()
        assert valid
    """

    def __init__(self, db_path: str = "data/audit_chain.db"):
        self._db_path = db_path
        self._lock = threading.Lock()
        self._init_db()

    def _init_db(self) -> None:
        """Initialize the SQLite database for audit storage."""
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self._db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS audit_chain (
                    entry_id TEXT PRIMARY KEY,
                    sequence INTEGER NOT NULL UNIQUE,
                    timestamp REAL NOT NULL,
                    event_type TEXT NOT NULL,
                    actor_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    data TEXT NOT NULL,
                    previous_hash TEXT NOT NULL,
                    entry_hash TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_audit_session
                ON audit_chain(session_id)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_audit_sequence
                ON audit_chain(sequence)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_audit_event_type
                ON audit_chain(event_type)
            """)
            conn.commit()

    def _get_last_entry(self, conn: sqlite3.Connection) -> Optional[Tuple[int, str]]:
        """Get the sequence and hash of the last entry in the chain."""
        row = conn.execute(
            "SELECT sequence, entry_hash FROM audit_chain ORDER BY sequence DESC LIMIT 1"
        ).fetchone()
        if row:
            return (row[0], row[1])
        return None

    def append(
        self,
        event_type: str,
        actor_id: str,
        session_id: str,
        data: Optional[Dict[str, Any]] = None,
    ) -> AuditEntry:
        """Append a new entry to the audit chain.

        Thread-safe. The entry's hash is computed from the previous entry's hash,
        the timestamp, and the event data — creating an unbreakable chain.

        Args:
            event_type: Type of audit event (e.g., "exam_started", "answer_submitted")
            actor_id: ID of the actor (student, examiner, or "system")
            session_id: Exam session ID
            data: Event-specific data payload

        Returns:
            The created AuditEntry with its computed hash
        """
        data = data or {}

        with self._lock:
            with sqlite3.connect(self._db_path) as conn:
                last = self._get_last_entry(conn)
                if last:
                    sequence = last[0] + 1
                    previous_hash = last[1]
                else:
                    sequence = 0
                    previous_hash = GENESIS_HASH

                entry_id = str(uuid.uuid4())[:12]
                timestamp = time.time()

                entry_hash = compute_entry_hash(
                    previous_hash=previous_hash,
                    timestamp=timestamp,
                    event_type=event_type,
                    actor_id=actor_id,
                    session_id=session_id,
                    data=data,
                    sequence=sequence,
                )

                entry = AuditEntry(
                    entry_id=entry_id,
                    sequence=sequence,
                    timestamp=timestamp,
                    event_type=event_type,
                    actor_id=actor_id,
                    session_id=session_id,
                    data=data,
                    previous_hash=previous_hash,
                    entry_hash=entry_hash,
                )

                conn.execute(
                    """INSERT INTO audit_chain
                       (entry_id, sequence, timestamp, event_type, actor_id,
                        session_id, data, previous_hash, entry_hash)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        entry.entry_id,
                        entry.sequence,
                        entry.timestamp,
                        entry.event_type,
                        entry.actor_id,
                        entry.session_id,
                        json.dumps(entry.data, sort_keys=True, default=str),
                        entry.previous_hash,
                        entry.entry_hash,
                    ),
                )
                conn.commit()

        return entry

    def verify_integrity(self, session_id: Optional[str] = None) -> Tuple[bool, Optional[int]]:
        """Verify the integrity of the entire audit chain (or a session's chain).

        Recomputes every hash from scratch and checks that each entry's
        `previous_hash` matches the actual hash of its predecessor.

        Args:
            session_id: If provided, verify only entries for this session.

        Returns:
            (is_valid, broken_at_sequence) — True if chain is intact,
            otherwise the sequence number where the break was detected.
        """
        with sqlite3.connect(self._db_path) as conn:
            if session_id:
                rows = conn.execute(
                    "SELECT * FROM audit_chain WHERE session_id = ? ORDER BY sequence",
                    (session_id,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM audit_chain ORDER BY sequence"
                ).fetchall()

        if not rows:
            return (True, None)

        expected_prev_hash = GENESIS_HASH

        for row in rows:
            entry_id, sequence, timestamp, event_type, actor_id, \
                session_id_val, data_str, previous_hash, entry_hash = row

            # Check that previous_hash links correctly
            if previous_hash != expected_prev_hash:
                return (False, sequence)

            # Recompute hash and compare
            data = json.loads(data_str)
            recomputed = compute_entry_hash(
                previous_hash=previous_hash,
                timestamp=timestamp,
                event_type=event_type,
                actor_id=actor_id,
                session_id=session_id_val,
                data=data,
                sequence=sequence,
            )

            if recomputed != entry_hash:
                return (False, sequence)

            expected_prev_hash = entry_hash

        return (True, None)

    def get_entries(
        self,
        session_id: Optional[str] = None,
        event_type: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[AuditEntry]:
        """Retrieve audit entries with optional filtering."""
        query = "SELECT * FROM audit_chain WHERE 1=1"
        params: list = []

        if session_id:
            query += " AND session_id = ?"
            params.append(session_id)
        if event_type:
            query += " AND event_type = ?"
            params.append(event_type)

        query += " ORDER BY sequence DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        with sqlite3.connect(self._db_path) as conn:
            rows = conn.execute(query, params).fetchall()

        entries = []
        for row in rows:
            entry_id, sequence, timestamp, event_type_val, actor_id, \
                session_id_val, data_str, previous_hash, entry_hash = row
            entries.append(AuditEntry(
                entry_id=entry_id,
                sequence=sequence,
                timestamp=timestamp,
                event_type=event_type_val,
                actor_id=actor_id,
                session_id=session_id_val,
                data=json.loads(data_str),
                previous_hash=previous_hash,
                entry_hash=entry_hash,
            ))

        return entries

    def get_chain_length(self, session_id: Optional[str] = None) -> int:
        """Get total number of entries in the chain."""
        with sqlite3.connect(self._db_path) as conn:
            if session_id:
                row = conn.execute(
                    "SELECT COUNT(*) FROM audit_chain WHERE session_id = ?",
                    (session_id,),
                ).fetchone()
            else:
                row = conn.execute("SELECT COUNT(*) FROM audit_chain").fetchone()
        return row[0] if row else 0

    def export_chain(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Export the full chain with verification status for human review.

        Returns a complete audit report including:
        - All entries with their hashes
        - Chain integrity verification result
        - Summary statistics
        """
        entries = self.get_entries(session_id=session_id, limit=10000)
        is_valid, broken_at = self.verify_integrity(session_id=session_id)

        # Summary stats
        event_types = {}
        actors = set()
        for entry in entries:
            event_types[entry.event_type] = event_types.get(entry.event_type, 0) + 1
            actors.add(entry.actor_id)

        return {
            "chain_length": len(entries),
            "integrity_valid": is_valid,
            "broken_at_sequence": broken_at,
            "session_id": session_id,
            "event_type_counts": event_types,
            "unique_actors": list(actors),
            "entries": [e.to_dict() for e in reversed(entries)],  # Chronological order
            "exported_at": datetime.now().isoformat(),
            "verification_hash": hashlib.sha256(
                json.dumps([e.entry_hash for e in entries], sort_keys=True).encode()
            ).hexdigest(),
        }
"""Module: trace/audit_chain.py"""

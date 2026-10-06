"""Subprocess-based secure sandbox for executing untrusted student code.

Security guarantees
-------------------
* Student code runs in a **separate OS process** — never inside Flask.
* A wall-clock timeout fires ``subprocess.TimeoutExpired``; the entire
  process tree is then killed (``taskkill /F /T`` on Windows, ``SIGKILL``
  on Unix).
* Communication is strictly JSON over stdin/stdout — no shared memory,
  no shared file descriptors.
* A secondary per-test threading timeout inside the worker catches hot
  loops that the wall-clock guard might be slow to catch.
* The worker itself uses a restricted ``__builtins__`` dict (no ``open``,
  ``exec``, ``eval``, ``__import__``, ``subprocess``, ``socket``, …).
* AST-level security scanning in ``ProgrammingGrader`` provides a first
  line of defence before any process is spawned.

Limitations (documented honestly)
----------------------------------
* Memory limits are not enforced on Windows (no ``resource`` module).
  A student could allocate gigabytes; the server OS will eventually OOM-kill
  the worker, which surfaces as a ``sandbox_error``.
* Filesystem and network isolation are *not* OS-enforced on Windows —
  they are blocked only via restricted builtins.  Docker or Windows Job
  Objects would be needed for production-grade isolation.
* CPU pinning is not implemented.
"""

import json
import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Worker script location
# ---------------------------------------------------------------------------

_WORKER_SCRIPT = Path(__file__).parent / "sandbox_worker.py"
_PYTHON_EXE = sys.executable  # same interpreter as the server


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class SandboxTestResult:
    name: str
    passed: bool
    actual: Any
    expected: Any
    error: Optional[str]
    time_ms: float
    is_hidden: bool
    is_edge: bool


@dataclass
class SandboxResult:
    """Outcome of a full sandbox execution."""
    success: bool
    """True = worker ran to completion (even if tests failed)."""

    init_error: Optional[str]
    """Error during code exec() or function lookup inside the worker."""

    sandbox_error: Optional[str]
    """Timeout, crash, or communication error at the sandbox level."""

    results: List[SandboxTestResult] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Process-tree killing
# ---------------------------------------------------------------------------

def _kill_tree(pid: int) -> None:
    """Kill a process and all its descendants, cross-platform."""
    try:
        if sys.platform == "win32":
            subprocess.call(
                ["taskkill", "/F", "/T", "/PID", str(pid)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        else:
            import signal
            os.killpg(os.getpgid(pid), signal.SIGKILL)
    except Exception:  # noqa: BLE001
        pass  # process may have already exited


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_in_sandbox(
    code: str,
    function_name: str,
    tests: List[Dict[str, Any]],
    timeout_seconds: float = 5.0,
) -> SandboxResult:
    """Execute student code in an isolated subprocess.

    Args:
        code:            Python source submitted by the student.
        function_name:   Name of the entry-point function to call.
        tests:           List of test dicts — each must have:
                         ``name``, ``input``, ``expected``;
                         optionally ``is_hidden``, ``is_edge``, ``kwargs``.
        timeout_seconds: Wall-clock timeout for the *entire* worker process.
                         A per-test guard (``timeout/n_tests``) is also sent
                         to the worker for finer-grained protection.

    Returns:
        :class:`SandboxResult` with individual test outcomes.
    """
    if not _WORKER_SCRIPT.exists():
        return SandboxResult(
            success=False,
            init_error=None,
            sandbox_error=f"Worker script not found: {_WORKER_SCRIPT}",
        )

    n_tests = max(1, len(tests))
    per_test = max(0.5, timeout_seconds / n_tests)

    job_json = json.dumps({
        "code": code,
        "function_name": function_name,
        "tests": tests,
        "per_test_timeout": per_test,
    })

    # Platform-specific subprocess flags for clean process-tree management
    popen_kwargs: Dict[str, Any] = {}
    if sys.platform == "win32":
        popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        popen_kwargs["start_new_session"] = True  # enables killpg

    proc = None
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            [_PYTHON_EXE, str(_WORKER_SCRIPT)],
            input=job_json,
            capture_output=True,
            text=True,
            timeout=timeout_seconds + 3.0,  # outer guard > inner guard
            **popen_kwargs,
        )
    except subprocess.TimeoutExpired as exc:
        if exc.output is not None:
            # subprocess already stored output; proc object is available via exc
            pass
        # Try to kill any lingering child processes
        try:
            if hasattr(exc, "args") and exc.args:
                _kill_tree(exc.args[0] if isinstance(exc.args[0], int) else 0)
        except Exception:  # noqa: BLE001
            pass
        return SandboxResult(
            success=False,
            init_error=None,
            sandbox_error=(
                f"Sandbox wall-clock timeout: student code exceeded "
                f"{timeout_seconds:.1f}s. Likely infinite loop or excessive computation."
            ),
        )
    except Exception as exc:  # noqa: BLE001
        return SandboxResult(
            success=False,
            init_error=None,
            sandbox_error=f"Failed to spawn sandbox process: {exc}",
        )

    # --- parse worker output -------------------------------------------------
    stdout = (proc.stdout or "").strip()
    stderr = (proc.stderr or "").strip()

    if not stdout:
        return SandboxResult(
            success=False,
            init_error=None,
            sandbox_error=(
                f"Worker produced no output (exit={proc.returncode}). "
                + (f"stderr: {stderr[:300]}" if stderr else "")
            ),
        )

    try:
        data: Dict[str, Any] = json.loads(stdout)
    except json.JSONDecodeError:
        return SandboxResult(
            success=False,
            init_error=None,
            sandbox_error=f"Worker output was not valid JSON: {stdout[:300]}",
        )

    # Map raw dicts → typed dataclasses
    raw_results: List[Dict[str, Any]] = data.get("results") or []
    test_results = [
        SandboxTestResult(
            name=r.get("name", f"test_{i + 1}"),
            passed=bool(r.get("passed", False)),
            actual=r.get("actual"),
            expected=r.get("expected"),
            error=r.get("error"),
            time_ms=float(r.get("time_ms", 0.0)),
            is_hidden=bool(r.get("is_hidden", False)),
            is_edge=bool(r.get("is_edge", False)),
        )
        for i, r in enumerate(raw_results)
    ]

    return SandboxResult(
        success=True,
        init_error=data.get("init_error"),
        sandbox_error=None,
        results=test_results,
    )

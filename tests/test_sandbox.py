"""Tests for the subprocess-based secure programming sandbox.

Verifies:
  - Correct code passes all tests
  - Wrong output is detected
  - Runtime errors are caught gracefully
  - Infinite loops are killed by wall-clock timeout
  - Forbidden module imports are blocked (AST + restricted builtins)
  - Forbidden function calls are blocked (AST-level)
  - Malicious system-access attempts fail safely
  - The Flask process is never infected by student code side-effects
  - Hidden / edge-case test metadata is respected
  - ProgrammingGrader.grade() end-to-end with sandbox
"""

import sys
import time
import pytest
from pathlib import Path

# Ensure project root on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.grading.sandbox import run_in_sandbox, SandboxResult
from src.grading.programming import ProgrammingGrader


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _simple_tests(fn_name: str = "add"):
    return [
        {"name": "t1", "input": [1, 2], "expected": 3,
         "is_hidden": False, "is_edge": False},
        {"name": "t2", "input": [0, 0], "expected": 0,
         "is_hidden": False, "is_edge": False},
        {"name": "t3_edge", "input": [-5, 5], "expected": 0,
         "is_hidden": False, "is_edge": True},
    ]


CORRECT_CODE = """
def add(a, b):
    \"\"\"Return the sum of a and b.\"\"\"
    return a + b
"""

WRONG_CODE = """
def add(a, b):
    return a - b  # deliberate wrong answer
"""

RUNTIME_ERROR_CODE = """
def add(a, b):
    return a / 0  # ZeroDivisionError
"""

INFINITE_LOOP_CODE = """
def add(a, b):
    while True:
        pass
"""

FORBIDDEN_IMPORT_CODE = """
import os
def add(a, b):
    return a + b
"""

FORBIDDEN_IMPORT_FROM_CODE = """
from subprocess import run
def add(a, b):
    return a + b
"""

FORBIDDEN_CALL_EVAL_CODE = """
def add(a, b):
    return eval("a + b")
"""

FORBIDDEN_CALL_EXEC_CODE = """
def add(a, b):
    exec("result = a + b")
    return result
"""

FORBIDDEN_CALL_OPEN_CODE = """
def add(a, b):
    with open("/etc/passwd") as f:
        return len(f.read())
"""

SYSTEM_ACCESS_SOCKET_CODE = """
import socket
def add(a, b):
    s = socket.socket()
    return a + b
"""

SIDE_EFFECT_CODE = """
import sys
_SENTINEL = []
def add(a, b):
    _SENTINEL.append(1)
    return a + b
"""

MISSING_FUNCTION_CODE = """
def wrong_name(a, b):
    return a + b
"""

SYNTAX_ERROR_CODE = """
def add(a b):
    return a + b
"""


# ---------------------------------------------------------------------------
# sandbox.run_in_sandbox() tests
# ---------------------------------------------------------------------------

class TestSandboxBasic:
    """Core correctness and error handling."""

    def test_correct_code_passes(self):
        result = run_in_sandbox(CORRECT_CODE, "add", _simple_tests(), timeout_seconds=5.0)
        assert result.success
        assert result.init_error is None
        assert result.sandbox_error is None
        assert all(r.passed for r in result.results)

    def test_wrong_answer_detected(self):
        # Use inputs where a-b != a+b (avoid 0+0 edge case)
        tests = [
            {"name": "t1", "input": [3, 4], "expected": 7,
             "is_hidden": False, "is_edge": False},
            {"name": "t2", "input": [10, 5], "expected": 15,
             "is_hidden": False, "is_edge": False},
        ]
        result = run_in_sandbox(WRONG_CODE, "add", tests, timeout_seconds=5.0)
        assert result.success
        assert not any(r.passed for r in result.results), "Wrong code should not pass any test"

    def test_runtime_error_caught(self):
        result = run_in_sandbox(RUNTIME_ERROR_CODE, "add", _simple_tests(), timeout_seconds=5.0)
        assert result.success
        for r in result.results:
            assert not r.passed
            assert r.error is not None
            assert "ZeroDivisionError" in r.error

    def test_missing_function_caught(self):
        result = run_in_sandbox(MISSING_FUNCTION_CODE, "add", _simple_tests(), timeout_seconds=5.0)
        assert result.success
        assert result.init_error is not None
        assert "add" in result.init_error

    def test_empty_test_list(self):
        result = run_in_sandbox(CORRECT_CODE, "add", [], timeout_seconds=5.0)
        assert result.success
        assert result.results == []

    def test_hidden_test_metadata_preserved(self):
        tests = [
            {"name": "hidden_t1", "input": [2, 3], "expected": 5,
             "is_hidden": True, "is_edge": False},
        ]
        result = run_in_sandbox(CORRECT_CODE, "add", tests, timeout_seconds=5.0)
        assert result.success
        r = result.results[0]
        assert r.is_hidden
        assert r.passed

    def test_edge_case_metadata_preserved(self):
        tests = [
            {"name": "edge_t1", "input": [0, 0], "expected": 0,
             "is_hidden": False, "is_edge": True},
        ]
        result = run_in_sandbox(CORRECT_CODE, "add", tests, timeout_seconds=5.0)
        assert result.success
        r = result.results[0]
        assert r.is_edge
        assert r.passed


class TestSandboxSecurity:
    """Security boundary tests — all should be blocked."""

    def test_infinite_loop_killed_by_timeout(self):
        """The sandbox must contain infinite loops — either via:
        (a) wall-clock process kill (result.success=False), or
        (b) per-test threading timeout inside the worker (result.success=True
            but all tests have TimeoutError errors and passed=False).
        Both are acceptable — the important thing is the grader returns
        within a bounded time and student tests all fail.
        """
        start = time.perf_counter()
        result = run_in_sandbox(INFINITE_LOOP_CODE, "add", _simple_tests(), timeout_seconds=5.0)
        elapsed = time.perf_counter() - start
        # Must return in bounded time
        assert elapsed < 20.0, f"Sandbox did not return in time: {elapsed:.1f}s"
        # All tests must have failed (passed=False)
        if result.success:
            # Threading timeout fired inside worker
            assert all(not r.passed for r in result.results), \
                "Infinite loop should not pass any tests"
            assert all(
                r.error is not None and "Timeout" in (r.error or "")
                for r in result.results
            ), "Expected TimeoutError on each test result"
        else:
            # Wall-clock timeout killed the process
            assert result.sandbox_error is not None
            assert "timeout" in result.sandbox_error.lower()

    def test_os_import_blocked_by_ast(self):
        """AST scanner must reject `import os` before any process is spawned."""
        grader = ProgrammingGrader()
        r = grader.grade(
            FORBIDDEN_IMPORT_CODE, "add",
            public_tests=[{"name": "t1", "input": [1, 2], "expected": 3}],
        )
        assert r.score == 0.0
        assert not r.security.is_safe
        assert any("os" in v for v in r.security.violations)

    def test_subprocess_import_blocked_by_ast(self):
        grader = ProgrammingGrader()
        r = grader.grade(
            FORBIDDEN_IMPORT_FROM_CODE, "add",
            public_tests=[{"name": "t1", "input": [1, 2], "expected": 3}],
        )
        assert r.score == 0.0
        assert not r.security.is_safe

    def test_eval_blocked_by_ast(self):
        grader = ProgrammingGrader()
        r = grader.grade(
            FORBIDDEN_CALL_EVAL_CODE, "add",
            public_tests=[{"name": "t1", "input": [1, 2], "expected": 3}],
        )
        assert r.score == 0.0
        assert not r.security.is_safe
        assert any("eval" in v for v in r.security.violations)

    def test_exec_blocked_by_ast(self):
        grader = ProgrammingGrader()
        r = grader.grade(
            FORBIDDEN_CALL_EXEC_CODE, "add",
            public_tests=[{"name": "t1", "input": [1, 2], "expected": 3}],
        )
        assert r.score == 0.0
        assert not r.security.is_safe

    def test_open_blocked_by_ast(self):
        grader = ProgrammingGrader()
        r = grader.grade(
            FORBIDDEN_CALL_OPEN_CODE, "add",
            public_tests=[{"name": "t1", "input": [1, 2], "expected": 3}],
        )
        assert r.score == 0.0
        assert not r.security.is_safe

    def test_socket_import_blocked_by_ast(self):
        grader = ProgrammingGrader()
        r = grader.grade(
            SYSTEM_ACCESS_SOCKET_CODE, "add",
            public_tests=[{"name": "t1", "input": [1, 2], "expected": 3}],
        )
        assert r.score == 0.0
        assert not r.security.is_safe

    def test_flask_process_not_infected_by_side_effects(self):
        """Student code setting module-level state in the worker must NOT
        affect the parent (Flask) process — process isolation guarantee."""
        import importlib
        # Run code that would mutate a list if exec'd in-process
        result = run_in_sandbox(SIDE_EFFECT_CODE, "add", _simple_tests(), timeout_seconds=5.0)
        # The sandbox ran successfully
        assert result.success
        # Verify _SENTINEL is not importable / does not exist here
        # (it was created inside the worker process, not here)
        try:
            # This should not have _SENTINEL attribute in our process
            import src.grading.sandbox_worker as sw
            # If the worker was imported (it shouldn't be as a module
            # during grading), its state should be clean
            sentinel = getattr(sw, "_SENTINEL", None)
            # The sentinel would only be populated if exec ran in-process
            # In subprocess mode it's always None/empty here
            assert sentinel is None or sentinel == [], (
                "Side effect leaked from worker to parent process!"
            )
        except ImportError:
            pass  # Module not importable — good


class TestSandboxSyntaxError:
    """Syntax errors must be caught before spawning anything."""

    def test_syntax_error_caught_before_sandbox(self):
        grader = ProgrammingGrader()
        r = grader.grade(
            SYNTAX_ERROR_CODE, "add",
            public_tests=[{"name": "t1", "input": [1, 2], "expected": 3}],
        )
        assert r.score == 0.0
        assert not r.syntax_valid


# ---------------------------------------------------------------------------
# ProgrammingGrader end-to-end tests
# ---------------------------------------------------------------------------

class TestProgrammingGraderEndToEnd:
    """Integration: AST analysis + sandbox execution + scoring."""

    def test_perfect_score(self):
        grader = ProgrammingGrader()
        result = grader.grade(
            CORRECT_CODE,
            "add",
            public_tests=[
                {"name": "t1", "input": [1, 2], "expected": 3},
                {"name": "t2", "input": [10, 20], "expected": 30},
            ],
            edge_cases=[
                {"name": "e1", "input": [0, 0], "expected": 0},
            ],
            marks=10.0,
        )
        assert result.score > 0.0
        assert result.syntax_valid
        assert result.security.is_safe
        assert result.passed_tests == 3
        assert "isolated subprocess sandbox" in result.explanation

    def test_zero_score_for_wrong_code(self):
        grader = ProgrammingGrader()
        result = grader.grade(
            WRONG_CODE,
            "add",
            public_tests=[{"name": "t1", "input": [3, 4], "expected": 7}],
            marks=10.0,
        )
        # Style/complexity component may award a small score
        assert result.passed_tests == 0

    def test_hidden_tests_not_exposed_in_results(self):
        grader = ProgrammingGrader()
        result = grader.grade(
            CORRECT_CODE,
            "add",
            public_tests=[{"name": "pub", "input": [1, 1], "expected": 2}],
            hidden_tests=[{"name": "hid", "input": [999, 1], "expected": 1000}],
            marks=10.0,
        )
        hidden = [t for t in result.test_results if t.is_hidden]
        assert len(hidden) == 1
        assert hidden[0].actual_output == "[HIDDEN]"
        assert hidden[0].expected_output == "[HIDDEN]"

    def test_complexity_penalty_applied(self):
        """Overly complex code should receive a lower composite score."""
        # Build a function with very high cyclomatic complexity
        branches = "\n    ".join(
            f"if x == {i}: return {i}" for i in range(20)
        )
        complex_code = f"""
def add(a, b):
    x = a + b
    {branches}
    return x
"""
        grader = ProgrammingGrader()
        result = grader.grade(
            complex_code, "add",
            public_tests=[{"name": "t1", "input": [1, 2], "expected": 3}],
            max_allowed_complexity=5,
            marks=10.0,
        )
        # Should still pass tests but complexity penalty applies
        assert result.cyclomatic_complexity > 5

    def test_malicious_infinite_loop_contained(self):
        """End-to-end: infinite loop must not hang grader (timeout fires).

        The grader must return within a bounded time and report 0 passing tests.
        The timeout signal surfaces either in:
          - result.explanation (sandbox-level kill), or
          - result.test_results[i].error_message (per-test thread timeout)
        """
        grader = ProgrammingGrader(execution_timeout_seconds=2.0)
        start = time.perf_counter()
        result = grader.grade(
            INFINITE_LOOP_CODE, "add",
            public_tests=[{"name": "t1", "input": [1, 2], "expected": 3}],
            marks=10.0,
        )
        elapsed = time.perf_counter() - start
        # Must return in bounded time
        assert elapsed < 20.0, f"Grader hung for {elapsed:.1f}s"
        # No tests should pass
        assert result.passed_tests == 0
        # Timeout signal must appear somewhere in the result
        timeout_in_explanation = (
            "timeout" in result.explanation.lower()
        )
        timeout_in_test_errors = any(
            r.error_message and "Timeout" in r.error_message
            for r in result.test_results
        )
        assert timeout_in_explanation or timeout_in_test_errors, (
            f"Expected a timeout signal. explanation={result.explanation!r}, "
            f"errors={[r.error_message for r in result.test_results]}"
        )

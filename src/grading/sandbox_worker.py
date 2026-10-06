"""Secure sandbox worker — executed as a child subprocess, never imported.

Reads a JSON job from stdin:
  {
    "code":          "<student Python source>",
    "function_name": "<entry point>",
    "tests":         [ {"name":..., "input":..., "expected":...,
                        "is_hidden":bool, "is_edge":bool, "kwargs":{}} ],
    "per_test_timeout": float   # seconds per individual test call
  }

Writes a JSON result to stdout:
  {
    "init_error": str | null,   # exec() / function-lookup failure
    "results":    [ {name, passed, actual, expected, error, time_ms,
                     is_hidden, is_edge} ]
  }

Security contract:
  - Only safe builtins are exposed (no open, exec, eval, __import__, …)
  - No network, filesystem, or subprocess access inside student code
  - The parent process kills this entire process if wall-clock timeout fires
  - Per-test threading timeout is a secondary, best-effort guard
"""

import json
import math
import sys
import time
import threading
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Restricted builtins
# ---------------------------------------------------------------------------

def _make_safe_builtins() -> dict:
    """Return a minimal builtins dict safe for untrusted student code."""
    ALLOWED = {
        # Type constructors
        "bool", "bytes", "bytearray", "complex", "dict", "float", "frozenset",
        "int", "list", "memoryview", "object", "set", "str", "tuple", "type",
        # Iterators / functional
        "all", "any", "enumerate", "filter", "iter", "map", "next",
        "range", "reversed", "sorted", "zip",
        # Math / numeric
        "abs", "bin", "chr", "divmod", "hex", "max", "min", "oct", "ord",
        "pow", "round", "sum",
        # Introspection (read-only)
        "callable", "dir", "getattr", "hasattr", "hash", "id",
        "isinstance", "issubclass", "len", "repr", "vars",
        # Formatting / conversion
        "ascii", "format", "print",
        # OOP helpers
        "classmethod", "property", "staticmethod", "super",
        # Misc safe
        "NotImplemented", "Ellipsis", "None", "True", "False",
        "slice",
        # Exception base classes (needed for raise / except)
        "Exception", "BaseException", "ArithmeticError", "AssertionError",
        "AttributeError", "EOFError", "EnvironmentError", "FileExistsError",
        "FileNotFoundError", "FloatingPointError", "GeneratorExit",
        "IOError", "ImportError", "IndexError", "KeyError",
        "KeyboardInterrupt", "LookupError", "MemoryError",
        "ModuleNotFoundError", "NameError", "NotImplementedError", "OSError",
        "OverflowError", "RecursionError", "ReferenceError", "RuntimeError",
        "StopIteration", "StopAsyncIteration", "SyntaxError", "SystemExit",
        "TimeoutError", "TypeError", "UnboundLocalError", "UnicodeError",
        "UnicodeDecodeError", "UnicodeEncodeError", "ValueError",
        "ZeroDivisionError",
    }
    import builtins as _b
    return {k: getattr(_b, k) for k in ALLOWED if hasattr(_b, k)}


_SAFE_BUILTINS = _make_safe_builtins()


# ---------------------------------------------------------------------------
# Per-test execution with secondary timeout
# ---------------------------------------------------------------------------

def _call_with_timeout(fn, args, kwargs, timeout_sec: float):
    """Call fn(*args, **kwargs) with a threading timeout guard.

    Returns (result, error_str).  The parent process provides the primary
    wall-clock kill; this is a secondary, best-effort guard for hot loops.
    """
    result_box: List[Any] = [None]
    error_box: List[Optional[str]] = [None]
    done = threading.Event()

    def _target():
        try:
            result_box[0] = fn(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001
            error_box[0] = f"{type(exc).__name__}: {exc}"
        finally:
            done.set()

    t = threading.Thread(target=_target, daemon=True)
    t.start()
    finished = done.wait(timeout=timeout_sec)
    if not finished:
        return None, f"TimeoutError: test exceeded {timeout_sec:.1f}s per-call limit"
    return result_box[0], error_box[0]


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    # --- read job from stdin -------------------------------------------------
    try:
        raw = sys.stdin.read()
        job: Dict[str, Any] = json.loads(raw)
    except Exception as exc:
        sys.stdout.write(json.dumps({"init_error": f"Job parse error: {exc}", "results": []}))
        sys.exit(0)

    code: str = job.get("code", "")
    function_name: str = job.get("function_name", "")
    tests: List[Dict[str, Any]] = job.get("tests", [])
    per_test_timeout: float = float(job.get("per_test_timeout", 2.0))

    # --- build restricted execution namespace --------------------------------
    safe_globals: Dict[str, Any] = {
        "__builtins__": _SAFE_BUILTINS,
        "math": math,
    }
    local_scope: Dict[str, Any] = {}

    # --- execute student code ------------------------------------------------
    try:
        exec(code, safe_globals, local_scope)  # noqa: S102
    except Exception as exc:
        sys.stdout.write(json.dumps({
            "init_error": f"{type(exc).__name__}: {exc}",
            "results": [],
        }))
        sys.exit(0)

    # --- locate entry function -----------------------------------------------
    fn = local_scope.get(function_name)
    if not callable(fn):
        sys.stdout.write(json.dumps({
            "init_error": f"Function '{function_name}' not defined or not callable",
            "results": [],
        }))
        sys.exit(0)

    # --- run each test case --------------------------------------------------
    results: List[Dict[str, Any]] = []
    for test in tests:
        args = test.get("input", ())
        if not isinstance(args, (list, tuple)):
            args = (args,)
        kwargs: Dict[str, Any] = test.get("kwargs", {}) or {}
        expected = test.get("expected")
        name: str = test.get("name", f"test_{len(results) + 1}")
        is_hidden: bool = bool(test.get("is_hidden", False))
        is_edge: bool = bool(test.get("is_edge", False))

        t0 = time.perf_counter()
        actual, error = _call_with_timeout(fn, args, kwargs, per_test_timeout)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        passed = (error is None) and (actual == expected)

        results.append({
            "name": name,
            "passed": passed,
            "actual": None if is_hidden else actual,
            "expected": None if is_hidden else expected,
            "error": error,
            "time_ms": round(elapsed_ms, 2),
            "is_hidden": is_hidden,
            "is_edge": is_edge,
        })

    sys.stdout.write(json.dumps({"init_error": None, "results": results}))
    sys.exit(0)


if __name__ == "__main__":
    main()

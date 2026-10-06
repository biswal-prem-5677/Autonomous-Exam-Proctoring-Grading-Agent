"""Programming Code Evaluation & AST Sandbox Grading Engine.

Evaluates student programming submissions without any third-party APIs:
    1. Syntax & AST Validation (ast.parse)
    2. Static Security & Vulnerability Analysis (ast.NodeVisitor looking for dangerous calls)
    3. Algorithmic Complexity Estimation (Cyclomatic Complexity, loop nesting depth)
    4. Execution against Public, Hidden, and Edge-Case Test Suites with timeout
    5. Clean Code & Style Heuristics (naming, docstrings, modularity)

Zero external dependencies — pure local Python execution.
"""

import ast
import time
import traceback
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable

from src.grading.sandbox import run_in_sandbox, SandboxResult


@dataclass
class TestCaseResult:
    name: str
    is_hidden: bool
    is_edge_case: bool
    passed: bool
    input_args: Any
    expected_output: Any
    actual_output: Any
    execution_time_ms: float
    error_message: Optional[str] = None


@dataclass
class CodeSecurityReport:
    is_safe: bool
    violations: List[str]
    forbidden_modules: List[str]
    forbidden_calls: List[str]


@dataclass
class ProgrammingGradingResult:
    score: float
    max_score: float
    percentage: float
    syntax_valid: bool
    security: CodeSecurityReport
    cyclomatic_complexity: int
    loop_nesting_depth: int
    total_tests: int
    passed_tests: int
    test_results: List[TestCaseResult]
    style_score: float
    explanation: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "score": round(self.score, 2),
            "max_score": round(self.max_score, 2),
            "percentage": round(self.percentage, 2),
            "syntax_valid": self.syntax_valid,
            "security": {
                "is_safe": self.security.is_safe,
                "violations": self.security.violations,
            },
            "metrics": {
                "cyclomatic_complexity": self.cyclomatic_complexity,
                "loop_nesting_depth": self.loop_nesting_depth,
                "total_tests": self.total_tests,
                "passed_tests": self.passed_tests,
            },
            "style_score": round(self.style_score, 2),
            "test_details": [
                {
                    "name": t.name,
                    "hidden": t.is_hidden,
                    "passed": t.passed,
                    "time_ms": round(t.execution_time_ms, 2),
                    "error": t.error_message,
                }
                for t in self.test_results
            ],
            "explanation": self.explanation,
        }


class _SecurityAuditor(ast.NodeVisitor):
    """Inspects AST for unsafe imports, file access, and execution primitives."""

    FORBIDDEN_MODULES = {
        "os", "sys", "subprocess", "shutil", "socket", "http", "urllib",
        "requests", "ctypes", "multiprocessing", "threading", "importlib",
        "builtins", "posix", "nt"
    }

    FORBIDDEN_CALLS = {
        "eval", "exec", "open", "__import__", "globals", "locals",
        "compile", "input", "exit", "quit"
    }

    def __init__(self):
        self.violations: List[str] = []
        self.forbidden_modules_found: List[str] = []
        self.forbidden_calls_found: List[str] = []

    def visit_Import(self, node: ast.Import):
        for alias in node.names:
            root_mod = alias.name.split('.')[0]
            if root_mod in self.FORBIDDEN_MODULES:
                self.violations.append(f"Forbidden module import: '{alias.name}'")
                self.forbidden_modules_found.append(alias.name)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        if node.module:
            root_mod = node.module.split('.')[0]
            if root_mod in self.FORBIDDEN_MODULES:
                self.violations.append(f"Forbidden module import from: '{node.module}'")
                self.forbidden_modules_found.append(node.module)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        func_name = None
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            func_name = node.func.attr

        if func_name in self.FORBIDDEN_CALLS:
            self.violations.append(f"Forbidden function invocation: '{func_name}()'")
            self.forbidden_calls_found.append(func_name)
        self.generic_visit(node)


class _ComplexityAnalyzer(ast.NodeVisitor):
    """Calculates McCabe Cyclomatic Complexity and maximum loop nesting depth."""

    def __init__(self):
        self.complexity = 1
        self.current_depth = 0
        self.max_depth = 0

    def visit_If(self, node: ast.If):
        self.complexity += 1
        self.generic_visit(node)

    def visit_For(self, node: ast.For):
        self.complexity += 1
        self.current_depth += 1
        self.max_depth = max(self.max_depth, self.current_depth)
        self.generic_visit(node)
        self.current_depth -= 1

    def visit_While(self, node: ast.While):
        self.complexity += 1
        self.current_depth += 1
        self.max_depth = max(self.max_depth, self.current_depth)
        self.generic_visit(node)
        self.current_depth -= 1

    def visit_ExceptHandler(self, node: ast.ExceptHandler):
        self.complexity += 1
        self.generic_visit(node)

    def visit_BoolOp(self, node: ast.BoolOp):
        # Each 'and' or 'or' adds an alternate branch
        self.complexity += len(node.values) - 1
        self.generic_visit(node)


class ProgrammingGrader:
    """Robust AST analyzer and test executor for coding questions."""

    def __init__(
        self,
        execution_timeout_seconds: float = 1.5,
        weight_tests: float = 0.65,
        weight_edge_cases: float = 0.15,
        weight_complexity: float = 0.10,
        weight_style: float = 0.10,
    ):
        self.timeout = execution_timeout_seconds
        self.w_tests = weight_tests
        self.w_edge = weight_edge_cases
        self.w_comp = weight_complexity
        self.w_style = weight_style

    def grade(
        self,
        code_str: str,
        entry_function_name: str,
        public_tests: List[Dict[str, Any]],
        hidden_tests: Optional[List[Dict[str, Any]]] = None,
        edge_cases: Optional[List[Dict[str, Any]]] = None,
        marks: float = 10.0,
        max_allowed_complexity: int = 15,
    ) -> ProgrammingGradingResult:
        """Grade a student programming submission.

        Args:
            code_str: Python source code submitted by the student
            entry_function_name: Function to invoke e.g. "two_sum", "reverse_list"
            public_tests: List of dicts e.g. [{"input": [1, 2], "expected": 3, "name": "t1"}]
            hidden_tests: Test cases not displayed to the student
            edge_cases: Edge cases (empty array, single element, negative numbers)
            marks: Maximum score
            max_allowed_complexity: Cyclomatic complexity ceiling
        """
        hidden_tests = hidden_tests or []
        edge_cases = edge_cases or []

        # 1. Parse AST & check syntax
        try:
            tree = ast.parse(code_str)
            syntax_valid = True
        except SyntaxError as e:
            sec_empty = CodeSecurityReport(is_safe=False, violations=[f"SyntaxError: {str(e)}"], forbidden_modules=[], forbidden_calls=[])
            return ProgrammingGradingResult(
                score=0.0,
                max_score=marks,
                percentage=0.0,
                syntax_valid=False,
                security=sec_empty,
                cyclomatic_complexity=0,
                loop_nesting_depth=0,
                total_tests=len(public_tests) + len(hidden_tests) + len(edge_cases),
                passed_tests=0,
                test_results=[],
                style_score=0.0,
                explanation=f"Code failed syntax check: {str(e)}",
            )

        # 2. Security scan
        auditor = _SecurityAuditor()
        auditor.visit(tree)
        is_safe = len(auditor.violations) == 0
        security_report = CodeSecurityReport(
            is_safe=is_safe,
            violations=auditor.violations,
            forbidden_modules=auditor.forbidden_modules_found,
            forbidden_calls=auditor.forbidden_calls_found,
        )

        if not is_safe:
            return ProgrammingGradingResult(
                score=0.0,
                max_score=marks,
                percentage=0.0,
                syntax_valid=True,
                security=security_report,
                cyclomatic_complexity=0,
                loop_nesting_depth=0,
                total_tests=len(public_tests) + len(hidden_tests) + len(edge_cases),
                passed_tests=0,
                test_results=[],
                style_score=0.0,
                explanation=f"Security violation detected: {'; '.join(auditor.violations)}",
            )

        # 3. Complexity & AST metrics
        complexity_analyzer = _ComplexityAnalyzer()
        complexity_analyzer.visit(tree)
        cyclomatic = complexity_analyzer.complexity
        loop_depth = complexity_analyzer.max_depth

        # Style score
        has_docstring = ast.get_docstring(tree) is not None
        has_entry = any(
            isinstance(node, ast.FunctionDef) and node.name == entry_function_name
            for node in tree.body
        )

        style_score = 1.0
        if not has_entry:
            style_score -= 0.5
        if not has_docstring:
            style_score -= 0.2
        if loop_depth > 3:
            style_score -= 0.3
        style_score = max(0.0, min(1.0, style_score))

        # Complexity penalty if code is excessively convoluted
        complexity_ratio = 1.0
        if cyclomatic > max_allowed_complexity:
            complexity_ratio = max(0.2, 1.0 - (cyclomatic - max_allowed_complexity) * 0.05)

        # 4. Execute tests in isolated subprocess sandbox
        #    Student code never runs inside the Flask process.
        all_tests = [
            {"name": t.get("name", f"pub_{i}"),
             "input": t.get("input", ()),
             "expected": t.get("expected"),
             "kwargs": t.get("kwargs", {}),
             "is_hidden": False,
             "is_edge": False}
            for i, t in enumerate(public_tests)
        ] + [
            {"name": t.get("name", f"hid_{i}"),
             "input": t.get("input", ()),
             "expected": t.get("expected"),
             "kwargs": t.get("kwargs", {}),
             "is_hidden": True,
             "is_edge": False}
            for i, t in enumerate(hidden_tests)
        ] + [
            {"name": t.get("name", f"edge_{i}"),
             "input": t.get("input", ()),
             "expected": t.get("expected"),
             "kwargs": t.get("kwargs", {}),
             "is_hidden": False,
             "is_edge": True}
            for i, t in enumerate(edge_cases)
        ]

        sandbox_result: SandboxResult = run_in_sandbox(
            code=code_str,
            function_name=entry_function_name,
            tests=all_tests,
            timeout_seconds=max(3.0, self.timeout * max(1, len(all_tests))),
        )

        # --- Handle sandbox-level failures -----------------------------------
        if not sandbox_result.success:
            error_msg = sandbox_result.sandbox_error or "Unknown sandbox error"
            return ProgrammingGradingResult(
                score=0.0,
                max_score=marks,
                percentage=0.0,
                syntax_valid=True,
                security=security_report,
                cyclomatic_complexity=cyclomatic,
                loop_nesting_depth=loop_depth,
                total_tests=len(all_tests),
                passed_tests=0,
                test_results=[],
                style_score=style_score,
                explanation=error_msg,
            )

        # --- Handle worker-level init error (exec / function not found) ------
        if sandbox_result.init_error:
            return ProgrammingGradingResult(
                score=0.0,
                max_score=marks,
                percentage=0.0,
                syntax_valid=True,
                security=security_report,
                cyclomatic_complexity=cyclomatic,
                loop_nesting_depth=loop_depth,
                total_tests=len(all_tests),
                passed_tests=0,
                test_results=[],
                style_score=style_score,
                explanation=f"Runtime error: {sandbox_result.init_error}",
            )

        # --- Map sandbox results to TestCaseResult ---------------------------
        test_results: List[TestCaseResult] = []
        regular_passed = 0
        regular_total = len(public_tests) + len(hidden_tests)
        edge_passed = 0
        edge_total = len(edge_cases)

        for sr in sandbox_result.results:
            test_results.append(TestCaseResult(
                name=sr.name,
                is_hidden=sr.is_hidden,
                is_edge_case=sr.is_edge,
                passed=sr.passed,
                input_args="[HIDDEN]" if sr.is_hidden else None,
                expected_output="[HIDDEN]" if sr.is_hidden else sr.expected,
                actual_output="[HIDDEN]" if sr.is_hidden else sr.actual,
                execution_time_ms=sr.time_ms,
                error_message=sr.error,
            ))
            if sr.passed:
                if sr.is_edge:
                    edge_passed += 1
                else:
                    regular_passed += 1

        # 5. Compute composite score
        reg_ratio = (regular_passed / regular_total) if regular_total > 0 else 1.0
        edge_ratio = (edge_passed / edge_total) if edge_total > 0 else 1.0

        total_ratio = (
            self.w_tests * reg_ratio
            + self.w_edge * edge_ratio
            + self.w_comp * complexity_ratio
            + self.w_style * style_score
        )
        awarded_marks = min(marks, max(0.0, total_ratio * marks))
        total_tests_count = regular_total + edge_total
        passed_tests_count = regular_passed + edge_passed
        percentage = (awarded_marks / marks * 100.0) if marks > 0 else 0.0

        expl = (
            f"Score: {awarded_marks:.1f}/{marks:.1f} ({percentage:.0f}%). "
            f"Passed {passed_tests_count}/{total_tests_count} test cases "
            f"(Regular: {regular_passed}/{regular_total}, Edge: {edge_passed}/{edge_total}). "
            f"Cyclomatic complexity: {cyclomatic}, Max loop depth: {loop_depth}. "
            f"[Executed in isolated subprocess sandbox]"
        )

        return ProgrammingGradingResult(
            score=round(awarded_marks, 2),
            max_score=marks,
            percentage=round(percentage, 2),
            syntax_valid=True,
            security=security_report,
            cyclomatic_complexity=cyclomatic,
            loop_nesting_depth=loop_depth,
            total_tests=total_tests_count,
            passed_tests=passed_tests_count,
            test_results=test_results,
            style_score=style_score,
            explanation=expl,
        )

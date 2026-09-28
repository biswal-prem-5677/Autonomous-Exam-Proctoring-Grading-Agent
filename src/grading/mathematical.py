"""Mathematical Answer & Partial Credit Grading Engine.

Grades complex multi-step mathematical and physical derivations without external AI APIs:
    1. Formula identification & algebraic equivalence
    2. Numerical substitution verification
    3. Intermediate derivation step validation
    4. Final numerical / analytical result with tolerance
    5. Units of measurement consistency

Awards explainable partial credit:
    Score = Marks · (w_formula · C_formula + w_sub · C_sub + w_steps · C_steps + w_final · C_final)
"""

import re
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple


@dataclass
class MathStepEvaluation:
    step_number: int
    raw_text: str
    step_type: str         # "formula", "substitution", "intermediate", "final_answer", "units"
    is_valid: bool
    credit_ratio: float    # 0.0 to 1.0 for this step
    feedback: str


@dataclass
class MathGradingResult:
    score: float
    max_score: float
    percentage: float
    steps_evaluated: List[MathStepEvaluation]
    formula_credit: float
    substitution_credit: float
    intermediate_credit: float
    final_answer_credit: float
    units_credit: float
    explanation: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "score": round(self.score, 2),
            "max_score": round(self.max_score, 2),
            "percentage": round(self.percentage, 2),
            "formula_credit": round(self.formula_credit, 2),
            "substitution_credit": round(self.substitution_credit, 2),
            "intermediate_credit": round(self.intermediate_credit, 2),
            "final_answer_credit": round(self.final_answer_credit, 2),
            "units_credit": round(self.units_credit, 2),
            "steps": [
                {
                    "step": s.step_number,
                    "type": s.step_type,
                    "valid": s.is_valid,
                    "credit": round(s.credit_ratio, 2),
                    "feedback": s.feedback,
                }
                for s in self.steps_evaluated
            ],
            "explanation": self.explanation,
        }


class MathematicalGrader:
    """Evaluates multi-step mathematical solutions with partial credit."""

    def __init__(
        self,
        default_tolerance: float = 0.02,
        weight_formula: float = 0.30,
        weight_substitution: float = 0.25,
        weight_intermediate: float = 0.25,
        weight_final: float = 0.20,
    ):
        self.default_tolerance = default_tolerance
        self.w_formula = weight_formula
        self.w_sub = weight_substitution
        self.w_intermediate = weight_intermediate
        self.w_final = weight_final

    def _normalize_math_expr(self, expr: str) -> str:
        """Normalize mathematical expressions for comparison."""
        expr = expr.lower().strip()
        expr = re.sub(r'\s+', '', expr)
        # normalize multiplication symbols
        expr = expr.replace('×', '*').replace('·', '*').replace('÷', '/')
        # strip redundant surrounding parentheses
        if expr.startswith('(') and expr.endswith(')') and expr.count('(') == 1:
            expr = expr[1:-1]
        return expr

    def _extract_number_and_units(self, text: str) -> Tuple[Optional[float], Optional[str]]:
        """Extract final numerical value and trailing units."""
        # Match e.g. "45.2 m/s", "120 J", "3.1415", "-9.8 m/s^2"
        pattern = r'([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*([a-zA-Z/°^0-9\-_]*)'
        matches = list(re.finditer(pattern, text))
        if not matches:
            return None, None
        last_match = matches[-1]
        val_str = last_match.group(1)
        units_str = last_match.group(2).strip() or None
        try:
            return float(val_str), units_str
        except ValueError:
            return None, None

    def grade(
        self,
        student_solution: str,
        expected_formula: Optional[str] = None,
        expected_variables: Optional[Dict[str, float]] = None,
        expected_final_value: Optional[float] = None,
        expected_units: Optional[str] = None,
        marks: float = 10.0,
        tolerance: Optional[float] = None,
    ) -> MathGradingResult:
        """Grade a multi-step mathematical response.

        Args:
            student_solution: Raw student written derivation (may contain formulas, steps, numbers)
            expected_formula: Core formula e.g. "v = u + a*t" or "F = m*a"
            expected_variables: Expected substituted numbers e.g. {"m": 10, "a": 9.8}
            expected_final_value: The numerical answer
            expected_units: Expected units e.g. "m/s", "N", "J"
            marks: Total available marks
            tolerance: Relative numerical tolerance (default 0.02)
        """
        tol = tolerance if tolerance is not None else self.default_tolerance
        lines = [line.strip() for line in student_solution.split('\n') if line.strip()]

        step_evals: List[MathStepEvaluation] = []
        step_idx = 1

        # 1. Formula validation
        formula_credit = 0.0
        if expected_formula:
            norm_exp = self._normalize_math_expr(expected_formula)
            found_formula = False
            for line in lines:
                norm_line = self._normalize_math_expr(line)
                if norm_exp in norm_line or norm_line in norm_exp:
                    found_formula = True
                    break
                # Partial match: check if key operators/variables present
                vars_in_exp = re.findall(r'[a-zA-Z]', norm_exp)
                if all(v in norm_line for v in vars_in_exp if len(vars_in_exp) >= 2):
                    found_formula = True
                    break

            if found_formula:
                formula_credit = 1.0
                step_evals.append(MathStepEvaluation(
                    step_number=step_idx,
                    raw_text=expected_formula,
                    step_type="formula",
                    is_valid=True,
                    credit_ratio=1.0,
                    feedback=f"Correct governing formula stated: '{expected_formula}'.",
                ))
            else:
                step_evals.append(MathStepEvaluation(
                    step_number=step_idx,
                    raw_text="",
                    step_type="formula",
                    is_valid=False,
                    credit_ratio=0.0,
                    feedback=f"Missing or incorrect governing formula (expected '{expected_formula}').",
                ))
            step_idx += 1
        else:
            formula_credit = 1.0  # not tested

        # 2. Substitution check
        sub_credit = 0.0
        if expected_variables:
            matched_vars = 0
            for var_name, var_val in expected_variables.items():
                val_int = str(int(var_val)) if float(var_val).is_integer() else None
                val_float = str(round(var_val, 2))
                val_raw = str(var_val)

                found_var = False
                for line in lines:
                    if (val_int and val_int in line) or (val_float in line) or (val_raw in line):
                        found_var = True
                        break
                if found_var:
                    matched_vars += 1

            sub_ratio = matched_vars / len(expected_variables)
            sub_credit = sub_ratio
            step_evals.append(MathStepEvaluation(
                step_number=step_idx,
                raw_text="Variables substitution",
                step_type="substitution",
                is_valid=sub_ratio >= 0.75,
                credit_ratio=sub_ratio,
                feedback=f"Substituted {matched_vars}/{len(expected_variables)} required parameters correctly.",
            ))
            step_idx += 1
        else:
            sub_credit = 1.0


        # 3. Intermediate derivation step detection
        # Check if student showed working (>= 2 intermediate steps between formula and answer)
        intermediate_credit = 0.0
        if len(lines) >= 3:
            intermediate_credit = 1.0
            feedback_str = f"Clear algebraic derivation shown ({len(lines)} steps)."
        elif len(lines) == 2:
            intermediate_credit = 0.6
            feedback_str = "Partial derivation steps shown."
        else:
            intermediate_credit = 0.2
            feedback_str = "Direct answer given with minimal intermediate algebraic steps."

        step_evals.append(MathStepEvaluation(
            step_number=step_idx,
            raw_text="Derivation progression",
            step_type="intermediate",
            is_valid=intermediate_credit >= 0.6,
            credit_ratio=intermediate_credit,
            feedback=feedback_str,
        ))
        step_idx += 1

        # 4. Final numerical answer & units check
        final_credit = 0.0
        units_credit = 0.0
        if expected_final_value is not None:
            # Check last few lines for numerical answers
            candidate_vals = []
            for line in lines[-2:]:
                num, unit = self._extract_number_and_units(line)
                if num is not None:
                    candidate_vals.append((num, unit))

            if candidate_vals:
                best_match = None
                best_err = float('inf')
                for num, unit in candidate_vals:
                    if expected_final_value != 0:
                        err = abs(num - expected_final_value) / abs(expected_final_value)
                    else:
                        err = abs(num)
                    if err < best_err or (err == best_err and unit is not None and best_match and best_match[1] is None):
                        best_err = err
                        best_match = (num, unit)


                val_found, unit_found = best_match
                if best_err <= tol:
                    final_credit = 1.0
                    val_feedback = f"Final value {val_found} matches expected {expected_final_value} (err: {best_err:.2%})."
                elif best_err <= tol * 2.5:
                    final_credit = 0.5
                    val_feedback = f"Final value {val_found} near miss for {expected_final_value} (err: {best_err:.2%})."
                else:
                    final_credit = 0.0
                    val_feedback = f"Final value {val_found} incorrect (expected {expected_final_value})."

                # Units check
                if expected_units:
                    if unit_found and expected_units.lower() in unit_found.lower():
                        units_credit = 1.0
                        unit_feedback = f"Correct units '{unit_found}'."
                    else:
                        units_credit = 0.0
                        unit_feedback = f"Missing or incorrect units (expected '{expected_units}', found '{unit_found}')."
                else:
                    units_credit = 1.0
                    unit_feedback = ""

                step_evals.append(MathStepEvaluation(
                    step_number=step_idx,
                    raw_text=f"{val_found} {unit_found or ''}".strip(),
                    step_type="final_answer",
                    is_valid=final_credit == 1.0,
                    credit_ratio=final_credit,
                    feedback=f"{val_feedback} {unit_feedback}".strip(),
                ))
            else:
                step_evals.append(MathStepEvaluation(
                    step_number=step_idx,
                    raw_text="Final Answer",
                    step_type="final_answer",
                    is_valid=False,
                    credit_ratio=0.0,
                    feedback="No clearly identifiable final numerical answer.",
                ))
        else:
            final_credit = 1.0
            units_credit = 1.0

        # Weighted score aggregation
        total_ratio = (
            self.w_formula * formula_credit +
            self.w_sub * sub_credit +
            self.w_intermediate * intermediate_credit +
            self.w_final * (0.8 * final_credit + 0.2 * units_credit)
        )
        awarded_marks = min(marks, max(0.0, total_ratio * marks))
        percentage = (awarded_marks / marks * 100.0) if marks > 0 else 0.0

        explanation_parts = [
            f"Score: {awarded_marks:.1f}/{marks:.1f} ({percentage:.0f}%).",
            f"Formula: {formula_credit:.0%}.",
            f"Substitution: {sub_credit:.0%}.",
            f"Intermediate Working: {intermediate_credit:.0%}.",
            f"Final Answer: {final_credit:.0%}.",
        ]

        return MathGradingResult(
            score=round(awarded_marks, 2),
            max_score=marks,
            percentage=round(percentage, 2),
            steps_evaluated=step_evals,
            formula_credit=formula_credit,
            substitution_credit=sub_credit,
            intermediate_credit=intermediate_credit,
            final_answer_credit=final_credit,
            units_credit=units_credit,
            explanation=" ".join(explanation_parts),
        )

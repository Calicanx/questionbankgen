"""Mathematical Answer Verification for Generated Questions.

This module provides comprehensive mathematical verification across all widget types:
- numeric-input: Evaluates expressions and verifies numeric answers
- input-number: Same as numeric-input
- expression: Verifies algebraic simplifications and equivalences
- radio: Verifies correct choice for math-based multiple choice
- dropdown: Basic verification for math-based dropdowns

Uses SymPy for symbolic mathematics to ensure accuracy.
"""

import re
import logging
from dataclasses import dataclass, field
from typing import Any, Optional, List, Dict, Tuple, Union
from fractions import Fraction

logger = logging.getLogger(__name__)

# Try importing SymPy
try:
    from sympy import (
        symbols, sympify, simplify, expand, factor, Eq, solve,
        latex, S, Symbol, Rational, Float, Integer, pi, E,
        sin, cos, tan, log, exp, sqrt, Abs, N, nsimplify
    )
    from sympy.parsing.latex import parse_latex
    from sympy.core.sympify import SympifyError
    SYMPY_AVAILABLE = True
except ImportError:
    SYMPY_AVAILABLE = False
    SympifyError = Exception
    logger.warning("SymPy not available - mathematical verification disabled")


@dataclass
class VerificationResult:
    """Result of mathematical verification."""
    is_valid: bool
    expected_answer: Optional[Any] = None
    actual_answer: Optional[Any] = None
    error_message: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ParsedQuestion:
    """Parsed mathematical content from a question."""
    expression: Optional[str] = None  # The main expression to evaluate
    variables: Dict[str, Any] = field(default_factory=dict)  # Variable assignments
    operation: Optional[str] = None  # Type of operation (evaluate, simplify, solve, etc.)
    raw_content: str = ""


class MathExpressionParser:
    """Parses mathematical expressions from question content."""

    # Patterns for extracting math content
    LATEX_PATTERNS = [
        r'\$([^$]+)\$',           # $...$
        r'\\\(([^)]+)\\\)',       # \(...\)
        r'\\\[([^\]]+)\\\]',      # \[...\]
    ]

    # Patterns for variable assignments
    ASSIGNMENT_PATTERNS = [
        r'when\s+\$?([a-zA-Z])\s*=\s*(-?\d+(?:\.\d+)?)\$?',  # when x = 5
        r'\$([a-zA-Z])\s*=\s*(-?\d+(?:\.\d+)?)\$',           # $x = 5$
        r'(?:let|if|where)\s+\$?([a-zA-Z])\s*=\s*(-?\d+(?:\.\d+)?)\$?',  # let x = 5
        r'\$([a-zA-Z])\$\s*=\s*(-?\d+(?:\.\d+)?)',           # $x$ = 5
        r'and\s+\$?([a-zA-Z])\s*=\s*(-?\d+(?:\.\d+)?)\$?',   # and y = 6
        r'([a-zA-Z])\s*=\s*(-?\d+(?:\.\d+)?)',               # x = 5 (plain)
    ]

    # Patterns for function definitions and evaluations
    FUNCTION_PATTERNS = [
        r'\$?([a-zA-Z])\(([a-zA-Z])\)\s*=\s*([^$\n]+?)\$?(?:\s|$)',  # f(x) = 5x-3
        r'\$?([a-zA-Z])\((-?\d+(?:\.\d+)?)\)',  # f(5) - function call
    ]

    # Patterns for operation detection
    OPERATION_PATTERNS = {
        'evaluate': [r'evaluate', r'find the value', r'calculate', r'compute', r'what is'],
        'simplify': [r'simplify', r'combine like terms', r'reduce'],
        'solve': [r'solve for', r'find\s+\$?[a-zA-Z]\$?'],
        'factor': [r'factor'],
        'expand': [r'expand', r'multiply out'],
    }

    def parse(self, content: str) -> ParsedQuestion:
        """Parse question content to extract mathematical components."""
        result = ParsedQuestion(raw_content=content)

        # Check for function definitions first (e.g., f(x) = 5x-3, f(5) = ?)
        func_result = self._parse_function_evaluation(content)
        if func_result:
            result.expression = func_result['expression']
            result.variables = func_result['variables']
            result.operation = 'evaluate'
            return result

        # Extract variable assignments
        result.variables = self._extract_variables(content)

        # Detect operation type
        result.operation = self._detect_operation(content)

        # Extract the main expression
        result.expression = self._extract_main_expression(content, result.variables)

        return result

    def _parse_function_evaluation(self, content: str) -> Optional[Dict[str, Any]]:
        """Parse function definitions and evaluations like f(x) = 5x-3, f(5)."""
        # Look for function definition: f(x) = expression
        # More robust pattern that captures until end of line or next $ or next f(
        func_def_patterns = [
            r'\$([a-zA-Z])\(([a-zA-Z])\)\s*=\s*([^$]+?)\$',  # $f(x) = 3x + 2$
            r'([a-zA-Z])\(([a-zA-Z])\)\s*=\s*([^\n$]+)',     # f(x) = 3x + 2
        ]
        func_call_pattern = r'\$?([a-zA-Z])\((-?\d+(?:\.\d+)?)\)'

        func_def_match = None
        for pattern in func_def_patterns:
            func_def_match = re.search(pattern, content)
            if func_def_match:
                break

        func_call_matches = re.findall(func_call_pattern, content)

        if func_def_match and func_call_matches:
            func_name = func_def_match.group(1)
            var_name = func_def_match.group(2)
            expression = func_def_match.group(3).strip()

            # Clean up expression (remove trailing punctuation or whitespace)
            expression = re.sub(r'[\s,;]+$', '', expression)

            # Find the function call for this function
            for call_name, call_value in func_call_matches:
                if call_name == func_name:
                    try:
                        value = float(call_value)
                        if value == int(value):
                            value = int(value)
                        return {
                            'expression': expression,
                            'variables': {var_name: value}
                        }
                    except ValueError:
                        pass

        return None

    def _extract_variables(self, content: str) -> Dict[str, float]:
        """Extract variable assignments from content."""
        variables = {}
        content_lower = content.lower()

        for pattern in self.ASSIGNMENT_PATTERNS:
            matches = re.findall(pattern, content_lower, re.IGNORECASE)
            for match in matches:
                if len(match) >= 2:
                    var_name = match[0].lower()
                    try:
                        value = float(match[1])
                        # Store as int if it's a whole number
                        if value == int(value):
                            value = int(value)
                        variables[var_name] = value
                    except ValueError:
                        pass

        return variables

    def _detect_operation(self, content: str) -> str:
        """Detect the type of mathematical operation requested."""
        content_lower = content.lower()

        for operation, patterns in self.OPERATION_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, content_lower):
                    return operation

        # Default to evaluate if variables are present
        return 'evaluate'

    def _extract_main_expression(
        self,
        content: str,
        variables: Dict[str, float]
    ) -> Optional[str]:
        """Extract the main mathematical expression from content."""
        # Find all LaTeX expressions
        expressions = []
        for pattern in self.LATEX_PATTERNS:
            matches = re.findall(pattern, content)
            expressions.extend(matches)

        if not expressions:
            return None

        # Filter out pure variable assignments
        var_names = set(variables.keys())
        filtered = []
        for expr in expressions:
            # Skip if it's just a variable assignment
            expr_clean = expr.strip()
            is_assignment = False
            for var in var_names:
                if re.match(rf'^{var}\s*=\s*-?\d+', expr_clean, re.IGNORECASE):
                    is_assignment = True
                    break

            if not is_assignment and len(expr_clean) > 1:
                filtered.append(expr_clean)

        # Return the first non-assignment expression (usually the main one)
        if filtered:
            return filtered[0]
        elif expressions:
            return expressions[0]

        return None


class MathEvaluator:
    """Evaluates mathematical expressions using SymPy."""

    def __init__(self):
        self.parser = MathExpressionParser()

    def evaluate_expression(
        self,
        expression: str,
        variables: Dict[str, Any],
    ) -> Optional[Union[float, int, str]]:
        """Evaluate a mathematical expression with given variable values."""
        if not SYMPY_AVAILABLE:
            return None

        try:
            # Clean and convert LaTeX to SymPy-compatible format
            expr_clean = self._latex_to_sympy(expression)

            # Create symbol dict
            symbol_dict = {}
            for var_name, value in variables.items():
                symbol_dict[var_name] = sympify(value)

            # Parse and evaluate
            expr = sympify(expr_clean, locals=symbol_dict)

            # Substitute variables
            for var_name, value in variables.items():
                var_symbol = Symbol(var_name)
                expr = expr.subs(var_symbol, value)

            # Evaluate to number
            result = expr.evalf()

            # Convert to Python number
            if result.is_integer:
                return int(result)
            else:
                return float(result)

        except Exception as e:
            logger.debug(f"Expression evaluation failed: {e}")
            return None

    def _latex_to_sympy(self, latex_str: str) -> str:
        """Convert LaTeX notation to SymPy-compatible format."""
        result = latex_str

        # Remove text formatting
        result = re.sub(r'\\text\{[^}]*\}', '', result)
        result = re.sub(r'\\textbf\{[^}]*\}', '', result)

        # Convert fractions with single-char denominator: \dfrac{24}b -> (24)/(b)
        # Must come before the general fraction pattern
        result = re.sub(r'\\d?frac\{([^{}]+)\}([a-zA-Z])', r'((\1)/(\2))', result)

        # Convert fractions: \frac{a}{b} or \dfrac{a}{b} -> (a)/(b)
        while re.search(r'\\d?frac\{([^{}]+)\}\{([^{}]+)\}', result):
            result = re.sub(r'\\d?frac\{([^{}]+)\}\{([^{}]+)\}', r'((\1)/(\2))', result)

        # Convert sqrt: \sqrt{x} -> sqrt(x)
        result = re.sub(r'\\sqrt\{([^{}]+)\}', r'sqrt(\1)', result)

        # Convert powers: ^{n} -> **n, x^2 -> x**2
        result = re.sub(r'\^\{([^{}]+)\}', r'**(\1)', result)
        result = re.sub(r'\^(\d+)', r'**\1', result)
        result = re.sub(r'\^([a-zA-Z])', r'**\1', result)

        # Convert multiplication: \cdot and \times -> *
        result = re.sub(r'\\cdot', '*', result)
        result = re.sub(r'\\times', '*', result)

        # Remove remaining LaTeX commands but keep content
        result = re.sub(r'\\[a-zA-Z]+\{([^{}]*)\}', r'\1', result)
        result = re.sub(r'\\[a-zA-Z]+', '', result)

        # Clean up braces
        result = result.replace('{', '(').replace('}', ')')

        # Add implicit multiplication: 2x -> 2*x
        result = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', result)
        # x2 -> x*2 but not for x**2
        result = re.sub(r'([a-zA-Z])(\d)(?!\*)', r'\1*\2', result)
        # )(  -> )*(
        result = re.sub(r'\)\(', r')*(', result)
        # )x -> )*x
        result = re.sub(r'\)([a-zA-Z])', r')*\1', result)
        # x( -> x*(  but NOT for known functions
        known_funcs = ['sin', 'cos', 'tan', 'log', 'exp', 'sqrt', 'abs']
        for func in known_funcs:
            # Temporarily replace function names
            result = result.replace(f'{func}(', f'__{func}__(')
        # Now add * before remaining (
        result = re.sub(r'([a-zA-Z])\(', r'\1*(', result)
        # Restore function names
        for func in known_funcs:
            result = result.replace(f'__{func}__(', f'{func}(')
        # 2( -> 2*(
        result = re.sub(r'(\d)\(', r'\1*(', result)
        # )2 -> )*2
        result = re.sub(r'\)(\d)', r')*\1', result)

        # Clean up spaces
        result = ' '.join(result.split())

        return result

    def simplify_expression(self, expression: str) -> Optional[str]:
        """Simplify an algebraic expression and return the result."""
        if not SYMPY_AVAILABLE:
            return None

        try:
            expr_clean = self._latex_to_sympy(expression)
            expr = sympify(expr_clean)
            simplified = simplify(expr)
            return str(simplified)
        except Exception as e:
            logger.debug(f"Simplification failed: {e}")
            return None

    def expressions_equivalent(self, expr1: str, expr2: str) -> bool:
        """Check if two expressions are mathematically equivalent."""
        if not SYMPY_AVAILABLE:
            return False

        try:
            e1_clean = self._latex_to_sympy(expr1)
            e2_clean = self._latex_to_sympy(expr2)

            e1 = sympify(e1_clean)
            e2 = sympify(e2_clean)

            # Check if difference simplifies to zero
            diff = simplify(e1 - e2)
            return diff == 0
        except Exception:
            return False


class MathAnswerVerifier:
    """Verifies mathematical answers in generated questions."""

    def __init__(self):
        self.parser = MathExpressionParser()
        self.evaluator = MathEvaluator()

    def verify_question(self, question_data: Dict[str, Any]) -> VerificationResult:
        """Verify all answers in a question."""
        question = question_data.get("question", {})
        content = question.get("content", "")
        widgets = question.get("widgets", {})

        # Parse the question content
        parsed = self.parser.parse(content)

        # Verify each graded widget
        all_valid = True
        errors = []
        details = {}

        for widget_id, widget in widgets.items():
            if not widget.get("graded", True):
                continue

            widget_type = widget.get("type", "")
            result = self._verify_widget(widget_type, widget, parsed, content)

            details[widget_id] = {
                "type": widget_type,
                "valid": result.is_valid,
                "expected": result.expected_answer,
                "actual": result.actual_answer,
                "error": result.error_message,
            }

            if not result.is_valid:
                all_valid = False
                if result.error_message:
                    errors.append(f"{widget_id}: {result.error_message}")

        return VerificationResult(
            is_valid=all_valid,
            error_message="; ".join(errors) if errors else None,
            details=details,
        )

    def _verify_widget(
        self,
        widget_type: str,
        widget: Dict[str, Any],
        parsed: ParsedQuestion,
        content: str,
    ) -> VerificationResult:
        """Verify a specific widget's answer."""

        if widget_type == "numeric-input":
            return self._verify_numeric_input(widget, parsed)
        elif widget_type == "input-number":
            return self._verify_input_number(widget, parsed)
        elif widget_type == "expression":
            return self._verify_expression(widget, parsed)
        elif widget_type == "radio":
            return self._verify_radio(widget, parsed, content)
        elif widget_type == "dropdown":
            return self._verify_dropdown(widget, parsed)
        else:
            # Skip verification for complex/non-math widgets
            return VerificationResult(is_valid=True, details={"skipped": True})

    def _verify_numeric_input(
        self,
        widget: Dict[str, Any],
        parsed: ParsedQuestion,
    ) -> VerificationResult:
        """Verify numeric-input widget answer."""
        options = widget.get("options", {})
        answers = options.get("answers", [])

        # Get the stated correct answer
        stated_answer = None
        for answer in answers:
            if answer.get("status") == "correct":
                stated_answer = answer.get("value")
                break

        if stated_answer is None:
            return VerificationResult(
                is_valid=False,
                error_message="No correct answer defined"
            )

        # Try to compute the expected answer
        if parsed.expression and parsed.variables:
            computed = self.evaluator.evaluate_expression(
                parsed.expression,
                parsed.variables
            )

            if computed is not None:
                # Compare answers with tolerance
                try:
                    stated_float = float(stated_answer)
                    computed_float = float(computed)

                    # Allow small floating point differences
                    tolerance = 0.0001
                    is_close = abs(stated_float - computed_float) < tolerance

                    if not is_close:
                        return VerificationResult(
                            is_valid=False,
                            expected_answer=computed,
                            actual_answer=stated_answer,
                            error_message=f"Answer mismatch: expected {computed}, got {stated_answer}"
                        )

                    return VerificationResult(
                        is_valid=True,
                        expected_answer=computed,
                        actual_answer=stated_answer,
                    )
                except (ValueError, TypeError):
                    pass

        # Could not verify computationally, accept if answer exists
        return VerificationResult(
            is_valid=True,
            actual_answer=stated_answer,
            details={"verification": "structural_only"}
        )

    def _verify_input_number(
        self,
        widget: Dict[str, Any],
        parsed: ParsedQuestion,
    ) -> VerificationResult:
        """Verify input-number widget answer (similar to numeric-input)."""
        options = widget.get("options", {})
        stated_answer = options.get("value")

        if stated_answer is None:
            return VerificationResult(
                is_valid=False,
                error_message="No answer value defined"
            )

        # Try to compute the expected answer
        if parsed.expression and parsed.variables:
            computed = self.evaluator.evaluate_expression(
                parsed.expression,
                parsed.variables
            )

            if computed is not None:
                try:
                    stated_float = float(stated_answer)
                    computed_float = float(computed)

                    tolerance = 0.0001
                    is_close = abs(stated_float - computed_float) < tolerance

                    if not is_close:
                        return VerificationResult(
                            is_valid=False,
                            expected_answer=computed,
                            actual_answer=stated_answer,
                            error_message=f"Answer mismatch: expected {computed}, got {stated_answer}"
                        )

                    return VerificationResult(
                        is_valid=True,
                        expected_answer=computed,
                        actual_answer=stated_answer,
                    )
                except (ValueError, TypeError):
                    pass

        return VerificationResult(
            is_valid=True,
            actual_answer=stated_answer,
            details={"verification": "structural_only"}
        )

    def _verify_expression(
        self,
        widget: Dict[str, Any],
        parsed: ParsedQuestion,
    ) -> VerificationResult:
        """Verify expression widget answer."""
        options = widget.get("options", {})
        answer_forms = options.get("answerForms", [])

        # Get the correct answer form
        correct_value = None
        for form in answer_forms:
            if form.get("considered") == "correct":
                correct_value = form.get("value")
                break

        if not correct_value:
            return VerificationResult(
                is_valid=False,
                error_message="No correct answer form defined"
            )

        # For simplification questions, verify the answer is simplified correctly
        if parsed.expression and parsed.operation == 'simplify':
            simplified = self.evaluator.simplify_expression(parsed.expression)

            if simplified:
                # Check if the stated answer is equivalent
                is_equiv = self.evaluator.expressions_equivalent(
                    correct_value,
                    simplified
                )

                if not is_equiv:
                    # Also try checking against the original
                    is_equiv = self.evaluator.expressions_equivalent(
                        correct_value,
                        parsed.expression
                    )

                if not is_equiv:
                    return VerificationResult(
                        is_valid=False,
                        expected_answer=simplified,
                        actual_answer=correct_value,
                        error_message=f"Expression mismatch: expected {simplified}"
                    )

        return VerificationResult(
            is_valid=True,
            actual_answer=correct_value,
        )

    def _verify_radio(
        self,
        widget: Dict[str, Any],
        parsed: ParsedQuestion,
        content: str,
    ) -> VerificationResult:
        """Verify radio widget (multiple choice) answer."""
        options = widget.get("options", {})
        choices = options.get("choices", [])

        # Find the correct choice
        correct_choice = None
        correct_index = None
        for i, choice in enumerate(choices):
            if choice.get("correct", False):
                correct_choice = choice
                correct_index = i
                break

        if correct_choice is None:
            return VerificationResult(
                is_valid=False,
                error_message="No correct choice marked"
            )

        # For math evaluation questions, try to verify the correct choice
        if parsed.expression and parsed.variables:
            computed = self.evaluator.evaluate_expression(
                parsed.expression,
                parsed.variables
            )

            if computed is not None:
                # Check if correct choice content matches computed value
                choice_content = correct_choice.get("content", "")

                # Extract number from choice content
                numbers = re.findall(r'-?\d+(?:\.\d+)?', choice_content)
                if numbers:
                    try:
                        choice_value = float(numbers[0])
                        computed_float = float(computed)

                        if abs(choice_value - computed_float) > 0.0001:
                            # Find which choice has the right answer
                            for i, ch in enumerate(choices):
                                ch_content = ch.get("content", "")
                                ch_numbers = re.findall(r'-?\d+(?:\.\d+)?', ch_content)
                                if ch_numbers:
                                    if abs(float(ch_numbers[0]) - computed_float) < 0.0001:
                                        return VerificationResult(
                                            is_valid=False,
                                            expected_answer=computed,
                                            actual_answer=choice_value,
                                            error_message=f"Wrong choice marked correct. Expected {computed}, choice {correct_index} has {choice_value}"
                                        )
                    except (ValueError, TypeError):
                        pass

        # Structural verification passed
        return VerificationResult(
            is_valid=True,
            actual_answer=correct_choice.get("content"),
            details={"correct_index": correct_index}
        )

    def _verify_dropdown(
        self,
        widget: Dict[str, Any],
        parsed: ParsedQuestion,
    ) -> VerificationResult:
        """Verify dropdown widget answer."""
        options = widget.get("options", {})
        choices = options.get("choices", [])

        # Find correct choice
        correct_choice = None
        for choice in choices:
            if choice.get("correct", False):
                correct_choice = choice
                break

        if correct_choice is None:
            return VerificationResult(
                is_valid=False,
                error_message="No correct choice marked"
            )

        # Dropdown answers are often conceptual, hard to verify mathematically
        return VerificationResult(
            is_valid=True,
            actual_answer=correct_choice.get("content"),
            details={"verification": "structural_only"}
        )

    def compute_correct_answer(
        self,
        content: str,
        widget_type: str,
    ) -> Optional[Union[float, int, str]]:
        """Compute what the correct answer should be based on question content."""
        parsed = self.parser.parse(content)

        if not parsed.expression or not parsed.variables:
            return None

        return self.evaluator.evaluate_expression(
            parsed.expression,
            parsed.variables
        )


# Singleton instance
math_verifier = MathAnswerVerifier()


def verify_and_fix_answer(
    question_data: Dict[str, Any],
) -> Tuple[Dict[str, Any], VerificationResult]:
    """Verify a question's answers and fix them if incorrect.

    Returns:
        Tuple of (potentially fixed question data, verification result)
    """
    import copy
    result = math_verifier.verify_question(question_data)

    if result.is_valid:
        return question_data, result

    # Try to fix incorrect answers
    fixed_data = copy.deepcopy(question_data)
    question = fixed_data.get("question", {})
    content = question.get("content", "")
    widgets = question.get("widgets", {})

    # Parse to get expected values
    parser = MathExpressionParser()
    evaluator = MathEvaluator()
    parsed = parser.parse(content)

    if not parsed.expression or not parsed.variables:
        return question_data, result

    computed = evaluator.evaluate_expression(parsed.expression, parsed.variables)
    if computed is None:
        return question_data, result

    # Fix each widget
    fixes_made = []
    for widget_id, widget in widgets.items():
        widget_type = widget.get("type", "")

        if widget_type == "numeric-input":
            answers = widget.get("options", {}).get("answers", [])
            for answer in answers:
                if answer.get("status") == "correct":
                    old_value = answer.get("value")
                    # Store as int if whole number
                    if computed == int(computed):
                        answer["value"] = int(computed)
                    else:
                        answer["value"] = computed
                    fixes_made.append(f"{widget_id}: {old_value} -> {computed}")
                    break

        elif widget_type == "input-number":
            old_value = widget.get("options", {}).get("value")
            if computed == int(computed):
                widget["options"]["value"] = int(computed)
            else:
                widget["options"]["value"] = computed
            fixes_made.append(f"{widget_id}: {old_value} -> {computed}")

    if fixes_made:
        logger.info(f"Fixed answers: {fixes_made}")
        # Re-verify
        new_result = math_verifier.verify_question(fixed_data)
        new_result.details["fixes_applied"] = fixes_made
        return fixed_data, new_result

    return question_data, result


if __name__ == "__main__":
    # Test the verifier
    print("Testing Math Verifier...")

    # Test expression parsing
    parser = MathExpressionParser()

    test_cases = [
        "**Evaluate $5a + \\dfrac{24}b$ when $a=4$ and $b=6$.**",
        "**Evaluate $6+x$ when $x=3$.**",
        "$f(x) = 5x-3$\n\n$f(5)=$",
        "**Evaluate $\\dfrac{2}{5}g+3h-6$ when $g=10$ and $h=6$.**",
    ]

    evaluator = MathEvaluator()

    for test in test_cases:
        print(f"\n{'='*50}")
        print(f"Input: {test[:60]}...")
        parsed = parser.parse(test)
        print(f"Expression: {parsed.expression}")
        print(f"Variables: {parsed.variables}")
        print(f"Operation: {parsed.operation}")

        if parsed.expression and parsed.variables:
            result = evaluator.evaluate_expression(parsed.expression, parsed.variables)
            print(f"Computed Answer: {result}")

    print("\n" + "="*50)
    print("Tests complete!")

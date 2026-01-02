"""Solution Generator - Step-by-step math problem solving using symbolic math.

Uses SymPy for accurate, symbolic mathematics instead of AI hallucination.
This follows the Photomath approach of using a Computer Algebra System (CAS)
to generate precise, step-by-step solutions.

Supported problem types:
- Linear equations (2x + 5 = 11)
- Quadratic equations (x^2 - 5x + 6 = 0)
- Systems of equations
- Simplification
- Factoring
- Basic calculus (derivatives, integrals)
"""

import re
import logging
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass

try:
    from sympy import (
        symbols, solve, simplify, expand, factor, Eq, sympify,
        Derivative, Integral, sin, cos, tan, log, exp, sqrt, Abs,
        latex, S, Symbol, Add, Mul, Pow, Rational, Integer
    )
    from sympy.parsing.latex import parse_latex
    from sympy.core.sympify import SympifyError
    SYMPY_AVAILABLE = True
except ImportError:
    SYMPY_AVAILABLE = False
    # Define placeholders for type hints when SymPy not available
    Symbol = Any
    SympifyError = Exception

logger = logging.getLogger(__name__)


@dataclass
class SolutionStep:
    """Represents a single step in a solution."""
    step: int
    content: str  # LaTeX/Math content
    action: str   # What operation was performed


@dataclass
class Solution:
    """Complete solution with steps and final answer."""
    steps: List[SolutionStep]
    final_answer: str
    problem_type: str


def generate_solution(
    equation_or_expression: str,
    variable: Optional[str] = None,
    problem_type: Optional[str] = None,
) -> Optional[Solution]:
    """Generate step-by-step solution for a math problem.

    Args:
        equation_or_expression: The math problem (LaTeX or plain text)
        variable: The variable to solve for (auto-detected if None)
        problem_type: Optional hint about problem type ("linear", "quadratic", etc.)

    Returns:
        Solution object with steps and answer, or None if failed
    """
    if not SYMPY_AVAILABLE:
        logger.error("SymPy not available for solution generation")
        return None

    try:
        # Parse the input
        parsed = _parse_equation(equation_or_expression)
        if parsed is None:
            return None

        lhs, rhs = parsed

        # Auto-detect variable if not specified
        if variable is None:
            variable = _detect_variable(lhs, rhs)

        x = Symbol(variable)

        # Detect problem type if not specified
        if problem_type is None:
            problem_type = _detect_problem_type(lhs - rhs, x)

        # Generate solution based on type
        if problem_type == "linear":
            return _solve_linear(lhs, rhs, x)
        elif problem_type == "quadratic":
            return _solve_quadratic(lhs, rhs, x)
        elif problem_type == "simplify":
            return _solve_simplify(lhs, x)
        elif problem_type == "factor":
            return _solve_factor(lhs, x)
        else:
            return _solve_generic(lhs, rhs, x)

    except Exception as e:
        logger.error(f"Error generating solution: {e}")
        return None


def _parse_equation(equation_str: str) -> Optional[Tuple[Any, Any]]:
    """Parse an equation string into LHS and RHS expressions."""
    try:
        # Clean up the string
        equation_str = equation_str.strip()
        equation_str = equation_str.replace('$', '')  # Remove LaTeX delimiters

        # Try LaTeX parsing first if it looks like LaTeX
        if '\\' in equation_str or '{' in equation_str:
            try:
                expr = parse_latex(equation_str)
                if hasattr(expr, 'lhs') and hasattr(expr, 'rhs'):
                    return expr.lhs, expr.rhs
                else:
                    return expr, S.Zero
            except Exception:
                pass

        # Pre-process for implicit multiplication (2x -> 2*x)
        equation_str = _add_implicit_multiplication(equation_str)

        # Check for equals sign
        if '=' in equation_str:
            parts = equation_str.split('=')
            if len(parts) == 2:
                lhs = sympify(parts[0].strip())
                rhs = sympify(parts[1].strip())
                return lhs, rhs

        # No equals sign - treat as expression
        expr = sympify(equation_str)
        return expr, S.Zero

    except SympifyError as e:
        logger.error(f"Could not parse equation: {e}")
        return None


def _add_implicit_multiplication(expr_str: str) -> str:
    """Add explicit multiplication where implicit (e.g., 2x -> 2*x)."""
    result = expr_str

    # Add * between number and letter (2x -> 2*x)
    result = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', result)

    # Add * between letter and number (x2 -> x*2, but not x^2)
    result = re.sub(r'([a-zA-Z])(\d)(?!\^)', r'\1*\2', result)

    # Add * between closing paren and letter or number ((2)x -> (2)*x)
    result = re.sub(r'\)([a-zA-Z0-9])', r')*\1', result)

    # Add * between letter/number and opening paren (2(x -> 2*(x)
    result = re.sub(r'([a-zA-Z0-9])\(', r'\1*(', result)

    # Add * between closing and opening parens ()() -> ()*()
    result = re.sub(r'\)\(', r')*(', result)

    return result


def _detect_variable(lhs: Any, rhs: Any) -> str:
    """Detect the variable to solve for in the equation."""
    # Get all free symbols from both sides
    all_symbols = lhs.free_symbols.union(rhs.free_symbols)

    # Filter to single-letter symbols (likely variables)
    variables = [s for s in all_symbols if len(str(s)) == 1 and str(s).isalpha()]

    if variables:
        # Prefer common variable names
        preferred = ['x', 'y', 'z', 'n', 'm', 'k', 'w', 'a', 'b', 'c']
        for pref in preferred:
            for var in variables:
                if str(var).lower() == pref:
                    return str(var)
        # Return first found variable
        return str(variables[0])

    return 'x'  # Default to x


def _detect_problem_type(expr: Any, x: Symbol) -> str:
    """Detect the type of math problem."""
    try:
        # Get the degree of the polynomial
        poly_expr = expr.as_poly(x)
        if poly_expr is not None:
            degree = poly_expr.degree()
            if degree == 1:
                return "linear"
            elif degree == 2:
                return "quadratic"
    except Exception:
        pass

    return "generic"


def _solve_linear(lhs: Any, rhs: Any, x: Symbol) -> Solution:
    """Solve a linear equation with step-by-step explanation."""
    steps = []
    step_num = 1
    var_name = str(x)  # Get actual variable name

    # Step 1: Original equation
    eq = Eq(lhs, rhs)
    steps.append(SolutionStep(
        step=step_num,
        content=f"${latex(eq)}$",
        action="Original equation"
    ))
    step_num += 1

    # Move everything to LHS
    expr = lhs - rhs

    # Collect terms
    collected = expr.collect(x)

    # Get coefficient of x and constant
    coeff = collected.coeff(x)
    constant = collected.subs(x, 0)

    if coeff != 0:
        # Step 2: Isolate variable terms
        if constant != 0:
            steps.append(SolutionStep(
                step=step_num,
                content=f"${latex(coeff)}{var_name} = {latex(-constant)}$",
                action=f"Move constant to right side"
            ))
            step_num += 1

        # Step 3: Divide by coefficient
        solution = -constant / coeff
        if coeff != 1:
            steps.append(SolutionStep(
                step=step_num,
                content=f"${var_name} = \\frac{{{latex(-constant)}}}{{{latex(coeff)}}}$",
                action=f"Divide both sides by {latex(coeff)}"
            ))
            step_num += 1

        # Step 4: Simplify
        solution = simplify(solution)
        steps.append(SolutionStep(
            step=step_num,
            content=f"${var_name} = {latex(solution)}$",
            action="Simplify"
        ))

        return Solution(
            steps=steps,
            final_answer=f"${var_name} = {latex(solution)}$",
            problem_type="linear"
        )
    else:
        return Solution(
            steps=steps,
            final_answer="No solution (coefficient of x is 0)",
            problem_type="linear"
        )


def _solve_quadratic(lhs: Any, rhs: Any, x: Symbol) -> Solution:
    """Solve a quadratic equation with step-by-step explanation."""
    steps = []
    step_num = 1

    # Step 1: Original equation
    eq = Eq(lhs, rhs)
    steps.append(SolutionStep(
        step=step_num,
        content=f"${latex(eq)}$",
        action="Original equation"
    ))
    step_num += 1

    # Move to standard form ax² + bx + c = 0
    expr = expand(lhs - rhs)

    # Step 2: Standard form
    steps.append(SolutionStep(
        step=step_num,
        content=f"${latex(expr)} = 0$",
        action="Write in standard form"
    ))
    step_num += 1

    # Get coefficients
    poly = expr.as_poly(x)
    if poly is None:
        return _solve_generic(lhs, rhs, x)

    coeffs = poly.all_coeffs()
    a = coeffs[0] if len(coeffs) > 0 else 0
    b = coeffs[1] if len(coeffs) > 1 else 0
    c = coeffs[2] if len(coeffs) > 2 else 0

    # Try factoring first
    factored = factor(expr)
    if factored != expr:
        steps.append(SolutionStep(
            step=step_num,
            content=f"${latex(factored)} = 0$",
            action="Factor the expression"
        ))
        step_num += 1

    # Step 3: Apply quadratic formula or solve
    discriminant = b**2 - 4*a*c

    steps.append(SolutionStep(
        step=step_num,
        content=f"Discriminant: $b^2 - 4ac = ({latex(b)})^2 - 4({latex(a)})({latex(c)}) = {latex(discriminant)}$",
        action="Calculate discriminant"
    ))
    step_num += 1

    # Solve
    solutions = solve(expr, x)

    if len(solutions) == 2:
        steps.append(SolutionStep(
            step=step_num,
            content=f"$x = \\frac{{-b \\pm \\sqrt{{b^2-4ac}}}}{{2a}} = \\frac{{-({latex(b)}) \\pm \\sqrt{{{latex(discriminant)}}}}}{{2({latex(a)})}}$",
            action="Apply quadratic formula"
        ))
        step_num += 1

        sol_strs = [latex(s) for s in solutions]
        steps.append(SolutionStep(
            step=step_num,
            content=f"$x = {sol_strs[0]}$ or $x = {sol_strs[1]}$",
            action="Simplify"
        ))

        return Solution(
            steps=steps,
            final_answer=f"$x = {sol_strs[0]}$ or $x = {sol_strs[1]}$",
            problem_type="quadratic"
        )
    elif len(solutions) == 1:
        steps.append(SolutionStep(
            step=step_num,
            content=f"$x = {latex(solutions[0])}$",
            action="One repeated solution"
        ))
        return Solution(
            steps=steps,
            final_answer=f"$x = {latex(solutions[0])}$",
            problem_type="quadratic"
        )
    else:
        return Solution(
            steps=steps,
            final_answer="No real solutions",
            problem_type="quadratic"
        )


def _solve_simplify(expr: Any, x: Symbol) -> Solution:
    """Simplify an expression with steps."""
    steps = []

    steps.append(SolutionStep(
        step=1,
        content=f"${latex(expr)}$",
        action="Original expression"
    ))

    simplified = simplify(expr)
    steps.append(SolutionStep(
        step=2,
        content=f"${latex(simplified)}$",
        action="Simplify"
    ))

    return Solution(
        steps=steps,
        final_answer=f"${latex(simplified)}$",
        problem_type="simplify"
    )


def _solve_factor(expr: Any, x: Symbol) -> Solution:
    """Factor an expression with steps."""
    steps = []

    steps.append(SolutionStep(
        step=1,
        content=f"${latex(expr)}$",
        action="Original expression"
    ))

    factored = factor(expr)
    steps.append(SolutionStep(
        step=2,
        content=f"${latex(factored)}$",
        action="Factor"
    ))

    return Solution(
        steps=steps,
        final_answer=f"${latex(factored)}$",
        problem_type="factor"
    )


def _solve_generic(lhs: Any, rhs: Any, x: Symbol) -> Solution:
    """Generic solver for equations."""
    steps = []
    step_num = 1

    # Step 1: Original equation
    eq = Eq(lhs, rhs)
    steps.append(SolutionStep(
        step=step_num,
        content=f"${latex(eq)}$",
        action="Original equation"
    ))
    step_num += 1

    # Solve
    solutions = solve(eq, x)

    if solutions:
        if len(solutions) == 1:
            steps.append(SolutionStep(
                step=step_num,
                content=f"$x = {latex(solutions[0])}$",
                action="Solve for x"
            ))
            final = f"$x = {latex(solutions[0])}$"
        else:
            sol_strs = [f"x = {latex(s)}" for s in solutions]
            steps.append(SolutionStep(
                step=step_num,
                content=f"${', '.join(sol_strs)}$",
                action="Find all solutions"
            ))
            final = f"${', '.join(sol_strs)}$"
    else:
        final = "No solution found"

    return Solution(
        steps=steps,
        final_answer=final,
        problem_type="generic"
    )


def solution_to_dict(solution: Solution) -> Dict[str, Any]:
    """Convert Solution object to dictionary for JSON serialization."""
    return {
        "steps": [
            {"step": s.step, "content": s.content, "action": s.action}
            for s in solution.steps
        ],
        "final_answer": solution.final_answer,
        "problem_type": solution.problem_type,
    }


def extract_equation_from_question(question_content: str) -> Optional[str]:
    """Extract mathematical equation from question content.

    Looks for LaTeX math expressions like $2x + 5 = 11$ or \\(ax + b = c\\)
    Also handles fractions and more complex LaTeX.
    """
    # Match LaTeX delimited expressions (allowing spaces)
    patterns = [
        r'\$\s*([^$]+?)\s*\$',           # $...$ (with optional spaces)
        r'\\\(\s*([^)]+?)\s*\\\)',        # \(...\)
        r'\\\[\s*([^\]]+?)\s*\\\]',       # \[...\]
    ]

    for pattern in patterns:
        matches = re.findall(pattern, question_content)
        for match in matches:
            # Check if it contains an equals sign (it's an equation)
            if '=' in match:
                # Clean up the equation
                cleaned = match.strip()
                # Convert LaTeX fractions to Python-parseable format
                cleaned = _convert_latex_fractions(cleaned)
                return cleaned

    return None


def _convert_latex_fractions(latex_str: str) -> str:
    """Convert LaTeX fractions to Python-parseable format.

    e.g., \\dfrac{15}{7} = \\dfrac{x}{4} -> 15/7 = x/4
    """
    result = latex_str

    # Convert \dfrac{a}{b} or \frac{a}{b} to (a)/(b)
    frac_pattern = r'\\d?frac\{([^{}]+)\}\{([^{}]+)\}'

    while re.search(frac_pattern, result):
        result = re.sub(frac_pattern, r'(\1)/(\2)', result)

    # Remove remaining LaTeX commands
    result = re.sub(r'\\[a-zA-Z]+', '', result)

    # Clean up extra spaces
    result = ' '.join(result.split())

    return result


if __name__ == "__main__":
    # Test the solution generator
    print("Testing solution generator...\n")

    # Test linear equation
    print("Linear: 2x + 5 = 11")
    sol = generate_solution("2x + 5 = 11")
    if sol:
        for step in sol.steps:
            print(f"  Step {step.step}: {step.action}")
            print(f"    {step.content}")
        print(f"  Answer: {sol.final_answer}\n")

    # Test quadratic equation
    print("Quadratic: x^2 - 5x + 6 = 0")
    sol = generate_solution("x**2 - 5*x + 6 = 0")
    if sol:
        for step in sol.steps:
            print(f"  Step {step.step}: {step.action}")
            print(f"    {step.content}")
        print(f"  Answer: {sol.final_answer}\n")

    # Test another linear
    print("Linear: 3x - 7 = 14")
    sol = generate_solution("3x - 7 = 14")
    if sol:
        for step in sol.steps:
            print(f"  Step {step.step}: {step.action}")
            print(f"    {step.content}")
        print(f"  Answer: {sol.final_answer}\n")

    print("All tests completed!")

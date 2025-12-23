"""LaTeX/KaTeX syntax validator for Perseus questions."""

import logging
import re
from dataclasses import dataclass, field
from typing import Any

from questionbank.utils.khan_colors import VALID_COLOR_COMMANDS

logger = logging.getLogger(__name__)


@dataclass
class LaTeXValidationResult:
    """Result of LaTeX validation."""

    is_valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class LaTeXValidator:
    """Validates LaTeX syntax in Perseus content for KaTeX compatibility."""

    # KaTeX supported commands (subset of most common)
    KATEX_SUPPORTED_COMMANDS = {
        # Text formatting
        "text", "textbf", "textit", "textrm", "textsf", "texttt",
        "textcolor", "color", "colorbox",
        # Math formatting
        "mathbf", "mathit", "mathrm", "mathsf", "mathtt", "mathbb", "mathcal",
        "boldsymbol", "bm",
        # Fractions and roots
        "frac", "dfrac", "tfrac", "cfrac", "sqrt", "root",
        # Greek letters
        "alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta",
        "iota", "kappa", "lambda", "mu", "nu", "xi", "pi", "rho", "sigma",
        "tau", "upsilon", "phi", "chi", "psi", "omega",
        "Alpha", "Beta", "Gamma", "Delta", "Epsilon", "Zeta", "Eta", "Theta",
        "Iota", "Kappa", "Lambda", "Mu", "Nu", "Xi", "Pi", "Rho", "Sigma",
        "Tau", "Upsilon", "Phi", "Chi", "Psi", "Omega",
        "varepsilon", "vartheta", "varpi", "varrho", "varsigma", "varphi",
        # Operators
        "times", "div", "cdot", "pm", "mp", "ast", "star", "circ", "bullet",
        "oplus", "otimes", "odot", "oslash", "ominus",
        # Relations
        "eq", "ne", "neq", "lt", "gt", "le", "ge", "leq", "geq",
        "approx", "sim", "simeq", "equiv", "cong", "propto",
        "subset", "supset", "subseteq", "supseteq", "in", "notin", "ni",
        # Arrows
        "leftarrow", "rightarrow", "leftrightarrow",
        "Leftarrow", "Rightarrow", "Leftrightarrow",
        "longleftarrow", "longrightarrow", "longleftrightarrow",
        "to", "gets", "mapsto",
        # Big operators
        "sum", "prod", "int", "oint", "bigcup", "bigcap", "bigoplus", "bigotimes",
        # Delimiters
        "left", "right", "big", "Big", "bigg", "Bigg",
        "langle", "rangle", "lfloor", "rfloor", "lceil", "rceil",
        "lvert", "rvert", "lVert", "rVert",
        # Accents
        "hat", "check", "tilde", "acute", "grave", "dot", "ddot", "breve",
        "bar", "vec", "overline", "underline", "widehat", "widetilde",
        "overrightarrow", "overleftarrow", "overbrace", "underbrace",
        # Functions
        "sin", "cos", "tan", "cot", "sec", "csc",
        "arcsin", "arccos", "arctan", "arccot", "arcsec", "arccsc",
        "sinh", "cosh", "tanh", "coth", "sech", "csch",
        "log", "ln", "lg", "exp", "lim", "limsup", "liminf",
        "max", "min", "sup", "inf", "arg", "det", "dim", "gcd", "hom", "ker",
        "Pr", "deg", "mod", "bmod", "pmod",
        # Spacing
        "quad", "qquad", "enspace", "thinspace", "negthinspace",
        "hspace", "vspace", "phantom", "hphantom", "vphantom",
        # Layout
        "fbox", "boxed", "underset", "overset", "stackrel",
        # Environments
        "begin", "end",
        # Matrices
        "matrix", "pmatrix", "bmatrix", "Bmatrix", "vmatrix", "Vmatrix",
        "array", "cases",
        # Align
        "align", "align*", "gather", "gather*", "equation", "equation*",
        # Misc
        "ldots", "cdots", "vdots", "ddots", "dots",
        "infty", "nabla", "partial", "forall", "exists", "nexists",
        "emptyset", "varnothing", "setminus", "cap", "cup",
        "land", "lor", "lnot", "neg",
        "triangle", "angle", "measuredangle", "sphericalangle",
        "degree", "prime", "backprime",
        "cancel", "bcancel", "xcancel", "sout",
    }

    # Add Khan Academy color commands
    KATEX_SUPPORTED_COMMANDS.update(VALID_COLOR_COMMANDS)

    def __init__(self) -> None:
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def validate(self, data: dict[str, Any]) -> LaTeXValidationResult:
        """Validate LaTeX in all content fields of a Perseus question."""
        self.errors = []
        self.warnings = []

        # Validate question content
        if "question" in data:
            question = data["question"]
            if "content" in question:
                self._validate_content(question["content"], "question.content")

            # Validate widget content
            if "widgets" in question:
                for widget_id, widget in question["widgets"].items():
                    self._validate_widget_latex(widget, f"question.widgets['{widget_id}']")

        # Validate hints
        if "hints" in data:
            for i, hint in enumerate(data["hints"]):
                if "content" in hint:
                    self._validate_content(hint["content"], f"hints[{i}].content")

        return LaTeXValidationResult(
            is_valid=len(self.errors) == 0,
            errors=self.errors.copy(),
            warnings=self.warnings.copy(),
        )

    def _validate_content(self, content: str, context: str) -> None:
        """Validate LaTeX in a content string."""
        if not content:
            return

        # Check for balanced math delimiters
        self._check_balanced_delimiters(content, context)

        # Extract and validate math regions
        self._validate_math_regions(content, context)

    def _check_balanced_delimiters(self, content: str, context: str) -> None:
        """Check for balanced math delimiters."""
        # Check inline math ($...$)
        # Count unescaped $ signs
        inline_count = len(re.findall(r"(?<!\\)\$(?!\$)", content))
        if inline_count % 2 != 0:
            self.errors.append(f"{context}: Unbalanced inline math delimiters ($...$)")

        # Check display math ($$...$$)
        display_count = len(re.findall(r"\$\$", content))
        if display_count % 2 != 0:
            self.errors.append(f"{context}: Unbalanced display math delimiters ($$...$$)")

        # Check for balanced braces in math regions
        math_regions = self._extract_math_regions(content)
        for math_content in math_regions:
            brace_count = math_content.count("{") - math_content.count("}")
            if brace_count != 0:
                self.errors.append(f"{context}: Unbalanced braces in math expression")
                break

    def _extract_math_regions(self, content: str) -> list[str]:
        """Extract all math regions from content."""
        regions = []

        # Extract display math ($$...$$)
        display_pattern = r"\$\$(.*?)\$\$"
        regions.extend(re.findall(display_pattern, content, re.DOTALL))

        # Extract inline math ($...$) - but not $$
        inline_pattern = r"(?<!\$)\$(?!\$)(.*?)(?<!\$)\$(?!\$)"
        regions.extend(re.findall(inline_pattern, content, re.DOTALL))

        # Extract \begin{...}...\end{...} environments
        env_pattern = r"\\begin\{(\w+)\}(.*?)\\end\{\1\}"
        for match in re.finditer(env_pattern, content, re.DOTALL):
            regions.append(match.group(2))

        return regions

    def _validate_math_regions(self, content: str, context: str) -> None:
        """Validate LaTeX commands in math regions."""
        math_regions = self._extract_math_regions(content)

        for region in math_regions:
            # Find all LaTeX commands
            commands = re.findall(r"\\([a-zA-Z]+)", region)

            for cmd in commands:
                if cmd not in self.KATEX_SUPPORTED_COMMANDS:
                    # Check if it might be a Khan Academy color without braces
                    if not any(cmd.startswith(color) for color in VALID_COLOR_COMMANDS):
                        self.warnings.append(
                            f"{context}: Potentially unsupported LaTeX command: \\{cmd}"
                        )

    def _validate_widget_latex(self, widget: dict[str, Any], context: str) -> None:
        """Validate LaTeX in widget options."""
        widget_type = widget.get("type", "")
        options = widget.get("options", {})

        if widget_type == "radio":
            choices = options.get("choices", [])
            for i, choice in enumerate(choices):
                content = choice.get("content", "")
                self._validate_content(content, f"{context}.options.choices[{i}].content")

        elif widget_type == "expression":
            answer_forms = options.get("answerForms", [])
            for i, form in enumerate(answer_forms):
                value = form.get("value", "")
                # Expression values are LaTeX
                self._validate_expression_latex(value, f"{context}.options.answerForms[{i}].value")

        elif widget_type in {"numeric-input", "input-number"}:
            # Check label text
            label = options.get("labelText", "")
            if label:
                self._validate_content(label, f"{context}.options.labelText")

    def _validate_expression_latex(self, latex: str, context: str) -> None:
        """Validate LaTeX expression (used in expression widget answers)."""
        if not latex:
            return

        # Check balanced braces
        brace_count = latex.count("{") - latex.count("}")
        if brace_count != 0:
            self.errors.append(f"{context}: Unbalanced braces in expression")

        # Find commands
        commands = re.findall(r"\\([a-zA-Z]+)", latex)
        for cmd in commands:
            if cmd not in self.KATEX_SUPPORTED_COMMANDS:
                self.warnings.append(f"{context}: Potentially unsupported command: \\{cmd}")


# Singleton instance
latex_validator = LaTeXValidator()

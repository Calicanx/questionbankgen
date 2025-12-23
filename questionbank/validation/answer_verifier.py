"""Answer verification for generated questions."""

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class AnswerVerificationResult:
    """Result of answer verification."""

    is_valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class AnswerVerifier:
    """Verifies that answers in generated questions are correct."""

    def verify(self, data: dict[str, Any]) -> AnswerVerificationResult:
        """Verify answers in a Perseus question."""
        errors: list[str] = []
        warnings: list[str] = []

        question = data.get("question", {})
        widgets = question.get("widgets", {})

        for widget_id, widget in widgets.items():
            widget_type = widget.get("type", "")
            options = widget.get("options", {})
            graded = widget.get("graded", True)

            if not graded:
                continue

            context = f"widgets['{widget_id}']"

            if widget_type == "numeric-input":
                result = self._verify_numeric_input(options, context)
            elif widget_type == "input-number":
                result = self._verify_input_number(options, context)
            elif widget_type == "radio":
                result = self._verify_radio(options, context)
            elif widget_type == "expression":
                result = self._verify_expression(options, context)
            elif widget_type == "dropdown":
                result = self._verify_dropdown(options, context)
            elif widget_type in {"sorter", "orderer"}:
                result = self._verify_orderer(options, context)
            else:
                # Skip verification for complex widgets
                result = AnswerVerificationResult(is_valid=True)

            errors.extend(result.errors)
            warnings.extend(result.warnings)

        return AnswerVerificationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
        )

    def _verify_numeric_input(
        self, options: dict[str, Any], context: str
    ) -> AnswerVerificationResult:
        """Verify numeric-input widget answers."""
        errors: list[str] = []
        warnings: list[str] = []

        answers = options.get("answers", [])

        if not answers:
            errors.append(f"{context}: No answers defined")
            return AnswerVerificationResult(is_valid=False, errors=errors)

        has_correct = False
        for i, answer in enumerate(answers):
            status = answer.get("status", "")
            value = answer.get("value")

            if status == "correct":
                has_correct = True

                # Check if value is a valid number
                if value is None:
                    errors.append(f"{context}.answers[{i}]: Missing value for correct answer")
                elif not isinstance(value, (int, float)):
                    try:
                        float(value)
                    except (ValueError, TypeError):
                        errors.append(f"{context}.answers[{i}]: Invalid numeric value: {value}")

        if not has_correct:
            errors.append(f"{context}: No correct answer defined")

        return AnswerVerificationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
        )

    def _verify_input_number(
        self, options: dict[str, Any], context: str
    ) -> AnswerVerificationResult:
        """Verify input-number widget answers."""
        errors: list[str] = []

        value = options.get("value")

        if value is None:
            errors.append(f"{context}: Missing value for input-number")
        elif not isinstance(value, (int, float)):
            try:
                float(value)
            except (ValueError, TypeError):
                errors.append(f"{context}: Invalid numeric value: {value}")

        return AnswerVerificationResult(is_valid=len(errors) == 0, errors=errors)

    def _verify_radio(
        self, options: dict[str, Any], context: str
    ) -> AnswerVerificationResult:
        """Verify radio widget answers."""
        errors: list[str] = []
        warnings: list[str] = []

        choices = options.get("choices", [])

        if not choices:
            errors.append(f"{context}: No choices defined")
            return AnswerVerificationResult(is_valid=False, errors=errors)

        correct_count = 0
        for i, choice in enumerate(choices):
            if choice.get("correct", False):
                correct_count += 1

        if correct_count == 0:
            errors.append(f"{context}: No correct choice marked")
        elif correct_count > 1:
            warnings.append(f"{context}: Multiple correct choices ({correct_count})")

        return AnswerVerificationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
        )

    def _verify_expression(
        self, options: dict[str, Any], context: str
    ) -> AnswerVerificationResult:
        """Verify expression widget answers."""
        errors: list[str] = []
        warnings: list[str] = []

        answer_forms = options.get("answerForms", [])

        if not answer_forms:
            errors.append(f"{context}: No answer forms defined")
            return AnswerVerificationResult(is_valid=False, errors=errors)

        has_correct = False
        for i, form in enumerate(answer_forms):
            considered = form.get("considered", "")
            value = form.get("value", "")

            if considered == "correct":
                has_correct = True

                if not value:
                    errors.append(f"{context}.answerForms[{i}]: Empty value for correct answer")

        if not has_correct:
            warnings.append(f"{context}: No 'correct' answer form (might use 'ungraded')")

        return AnswerVerificationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
        )

    def _verify_dropdown(
        self, options: dict[str, Any], context: str
    ) -> AnswerVerificationResult:
        """Verify dropdown widget answers."""
        errors: list[str] = []

        choices = options.get("choices", [])
        correct_idx = options.get("correct")

        if not choices:
            errors.append(f"{context}: No choices defined")
        elif correct_idx is not None:
            if not isinstance(correct_idx, int) or correct_idx < 0 or correct_idx >= len(choices):
                errors.append(f"{context}: Invalid correct index: {correct_idx}")

        return AnswerVerificationResult(is_valid=len(errors) == 0, errors=errors)

    def _verify_orderer(
        self, options: dict[str, Any], context: str
    ) -> AnswerVerificationResult:
        """Verify sorter/orderer widget answers."""
        errors: list[str] = []

        # Sorter uses 'correct', orderer uses 'options' or 'correctOptions'
        correct_options = options.get("correct", options.get("correctOptions", options.get("options", [])))

        if not correct_options:
            errors.append(f"{context}: No items to order")

        return AnswerVerificationResult(is_valid=len(errors) == 0, errors=errors)


# Singleton instance
answer_verifier = AnswerVerifier()

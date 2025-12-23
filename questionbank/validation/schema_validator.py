"""Perseus v2.0 JSON Schema Validator."""

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of a validation check."""

    is_valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class PerseusSchemaValidator:
    """Validates Perseus v2.0 JSON schema compliance."""

    # Required top-level fields
    REQUIRED_FIELDS = ["answerArea", "hints", "itemDataVersion", "question"]

    # Expected answerArea keys
    ANSWER_AREA_KEYS = ["calculator", "chi2Table", "periodicTable", "tTable", "zTable"]

    # Supported widget types
    SUPPORTED_WIDGET_TYPES = {
        "radio",
        "numeric-input",
        "input-number",
        "expression",
        "dropdown",
        "sorter",
        "orderer",
        "matcher",
        "categorizer",
        "interactive-graph",
        "grapher",
        "plotter",
        "label-image",
        "image",
        "passage",
        "video",
        "definition",
        "explanation",
        "table",
        "matrix",
        "free-response",
    }

    def __init__(self) -> None:
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def validate(self, data: dict[str, Any]) -> ValidationResult:
        """Validate a Perseus question JSON."""
        self.errors = []
        self.warnings = []

        # Check required top-level fields
        for field_name in self.REQUIRED_FIELDS:
            if field_name not in data:
                self.errors.append(f"Missing required field: {field_name}")

        # Validate individual sections
        if "answerArea" in data:
            self._validate_answer_area(data["answerArea"])

        if "itemDataVersion" in data:
            self._validate_version(data["itemDataVersion"])

        if "question" in data:
            self._validate_question(data["question"])

        if "hints" in data:
            self._validate_hints(data["hints"])

        return ValidationResult(
            is_valid=len(self.errors) == 0,
            errors=self.errors.copy(),
            warnings=self.warnings.copy(),
        )

    def _validate_answer_area(self, answer_area: dict[str, Any]) -> None:
        """Validate answerArea structure."""
        if not isinstance(answer_area, dict):
            self.errors.append("answerArea must be an object")
            return

        for key in self.ANSWER_AREA_KEYS:
            if key not in answer_area:
                self.warnings.append(f"answerArea missing optional key: {key}")
            elif not isinstance(answer_area[key], bool):
                self.errors.append(f"answerArea.{key} must be boolean")

    def _validate_version(self, version: dict[str, Any]) -> None:
        """Validate itemDataVersion structure."""
        if not isinstance(version, dict):
            self.errors.append("itemDataVersion must be an object")
            return

        if "major" not in version or "minor" not in version:
            self.errors.append("itemDataVersion must have 'major' and 'minor' fields")
        else:
            if version["major"] != 2:
                self.warnings.append(f"itemDataVersion.major is {version['major']}, expected 2")
            if version["minor"] != 0:
                self.warnings.append(f"itemDataVersion.minor is {version['minor']}, expected 0")

    def _validate_question(self, question: dict[str, Any]) -> None:
        """Validate question structure."""
        if not isinstance(question, dict):
            self.errors.append("question must be an object")
            return

        if "content" not in question:
            self.errors.append("question missing 'content' field")

        if "widgets" not in question:
            self.errors.append("question missing 'widgets' field")
        else:
            self._validate_widgets(question["widgets"], "question")

        if "images" not in question:
            self.warnings.append("question missing 'images' field (should be {} if no images)")

    def _validate_hints(self, hints: list[dict[str, Any]]) -> None:
        """Validate hints array."""
        if not isinstance(hints, list):
            self.errors.append("hints must be an array")
            return

        for i, hint in enumerate(hints):
            if not isinstance(hint, dict):
                self.errors.append(f"hints[{i}] must be an object")
                continue

            if "content" not in hint:
                self.errors.append(f"hints[{i}] missing 'content' field")

            if "replace" not in hint:
                self.warnings.append(f"hints[{i}] missing 'replace' field")

            if "widgets" in hint:
                self._validate_widgets(hint["widgets"], f"hints[{i}]")

    def _validate_widgets(self, widgets: dict[str, Any], context: str) -> None:
        """Validate widgets object."""
        if not isinstance(widgets, dict):
            self.errors.append(f"{context}.widgets must be an object")
            return

        for widget_id, widget in widgets.items():
            if not isinstance(widget, dict):
                self.errors.append(f"{context}.widgets['{widget_id}'] must be an object")
                continue

            widget_context = f"{context}.widgets['{widget_id}']"

            if "type" not in widget:
                self.errors.append(f"{widget_context} missing 'type' field")
                continue

            widget_type = widget["type"]

            # Check if widget type is supported
            if widget_type not in self.SUPPORTED_WIDGET_TYPES:
                self.warnings.append(f"{widget_context} has unknown type: {widget_type}")

            # Validate version
            if "version" not in widget:
                self.warnings.append(f"{widget_context} missing 'version' field")

            # Validate graded
            if "graded" not in widget:
                self.warnings.append(f"{widget_context} missing 'graded' field")

            # Validate options
            if "options" not in widget:
                self.errors.append(f"{widget_context} missing 'options' field")
                continue

            # Widget-specific validation
            self._validate_widget_options(widget_type, widget["options"], widget_context)

    def _validate_widget_options(
        self, widget_type: str, options: dict[str, Any], context: str
    ) -> None:
        """Validate widget-specific options."""
        if widget_type == "numeric-input":
            self._validate_numeric_input(options, context)
        elif widget_type == "input-number":
            self._validate_input_number(options, context)
        elif widget_type == "radio":
            self._validate_radio(options, context)
        elif widget_type == "expression":
            self._validate_expression(options, context)
        elif widget_type == "dropdown":
            self._validate_dropdown(options, context)
        elif widget_type == "image":
            self._validate_image(options, context)
        elif widget_type in {"sorter", "orderer"}:
            self._validate_orderer(options, context)
        elif widget_type == "matcher":
            self._validate_matcher(options, context)
        elif widget_type == "categorizer":
            self._validate_categorizer(options, context)

    def _validate_numeric_input(self, options: dict[str, Any], context: str) -> None:
        """Validate numeric-input widget options."""
        if "answers" not in options:
            self.errors.append(f"{context}.options missing 'answers' field")
            return

        answers = options["answers"]
        if not isinstance(answers, list) or len(answers) == 0:
            self.errors.append(f"{context}.options.answers must be a non-empty array")
            return

        for i, answer in enumerate(answers):
            if "value" not in answer:
                self.errors.append(f"{context}.options.answers[{i}] missing 'value' field")
            if "status" not in answer:
                self.errors.append(f"{context}.options.answers[{i}] missing 'status' field")

    def _validate_input_number(self, options: dict[str, Any], context: str) -> None:
        """Validate input-number widget options."""
        if "value" not in options:
            self.errors.append(f"{context}.options missing 'value' field")

    def _validate_radio(self, options: dict[str, Any], context: str) -> None:
        """Validate radio widget options."""
        if "choices" not in options:
            self.errors.append(f"{context}.options missing 'choices' field")
            return

        choices = options["choices"]
        if not isinstance(choices, list) or len(choices) == 0:
            self.errors.append(f"{context}.options.choices must be a non-empty array")
            return

        # Validate choice count (typical range: 2-10)
        if len(choices) < 2:
            self.errors.append(f"{context}.options.choices must have at least 2 choices")
        elif len(choices) > 10:
            self.errors.append(f"{context}.options.choices has {len(choices)} choices (max 10 allowed)")

        has_correct = False
        correct_count = 0
        for i, choice in enumerate(choices):
            if "content" not in choice:
                self.errors.append(f"{context}.options.choices[{i}] missing 'content' field")
            if "correct" not in choice:
                self.errors.append(f"{context}.options.choices[{i}] missing 'correct' field")
            elif choice["correct"]:
                has_correct = True
                correct_count += 1

        if not has_correct:
            self.errors.append(f"{context}.options.choices has no correct answer")
        elif correct_count > 1:
            self.warnings.append(f"{context}.options.choices has {correct_count} correct answers (expected 1)")

    def _validate_expression(self, options: dict[str, Any], context: str) -> None:
        """Validate expression widget options."""
        if "answerForms" not in options:
            self.errors.append(f"{context}.options missing 'answerForms' field")
            return

        answer_forms = options["answerForms"]
        if not isinstance(answer_forms, list) or len(answer_forms) == 0:
            self.errors.append(f"{context}.options.answerForms must be a non-empty array")
            return

        has_correct = False
        for i, form in enumerate(answer_forms):
            if "value" not in form:
                self.errors.append(f"{context}.options.answerForms[{i}] missing 'value' field")
            if "considered" not in form:
                self.warnings.append(f"{context}.options.answerForms[{i}] missing 'considered' field")
            elif form["considered"] == "correct":
                has_correct = True

        if not has_correct:
            self.warnings.append(f"{context}.options.answerForms has no 'correct' answer")

    def _validate_dropdown(self, options: dict[str, Any], context: str) -> None:
        """Validate dropdown widget options."""
        if "choices" not in options:
            self.errors.append(f"{context}.options missing 'choices' field")
            return

        choices = options["choices"]
        if not isinstance(choices, list) or len(choices) == 0:
            self.errors.append(f"{context}.options.choices must be a non-empty array")

    def _validate_image(self, options: dict[str, Any], context: str) -> None:
        """Validate image widget options."""
        if "backgroundImage" not in options:
            self.errors.append(f"{context}.options missing 'backgroundImage' field")
            return

        bg = options["backgroundImage"]
        if not isinstance(bg, dict):
            self.errors.append(f"{context}.options.backgroundImage must be an object")
            return

        if "url" not in bg:
            self.errors.append(f"{context}.options.backgroundImage missing 'url' field")
        if "width" not in bg or "height" not in bg:
            self.warnings.append(f"{context}.options.backgroundImage missing width/height")

    def _validate_orderer(self, options: dict[str, Any], context: str) -> None:
        """Validate sorter/orderer widget options."""
        # Sorter uses 'correct', orderer uses 'options' or 'correctOptions'
        if "correct" not in options and "options" not in options and "correctOptions" not in options:
            self.errors.append(f"{context}.options missing items to order")

    def _validate_matcher(self, options: dict[str, Any], context: str) -> None:
        """Validate matcher widget options."""
        if "left" not in options or "right" not in options:
            self.errors.append(f"{context}.options missing 'left' or 'right' arrays")

    def _validate_categorizer(self, options: dict[str, Any], context: str) -> None:
        """Validate categorizer widget options."""
        if "categories" not in options:
            self.errors.append(f"{context}.options missing 'categories' field")
        if "items" not in options:
            self.errors.append(f"{context}.options missing 'items' field")


# Singleton instance
schema_validator = PerseusSchemaValidator()

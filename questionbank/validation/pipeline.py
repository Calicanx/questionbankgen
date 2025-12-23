"""Validation pipeline for Perseus questions."""

import logging
from dataclasses import dataclass, field
from typing import Any

from questionbank.validation.schema_validator import PerseusSchemaValidator
from questionbank.validation.latex_validator import LaTeXValidator
from questionbank.validation.answer_verifier import AnswerVerifier

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Result of the validation pipeline."""

    is_valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    stage_results: dict[str, bool] = field(default_factory=dict)


class ValidationPipeline:
    """Multi-stage validation pipeline for Perseus questions."""

    def __init__(
        self,
        check_schema: bool = True,
        check_latex: bool = True,
        verify_answers: bool = True,
    ) -> None:
        self.check_schema = check_schema
        self.check_latex = check_latex
        self.verify_answers = verify_answers

        self.schema_validator = PerseusSchemaValidator()
        self.latex_validator = LaTeXValidator()
        self.answer_verifier = AnswerVerifier()

    def validate(self, data: dict[str, Any]) -> PipelineResult:
        """Run the full validation pipeline."""
        all_errors: list[str] = []
        all_warnings: list[str] = []
        stage_results: dict[str, bool] = {}

        # Stage 1: Schema validation
        if self.check_schema:
            result = self.schema_validator.validate(data)
            stage_results["schema"] = result.is_valid
            all_errors.extend(result.errors)
            all_warnings.extend(result.warnings)

            if not result.is_valid:
                logger.warning(f"Schema validation failed: {result.errors}")

        # Stage 2: LaTeX validation
        if self.check_latex:
            result = self.latex_validator.validate(data)
            stage_results["latex"] = result.is_valid
            all_errors.extend(result.errors)
            all_warnings.extend(result.warnings)

            if not result.is_valid:
                logger.warning(f"LaTeX validation failed: {result.errors}")

        # Stage 3: Answer verification
        if self.verify_answers:
            result = self.answer_verifier.verify(data)
            stage_results["answers"] = result.is_valid
            all_errors.extend(result.errors)
            all_warnings.extend(result.warnings)

            if not result.is_valid:
                logger.warning(f"Answer verification failed: {result.errors}")

        is_valid = len(all_errors) == 0

        return PipelineResult(
            is_valid=is_valid,
            errors=all_errors,
            warnings=all_warnings,
            stage_results=stage_results,
        )


# Default pipeline instance
validation_pipeline = ValidationPipeline()

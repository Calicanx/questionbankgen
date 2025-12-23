"""Intelligent validation pipeline for generated questions."""

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from .constraint_extractor import ConstraintExtractor, QuestionConstraints
from .coherence_validator import CoherenceValidator, CoherenceResult
from ..validation.schema_validator import PerseusSchemaValidator
from ..validation.answer_verifier import AnswerVerifier
from ..validation.latex_validator import LaTeXValidator

logger = logging.getLogger(__name__)


@dataclass
class PipelineStage:
    """Result from a single pipeline stage."""
    name: str
    passed: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    can_fix: bool = False
    fix_applied: bool = False


@dataclass
class PipelineResult:
    """Complete result from validation pipeline."""
    is_valid: bool
    overall_score: float  # 0.0 to 1.0
    stages: list[PipelineStage] = field(default_factory=list)
    constraints: Optional[QuestionConstraints] = None
    coherence: Optional[CoherenceResult] = None
    fixed_data: Optional[dict] = None

    def add_stage(self, stage: PipelineStage):
        self.stages.append(stage)
        if not stage.passed:
            self.is_valid = False

    def get_all_errors(self) -> list[str]:
        errors = []
        for stage in self.stages:
            errors.extend([f"[{stage.name}] {e}" for e in stage.errors])
        return errors

    def get_all_warnings(self) -> list[str]:
        warnings = []
        for stage in self.stages:
            warnings.extend([f"[{stage.name}] {w}" for w in stage.warnings])
        return warnings


class IntelligentValidationPipeline:
    """
    Multi-stage validation pipeline for generated questions.

    Stages:
    1. Schema Validation - Check Perseus JSON structure
    2. LaTeX Validation - Check mathematical notation
    3. Answer Verification - Check answer correctness
    4. Constraint Extraction - Extract semantic constraints
    5. Coherence Validation - Check text-image-widget alignment

    Each stage can optionally attempt fixes.
    """

    def __init__(self, attempt_fixes: bool = True):
        self.attempt_fixes = attempt_fixes

        # Initialize validators
        self.schema_validator = PerseusSchemaValidator()
        self.latex_validator = LaTeXValidator()
        self.answer_verifier = AnswerVerifier()
        self.constraint_extractor = ConstraintExtractor()
        self.coherence_validator = CoherenceValidator()

    def validate(
        self,
        generated_data: dict[str, Any],
        source_data: Optional[dict[str, Any]] = None
    ) -> PipelineResult:
        """
        Run complete validation pipeline on generated question.

        Args:
            generated_data: The generated Perseus question
            source_data: Optional source question for comparison

        Returns:
            PipelineResult with stage-by-stage results
        """
        result = PipelineResult(is_valid=True, overall_score=1.0)
        working_data = generated_data.copy()

        # Stage 1: Schema Validation
        schema_stage = self._validate_schema(working_data)
        result.add_stage(schema_stage)

        if not schema_stage.passed and self.attempt_fixes:
            fixed = self._attempt_schema_fix(working_data, schema_stage)
            if fixed:
                working_data = fixed
                result.fixed_data = fixed

        # Stage 2: LaTeX Validation
        latex_stage = self._validate_latex(working_data)
        result.add_stage(latex_stage)

        if not latex_stage.passed and self.attempt_fixes:
            fixed = self._attempt_latex_fix(working_data, latex_stage)
            if fixed:
                working_data = fixed
                result.fixed_data = fixed

        # Stage 3: Answer Verification
        answer_stage = self._verify_answers(working_data)
        result.add_stage(answer_stage)

        # Stage 4: Constraint Extraction
        try:
            constraints = self.constraint_extractor.extract(working_data)
            result.constraints = constraints
            constraint_stage = PipelineStage(
                name="Constraint Extraction",
                passed=True
            )
        except Exception as e:
            constraint_stage = PipelineStage(
                name="Constraint Extraction",
                passed=False,
                errors=[str(e)]
            )
        result.add_stage(constraint_stage)

        # Stage 5: Coherence Validation (if source provided)
        if source_data:
            coherence_stage = self._validate_coherence(working_data, source_data)
            result.add_stage(coherence_stage)
            result.coherence = self.coherence_validator.validate(source_data, working_data)

            if not coherence_stage.passed and self.attempt_fixes:
                fixed = self._attempt_coherence_fix(working_data, source_data, result.coherence)
                if fixed:
                    working_data = fixed
                    result.fixed_data = fixed
        else:
            # Quick coherence check without source
            quick_check = self.coherence_validator.quick_check(working_data)
            coherence_stage = PipelineStage(
                name="Coherence Check",
                passed=quick_check,
                warnings=[] if quick_check else ["Question may have coherence issues"]
            )
            result.add_stage(coherence_stage)

        # Calculate overall score
        passed_stages = sum(1 for s in result.stages if s.passed)
        total_stages = len(result.stages)
        error_count = sum(len(s.errors) for s in result.stages)
        warning_count = sum(len(s.warnings) for s in result.stages)

        result.overall_score = max(0.0,
            (passed_stages / total_stages) - (error_count * 0.1) - (warning_count * 0.02)
        )

        return result

    def _validate_schema(self, data: dict[str, Any]) -> PipelineStage:
        """Run schema validation."""
        result = self.schema_validator.validate(data)

        return PipelineStage(
            name="Schema Validation",
            passed=result.is_valid,
            errors=result.errors,
            warnings=result.warnings,
            can_fix=True
        )

    def _validate_latex(self, data: dict[str, Any]) -> PipelineStage:
        """Run LaTeX validation."""
        content = data.get('question', {}).get('content', '')
        result = self.latex_validator.validate(content)

        # Also check hints
        hints = data.get('hints', [])
        for i, hint in enumerate(hints):
            hint_result = self.latex_validator.validate(hint.get('content', ''))
            if not hint_result.is_valid:
                result.errors.extend([f"hint[{i}]: {e}" for e in hint_result.errors])
                result.is_valid = False

        return PipelineStage(
            name="LaTeX Validation",
            passed=result.is_valid,
            errors=result.errors,
            warnings=result.warnings,
            can_fix=True
        )

    def _verify_answers(self, data: dict[str, Any]) -> PipelineStage:
        """Run answer verification."""
        result = self.answer_verifier.verify(data)

        return PipelineStage(
            name="Answer Verification",
            passed=result.is_valid,
            errors=result.errors,
            warnings=result.warnings,
            can_fix=False
        )

    def _validate_coherence(
        self,
        generated: dict[str, Any],
        source: dict[str, Any]
    ) -> PipelineStage:
        """Run coherence validation."""
        result = self.coherence_validator.validate(source, generated)

        errors = [i.message for i in result.issues if i.severity == 'error']
        warnings = [i.message for i in result.issues if i.severity == 'warning']

        return PipelineStage(
            name="Coherence Validation",
            passed=result.is_coherent,
            errors=errors,
            warnings=warnings,
            can_fix=True
        )

    def _attempt_schema_fix(
        self,
        data: dict[str, Any],
        stage: PipelineStage
    ) -> Optional[dict[str, Any]]:
        """Attempt to fix schema issues."""
        fixed = data.copy()
        fixed_any = False

        # Add missing required fields with defaults
        if 'answerArea' not in fixed:
            fixed['answerArea'] = {
                'calculator': False,
                'periodicTable': False,
                'chi2Table': False,
                'tTable': False,
                'zTable': False
            }
            fixed_any = True

        if 'hints' not in fixed:
            fixed['hints'] = []
            fixed_any = True

        if 'itemDataVersion' not in fixed:
            fixed['itemDataVersion'] = {'major': 2, 'minor': 0}
            fixed_any = True

        if 'question' not in fixed:
            fixed['question'] = {'content': '', 'widgets': {}, 'images': {}}
            fixed_any = True
        else:
            if 'widgets' not in fixed['question']:
                fixed['question']['widgets'] = {}
                fixed_any = True
            if 'images' not in fixed['question']:
                fixed['question']['images'] = {}
                fixed_any = True

        if fixed_any:
            stage.fix_applied = True
            return fixed

        return None

    def _attempt_latex_fix(
        self,
        data: dict[str, Any],
        stage: PipelineStage
    ) -> Optional[dict[str, Any]]:
        """Attempt to fix LaTeX issues."""
        fixed = data.copy()
        content = fixed.get('question', {}).get('content', '')

        # Common fixes
        # Balance dollar signs
        dollar_count = content.count('$')
        if dollar_count % 2 != 0:
            # Try to fix unbalanced dollars
            # Simple heuristic: add $ at end if odd
            content = content + '$'
            fixed['question']['content'] = content
            stage.fix_applied = True
            return fixed

        return None

    def _attempt_coherence_fix(
        self,
        generated: dict[str, Any],
        source: dict[str, Any],
        coherence: CoherenceResult
    ) -> Optional[dict[str, Any]]:
        """Attempt to fix coherence issues."""
        from .smart_generator import SmartQuestionGenerator

        generator = SmartQuestionGenerator()
        result = generator._fix_coherence_issues(
            source,
            generated,
            coherence,
            self.constraint_extractor.extract(source)
        )

        return result


# Singleton instance
validation_pipeline = IntelligentValidationPipeline()

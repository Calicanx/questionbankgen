"""Image-text coherence validation for generated questions."""

import re
import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from .constraint_extractor import (
    ConstraintExtractor,
    QuestionConstraints,
    SubjectGender,
    ImageDependency,
)

logger = logging.getLogger(__name__)


@dataclass
class CoherenceIssue:
    """A specific coherence issue found."""
    severity: str  # 'error', 'warning'
    category: str  # 'gender_mismatch', 'image_text_mismatch', etc.
    message: str
    fix_suggestion: Optional[str] = None


@dataclass
class CoherenceResult:
    """Result of coherence validation."""
    is_coherent: bool
    score: float  # 0.0 to 1.0
    issues: list[CoherenceIssue] = field(default_factory=list)

    def add_error(self, category: str, message: str, fix: Optional[str] = None):
        self.issues.append(CoherenceIssue('error', category, message, fix))
        self.is_coherent = False

    def add_warning(self, category: str, message: str, fix: Optional[str] = None):
        self.issues.append(CoherenceIssue('warning', category, message, fix))


class CoherenceValidator:
    """Validates coherence between text, images, and widgets in questions."""

    def __init__(self):
        self.extractor = ConstraintExtractor()

    def validate(
        self,
        source_data: dict[str, Any],
        generated_data: dict[str, Any]
    ) -> CoherenceResult:
        """
        Validate coherence of generated question against source.

        Checks:
        1. Gender consistency (if source image shows female, text should too)
        2. Image-text alignment (question text matches what image shows)
        3. Widget-content alignment (widget type matches question type)
        4. Numerical consistency (answer matches question)
        5. Logical sense (question is answerable)
        """
        result = CoherenceResult(is_coherent=True, score=1.0)

        # Extract constraints from both
        source_constraints = self.extractor.extract(source_data)
        gen_constraints = self.extractor.extract(generated_data)

        # Check gender consistency
        self._check_gender_consistency(
            source_constraints, gen_constraints, generated_data, result
        )

        # Check image-text alignment
        self._check_image_text_alignment(
            source_constraints, gen_constraints, generated_data, result
        )

        # Check widget-content alignment
        self._check_widget_content_alignment(
            gen_constraints, generated_data, result
        )

        # Check numerical consistency
        self._check_numerical_consistency(
            gen_constraints, generated_data, result
        )

        # Check logical sense
        self._check_logical_sense(generated_data, result)

        # Calculate final score
        error_count = sum(1 for i in result.issues if i.severity == 'error')
        warning_count = sum(1 for i in result.issues if i.severity == 'warning')
        result.score = max(0.0, 1.0 - (error_count * 0.3) - (warning_count * 0.1))

        return result

    def _check_gender_consistency(
        self,
        source: QuestionConstraints,
        generated: QuestionConstraints,
        gen_data: dict[str, Any],
        result: CoherenceResult
    ) -> None:
        """Check if gender references match between image and text."""
        # If source has gender-specific image
        if source.image.dependency == ImageDependency.REQUIRED:
            if source.image.gender in [SubjectGender.MALE, SubjectGender.FEMALE]:
                # Check if generated uses same image (by comparing URLs)
                source_urls = set(source.image.urls)
                gen_urls = set(generated.image.urls)

                if source_urls & gen_urls:  # Same image used
                    # Text gender must match source image gender
                    gen_content = gen_data.get('question', {}).get('content', '').lower()

                    if source.image.gender == SubjectGender.FEMALE:
                        if any(word in gen_content for word in ['male', 'boy', 'man', ' he ', ' his ']):
                            result.add_error(
                                'gender_mismatch',
                                f'Text refers to male but image shows female',
                                'Change text to match image gender or generate new image'
                            )
                    elif source.image.gender == SubjectGender.MALE:
                        if any(word in gen_content for word in ['female', 'girl', 'woman', ' she ', ' her ']):
                            result.add_error(
                                'gender_mismatch',
                                f'Text refers to female but image shows male',
                                'Change text to match image gender or generate new image'
                            )

    def _check_image_text_alignment(
        self,
        source: QuestionConstraints,
        generated: QuestionConstraints,
        gen_data: dict[str, Any],
        result: CoherenceResult
    ) -> None:
        """Check if question text aligns with image content."""
        gen_content = gen_data.get('question', {}).get('content', '').lower()

        # If source has specific image subjects
        if source.image.subjects:
            # Check if generated text references things not in image
            for subject in source.image.subjects:
                if subject == 'place_value_blocks':
                    # For place value, check if numbers in text match image
                    # This is handled by numerical consistency
                    pass

        # Check for image references without images
        if 'image' in gen_content or 'diagram' in gen_content or 'picture' in gen_content:
            if not generated.image.urls:
                result.add_warning(
                    'missing_image',
                    'Question references image/diagram but no image found',
                    'Add an appropriate image or remove image reference'
                )

        # Check for "below"/"above" references
        if 'below' in gen_content or 'above' in gen_content:
            if not generated.image.urls and '[[' not in gen_content:
                result.add_warning(
                    'spatial_reference',
                    'Question uses spatial reference (below/above) without visual element',
                    'Add visual element or rephrase question'
                )

    def _check_widget_content_alignment(
        self,
        constraints: QuestionConstraints,
        gen_data: dict[str, Any],
        result: CoherenceResult
    ) -> None:
        """Check if widget type matches the question content."""
        content = gen_data.get('question', {}).get('content', '').lower()
        widgets = gen_data.get('question', {}).get('widgets', {})

        for widget_id, widget in widgets.items():
            widget_type = widget.get('type', '')

            # Check for mismatches
            if widget_type == 'radio':
                # Radio should have choices
                choices = widget.get('options', {}).get('choices', [])
                if not choices:
                    result.add_error(
                        'widget_config',
                        f'Radio widget {widget_id} has no choices',
                        'Add choices to radio widget'
                    )
                # Check if any choice is marked correct
                has_correct = any(c.get('correct') for c in choices)
                if not has_correct:
                    result.add_error(
                        'no_correct_answer',
                        f'Radio widget {widget_id} has no correct answer',
                        'Mark one choice as correct'
                    )

            elif widget_type == 'numeric-input':
                answers = widget.get('options', {}).get('answers', [])
                if not answers:
                    result.add_error(
                        'widget_config',
                        f'Numeric input {widget_id} has no answers defined',
                        'Add correct answer configuration'
                    )

            elif widget_type in ['orderer', 'sorter']:
                options = widget.get('options', {})
                items = options.get('correct') or options.get('options') or options.get('correctOptions')
                if not items:
                    result.add_error(
                        'widget_config',
                        f'{widget_type} widget {widget_id} has no items to order',
                        'Add items to order'
                    )

    def _check_numerical_consistency(
        self,
        constraints: QuestionConstraints,
        gen_data: dict[str, Any],
        result: CoherenceResult
    ) -> None:
        """Check if numerical values in question match expected answer."""
        # This is a basic check - more sophisticated math verification
        # would require symbolic computation

        if constraints.numerical.target_answer is not None:
            # Check if answer is reasonable given the numbers in question
            if constraints.numerical.values:
                answer = constraints.numerical.target_answer
                values = constraints.numerical.values

                # Simple sanity checks
                # Answer shouldn't be wildly different from input values
                if values:
                    max_val = max(values)
                    min_val = min(values)

                    # For most operations, answer should be within reasonable range
                    if answer > max_val * 1000 or (answer != 0 and answer < min_val / 1000):
                        result.add_warning(
                            'numerical_range',
                            f'Answer {answer} seems unusual given input values {values}',
                            'Verify the mathematical correctness'
                        )

    def _check_logical_sense(
        self,
        gen_data: dict[str, Any],
        result: CoherenceResult
    ) -> None:
        """Check if question makes logical sense."""
        content = gen_data.get('question', {}).get('content', '')
        widgets = gen_data.get('question', {}).get('widgets', {})

        # Check for empty or very short content
        if len(content.strip()) < 20:
            result.add_error(
                'empty_content',
                'Question content is too short or empty',
                'Add meaningful question content'
            )

        # Check for widget placeholders without widgets
        placeholder_pattern = r'\[\[☃\s*([^\]]+)\]\]'
        placeholders = re.findall(placeholder_pattern, content)

        for placeholder in placeholders:
            widget_id = placeholder.strip()
            if widget_id not in widgets:
                result.add_error(
                    'missing_widget',
                    f'Placeholder [[☃ {widget_id}]] has no corresponding widget',
                    f'Add widget definition for {widget_id}'
                )

        # Check for widgets not referenced in content
        for widget_id in widgets.keys():
            if f'[[☃ {widget_id}]]' not in content:
                result.add_warning(
                    'unreferenced_widget',
                    f'Widget {widget_id} is not referenced in question content',
                    f'Add [[☃ {widget_id}]] to content or remove widget'
                )

    def quick_check(self, gen_data: dict[str, Any]) -> bool:
        """Quick coherence check without source comparison."""
        result = CoherenceResult(is_coherent=True, score=1.0)

        constraints = self.extractor.extract(gen_data)
        self._check_widget_content_alignment(constraints, gen_data, result)
        self._check_logical_sense(gen_data, result)

        return result.is_coherent


# Singleton instance
coherence_validator = CoherenceValidator()

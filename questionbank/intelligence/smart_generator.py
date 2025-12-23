"""Smart question generator with widget-specific strategies."""

import re
import json
import random
import logging
from typing import Any, Optional
from dataclasses import dataclass, field
from copy import deepcopy

from .constraint_extractor import (
    ConstraintExtractor,
    QuestionConstraints,
    SubjectGender,
    ImageDependency,
)
from .coherence_validator import CoherenceValidator, CoherenceResult
from .image_generator import ImageGenerator, GeneratedImage

logger = logging.getLogger(__name__)


@dataclass
class GenerationConfig:
    """Configuration for smart generation."""
    max_retries: int = 3
    preserve_image_gender: bool = True
    generate_new_images: bool = True
    numerical_variation_range: float = 0.5  # ±50% of original values
    validate_coherence: bool = True


@dataclass
class GenerationResult:
    """Result of smart question generation."""
    success: bool
    generated_data: Optional[dict] = None
    coherence_result: Optional[CoherenceResult] = None
    strategy_used: str = ""
    modifications: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


class SmartQuestionGenerator:
    """
    Intelligent question generator that understands semantic constraints.

    Strategies by widget type:
    - numeric-input: Vary numbers while preserving operation type
    - radio: Vary choice content while preserving structure
    - radio-with-images: Generate new images OR preserve image-text alignment
    - orderer/sorter: Vary items while preserving category
    - expression: Vary coefficients/constants
    - image-dependent: Keep image, modify only safe text elements
    """

    def __init__(self, config: Optional[GenerationConfig] = None):
        self.config = config or GenerationConfig()
        self.extractor = ConstraintExtractor()
        self.validator = CoherenceValidator()
        self.image_generator = ImageGenerator()

    def generate(
        self,
        source_data: dict[str, Any],
        llm_response: Optional[dict[str, Any]] = None
    ) -> GenerationResult:
        """
        Generate a coherent variation of the source question.

        Args:
            source_data: Original Perseus question data
            llm_response: Optional LLM-generated variation to validate/fix

        Returns:
            GenerationResult with success status and generated data
        """
        result = GenerationResult(success=False)

        # Extract constraints from source
        constraints = self.extractor.extract(source_data)
        result.strategy_used = constraints.variation_strategy

        # Determine generation strategy
        strategy = self._select_strategy(constraints)
        logger.info(f"Using strategy: {strategy} for widget type: {constraints.widget_type}")

        try:
            if llm_response:
                # Validate and fix LLM response
                generated = self._validate_and_fix(
                    source_data, llm_response, constraints
                )
            else:
                # Generate programmatically based on strategy
                generated = self._generate_by_strategy(source_data, constraints, strategy)

            if generated:
                # Final coherence check
                if self.config.validate_coherence:
                    coherence = self.validator.validate(source_data, generated)
                    result.coherence_result = coherence

                    if not coherence.is_coherent:
                        # Try to fix coherence issues
                        generated = self._fix_coherence_issues(
                            source_data, generated, coherence, constraints
                        )
                        # Re-validate
                        result.coherence_result = self.validator.validate(source_data, generated)

                result.generated_data = generated
                result.success = result.coherence_result is None or result.coherence_result.is_coherent

        except Exception as e:
            logger.error(f"Generation failed: {e}")
            result.errors.append(str(e))

        return result

    def _select_strategy(self, constraints: QuestionConstraints) -> str:
        """Select the best generation strategy based on constraints."""
        # Image-dependent questions need special handling
        if constraints.image.dependency == ImageDependency.REQUIRED:
            if 'place_value_blocks' in constraints.image.subjects:
                return 'regenerate_place_value'
            if constraints.image.gender != SubjectGender.UNKNOWN:
                return 'preserve_image_gender'
            return 'preserve_image'

        # Widget-specific strategies
        widget_strategies = {
            'numeric-input': 'numerical_variation',
            'input-number': 'numerical_variation',
            'expression': 'expression_variation',
            'radio': 'choice_variation',
            'dropdown': 'choice_variation',
            'orderer': 'ordering_variation',
            'sorter': 'ordering_variation',
        }

        return widget_strategies.get(constraints.widget_type, 'general_variation')

    def _generate_by_strategy(
        self,
        source: dict[str, Any],
        constraints: QuestionConstraints,
        strategy: str
    ) -> dict[str, Any]:
        """Generate variation using specific strategy."""
        generated = deepcopy(source)

        if strategy == 'regenerate_place_value':
            generated = self._regenerate_place_value(source, constraints)

        elif strategy == 'numerical_variation':
            generated = self._apply_numerical_variation(source, constraints)

        elif strategy == 'preserve_image_gender':
            generated = self._preserve_image_gender_variation(source, constraints)

        elif strategy == 'ordering_variation':
            generated = self._apply_ordering_variation(source, constraints)

        elif strategy == 'choice_variation':
            generated = self._apply_choice_variation(source, constraints)

        return generated

    def _regenerate_place_value(
        self,
        source: dict[str, Any],
        constraints: QuestionConstraints
    ) -> dict[str, Any]:
        """Regenerate place value question with new number and image."""
        generated = deepcopy(source)

        # Find the target number from source
        source_answer = constraints.numerical.target_answer
        if source_answer is None:
            return generated

        # Generate a new number (different from source)
        new_number = self._generate_different_number(int(source_answer), 100, 999)

        # Generate new place value image
        new_image = self.image_generator.generate_place_value_blocks(new_number)

        if new_image:
            # Update question content with new image
            content = generated.get('question', {}).get('content', '')

            # Replace image URL
            old_img_pattern = r'!\[([^\]]*)\]\([^)]+\)'
            new_img_markdown = f'![{new_image.alt_text}]({new_image.url})'
            content = re.sub(old_img_pattern, new_img_markdown, content)

            generated['question']['content'] = content

            # Update images dict
            generated['question']['images'] = {
                new_image.url: {
                    'url': new_image.url,
                    'width': new_image.width,
                    'height': new_image.height,
                    'alt': new_image.alt_text
                }
            }

            # Update answer widget
            widgets = generated.get('question', {}).get('widgets', {})
            for widget_id, widget in widgets.items():
                if widget.get('type') == 'numeric-input':
                    answers = widget.get('options', {}).get('answers', [])
                    for ans in answers:
                        if ans.get('status') == 'correct':
                            ans['value'] = new_number
                            break

        return generated

    def _apply_numerical_variation(
        self,
        source: dict[str, Any],
        constraints: QuestionConstraints
    ) -> dict[str, Any]:
        """Apply numerical variation to question."""
        generated = deepcopy(source)

        content = generated.get('question', {}).get('content', '')
        widgets = generated.get('question', {}).get('widgets', {})

        # Find numbers in content and vary them
        # Skip numbers that are part of widget references
        number_pattern = r'(?<!\[\[☃\s)(?<!\d)(\d+\.?\d*)(?!\d)(?!\s*\]\])'

        def vary_number(match):
            original = float(match.group(1))
            variation = random.uniform(
                1 - self.config.numerical_variation_range,
                1 + self.config.numerical_variation_range
            )
            new_val = original * variation

            # Keep integers as integers
            if '.' not in match.group(1):
                new_val = int(round(new_val))

            return str(new_val)

        # Be careful not to change things like "2x" -> "3x" breaking meaning
        # Only vary standalone numbers
        new_content = re.sub(number_pattern, vary_number, content)
        generated['question']['content'] = new_content

        # Update answer based on new numbers
        # This requires understanding the mathematical relationship
        # For now, flag for manual verification
        generated['_needs_answer_recalculation'] = True

        return generated

    def _preserve_image_gender_variation(
        self,
        source: dict[str, Any],
        constraints: QuestionConstraints
    ) -> dict[str, Any]:
        """Vary question while preserving gender to match image."""
        generated = deepcopy(source)

        content = generated.get('question', {}).get('content', '')

        # Only vary non-gender-related elements
        # Keep: gender pronouns, gender-specific nouns
        # Vary: numbers, non-gender context

        # Apply numerical variation only
        number_pattern = r'(?<!\[\[☃\s)(\d+)(?!\s*\]\])'

        def safe_vary(match):
            original = int(match.group(1))
            new_val = original + random.randint(-5, 5)
            return str(max(1, new_val))

        new_content = re.sub(number_pattern, safe_vary, content)
        generated['question']['content'] = new_content

        return generated

    def _apply_ordering_variation(
        self,
        source: dict[str, Any],
        constraints: QuestionConstraints
    ) -> dict[str, Any]:
        """Vary orderer/sorter items while preserving category."""
        generated = deepcopy(source)

        widgets = generated.get('question', {}).get('widgets', {})

        for widget_id, widget in widgets.items():
            if widget.get('type') in ['orderer', 'sorter']:
                options = widget.get('options', {})

                # Get current items
                items = (
                    options.get('correct') or
                    options.get('options') or
                    options.get('correctOptions') or
                    []
                )

                if items:
                    # Try to generate semantically similar items
                    # This is simplified - real implementation would use LLM
                    new_items = self._vary_ordering_items(items)

                    # Update widget
                    if 'correct' in options:
                        options['correct'] = new_items
                    if 'options' in options:
                        options['options'] = new_items
                    if 'correctOptions' in options:
                        options['correctOptions'] = new_items

        return generated

    def _vary_ordering_items(self, items: list[str]) -> list[str]:
        """Generate variations of ordering items."""
        # Simple strategy: if items are numbers/fractions, vary them
        # If items are text, keep them (needs LLM for meaningful variation)

        new_items = []
        for item in items:
            # Check if item is a number or fraction
            if re.match(r'^[\d./]+$', item.strip()):
                try:
                    if '/' in item:
                        # Fraction
                        num, den = item.split('/')
                        new_num = int(num) + random.randint(-1, 1)
                        new_den = int(den)
                        new_items.append(f"{max(1, new_num)}/{new_den}")
                    else:
                        # Number
                        val = float(item)
                        new_val = val + random.uniform(-0.5, 0.5)
                        if '.' not in item:
                            new_val = int(round(new_val))
                        new_items.append(str(new_val))
                except:
                    new_items.append(item)
            else:
                # Keep text items as-is
                new_items.append(item)

        return new_items

    def _apply_choice_variation(
        self,
        source: dict[str, Any],
        constraints: QuestionConstraints
    ) -> dict[str, Any]:
        """Vary radio/dropdown choices."""
        generated = deepcopy(source)

        widgets = generated.get('question', {}).get('widgets', {})

        for widget_id, widget in widgets.items():
            if widget.get('type') in ['radio', 'dropdown']:
                choices = widget.get('options', {}).get('choices', [])

                for choice in choices:
                    content = choice.get('content', '')

                    # Vary numbers in choices
                    number_pattern = r'(\d+\.?\d*)'

                    def vary_choice_number(match):
                        original = float(match.group(1))
                        variation = random.uniform(0.8, 1.2)
                        new_val = original * variation
                        if '.' not in match.group(1):
                            new_val = int(round(new_val))
                        return str(new_val)

                    choice['content'] = re.sub(number_pattern, vary_choice_number, content)

        return generated

    def _validate_and_fix(
        self,
        source: dict[str, Any],
        llm_response: dict[str, Any],
        constraints: QuestionConstraints
    ) -> dict[str, Any]:
        """Validate LLM response and fix issues."""
        generated = deepcopy(llm_response)

        # Check coherence
        coherence = self.validator.validate(source, generated)

        if not coherence.is_coherent:
            generated = self._fix_coherence_issues(source, generated, coherence, constraints)

        return generated

    def _fix_coherence_issues(
        self,
        source: dict[str, Any],
        generated: dict[str, Any],
        coherence: CoherenceResult,
        constraints: QuestionConstraints
    ) -> dict[str, Any]:
        """Attempt to fix coherence issues in generated question."""
        fixed = deepcopy(generated)

        for issue in coherence.issues:
            if issue.severity != 'error':
                continue

            if issue.category == 'gender_mismatch':
                # Fix gender in text to match image
                fixed = self._fix_gender_mismatch(source, fixed, constraints)

            elif issue.category == 'missing_widget':
                # Copy widget from source
                fixed = self._fix_missing_widget(source, fixed, issue.message)

            elif issue.category == 'no_correct_answer':
                # Mark first choice as correct
                fixed = self._fix_missing_correct_answer(fixed)

        return fixed

    def _fix_gender_mismatch(
        self,
        source: dict[str, Any],
        generated: dict[str, Any],
        constraints: QuestionConstraints
    ) -> dict[str, Any]:
        """Fix gender references to match source image."""
        fixed = deepcopy(generated)
        content = fixed.get('question', {}).get('content', '')

        if constraints.image.gender == SubjectGender.FEMALE:
            # Replace male references with female
            replacements = [
                (r'\bmale\b', 'female'),
                (r'\bboy\b', 'girl'),
                (r'\bman\b', 'woman'),
                (r'\b[Hh]is\b', 'her'),
                (r'\b[Hh]e\b', 'she'),
            ]
        elif constraints.image.gender == SubjectGender.MALE:
            # Replace female references with male
            replacements = [
                (r'\bfemale\b', 'male'),
                (r'\bgirl\b', 'boy'),
                (r'\bwoman\b', 'man'),
                (r'\b[Hh]er\b', 'his'),
                (r'\b[Ss]he\b', 'he'),
            ]
        else:
            return fixed

        for pattern, replacement in replacements:
            content = re.sub(pattern, replacement, content)

        fixed['question']['content'] = content
        return fixed

    def _fix_missing_widget(
        self,
        source: dict[str, Any],
        generated: dict[str, Any],
        error_message: str
    ) -> dict[str, Any]:
        """Copy missing widget from source."""
        fixed = deepcopy(generated)

        # Extract widget ID from error message
        match = re.search(r'\[\[☃\s*([^\]]+)\]\]', error_message)
        if match:
            widget_id = match.group(1).strip()
            source_widgets = source.get('question', {}).get('widgets', {})

            if widget_id in source_widgets:
                fixed['question']['widgets'][widget_id] = deepcopy(source_widgets[widget_id])

        return fixed

    def _fix_missing_correct_answer(self, generated: dict[str, Any]) -> dict[str, Any]:
        """Mark first choice as correct if none marked."""
        fixed = deepcopy(generated)

        widgets = fixed.get('question', {}).get('widgets', {})
        for widget_id, widget in widgets.items():
            if widget.get('type') == 'radio':
                choices = widget.get('options', {}).get('choices', [])
                if choices and not any(c.get('correct') for c in choices):
                    choices[0]['correct'] = True

        return fixed

    def _generate_different_number(
        self,
        original: int,
        min_val: int,
        max_val: int
    ) -> int:
        """Generate a number different from original within range."""
        attempts = 0
        while attempts < 10:
            new_num = random.randint(min_val, max_val)
            if new_num != original:
                return new_num
            attempts += 1
        return original + 1 if original < max_val else original - 1


# Singleton instance
smart_generator = SmartQuestionGenerator()

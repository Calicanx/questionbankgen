"""Semantic constraint extraction from source questions."""

import re
import logging
from dataclasses import dataclass, field
from typing import Any, Optional
from enum import Enum

logger = logging.getLogger(__name__)


class SubjectGender(Enum):
    MALE = "male"
    FEMALE = "female"
    NEUTRAL = "neutral"
    UNKNOWN = "unknown"


class ImageDependency(Enum):
    REQUIRED = "required"  # Question meaningless without image
    SUPPLEMENTARY = "supplementary"  # Image helps but not essential
    NONE = "none"  # No image dependency


@dataclass
class ImageConstraints:
    """Constraints related to images in the question."""
    urls: list[str] = field(default_factory=list)
    alt_texts: list[str] = field(default_factory=list)
    subjects: list[str] = field(default_factory=list)  # What's depicted
    gender: SubjectGender = SubjectGender.UNKNOWN
    dependency: ImageDependency = ImageDependency.NONE
    is_diagram: bool = False
    is_graph: bool = False
    is_photo: bool = False


@dataclass
class NumericalConstraints:
    """Numerical values and their roles in the question."""
    values: list[float] = field(default_factory=list)
    target_answer: Optional[float] = None
    units: Optional[str] = None
    precision: int = 0  # Decimal places


@dataclass
class QuestionConstraints:
    """Complete constraints extracted from a source question."""
    widget_type: str = ""
    topic: str = ""
    subject_area: str = ""  # math, science, language, etc.

    # Image constraints
    image: ImageConstraints = field(default_factory=ImageConstraints)

    # Numerical constraints
    numerical: NumericalConstraints = field(default_factory=NumericalConstraints)

    # Text constraints
    key_entities: list[str] = field(default_factory=list)  # People, places, things referenced
    must_preserve: list[str] = field(default_factory=list)  # Elements that can't change
    can_vary: list[str] = field(default_factory=list)  # Elements that can be changed

    # Widget-specific
    choice_count: int = 0  # For radio/dropdown
    correct_index: Optional[int] = None

    # Generation guidance
    variation_strategy: str = "numerical"  # numerical, contextual, structural


class ConstraintExtractor:
    """Extracts semantic constraints from Perseus questions."""

    # Gender indicators
    FEMALE_INDICATORS = ['female', 'girl', 'woman', 'her', 'she', 'mother', 'sister', 'daughter', 'aunt']
    MALE_INDICATORS = ['male', 'boy', 'man', 'his', 'he', 'father', 'brother', 'son', 'uncle']

    # Subject area keywords
    MATH_KEYWORDS = ['equation', 'solve', 'calculate', 'graph', 'function', 'variable', 'expression', 'fraction', 'decimal']
    SCIENCE_KEYWORDS = ['cell', 'organism', 'reaction', 'force', 'energy', 'atom', 'molecule', 'photosynthesis']
    ANATOMY_KEYWORDS = ['kidney', 'heart', 'lung', 'muscle', 'bone', 'organ', 'tissue', 'blood', 'urinary']

    def extract(self, data: dict[str, Any]) -> QuestionConstraints:
        """Extract constraints from a Perseus question."""
        constraints = QuestionConstraints()

        question = data.get('question', {})
        content = question.get('content', '')
        widgets = question.get('widgets', {})
        images = question.get('images', {})

        # Extract widget type
        constraints.widget_type = self._get_primary_widget_type(widgets)

        # Extract image constraints
        constraints.image = self._extract_image_constraints(content, images, widgets)

        # Extract numerical constraints
        constraints.numerical = self._extract_numerical_constraints(content, widgets)

        # Extract subject area and topic
        constraints.subject_area = self._determine_subject_area(content)
        constraints.topic = self._extract_topic(content)

        # Extract key entities
        constraints.key_entities = self._extract_entities(content)

        # Determine what can/can't change
        constraints.must_preserve, constraints.can_vary = self._determine_variability(
            content, constraints.image, constraints.widget_type
        )

        # Widget-specific constraints
        self._extract_widget_specific(constraints, widgets)

        # Determine variation strategy
        constraints.variation_strategy = self._determine_strategy(constraints)

        return constraints

    def _get_primary_widget_type(self, widgets: dict[str, Any]) -> str:
        """Get the primary graded widget type."""
        for widget_id, widget in widgets.items():
            if widget.get('graded', True):
                return widget.get('type', '')
        # Return first widget type if none are graded
        if widgets:
            return list(widgets.values())[0].get('type', '')
        return ''

    def _extract_image_constraints(
        self,
        content: str,
        images: dict[str, Any],
        widgets: dict[str, Any]
    ) -> ImageConstraints:
        """Extract constraints related to images."""
        img_constraints = ImageConstraints()

        # Extract image URLs from content
        img_pattern = r'!\[([^\]]*)\]\(([^)]+)\)'
        matches = re.findall(img_pattern, content)

        for alt_text, url in matches:
            img_constraints.urls.append(url)
            if alt_text:
                img_constraints.alt_texts.append(alt_text)

        # Add images from images dict
        for url, img_data in images.items():
            if url not in img_constraints.urls:
                img_constraints.urls.append(url)
            if img_data.get('alt'):
                img_constraints.alt_texts.append(img_data['alt'])

        # Check for image widgets
        for widget_id, widget in widgets.items():
            if widget.get('type') == 'image':
                bg_img = widget.get('options', {}).get('backgroundImage', {})
                if bg_img.get('url'):
                    img_constraints.urls.append(bg_img['url'])

        # Determine image dependency
        if img_constraints.urls:
            content_lower = content.lower()
            if any(phrase in content_lower for phrase in ['look at', 'shown', 'below', 'above', 'diagram', 'image', 'picture', 'graph']):
                img_constraints.dependency = ImageDependency.REQUIRED
            else:
                img_constraints.dependency = ImageDependency.SUPPLEMENTARY

        # Detect gender from content and alt text
        img_constraints.gender = self._detect_gender(content + ' ' + ' '.join(img_constraints.alt_texts))

        # Detect image type
        content_lower = content.lower()
        if 'diagram' in content_lower or 'cell' in content_lower:
            img_constraints.is_diagram = True
        if 'graph' in content_lower or 'plot' in content_lower:
            img_constraints.is_graph = True
        if any(ext in str(img_constraints.urls) for ext in ['.jpg', '.jpeg', '.png', '.photo']):
            img_constraints.is_photo = True

        # Extract subjects from alt text
        for alt in img_constraints.alt_texts:
            # Parse place value blocks
            if 'cube' in alt.lower() or 'flat' in alt.lower() or 'rod' in alt.lower():
                img_constraints.subjects.append('place_value_blocks')
            # Parse other subjects
            words = re.findall(r'\b[A-Za-z]{4,}\b', alt)
            img_constraints.subjects.extend([w.lower() for w in words])

        return img_constraints

    def _detect_gender(self, text: str) -> SubjectGender:
        """Detect gender references in text."""
        text_lower = text.lower()

        female_count = sum(1 for word in self.FEMALE_INDICATORS if word in text_lower)
        male_count = sum(1 for word in self.MALE_INDICATORS if word in text_lower)

        if female_count > male_count:
            return SubjectGender.FEMALE
        elif male_count > female_count:
            return SubjectGender.MALE
        elif female_count == male_count and female_count > 0:
            return SubjectGender.NEUTRAL
        return SubjectGender.UNKNOWN

    def _extract_numerical_constraints(
        self,
        content: str,
        widgets: dict[str, Any]
    ) -> NumericalConstraints:
        """Extract numerical values and constraints."""
        num_constraints = NumericalConstraints()

        # Extract numbers from content
        # Match integers and decimals, including those in LaTeX
        number_pattern = r'(?<![a-zA-Z])(\d+\.?\d*)(?![a-zA-Z])'
        numbers = re.findall(number_pattern, content)
        num_constraints.values = [float(n) for n in numbers if n]

        # Extract target answer from widgets
        for widget_id, widget in widgets.items():
            widget_type = widget.get('type', '')
            options = widget.get('options', {})

            if widget_type == 'numeric-input':
                answers = options.get('answers', [])
                for ans in answers:
                    if ans.get('status') == 'correct':
                        num_constraints.target_answer = ans.get('value')
                        break

            elif widget_type == 'input-number':
                num_constraints.target_answer = options.get('value')

            elif widget_type == 'expression':
                forms = options.get('answerForms', [])
                for form in forms:
                    if form.get('considered') == 'correct':
                        # Try to parse numerical value from expression
                        try:
                            val = float(form.get('value', ''))
                            num_constraints.target_answer = val
                        except ValueError:
                            pass
                        break

        # Detect units
        unit_patterns = [
            r'(\d+)\s*(cm|m|km|mm|inch|ft|yard)',
            r'(\d+)\s*(kg|g|mg|lb|oz)',
            r'(\d+)\s*(L|mL|gal)',
            r'(\d+)\s*(°[CF]|degrees)',
            r'(\d+)\s*(hr|min|sec|hours|minutes|seconds)',
        ]
        for pattern in unit_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                num_constraints.units = match.group(2)
                break

        # Determine precision
        for val in num_constraints.values:
            str_val = str(val)
            if '.' in str_val:
                decimals = len(str_val.split('.')[1])
                num_constraints.precision = max(num_constraints.precision, decimals)

        return num_constraints

    def _determine_subject_area(self, content: str) -> str:
        """Determine the subject area of the question."""
        content_lower = content.lower()

        if any(kw in content_lower for kw in self.ANATOMY_KEYWORDS):
            return 'anatomy'
        if any(kw in content_lower for kw in self.SCIENCE_KEYWORDS):
            return 'science'
        if any(kw in content_lower for kw in self.MATH_KEYWORDS):
            return 'math'

        # Default based on content patterns
        if re.search(r'\$.*\$', content):  # Has LaTeX
            return 'math'

        return 'general'

    def _extract_topic(self, content: str) -> str:
        """Extract the specific topic of the question."""
        content_lower = content.lower()

        # Math topics
        if 'place value' in content_lower or 'blocks' in content_lower:
            return 'place_value'
        if 'fraction' in content_lower:
            return 'fractions'
        if 'decimal' in content_lower:
            return 'decimals'
        if 'equation' in content_lower:
            return 'equations'
        if 'graph' in content_lower:
            return 'graphing'

        # Science topics
        if 'cell' in content_lower:
            return 'cells'
        if 'photosynthesis' in content_lower:
            return 'photosynthesis'
        if 'urinary' in content_lower or 'kidney' in content_lower:
            return 'urinary_system'

        return 'general'

    def _extract_entities(self, content: str) -> list[str]:
        """Extract key entities (nouns, proper nouns) from content."""
        entities = []

        # Extract capitalized words (potential proper nouns)
        proper_nouns = re.findall(r'\b[A-Z][a-z]+\b', content)
        entities.extend(proper_nouns)

        # Extract specific entity patterns
        # Names
        name_pattern = r'\b(Mr\.|Mrs\.|Ms\.|Dr\.)\s+([A-Z][a-z]+)'
        names = re.findall(name_pattern, content)
        entities.extend([n[1] for n in names])

        return list(set(entities))

    def _determine_variability(
        self,
        content: str,
        image_constraints: ImageConstraints,
        widget_type: str
    ) -> tuple[list[str], list[str]]:
        """Determine what must be preserved vs what can vary."""
        must_preserve = []
        can_vary = []

        # If image is required, preserve image-related attributes
        if image_constraints.dependency == ImageDependency.REQUIRED:
            must_preserve.append('image_reference')
            if image_constraints.gender != SubjectGender.UNKNOWN:
                must_preserve.append(f'gender:{image_constraints.gender.value}')
            must_preserve.extend(image_constraints.subjects)

        # Numbers can usually vary
        can_vary.append('numerical_values')

        # Widget-specific variability
        if widget_type in ['radio', 'dropdown']:
            can_vary.append('choice_content')
            must_preserve.append('choice_count')

        if widget_type in ['orderer', 'sorter']:
            can_vary.append('item_content')
            must_preserve.append('ordering_logic')

        return must_preserve, can_vary

    def _extract_widget_specific(
        self,
        constraints: QuestionConstraints,
        widgets: dict[str, Any]
    ) -> None:
        """Extract widget-specific constraints."""
        for widget_id, widget in widgets.items():
            widget_type = widget.get('type', '')
            options = widget.get('options', {})

            if widget_type == 'radio':
                choices = options.get('choices', [])
                constraints.choice_count = len(choices)
                for i, choice in enumerate(choices):
                    if choice.get('correct'):
                        constraints.correct_index = i
                        break

            elif widget_type == 'dropdown':
                choices = options.get('choices', [])
                constraints.choice_count = len(choices)
                constraints.correct_index = options.get('correct')

    def _determine_strategy(self, constraints: QuestionConstraints) -> str:
        """Determine the best variation strategy."""
        # If image is required and shows specific content, be careful
        if constraints.image.dependency == ImageDependency.REQUIRED:
            if constraints.image.gender != SubjectGender.UNKNOWN:
                return 'image_preserving'  # Keep image, modify only safe elements
            if 'place_value_blocks' in constraints.image.subjects:
                return 'regenerate_image'  # Generate new image with new values

        # Math questions can usually vary numerically
        if constraints.subject_area == 'math':
            return 'numerical'

        # Science questions might need contextual variation
        if constraints.subject_area == 'science':
            return 'contextual'

        return 'structural'


# Singleton instance
constraint_extractor = ConstraintExtractor()

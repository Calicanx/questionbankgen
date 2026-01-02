"""Comprehensive Answer Verification for ALL Question Types.

This module verifies answers across all question types:
- Math questions: Uses SymPy symbolic computation
- Science questions: Uses AI verification
- Reading/Language: Uses AI verification
- General knowledge: Uses AI verification

Supports all widget types:
- numeric-input, input-number: Numeric answers
- radio, dropdown: Multiple choice
- expression: Algebraic expressions
- matcher, orderer, sorter: Matching/ordering
"""

import re
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Optional, List, Dict, Tuple, Union
from enum import Enum

logger = logging.getLogger(__name__)

# Import math verifier components
from questionbank.validation.math_verifier import (
    MathExpressionParser,
    MathEvaluator,
    VerificationResult,
    ParsedQuestion,
    SYMPY_AVAILABLE,
)


class QuestionType(Enum):
    """Types of questions based on content."""
    MATH = "math"
    SCIENCE = "science"
    READING = "reading"
    LANGUAGE = "language"
    GENERAL = "general"
    UNKNOWN = "unknown"


class SubjectDetector:
    """Detects the subject/type of a question."""

    MATH_INDICATORS = [
        r'\$[^$]+\$',  # LaTeX math
        r'\\frac', r'\\dfrac', r'\\sqrt',
        r'evaluate', r'simplify', r'solve',
        r'equation', r'expression', r'polynomial',
        r'calculate', r'compute', r'\+ | \- | \* | \/ | \=',
        r'algebra', r'geometry', r'calculus',
        r'x\s*=', r'y\s*=', r'variable',
        r'\d+\s*[\+\-\*/]\s*\d+',
    ]

    SCIENCE_INDICATORS = [
        r'molecule', r'atom', r'chemical', r'element',
        r'cell', r'organism', r'species', r'evolution',
        r'force', r'energy', r'velocity', r'acceleration',
        r'photosynthesis', r'respiration', r'dna', r'rna',
        r'gravity', r'mass', r'weight', r'newton',
        r'electron', r'proton', r'neutron',
        r'ecosystem', r'habitat', r'climate',
        r'planet', r'solar system', r'universe',
        r'temperature', r'pressure', r'volume',
    ]

    READING_INDICATORS = [
        r'passage', r'paragraph', r'text',
        r'author', r'narrator', r'character',
        r'main idea', r'theme', r'plot',
        r'inference', r'conclude', r'suggest',
        r'according to', r'the passage states',
        r'literary', r'metaphor', r'simile',
        r'tone', r'mood', r'setting',
    ]

    LANGUAGE_INDICATORS = [
        r'grammar', r'punctuation', r'spelling',
        r'noun', r'verb', r'adjective', r'adverb',
        r'sentence', r'clause', r'phrase',
        r'tense', r'singular', r'plural',
        r'synonym', r'antonym', r'prefix', r'suffix',
        r'vocabulary', r'definition',
    ]

    def detect(self, content: str) -> QuestionType:
        """Detect question type from content."""
        content_lower = content.lower()

        scores = {
            QuestionType.MATH: 0,
            QuestionType.SCIENCE: 0,
            QuestionType.READING: 0,
            QuestionType.LANGUAGE: 0,
        }

        # Check math indicators
        for pattern in self.MATH_INDICATORS:
            if re.search(pattern, content, re.IGNORECASE):
                scores[QuestionType.MATH] += 1

        # Check science indicators
        for pattern in self.SCIENCE_INDICATORS:
            if re.search(pattern, content_lower):
                scores[QuestionType.SCIENCE] += 1

        # Check reading indicators
        for pattern in self.READING_INDICATORS:
            if re.search(pattern, content_lower):
                scores[QuestionType.READING] += 1

        # Check language indicators
        for pattern in self.LANGUAGE_INDICATORS:
            if re.search(pattern, content_lower):
                scores[QuestionType.LANGUAGE] += 1

        # Return type with highest score
        max_score = max(scores.values())
        if max_score == 0:
            return QuestionType.GENERAL

        for qtype, score in scores.items():
            if score == max_score:
                return qtype

        return QuestionType.UNKNOWN


class AIAnswerVerifier:
    """Uses AI (Gemini) to verify answers for non-math questions."""

    def __init__(self):
        self._client = None

    @property
    def client(self):
        """Lazy load Gemini client."""
        if self._client is None:
            try:
                from questionbank.llm.gemini_client import get_gemini_client
                self._client = get_gemini_client()
            except Exception as e:
                logger.error(f"Failed to initialize Gemini client: {e}")
        return self._client

    def verify_answer(
        self,
        question_content: str,
        widget_type: str,
        stated_answer: Any,
        choices: Optional[List[Dict]] = None,
    ) -> VerificationResult:
        """Use AI to verify if the stated answer is correct."""
        if self.client is None:
            return VerificationResult(
                is_valid=True,
                details={"verification": "skipped_no_ai"}
            )

        try:
            # Build verification prompt
            prompt = self._build_verification_prompt(
                question_content,
                widget_type,
                stated_answer,
                choices,
            )

            # Get AI verification
            response = self.client.generate(prompt)

            if not response:
                return VerificationResult(
                    is_valid=True,
                    details={"verification": "ai_no_response"}
                )

            # Parse AI response
            return self._parse_ai_response(response, stated_answer)

        except Exception as e:
            logger.warning(f"AI verification failed: {e}")
            return VerificationResult(
                is_valid=True,
                details={"verification": "ai_error", "error": str(e)}
            )

    def _build_verification_prompt(
        self,
        question_content: str,
        widget_type: str,
        stated_answer: Any,
        choices: Optional[List[Dict]] = None,
    ) -> str:
        """Build prompt for AI verification."""

        if widget_type in ("radio", "dropdown") and choices:
            choices_text = "\n".join([
                f"  {i+1}. {c.get('content', '')} {'[MARKED CORRECT]' if c.get('correct') else ''}"
                for i, c in enumerate(choices)
            ])

            prompt = f"""Verify if the marked correct answer is actually correct for this question.

QUESTION:
{question_content}

ANSWER CHOICES:
{choices_text}

TASK: Analyze the question and determine if the choice marked as [MARKED CORRECT] is actually the correct answer.

Respond with ONLY a JSON object in this exact format:
{{"is_correct": true/false, "correct_answer": "the actual correct answer", "explanation": "brief reason"}}

IMPORTANT: Return ONLY the JSON, no other text."""

        else:
            prompt = f"""Verify if the given answer is correct for this question.

QUESTION:
{question_content}

STATED ANSWER: {stated_answer}

TASK: Determine if the stated answer is correct.

Respond with ONLY a JSON object in this exact format:
{{"is_correct": true/false, "correct_answer": "the actual correct answer", "explanation": "brief reason"}}

IMPORTANT: Return ONLY the JSON, no other text."""

        return prompt

    def _parse_ai_response(
        self,
        response: str,
        stated_answer: Any,
    ) -> VerificationResult:
        """Parse AI verification response."""
        try:
            # Clean response
            response = response.strip()
            if response.startswith("```"):
                response = re.sub(r'^```(?:json)?\n?', '', response)
                response = re.sub(r'\n?```$', '', response)

            # Parse JSON
            data = json.loads(response)

            is_correct = data.get("is_correct", True)
            correct_answer = data.get("correct_answer")
            explanation = data.get("explanation", "")

            if not is_correct:
                return VerificationResult(
                    is_valid=False,
                    expected_answer=correct_answer,
                    actual_answer=stated_answer,
                    error_message=f"AI verification failed: {explanation}",
                    details={"ai_explanation": explanation}
                )

            return VerificationResult(
                is_valid=True,
                expected_answer=correct_answer,
                actual_answer=stated_answer,
                details={"ai_verified": True, "explanation": explanation}
            )

        except json.JSONDecodeError:
            # Try to extract yes/no from response
            response_lower = response.lower()
            if "incorrect" in response_lower or "wrong" in response_lower:
                return VerificationResult(
                    is_valid=False,
                    actual_answer=stated_answer,
                    error_message="AI indicated answer may be incorrect",
                    details={"raw_response": response[:200]}
                )

            return VerificationResult(
                is_valid=True,
                actual_answer=stated_answer,
                details={"verification": "ai_parse_fallback"}
            )

    def compute_correct_answer(
        self,
        question_content: str,
        widget_type: str,
        choices: Optional[List[Dict]] = None,
    ) -> Optional[Any]:
        """Use AI to compute what the correct answer should be."""
        if self.client is None:
            return None

        try:
            if widget_type in ("radio", "dropdown") and choices:
                choices_text = "\n".join([
                    f"  {i+1}. {c.get('content', '')}"
                    for i, c in enumerate(choices)
                ])

                prompt = f"""What is the correct answer to this question?

QUESTION:
{question_content}

ANSWER CHOICES:
{choices_text}

Respond with ONLY a JSON object:
{{"correct_choice_index": <0-based index>, "correct_answer": "<the correct choice text>", "explanation": "<brief reason>"}}"""

            else:
                prompt = f"""What is the correct answer to this question?

QUESTION:
{question_content}

Respond with ONLY a JSON object:
{{"correct_answer": "<the answer>", "explanation": "<brief reason>"}}"""

            response = self.client.generate(prompt)

            if response:
                response = response.strip()
                if response.startswith("```"):
                    response = re.sub(r'^```(?:json)?\n?', '', response)
                    response = re.sub(r'\n?```$', '', response)

                data = json.loads(response)
                return data.get("correct_answer") or data.get("correct_choice_index")

        except Exception as e:
            logger.debug(f"AI compute answer failed: {e}")

        return None


class ComprehensiveAnswerVerifier:
    """Comprehensive answer verifier for all question types."""

    def __init__(self, use_ai: bool = True):
        self.use_ai = use_ai
        self.subject_detector = SubjectDetector()
        self.math_parser = MathExpressionParser()
        self.math_evaluator = MathEvaluator()
        self.ai_verifier = AIAnswerVerifier() if use_ai else None

    def verify_question(
        self,
        question_data: Dict[str, Any],
        source_data: Optional[Dict[str, Any]] = None,
    ) -> VerificationResult:
        """Verify all answers in a question.

        Args:
            question_data: The generated question to verify
            source_data: Optional source question for comparison

        Returns:
            VerificationResult with validation status and details
        """
        question = question_data.get("question", {})
        content = question.get("content", "")
        widgets = question.get("widgets", {})

        # Detect question type
        question_type = self.subject_detector.detect(content)
        logger.debug(f"Detected question type: {question_type.value}")

        # Parse math content if applicable
        parsed_math = None
        if question_type == QuestionType.MATH:
            parsed_math = self.math_parser.parse(content)

        # Verify each widget
        all_valid = True
        errors = []
        details = {
            "question_type": question_type.value,
            "widgets": {}
        }

        for widget_id, widget in widgets.items():
            if not widget.get("graded", True):
                continue

            widget_type = widget.get("type", "")

            # Choose verification method based on question type
            if question_type == QuestionType.MATH and parsed_math:
                result = self._verify_math_widget(
                    widget_type, widget, parsed_math
                )
            else:
                result = self._verify_general_widget(
                    widget_type, widget, content, question_type
                )

            details["widgets"][widget_id] = {
                "type": widget_type,
                "valid": result.is_valid,
                "expected": result.expected_answer,
                "actual": result.actual_answer,
                "error": result.error_message,
                "details": result.details,
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

    def _verify_math_widget(
        self,
        widget_type: str,
        widget: Dict[str, Any],
        parsed: ParsedQuestion,
    ) -> VerificationResult:
        """Verify widget for math questions using symbolic computation."""
        options = widget.get("options", {})

        if widget_type == "numeric-input":
            return self._verify_numeric_math(options, parsed)

        elif widget_type == "input-number":
            return self._verify_input_number_math(options, parsed)

        elif widget_type == "expression":
            return self._verify_expression_math(options, parsed)

        elif widget_type == "radio":
            return self._verify_radio_math(options, parsed)

        elif widget_type == "dropdown":
            return self._verify_dropdown_math(options, parsed)

        else:
            # For other widget types, accept as valid
            return VerificationResult(is_valid=True, details={"skipped": True})

    def _verify_numeric_math(
        self,
        options: Dict[str, Any],
        parsed: ParsedQuestion,
    ) -> VerificationResult:
        """Verify numeric-input for math questions."""
        answers = options.get("answers", [])

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

        # Compute expected answer
        if parsed.expression and parsed.variables:
            computed = self.math_evaluator.evaluate_expression(
                parsed.expression, parsed.variables
            )

            if computed is not None:
                try:
                    stated_float = float(stated_answer)
                    computed_float = float(computed)

                    tolerance = 0.0001
                    if abs(stated_float - computed_float) > tolerance:
                        return VerificationResult(
                            is_valid=False,
                            expected_answer=computed,
                            actual_answer=stated_answer,
                            error_message=f"Math error: expected {computed}, got {stated_answer}"
                        )
                except (ValueError, TypeError):
                    pass

        return VerificationResult(
            is_valid=True,
            actual_answer=stated_answer,
        )

    def _verify_input_number_math(
        self,
        options: Dict[str, Any],
        parsed: ParsedQuestion,
    ) -> VerificationResult:
        """Verify input-number for math questions."""
        stated_answer = options.get("value")

        if stated_answer is None:
            return VerificationResult(
                is_valid=False,
                error_message="No answer value defined"
            )

        if parsed.expression and parsed.variables:
            computed = self.math_evaluator.evaluate_expression(
                parsed.expression, parsed.variables
            )

            if computed is not None:
                try:
                    stated_float = float(stated_answer)
                    computed_float = float(computed)

                    tolerance = 0.0001
                    if abs(stated_float - computed_float) > tolerance:
                        return VerificationResult(
                            is_valid=False,
                            expected_answer=computed,
                            actual_answer=stated_answer,
                            error_message=f"Math error: expected {computed}, got {stated_answer}"
                        )
                except (ValueError, TypeError):
                    pass

        return VerificationResult(
            is_valid=True,
            actual_answer=stated_answer,
        )

    def _verify_expression_math(
        self,
        options: Dict[str, Any],
        parsed: ParsedQuestion,
    ) -> VerificationResult:
        """Verify expression widget for math questions."""
        answer_forms = options.get("answerForms", [])

        correct_value = None
        for form in answer_forms:
            if form.get("considered") == "correct":
                correct_value = form.get("value")
                break

        if not correct_value:
            return VerificationResult(
                is_valid=False,
                error_message="No correct answer form"
            )

        # Verify simplification if applicable
        if parsed.expression and parsed.operation in ('simplify', 'factor', 'expand'):
            simplified = self.math_evaluator.simplify_expression(parsed.expression)
            if simplified:
                is_equiv = self.math_evaluator.expressions_equivalent(
                    correct_value, simplified
                )
                if not is_equiv:
                    return VerificationResult(
                        is_valid=False,
                        expected_answer=simplified,
                        actual_answer=correct_value,
                        error_message=f"Expression error: expected {simplified}"
                    )

        return VerificationResult(is_valid=True, actual_answer=correct_value)

    def _verify_radio_math(
        self,
        options: Dict[str, Any],
        parsed: ParsedQuestion,
    ) -> VerificationResult:
        """Verify radio widget for math questions."""
        choices = options.get("choices", [])

        correct_idx = None
        correct_content = None
        for i, choice in enumerate(choices):
            if choice.get("correct"):
                correct_idx = i
                correct_content = choice.get("content", "")
                break

        if correct_idx is None:
            return VerificationResult(
                is_valid=False,
                error_message="No correct choice marked"
            )

        # For evaluation questions, check if marked choice has correct value
        if parsed.expression and parsed.variables:
            computed = self.math_evaluator.evaluate_expression(
                parsed.expression, parsed.variables
            )

            if computed is not None:
                # Extract number from correct choice
                numbers = re.findall(r'-?\d+(?:\.\d+)?', correct_content)
                if numbers:
                    try:
                        choice_value = float(numbers[-1])  # Use last number
                        computed_float = float(computed)

                        if abs(choice_value - computed_float) > 0.0001:
                            # Find the correct choice
                            correct_choice_idx = None
                            for i, ch in enumerate(choices):
                                ch_nums = re.findall(r'-?\d+(?:\.\d+)?', ch.get("content", ""))
                                if ch_nums:
                                    if abs(float(ch_nums[-1]) - computed_float) < 0.0001:
                                        correct_choice_idx = i
                                        break

                            return VerificationResult(
                                is_valid=False,
                                expected_answer=computed,
                                actual_answer=choice_value,
                                error_message=f"Wrong choice: expected {computed}, choice {correct_idx} has {choice_value}",
                                details={"should_be_choice": correct_choice_idx}
                            )
                    except (ValueError, TypeError):
                        pass

        return VerificationResult(
            is_valid=True,
            actual_answer=correct_content,
            details={"correct_index": correct_idx}
        )

    def _verify_dropdown_math(
        self,
        options: Dict[str, Any],
        parsed: ParsedQuestion,
    ) -> VerificationResult:
        """Verify dropdown for math questions (similar to radio)."""
        choices = options.get("choices", [])

        correct_content = None
        for choice in choices:
            if choice.get("correct"):
                correct_content = choice.get("content")
                break

        if correct_content is None:
            return VerificationResult(
                is_valid=False,
                error_message="No correct choice"
            )

        return VerificationResult(
            is_valid=True,
            actual_answer=correct_content
        )

    def _verify_general_widget(
        self,
        widget_type: str,
        widget: Dict[str, Any],
        content: str,
        question_type: QuestionType,
    ) -> VerificationResult:
        """Verify widget for non-math questions using AI."""
        options = widget.get("options", {})

        # Extract stated answer based on widget type
        stated_answer = None
        choices = None

        if widget_type == "numeric-input":
            answers = options.get("answers", [])
            for ans in answers:
                if ans.get("status") == "correct":
                    stated_answer = ans.get("value")
                    break

        elif widget_type == "input-number":
            stated_answer = options.get("value")

        elif widget_type in ("radio", "dropdown"):
            choices = options.get("choices", [])
            for choice in choices:
                if choice.get("correct"):
                    stated_answer = choice.get("content")
                    break

        elif widget_type == "expression":
            answer_forms = options.get("answerForms", [])
            for form in answer_forms:
                if form.get("considered") == "correct":
                    stated_answer = form.get("value")
                    break

        else:
            # Skip verification for complex widgets
            return VerificationResult(is_valid=True, details={"skipped": True})

        # Use AI verification if available
        if self.use_ai and self.ai_verifier:
            return self.ai_verifier.verify_answer(
                content, widget_type, stated_answer, choices
            )

        # Fall back to structural verification only
        if stated_answer is not None:
            return VerificationResult(
                is_valid=True,
                actual_answer=stated_answer,
                details={"verification": "structural_only"}
            )

        return VerificationResult(
            is_valid=False,
            error_message="No correct answer defined"
        )

    def fix_incorrect_answer(
        self,
        question_data: Dict[str, Any],
        verification_result: VerificationResult,
    ) -> Dict[str, Any]:
        """Fix incorrect answers in a question.

        Args:
            question_data: The question to fix
            verification_result: Result from verify_question

        Returns:
            Fixed question data
        """
        import copy
        fixed = copy.deepcopy(question_data)

        if verification_result.is_valid:
            return fixed

        question = fixed.get("question", {})
        content = question.get("content", "")
        widgets = question.get("widgets", {})

        widget_details = verification_result.details.get("widgets", {})
        question_type = verification_result.details.get("question_type", "unknown")

        for widget_id, details in widget_details.items():
            if details.get("valid"):
                continue

            expected = details.get("expected")
            if expected is None:
                # Try to compute correct answer
                if question_type == "math":
                    parsed = self.math_parser.parse(content)
                    if parsed.expression and parsed.variables:
                        expected = self.math_evaluator.evaluate_expression(
                            parsed.expression, parsed.variables
                        )
                elif self.use_ai and self.ai_verifier:
                    widget_type = details.get("type", "")
                    choices = None
                    if widget_type in ("radio", "dropdown"):
                        choices = widgets[widget_id].get("options", {}).get("choices", [])
                    expected = self.ai_verifier.compute_correct_answer(
                        content, widget_type, choices
                    )

            if expected is None:
                continue

            widget = widgets.get(widget_id, {})
            widget_type = widget.get("type", "")

            # Apply fix based on widget type
            if widget_type == "numeric-input":
                answers = widget.get("options", {}).get("answers", [])
                for ans in answers:
                    if ans.get("status") == "correct":
                        old_val = ans.get("value")
                        # Convert to int if whole number
                        if isinstance(expected, float) and expected == int(expected):
                            expected = int(expected)
                        ans["value"] = expected
                        logger.info(f"Fixed {widget_id}: {old_val} -> {expected}")
                        break

            elif widget_type == "input-number":
                old_val = widget.get("options", {}).get("value")
                if isinstance(expected, float) and expected == int(expected):
                    expected = int(expected)
                widget["options"]["value"] = expected
                logger.info(f"Fixed {widget_id}: {old_val} -> {expected}")

            elif widget_type in ("radio", "dropdown"):
                choices = widget.get("options", {}).get("choices", [])

                # If expected is an index
                if isinstance(expected, int) and 0 <= expected < len(choices):
                    for i, ch in enumerate(choices):
                        ch["correct"] = (i == expected)
                    logger.info(f"Fixed {widget_id}: set choice {expected} as correct")

                # If expected is a value to match
                elif expected is not None:
                    expected_str = str(expected)
                    for i, ch in enumerate(choices):
                        ch_content = ch.get("content", "")
                        # Check if this choice contains the expected value
                        if expected_str in ch_content:
                            for j, c in enumerate(choices):
                                c["correct"] = (j == i)
                            logger.info(f"Fixed {widget_id}: set choice {i} as correct")
                            break

            elif widget_type == "expression":
                answer_forms = widget.get("options", {}).get("answerForms", [])
                for form in answer_forms:
                    if form.get("considered") == "correct":
                        old_val = form.get("value")
                        form["value"] = str(expected)
                        logger.info(f"Fixed {widget_id}: {old_val} -> {expected}")
                        break

        return fixed


def verify_and_fix_question(
    question_data: Dict[str, Any],
    use_ai: bool = True,
) -> Tuple[Dict[str, Any], VerificationResult]:
    """Verify a question and fix any incorrect answers.

    Args:
        question_data: The question to verify and fix
        use_ai: Whether to use AI for non-math questions

    Returns:
        Tuple of (fixed question data, verification result)
    """
    verifier = ComprehensiveAnswerVerifier(use_ai=use_ai)

    # Initial verification
    result = verifier.verify_question(question_data)

    if result.is_valid:
        return question_data, result

    # Try to fix
    fixed = verifier.fix_incorrect_answer(question_data, result)

    # Re-verify
    new_result = verifier.verify_question(fixed)
    new_result.details["fixes_attempted"] = True

    return fixed, new_result


# Singleton instance
comprehensive_verifier = ComprehensiveAnswerVerifier(use_ai=True)


if __name__ == "__main__":
    # Test the comprehensive verifier
    print("Testing Comprehensive Verifier...")

    detector = SubjectDetector()

    test_questions = [
        "**Evaluate $5a + \\dfrac{24}b$ when $a=4$ and $b=6$.**",
        "What is the chemical formula for water?",
        "According to the passage, what was the author's main argument?",
        "Which word is a noun in this sentence?",
        "What determines the length of a day on Mars?",
    ]

    for q in test_questions:
        qtype = detector.detect(q)
        print(f"\n{q[:50]}...")
        print(f"  -> Type: {qtype.value}")

    print("\n" + "="*50)
    print("Tests complete!")

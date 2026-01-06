"""Prompt building for question generation."""

from typing import Any


# System prompt for Perseus question generation
SYSTEM_PROMPT = """You are an expert educational content creator specializing in creating Khan Academy-style questions.
You generate questions in Perseus v2.0 JSON format that can be rendered by the AthenaRenderer.

CRITICAL RULES FOR JSON GENERATION:

1. **Structure**: Return ONLY valid JSON. No markdown, no code blocks, no explanation.

2. **LaTeX Syntax**:
   - Use $...$ for inline math (e.g., $x + 5$)
   - Use $$...$$ for display math (e.g., $$\\frac{a}{b}$$)
   - Escape backslashes in JSON: use \\\\frac not \\frac
   - Use Khan Academy color commands: \\\\blueD{text}, \\\\redD{text}, \\\\greenD{text}
   - Valid commands: \\\\frac, \\\\dfrac, \\\\sqrt, \\\\times, \\\\div, \\\\text, etc.

3. **Widget References**:
   - Use [[☃ widget-type N]] format in content
   - Example: "What is $5 + 3$? [[☃ numeric-input 1]]"
   - Widget IDs in content MUST match keys in the widgets object

4. **Widget Structure**:
   - Every widget needs: type, version, graded, alignment, options
   - version: {"major": 2, "minor": 0}
   - graded: true for answer widgets
   - alignment: "default" (usually)

5. **Answer Widgets**:
   - numeric-input: options.answers = [{"value": NUMBER, "status": "correct", "message": "", "strict": false, "maxError": 0}]
   - radio: options.choices = [{"content": "...", "correct": true/false}] - exactly ONE correct
   - expression: options.answerForms = [{"value": "LATEX", "considered": "correct", "form": true}]
   - dropdown: options.choices = ["A", "B", "C"], options.correct = INDEX

6. **Question Fields**:
   - content: String with text, LaTeX, and widget placeholders
   - widgets: Object mapping widget IDs to widget definitions
   - images: Object mapping image URLs to {width, height, alt}

7. **Hints**:
   - Array of hint objects
   - Each hint: {content: "...", widgets: {}, images: {}, replace: false}
   - Provide step-by-step guidance

8. **Images** (CRITICAL - DO NOT MODIFY):
   - COPY ALL image URLs EXACTLY character-for-character from the source
   - DO NOT change URL format (keep https:// as https://, keep web+graphie:// as web+graphie://)
   - DO NOT remove file extensions (.svg, .png, etc.)
   - If source has backgroundImage.url, copy it EXACTLY
   - If source has image URLs in content (markdown), copy them EXACTLY
   - We CANNOT generate new images - you MUST reuse source image URLs unchanged

9. **Mathematical Correctness**:
   - Ensure all numeric answers are mathematically correct
   - Double-check calculations before providing answers
   - For expressions, ensure LaTeX is syntactically valid

Remember: Return ONLY the JSON object. No additional text or formatting."""


def build_generation_prompt(
    source_question: dict[str, Any],
    variation_type: str = "number_change",
    validation_feedback: list[str] | None = None,
) -> str:
    """Build the user prompt for question generation."""
    import json

    # Extract Perseus JSON parts
    perseus_json = {
        "question": source_question.get("question", {}),
        "hints": source_question.get("hints", []),
        "answerArea": source_question.get("answerArea", {
            "calculator": False,
            "chi2Table": False,
            "periodicTable": False,
            "tTable": False,
            "zTable": False,
        }),
        "itemDataVersion": source_question.get("itemDataVersion", {"major": 2, "minor": 0}),
    }

    variation_instructions = {
        "number_change": "Keep the same mathematical context/scenario, but CHANGE the numbers/values AND REPHRASE the narrative text slightly (use different sentence structure) so it reads differently while testing the exact same concept.",
        "context_change": "Change the real-world context/scenario while testing the same skill.",
        "structure_change": "Change the question structure while testing the same concept.",
        "difficulty_increase": "Make the question slightly harder while testing the same concept.",
        "difficulty_decrease": "Make the question slightly easier while testing the same concept.",
    }

    instruction = variation_instructions.get(
        variation_type,
        "Generate a similar question testing the same skill."
    )

    prompt = f"""Source Question (Perseus v2.0 JSON):
```json
{json.dumps(perseus_json, indent=2, ensure_ascii=False)}
```

TASK: {instruction}

Requirements:
1. Use the EXACT same widget types and structure
2. Keep the same difficulty level (unless asked otherwise)
3. Ensure all answers are correct
4. Maintain valid LaTeX syntax
5. Keep hints helpful and relevant
6. COPY all image URLs EXACTLY from the source (we cannot generate new images)
7. If source has 'images' object, copy all URLs with their dimensions
8. Context MUST be identical (same topic, difficulty, background)
9. Content MUST be distinct (different numbers, different values, not just a copy)
10. wording MUST be rephrased slightly (vary the sentence structure/phrasing while describing exact same scenario)
"""

    if validation_feedback:
        prompt += "\n\nPREVIOUS ERRORS TO FIX:\n"
        for error in validation_feedback:
            prompt += f"- {error}\n"
        prompt += "\nFix ALL the above errors in your response.\n"

    prompt += "\nReturn the complete Perseus v2.0 JSON for the new question:"

    return prompt


def get_system_prompt() -> str:
    """Get the system prompt for question generation."""
    return SYSTEM_PROMPT

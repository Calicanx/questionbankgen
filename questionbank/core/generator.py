"""Core question generator using LLM with intelligent generation capabilities."""

import logging
import os
import re
from datetime import datetime
from typing import Any, Optional

from questionbank.config import config
from questionbank.llm.gemini_client import get_gemini_client, GENERATED_IMAGES_DIR
from questionbank.mongodb.repository import question_repo
from questionbank.validation.pipeline import ValidationPipeline
from questionbank.core.prompt_builder import build_generation_prompt, get_system_prompt

# Import intelligent generation modules
from questionbank.intelligence.constraint_extractor import ConstraintExtractor, QuestionConstraints
from questionbank.intelligence.coherence_validator import CoherenceValidator
from questionbank.intelligence.smart_generator import SmartQuestionGenerator, GenerationConfig
from questionbank.intelligence.validation_pipeline import IntelligentValidationPipeline
from questionbank.intelligence.image_generator import ImageGenerator

# Import solution generator for step-by-step solutions
from questionbank.utils.solution_generator import (
    generate_solution,
    extract_equation_from_question,
    solution_to_dict,
)

# Import comprehensive answer verifier
from questionbank.validation.comprehensive_verifier import (
    verify_and_fix_question,
    ComprehensiveAnswerVerifier,
)

logger = logging.getLogger(__name__)

from questionbank.utils.gcs import get_gcs_client

# Base URL for serving generated images (set this based on your deployment)
GENERATED_IMAGES_BASE_URL = os.getenv(
    "GENERATED_IMAGES_BASE_URL",
    "http://localhost:8001/static/generated_images"
)



def _download_and_convert_image(url: str) -> Optional[str]:
    """Download a remote image and save it locally or to GCS.
    
    Args:
        url: Remote image URL (can be web+graphie:// or https://)
        
    Returns:
        Public URL (GCS/Local) if successful, None otherwise
    """
    import requests
    import uuid
    from questionbank.utils.gcs import get_gcs_client
    
    try:
        # Convert web+graphie:// URLs to https://
        download_url = url
        if url.startswith("web+graphie://"):
            download_url = url.replace("web+graphie://", "https://") + ".svg"
        
        # Download with browser-like headers
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
            'Accept': 'image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8',
            'Referer': 'https://www.khanacademy.org/',
        }
        
        response = requests.get(download_url, headers=headers, timeout=15)
        if response.status_code != 200:
            logger.warning(f"Failed to download image: {download_url[:60]}... (status {response.status_code})")
            return None
        
        # Determine file type
        content_type = response.headers.get('Content-Type', '')
        is_svg = 'svg' in content_type or download_url.endswith('.svg')
        
        # Generate unique filename
        file_ext = '.png' if is_svg else '.jpg'
        # Just filename, no path for GCS
        filename = f"downloaded_{uuid.uuid4().hex[:12]}{file_ext}"
        
        final_image_bytes = response.content
        mime_type = "image/png" if is_svg else "image/jpeg"

        if is_svg:
            # Try to convert SVG to PNG using Wand (ImageMagick)
            try:
                from wand.image import Image as WandImage
                with WandImage(blob=response.content, format='svg') as img:
                    img.format = 'png'
                    img.background_color = 'white'
                    img.alpha_channel = 'remove'
                    final_image_bytes = img.make_blob()
                    mime_type = "image/png"
                logger.info(f"Converted SVG to PNG")
            except Exception as e:
                logger.warning(f"SVG conversion failed: {e}, saving as SVG")
                file_ext = '.svg'
                filename = f"downloaded_{uuid.uuid4().hex[:12]}{file_ext}"
                mime_type = "image/svg+xml"

        # Try to upload to GCS first
        try:
            gcs_client = get_gcs_client()
            blob_name = f"generated_images/{filename}"
            gcs_url = gcs_client.upload_bytes(final_image_bytes, blob_name, content_type=mime_type)
            if gcs_url:
                logger.info(f"Uploaded downloaded image to GCS: {gcs_url}")
                return gcs_url
        except Exception as e:
            logger.error(f"Failed to upload to GCS: {e}")

        # Fallback to local save
        save_path = os.path.join(GENERATED_IMAGES_DIR, filename)
        os.makedirs(GENERATED_IMAGES_DIR, exist_ok=True)
            
        with open(save_path, 'wb') as f:
            f.write(final_image_bytes)
        logger.info(f"Downloaded image locally: {save_path}")
        
        # Return local URL
        return f"{GENERATED_IMAGES_BASE_URL}/{filename}"
        
    except Exception as e:
        logger.error(f"Error downloading image from {url[:60]}...: {e}")
        return None


class QuestionGenerator:
    """Generates new questions using LLM based on source questions.

    Supports two generation modes:
    1. Standard mode: Uses LLM with basic validation
    2. Intelligent mode: Uses semantic constraint extraction, coherence validation,
       and smart generation strategies for better results
    """

    def __init__(self, max_retries: int = 3, use_intelligent_mode: bool = True) -> None:
        self.max_retries = max_retries
        self.gemini = get_gemini_client()
        self.use_intelligent_mode = use_intelligent_mode

        # Standard validation pipeline
        self.validation_pipeline = ValidationPipeline(
            check_schema=config.validation.strict_mode,
            check_latex=config.validation.check_latex,
            verify_answers=config.validation.verify_answers,
        )

        # Intelligent generation components
        self.constraint_extractor = ConstraintExtractor()
        self.coherence_validator = CoherenceValidator()
        self.smart_generator = SmartQuestionGenerator(GenerationConfig(
            max_retries=max_retries,
            preserve_image_gender=True,
            generate_new_images=True,
            validate_coherence=True,
        ))
        self.intelligent_pipeline = IntelligentValidationPipeline(attempt_fixes=True)
        self.image_generator = ImageGenerator()

    def generate_from_source(
        self,
        source_question: dict[str, Any],
        variation_type: str = "number_change",
        save_to_db: bool = True,
        generated_id: Optional[str] = None,
    ) -> Optional[dict[str, Any]]:
        """Generate a new question from a source question."""
        source_id = str(source_question.get("_id", "unknown"))
        logger.info(f"Generating question from source: {source_id}")

        validation_feedback: list[str] = []
        attempt = 0

        while attempt < self.max_retries:
            attempt += 1
            logger.info(f"Generation attempt {attempt}/{self.max_retries}")

            # Build prompts
            system_prompt = get_system_prompt()
            user_prompt = build_generation_prompt(
                source_question,
                variation_type=variation_type,
                validation_feedback=validation_feedback if validation_feedback else None,
            )

            # Generate using Gemini
            generated_json = self.gemini.generate_question_json(
                source_question,
                system_prompt,
                validation_feedback=validation_feedback if validation_feedback else None,
                prompt=user_prompt,
            )

            if not generated_json:
                logger.warning(f"Attempt {attempt}: Failed to generate valid JSON")
                validation_feedback.append("Failed to generate valid JSON. Ensure response is pure JSON.")
                continue

            # Post-process: Copy image URLs from source to ensure they're preserved
            generated_json = self._copy_image_urls_from_source(source_question, generated_json)

            # Generate new images for radio choices with place value blocks
            # (copying source images won't work if the numbers have changed)
            generated_json = self._generate_radio_choice_images(source_question, generated_json)

            # Validate the generated question
            validation_result = self.validation_pipeline.validate(generated_json)

            if validation_result.is_valid:
                logger.info(f"Generated valid question after {attempt} attempt(s)")

                # Generate step-by-step solution for the generated question
                generated_json = self._add_solution_steps(generated_json)

                # Verify and fix answers using comprehensive verifier
                generated_json, answer_result = verify_and_fix_question(
                    generated_json, use_ai=True
                )

                if not answer_result.is_valid:
                    logger.warning(f"Answer verification issues: {answer_result.error_message}")
                    # Add to validation feedback for retry if fixes failed
                    if answer_result.error_message:
                        validation_feedback.append(f"Answer error: {answer_result.error_message}")
                        continue
                else:
                    if answer_result.details.get("fixes_attempted"):
                        logger.info("Answers were auto-corrected")

                # Optionally save to database
                if save_to_db:
                    self._save_generated_question(
                        source_id=source_id,
                        generated_json=generated_json,
                        attempt_count=attempt,
                        generated_id=generated_id,
                    )

                return generated_json
            else:
                logger.warning(f"Attempt {attempt}: Validation failed")
                logger.debug(f"Errors: {validation_result.errors}")
                validation_feedback = validation_result.errors

        logger.error(f"Failed to generate valid question after {self.max_retries} attempts")
        return None

    def generate_from_id(
        self,
        question_id: str,
        variation_type: str = "number_change",
        save_to_db: bool = True,
    ) -> Optional[dict[str, Any]]:
        """Generate a new question from a source question ID."""
        source_question = question_repo.get_question_by_id(question_id)

        if not source_question:
            logger.error(f"Source question not found: {question_id}")
            return None

        return self.generate_from_source(
            source_question,
            variation_type=variation_type,
            save_to_db=save_to_db,
        )

    def generate_random(
        self,
        widget_type: Optional[str] = None,
        skill_prefix: Optional[str] = None,
        variation_type: str = "number_change",
        save_to_db: bool = True,
    ) -> Optional[dict[str, Any]]:
        """Generate a new question from a random source question."""
        source_question = question_repo.get_random_question(
            widget_type=widget_type,
            skill_prefix=skill_prefix,
        )

        if not source_question:
            logger.error("No source questions found matching criteria")
            return None

        return self.generate_from_source(
            source_question,
            variation_type=variation_type,
            save_to_db=save_to_db,
        )

    def generate_batch(
        self,
        count: int = 5,
        widget_type: Optional[str] = None,
        skill_prefix: Optional[str] = None,
        variation_type: str = "number_change",
        save_to_db: bool = True,
    ) -> list[dict[str, Any]]:
        """Generate multiple questions."""
        results: list[dict[str, Any]] = []

        source_questions = question_repo.get_questions(
            widget_type=widget_type,
            skill_prefix=skill_prefix,
            limit=count,
        )

        logger.info(f"Generating {len(source_questions)} questions")

        for i, source in enumerate(source_questions):
            logger.info(f"Processing question {i + 1}/{len(source_questions)}")

            generated = self.generate_from_source(
                source,
                variation_type=variation_type,
                save_to_db=save_to_db,
            )

            if generated:
                results.append(generated)
            else:
                logger.warning(f"Failed to generate from source {source.get('_id')}")

        logger.info(f"Successfully generated {len(results)}/{len(source_questions)} questions")
        return results

    def _copy_image_urls_from_source(
        self,
        source: dict[str, Any],
        generated: dict[str, Any],
    ) -> dict[str, Any]:
        """Copy image URLs from source question to generated question.

        This ensures image URLs are preserved exactly, since the AI often
        modifies or corrupts them during generation.

        Handles both:
        1. Image widgets (backgroundImage URLs)
        2. Markdown-embedded images (![](web+graphie://...))
        """
        import copy
        result = copy.deepcopy(generated)

        source_question = source.get("question", {})
        gen_question = result.get("question", {})

        # Copy images dict if source has it (required for markdown images)
        if source_question.get("images"):
            gen_question["images"] = copy.deepcopy(source_question["images"])
            logger.debug(f"Copied images dict with {len(source_question['images'])} entries")

        # Handle markdown-embedded images (![](web+graphie://...))
        # Pattern: ![alt](web+graphie://...) or ![](web+graphie://...)
        source_content = source_question.get("content", "")
        gen_content = gen_question.get("content", "")

        # Find all graphie URLs in source
        graphie_pattern = r'!\[([^\]]*)\]\((web\+graphie://[^)]+)\)'
        source_images = re.findall(graphie_pattern, source_content)
        source_urls = {url for _, url in source_images}

        if source_images:
            # First, remove any FAKE graphie URLs that AI generated (not in source)
            gen_images = re.findall(graphie_pattern, gen_content)
            for alt, url in gen_images:
                if url not in source_urls:
                    # This is a fake URL generated by AI - remove it
                    fake_img = f"![{alt}]({url})"
                    gen_content = gen_content.replace(fake_img, "")
                    logger.info(f"Removed fake graphie URL: {url[:60]}...")

            # Clean up any double newlines from removal
            while "\n\n\n" in gen_content:
                gen_content = gen_content.replace("\n\n\n", "\n\n")

            # Download source images and replace with local/GCS URLs
            local_images_dict = {}
            for alt, url in source_images:
                # Download the image locally or to GCS
                result_path_or_url = _download_and_convert_image(url)
                
                if result_path_or_url:
                    # Check if it's already a URL (GCS)
                    if result_path_or_url.startswith("http"):
                        final_url = result_path_or_url
                    else:
                        # It's a local path, convert to local URL (fallback)
                        filename = os.path.basename(result_path_or_url)
                        final_url = f"{GENERATED_IMAGES_BASE_URL}/{filename}"
                    
                    # Add to images dict
                    local_images_dict[final_url] = {"width": 400, "height": 300}
                    
                    # Insert image markdown into content
                    markdown_img = f"![{alt}]({final_url})"
                    if final_url not in gen_content:
                        widget_match = re.search(r'\[\[☃[^\]]+\]\]', gen_content)
                        if widget_match:
                            insert_pos = widget_match.start()
                            gen_content = (
                                gen_content[:insert_pos].rstrip() +
                                "\n\n" + markdown_img + "\n\n" +
                                gen_content[insert_pos:]
                            )
                        else:
                            gen_content = gen_content.rstrip() + "\n\n" + markdown_img
                    logger.info(f"Downloaded and replaced image: {url[:60]}... -> {final_url}")
                else:
                    # Fallback: use original URL (will still fail, but at least we tried)
                    logger.warning(f"Failed to download image, keeping original URL: {url[:60]}...")
                    markdown_img = f"![{alt}]({url})"
                    if url not in gen_content:
                        widget_match = re.search(r'\[\[☃[^\]]+\]\]', gen_content)
                        if widget_match:
                            insert_pos = widget_match.start()
                            gen_content = (
                                gen_content[:insert_pos].rstrip() +
                                "\n\n" + markdown_img + "\n\n" +
                                gen_content[insert_pos:]
                            )
                        else:
                            gen_content = gen_content.rstrip() + "\n\n" + markdown_img

            # Update images dict with local URLs
            if local_images_dict:
                gen_question["images"] = local_images_dict

            gen_question["content"] = gen_content.strip()

        # Copy image widget backgroundImage URLs - more robust matching
        source_widgets = source_question.get("widgets", {})
        gen_widgets = gen_question.get("widgets", {})

        # Collect all source image widgets
        source_image_widgets = [
            (wid, w) for wid, w in source_widgets.items()
            if w.get("type") == "image" and w.get("options", {}).get("backgroundImage")
        ]

        # Collect all generated image widgets
        gen_image_widgets = [
            (wid, w) for wid, w in gen_widgets.items()
            if w.get("type") == "image"
        ]

        # Download and replace backgroundImage URLs from source to generated image widgets
        for i, (gen_wid, gen_widget) in enumerate(gen_image_widgets):
            # First try exact ID match
            matched_source = None
            if gen_wid in source_widgets and source_widgets[gen_wid].get("type") == "image":
                matched_source = source_widgets[gen_wid]
            # Then try position match
            elif i < len(source_image_widgets):
                matched_source = source_image_widgets[i][1]

            if matched_source:
                source_bg = matched_source.get("options", {}).get("backgroundImage")
                if source_bg and isinstance(source_bg, dict):
                    source_url = source_bg.get("url", "")
                    
                    # Download the image if it's a remote URL
                    if source_url and (source_url.startswith("web+graphie://") or source_url.startswith("https://")):
                        result_path_or_url = _download_and_convert_image(source_url)
                        
                        if result_path_or_url:
                            # Check if it's already a URL (GCS)
                            if result_path_or_url.startswith("http"):
                                final_url = result_path_or_url
                            else:
                                # It's a local path
                                filename = os.path.basename(result_path_or_url)
                                final_url = f"{GENERATED_IMAGES_BASE_URL}/{filename}"
                            
                            # Update widget with local URL
                            if "options" not in gen_widgets[gen_wid]:
                                gen_widgets[gen_wid]["options"] = {}
                            gen_widgets[gen_wid]["options"]["backgroundImage"] = {
                                "url": final_url,
                                "width": source_bg.get("width", 400),
                                "height": source_bg.get("height", 300)
                            }
                            logger.info(f"Downloaded and replaced backgroundImage for {gen_wid}: {source_url[:60]}... -> {final_url}")
                        else:
                            # Fallback: copy original URL
                            if "options" not in gen_widgets[gen_wid]:
                                gen_widgets[gen_wid]["options"] = {}
                            gen_widgets[gen_wid]["options"]["backgroundImage"] = copy.deepcopy(source_bg)
                    else:
                        # Non-remote URL, copy as-is
                        if "options" not in gen_widgets[gen_wid]:
                            gen_widgets[gen_wid]["options"] = {}
                        gen_widgets[gen_wid]["options"]["backgroundImage"] = copy.deepcopy(source_bg)

        # For questions where AI generates image widgets but source has none (or vice versa),
        # we need to ensure image widgets have valid URLs or are removed
        for gen_wid, gen_widget in list(gen_widgets.items()):
            if gen_widget.get("type") == "image":
                bg = gen_widget.get("options", {}).get("backgroundImage", {})
                url = bg.get("url", "") if isinstance(bg, dict) else ""
                # Check if URL is valid (not a fake AI-generated URL)
                if url and not url.startswith(("https://", "http://", "web+graphie://")):
                    # Invalid URL - copy from first source image or remove
                    if source_image_widgets:
                        gen_widgets[gen_wid]["options"]["backgroundImage"] = copy.deepcopy(
                            source_image_widgets[0][1].get("options", {}).get("backgroundImage")
                        )
                        logger.info(f"Fixed invalid URL in {gen_wid}")

        # Handle radio/dropdown widgets with images in choice content
        # These widgets can have embedded images in their choice content (like molecule structures)
        for widget_id, gen_widget in gen_widgets.items():
            widget_type = gen_widget.get("type", "")
            if widget_type in ("radio", "dropdown"):
                # Get source widget to copy choice content with images
                source_widget = source_widgets.get(widget_id, {})
                if source_widget.get("type") == widget_type:
                    source_choices = source_widget.get("options", {}).get("choices", [])
                    gen_choices = gen_widget.get("options", {}).get("choices", [])

                    # Copy images from source choices to generated choices
                    for i, (src_choice, gen_choice) in enumerate(zip(source_choices, gen_choices)):
                        src_content = src_choice.get("content", "")
                        # If source has images (graphie URLs), preserve them
                        if "web+graphie://" in src_content or "https://" in src_content:
                            # Keep the source content with images
                            gen_choice["content"] = src_content
                            logger.info(f"Preserved image in {widget_type} choice {i}")
                        # Also copy any images dict from source choices
                        if "images" in src_choice:
                            gen_choice["images"] = copy.deepcopy(src_choice["images"])

        # Also check for image URLs in content that might be in markdown format
        # and copy any graphie URLs from source hints to generated hints
        source_hints = source.get("hints", [])
        gen_hints = result.get("hints", [])

        for i, (src_hint, gen_hint) in enumerate(zip(source_hints, gen_hints)):
            # Copy images from hints
            if src_hint.get("images"):
                gen_hint["images"] = copy.deepcopy(src_hint["images"])

            # Copy hint widget images
            src_hint_widgets = src_hint.get("widgets", {})
            gen_hint_widgets = gen_hint.get("widgets", {})

            for widget_id, src_widget in src_hint_widgets.items():
                if src_widget.get("type") == "image" and widget_id in gen_hint_widgets:
                    src_bg = src_widget.get("options", {}).get("backgroundImage")
                    if src_bg:
                        if "options" not in gen_hint_widgets[widget_id]:
                            gen_hint_widgets[widget_id]["options"] = {}
                        gen_hint_widgets[widget_id]["options"]["backgroundImage"] = copy.deepcopy(src_bg)

        result["question"] = gen_question
        result["hints"] = gen_hints

        return result

    def _add_solution_steps(self, generated_json: dict[str, Any]) -> dict[str, Any]:
        """Add step-by-step solution to the generated question.

        Uses SymPy for accurate symbolic math solving (Photomath approach).
        """
        try:
            # Get the question content
            question_content = generated_json.get("question", {}).get("content", "")

            # Try to extract an equation from the question
            equation = extract_equation_from_question(question_content)

            if equation:
                logger.info(f"Extracted equation for solution: {equation}")
                solution = generate_solution(equation)

                if solution:
                    generated_json["solution"] = solution_to_dict(solution)
                    logger.info(f"Generated {len(solution.steps)} solution steps")
                else:
                    logger.debug("Could not generate solution for this equation")
            else:
                # For non-equation questions, try to generate from numeric-input answer
                widgets = generated_json.get("question", {}).get("widgets", {})
                for widget_id, widget in widgets.items():
                    if widget.get("type") == "numeric-input":
                        answers = widget.get("options", {}).get("answers", [])
                        if answers:
                            correct_value = answers[0].get("value")
                            if correct_value is not None:
                                # Create a simple solution showing the answer
                                generated_json["solution"] = {
                                    "steps": [
                                        {"step": 1, "content": question_content[:200], "action": "Read the problem"},
                                        {"step": 2, "content": f"The answer is ${correct_value}$", "action": "Calculate"},
                                    ],
                                    "final_answer": f"${correct_value}$",
                                    "problem_type": "numeric"
                                }
                                break

        except Exception as e:
            logger.warning(f"Error generating solution steps: {e}")
            # Don't fail the whole generation if solution fails

        return generated_json

    def _generate_hints_if_missing(
        self,
        generated_json: dict[str, Any],
        source_question: dict[str, Any],
    ) -> dict[str, Any]:
        """Generate hints if the source question doesn't have any.

        Uses AI to create helpful hints for the generated question.
        """
        # Check if generated already has hints
        gen_hints = generated_json.get("hints", [])
        source_hints = source_question.get("hints", [])

        # If we already have hints, no need to generate
        if gen_hints and len(gen_hints) > 0:
            return generated_json

        # If source had hints but they weren't copied, something is wrong
        if source_hints and len(source_hints) > 0:
            logger.warning("Source had hints but they weren't copied to generated question")
            return generated_json

        # Source has no hints - generate them using AI
        try:
            question_content = generated_json.get("question", {}).get("content", "")
            widgets = generated_json.get("question", {}).get("widgets", {})

            # Find the correct answer for context
            correct_answer = None
            for widget_id, widget in widgets.items():
                if widget.get("type") == "radio":
                    choices = widget.get("options", {}).get("choices", [])
                    for choice in choices:
                        if choice.get("correct"):
                            correct_answer = choice.get("content", "")
                            break
                elif widget.get("type") == "numeric-input":
                    answers = widget.get("options", {}).get("answers", [])
                    if answers:
                        correct_answer = str(answers[0].get("value", ""))

            hint_prompt = f"""Generate 2-3 helpful hints for this question. Each hint should guide the student step-by-step toward the answer WITHOUT giving it away directly.

Question:
{question_content}

{"Correct answer: " + correct_answer if correct_answer else ""}

Return a JSON array of hint objects in this EXACT format:
[
  {{"content": "First hint text here with $LaTeX$ if needed", "widgets": {{}}, "images": {{}}, "replace": false}},
  {{"content": "Second hint text here", "widgets": {{}}, "images": {{}}, "replace": false}}
]

Rules:
1. Each hint should reveal progressively more information
2. First hint: General strategy or approach
3. Second hint: More specific guidance
4. Third hint (if applicable): Almost direct pointer without giving answer
5. Use $...$ for math expressions
6. Return ONLY the JSON array, no markdown or explanation"""

            # Generate hints using Gemini
            import json
            response = self.gemini.generate(hint_prompt)

            if response:
                # Clean up response - remove markdown if present
                cleaned = response.strip()
                if cleaned.startswith("```"):
                    cleaned = re.sub(r'^```(?:json)?\n?', '', cleaned)
                    cleaned = re.sub(r'\n?```$', '', cleaned)

                hints = json.loads(cleaned)

                if isinstance(hints, list) and len(hints) > 0:
                    generated_json["hints"] = hints
                    logger.info(f"Generated {len(hints)} hints for question without source hints")

        except Exception as e:
            logger.warning(f"Error generating hints: {e}")
            # Don't fail generation if hint creation fails

        return generated_json

    def _save_generated_question(
        self,
        source_id: str,
        generated_json: dict[str, Any],
        attempt_count: int,
        generated_id: Optional[str] = None,
    ) -> Optional[str]:
        """Save a generated question to the database."""
        metadata = {
            "llm_model": config.gemini.model,
            "validation_status": "valid",
            "attempt_count": attempt_count,
            "generated_at": datetime.utcnow(),
        }

        if generated_id:
            # Update existing record
            success = question_repo.update_generated_question(
                generated_id=generated_id,
                perseus_json=generated_json,
                metadata=metadata,
            )
            inserted_id = generated_id if success else None
        else:
            # Create new record
            inserted_id = question_repo.insert_generated_question(
                source_question_id=source_id,
                perseus_json=generated_json,
                metadata=metadata,
            )

        if inserted_id:
            logger.info(f"Saved generated question: {inserted_id}")
        else:
            logger.warning("Failed to save generated question to database")

        return inserted_id

    def _generate_new_markdown_images(
        self,
        source: dict[str, Any],
        generated: dict[str, Any],
    ) -> dict[str, Any]:
        """Generate NEW images for markdown-embedded images.

        Uses programmatic generation for mathematical diagrams (place value blocks,
        counting objects, fractions) and falls back to Gemini for other types.
        """
        import copy
        result = copy.deepcopy(generated)

        source_question = source.get("question", {})
        gen_question = result.get("question", {})

        source_content = source_question.get("content", "")
        gen_content = gen_question.get("content", "")

        # Find markdown images in source (both graphie and standard)
        image_pattern = r'!\[([^\]]*)\]\(([^)]+)\)'
        source_images = re.findall(image_pattern, source_content)

        if not source_images:
            return result

        # Remove any old image URLs from generated content to avoid duplicates
        gen_images = re.findall(image_pattern, gen_content)
        source_urls = {url for _, url in source_images}

        for alt, url in gen_images:
            if url not in source_urls:
                fake_img = f"![{alt}]({url})"
                gen_content = gen_content.replace(fake_img, "")

        # Clean up newlines
        while "\n\n\n" in gen_content:
            gen_content = gen_content.replace("\n\n\n", "\n\n")

        # Generate new image based on question content
        question_text = gen_content.split("[[")[0].strip()
        question_text = re.sub(r'\*\*|\!\[\]\([^\)]+\)', '', question_text).strip()
        question_lower = question_text.lower()

        image_path = None

        # Try programmatic generation first for mathematical diagrams
        if "place value" in question_lower or "blocks" in question_lower:
            # Extract the answer value from the generated question's widget
            answer_value = self._extract_numeric_answer(generated)
            if answer_value:
                logger.info(f"Using programmatic generator for place value blocks: {answer_value}")
                from questionbank.utils.image_generator import generate_place_value_blocks
                image_path = generate_place_value_blocks(answer_value)

        elif any(word in question_lower for word in ["count", "how many"]):
            # Extract the count from the answer
            answer_value = self._extract_numeric_answer(generated)
            if answer_value:
                logger.info(f"Using programmatic generator for counting: {answer_value}")
                from questionbank.utils.image_generator import generate_counting_objects
                image_path = generate_counting_objects(answer_value)

        elif "fraction" in question_lower:
            # Try to extract fraction from question or answer
            fraction_match = re.search(r'(\d+)\s*/\s*(\d+)', question_text)
            if fraction_match:
                num, denom = int(fraction_match.group(1)), int(fraction_match.group(2))
                logger.info(f"Using programmatic generator for fraction: {num}/{denom}")
                from questionbank.utils.image_generator import generate_fraction_visual
                image_path = generate_fraction_visual(num, denom)

        # Fallback to Gemini for other question types
        if image_path is None:
            image_prompt = self._build_image_prompt_from_question(question_text, source_content)
            if image_prompt:
                logger.info(f"Using Gemini for image generation: {image_prompt[:80]}...")
                # This returns GCS URL or local path
                image_path = self.gemini.generate_educational_image(
                    description=image_prompt,
                    context="educational math diagram for Khan Academy style question",
                    style="clean, simple, colorful educational illustration with clear visuals",
                )

        if image_path:
            # Get the URL for serving
            image_url = None
            
            # If it's already a URL (from Gemini GCS upload), use it
            if image_path.startswith("http"):
                image_url = image_path
            else:
                # It's a local path (from programmatic generator), upload to GCS
                try:
                    import mimetypes
                    from questionbank.utils.gcs import get_gcs_client
                    from questionbank.config import config
                    gcs_client = get_gcs_client()
                    filename = os.path.basename(image_path)
                    blob_name = f"generated_images/{filename}"
                    
                    mime_type, _ = mimetypes.guess_type(image_path)
                    # Use upload_file since we have a file path
                    image_url = gcs_client.upload_file(image_path, blob_name, content_type=mime_type)
                    
                    if image_url:
                        logger.info(f"Uploaded programmatic image to GCS: {image_url}")
                    else:
                        # Upload failed, fallback to local URL
                        image_url = f"{GENERATED_IMAGES_BASE_URL}/{filename}"
                except Exception as e:
                    logger.error(f"Failed to upload programmatic image to GCS: {e}")
                    filename = os.path.basename(image_path)
                    image_url = f"{GENERATED_IMAGES_BASE_URL}/{filename}"

            # Insert the new image into content
            new_markdown = f"![]({image_url})"
            widget_match = re.search(r'\[\[☃[^\]]+\]\]', gen_content)
            if widget_match:
                insert_pos = widget_match.start()
                gen_content = (
                    gen_content[:insert_pos].rstrip() +
                    "\n\n" + new_markdown + "\n\n" +
                    gen_content[insert_pos:]
                )
            else:
                gen_content = gen_content.rstrip() + "\n\n" + new_markdown

            # Update images dict with new image dimensions
            gen_question["images"] = {
                image_url: {"width": 400, "height": 300}
            }
            logger.info(f"Generated new image: {image_url}")
        else:
            # Fallback to source image
            logger.warning("Failed to generate image, using source image")
            for alt, url in source_images:
                markdown_img = f"![{alt}]({url})"
                if url not in gen_content:
                    widget_match = re.search(r'\[\[☃[^\]]+\]\]', gen_content)
                    if widget_match:
                        insert_pos = widget_match.start()
                        gen_content = (
                            gen_content[:insert_pos].rstrip() +
                            "\n\n" + markdown_img + "\n\n" +
                            gen_content[insert_pos:]
                        )
                    else:
                        gen_content = gen_content.rstrip() + "\n\n" + markdown_img
            if source_question.get("images"):
                gen_question["images"] = copy.deepcopy(source_question["images"])

        gen_question["content"] = gen_content.strip()
        result["question"] = gen_question
        return result

    def _extract_numeric_answer(self, generated: dict[str, Any]) -> Optional[int]:
        """Extract the numeric answer from a generated question.

        Looks in the widget options for the correct answer value.
        """
        question = generated.get("question", {})
        widgets = question.get("widgets", {})

        for widget_id, widget in widgets.items():
            widget_type = widget.get("type", "")
            options = widget.get("options", {})

            if widget_type == "numeric-input":
                # Get answer from numeric-input widget
                answers = options.get("answers", [])
                for answer in answers:
                    if answer.get("status") == "correct":
                        value = answer.get("value")
                        if value is not None:
                            try:
                                return int(float(value))
                            except (ValueError, TypeError):
                                pass

            elif widget_type == "input-number":
                # Get answer from input-number widget
                value = options.get("value")
                if value is not None:
                    try:
                        return int(float(value))
                    except (ValueError, TypeError):
                        pass

            elif widget_type == "radio":
                # Get answer from radio choices
                choices = options.get("choices", [])
                for choice in choices:
                    if choice.get("correct"):
                        content = choice.get("content", "")
                        # Try to extract number from choice content
                        numbers = re.findall(r'\b(\d+)\b', content)
                        if numbers:
                            return int(numbers[0])

        # Final fallback: look for numbers in the question text itself
        # Especially for "How many blocks show X?" or "Which model shows X?"
        question_content = question.get("content", "")
        # Look for 2-4 digit numbers which are common in place value questions
        numbers = re.findall(r'\b(\d{2,4})\b', question_content)
        if numbers:
            return int(numbers[0])

        return None

    def _parse_place_value_from_alt_text(self, alt_text: str) -> Optional[int]:
        """Parse alt text to extract the number represented by place value blocks.

        Examples:
            "3, one-hundred-cube flats. 6, ten-cube rods. 2 unit cubes." -> 362
            "4 hundred flats, 8 ten rods, 5 ones" -> 485
        """
        if not alt_text:
            return None

        alt_lower = alt_text.lower()
        hundreds = 0
        tens = 0
        ones = 0

        # Pattern: "X, one-hundred-cube flats" or "X hundred flats"
        hundreds_match = re.search(
            r'(\d+)[\s,]*(?:one-?hundred-?cube\s*flats?|hundred\s*(?:flat|block|cube)s?)',
            alt_lower
        )
        if hundreds_match:
            hundreds = int(hundreds_match.group(1))

        # Pattern: "X, ten-cube rods" or "X ten rods"
        tens_match = re.search(
            r'(\d+)[\s,]*(?:ten-?cube\s*rods?|ten\s*(?:rod|stick|block)s?)',
            alt_lower
        )
        if tens_match:
            tens = int(tens_match.group(1))

        # Pattern: "X unit cubes" or "X ones"
        ones_match = re.search(
            r'(\d+)[\s,]*(?:unit\s*cubes?|ones?|single\s*cubes?)',
            alt_lower
        )
        if ones_match:
            ones = int(ones_match.group(1))

        total = hundreds * 100 + tens * 10 + ones

        if total > 0:
            logger.debug(f"Parsed place value: {hundreds}H + {tens}T + {ones}O = {total}")
            return total

        return None

    def _generate_radio_choice_images(
        self,
        source: dict[str, Any],
        generated: dict[str, Any],
    ) -> dict[str, Any]:
        """Generate new images for radio choices that contain place value blocks.

        For questions like "Which place value model shows X?", each radio choice
        has an embedded image. This method generates new programmatic images
        for each choice based on the alt text description.
        """
        import copy
        import random
        result = copy.deepcopy(generated)

        gen_question = result.get("question", {})
        source_question = source.get("question", {})
        gen_content = gen_question.get("content", "")
        gen_content_lower = gen_content.lower()

        # Only process "which model shows" type questions
        if "which" not in gen_content_lower or "model" not in gen_content_lower:
            if "place value" not in gen_content_lower and "blocks" not in gen_content_lower:
                return result

        # Extract the target number from the question (e.g., "Which place value model shows $263$?")
        target_number = None
        target_match = re.search(r'\$(\d{2,4})\$', gen_content)
        if target_match:
            target_number = int(target_match.group(1))
            logger.info(f"Target number from question: {target_number}")

        gen_widgets = gen_question.get("widgets", {})
        source_widgets = source_question.get("widgets", {})

        for widget_id, gen_widget in gen_widgets.items():
            if gen_widget.get("type") != "radio":
                continue

            source_widget = source_widgets.get(widget_id, {})
            source_choices = source_widget.get("options", {}).get("choices", [])
            gen_choices = gen_widget.get("options", {}).get("choices", [])

            # Check if source choices have embedded images (place value blocks)
            source_has_images = any(
                "web+graphie://" in choice.get("content", "") or
                "![" in choice.get("content", "")
                for choice in source_choices
            )

            if not source_has_images:
                continue

            logger.info(f"Generating images for radio widget {widget_id} with {len(gen_choices)} choices")

            # Import the programmatic generator
            from questionbank.utils.image_generator import generate_place_value_blocks

            # Generate distractors if we have a target number
            used_numbers = set()
            if target_number:
                used_numbers.add(target_number)

            for i, gen_choice in enumerate(gen_choices):
                choice_content = gen_choice.get("content", "")
                is_correct = gen_choice.get("correct", False)

                # For the correct answer, use the target number from the question
                if is_correct and target_number:
                    number = target_number
                    logger.info(f"Choice {i} (CORRECT): Using target number {number}")
                else:
                    # For wrong answers, parse from content or generate distractors
                    # First check for markdown image alt text
                    alt_match = re.search(r'!\[([^\]]+)\]\([^)]+\)', choice_content)
                    if alt_match:
                        alt_text = alt_match.group(1)
                        number = self._parse_place_value_from_alt_text(alt_text)
                    else:
                        # Try parsing the content directly
                        number = self._parse_place_value_from_alt_text(choice_content)

                    if number is None:
                        # Try to extract any number from the content
                        numbers = re.findall(r'\b(\d{2,4})\b', choice_content)
                        if numbers:
                            number = int(numbers[0])

                    # If still no number or it's the same as target, generate a distractor
                    if number is None or number == target_number or number in used_numbers:
                        if target_number:
                            # Generate a plausible distractor
                            h = target_number // 100
                            t = (target_number % 100) // 10
                            o = target_number % 10
                            # Swap digits to create distractor
                            attempts = 0
                            while attempts < 10:
                                swap_type = random.choice(['ht', 'ho', 'to', 'shift'])
                                if swap_type == 'ht':
                                    number = t * 100 + h * 10 + o
                                elif swap_type == 'ho':
                                    number = o * 100 + t * 10 + h
                                elif swap_type == 'to':
                                    number = h * 100 + o * 10 + t
                                else:
                                    number = target_number + random.choice([-100, 100, -10, 10, -1, 1])
                                if number not in used_numbers and 100 <= number <= 999:
                                    break
                                attempts += 1
                            logger.info(f"Choice {i}: Generated distractor {number}")

                if number and number > 0 and 10 <= number <= 9999:
                    used_numbers.add(number)
                    logger.info(f"Choice {i}: Generating place value image for {number}")
                    try:
                        image_path = generate_place_value_blocks(number)
                        if image_path:
                            filename = os.path.basename(image_path)
                            image_url = f"{GENERATED_IMAGES_BASE_URL}/{filename}"

                            # Create alt text describing the blocks
                            hundreds = number // 100
                            tens = (number % 100) // 10
                            ones = number % 10
                            alt_text = f"{hundreds} hundred flats, {tens} ten rods, {ones} unit cubes"

                            # Replace the choice content with new markdown image
                            new_content = f"![{alt_text}]({image_url})"
                            gen_choice["content"] = new_content
                            logger.info(f"Updated choice {i} with image: {image_url}")
                    except Exception as e:
                        logger.warning(f"Failed to generate image for choice {i}: {e}")
                else:
                    logger.warning(f"Choice {i}: Invalid number {number}, skipping")

        result["question"] = gen_question
        return result

    def _build_image_prompt_from_question(
        self,
        question_text: str,
        source_content: str,
        answer_value: Optional[int] = None,
    ) -> Optional[str]:
        """Build an image generation prompt based on question content.

        Args:
            question_text: The question text
            source_content: The source question content for reference
            answer_value: The numeric answer if known (for precise image generation)
        """
        question_lower = question_text.lower()

        # Place value blocks - VERY SPECIFIC prompt for base-10 manipulatives
        if "place value" in question_lower or "blocks" in question_lower:
            # Try to extract the answer from answer_value or question
            num = answer_value
            if num is None:
                numbers = re.findall(r'\b(\d{2,4})\b', question_text)
                if numbers:
                    num = int(numbers[0])

            if num:
                hundreds = num // 100
                tens = (num % 100) // 10
                ones = num % 10

                # Very detailed prompt for base-10 blocks with grid pattern
                prompt = f"""Educational base-10 place value blocks diagram showing the number {num}:

EXACT REQUIREMENTS:
- {hundreds} HUNDREDS blocks: Each is a flat 10x10 grid square with 100 small unit squares visible inside. Purple/violet colored with visible grid lines dividing it into 100 squares. Stack them slightly offset to show depth.
- {tens} TENS rods: Each is a vertical rectangular bar divided into 10 visible unit segments by horizontal lines. Same purple/violet color. Arranged side by side.
- {ones} ONES cubes: Each is a small single cube. Same purple/violet color.

STYLE:
- Match Khan Academy's place value blocks style exactly
- All purple/violet colored blocks with visible internal grid lines
- White/light gray background
- Clean 2D diagram view (slight isometric perspective OK)
- Grid lines clearly visible on all blocks to show unit squares
- No text, no labels, just the blocks
- Simple, clean educational illustration"""
                return prompt

        # Counting objects
        if any(word in question_lower for word in ["count", "how many", "objects"]):
            numbers = re.findall(r'\b(\d+)\b', question_text)
            if numbers:
                return f"""Counting diagram with EXACTLY {numbers[0]} objects:
- Arrange {numbers[0]} identical simple objects (dots, stars, or circles) in a clear grid pattern
- Make them easy to count - use rows of 5 or 10
- Clean white background
- All objects same size and color
- No text or numbers, just the objects to count
- Educational math worksheet style"""

        # Shapes/geometry
        if any(word in question_lower for word in ["triangle", "square", "circle", "rectangle", "shape"]):
            return f"""Geometric diagram for: {question_text}
- Clean black outlines on white background
- Labeled dimensions with measurement marks
- Educational math textbook style
- Precise angles and proportions
- Grid lines if helpful for measurement"""

        # Fractions
        if "fraction" in question_lower or "/" in question_text:
            return f"""Fraction visualization for: {question_text}
- Use clearly divided shapes (circles/rectangles divided into equal parts)
- Shaded portions clearly distinguishable
- Clean educational style
- No text labels, visual only
- Equal-sized divisions"""

        # Graphs/charts
        if any(word in question_lower for word in ["graph", "chart", "plot", "data"]):
            return f"""Mathematical graph/chart showing: {question_text}
- Clear labeled axes with numbers
- Grid lines for reading values
- Clean educational textbook style
- Easy to read data points or bars"""

        # Molecules/chemistry
        if any(word in question_lower for word in ["molecule", "atom", "chemical", "structure"]):
            return f"""Molecular structure diagram: {question_text}
- Ball-and-stick model
- Standard element colors (Carbon black, Hydrogen white, Oxygen red, Nitrogen blue)
- Clear bond lines
- Educational chemistry diagram style"""

        # Default: generate based on question text
        if len(question_text) > 10:
            return f"""Educational diagram illustrating: {question_text}
- Clean, simple, clear illustration for K-12 students
- White background
- Educational textbook style
- No unnecessary decoration"""

        return None

    def _should_generate_new_image(
        self,
        source_alt: str,
        generated_alt: str,
    ) -> bool:
        """Determine if a new image needs to be generated.

        Returns True if the alt text has changed significantly,
        indicating the content has changed and a new image is needed.
        """
        if not source_alt or not generated_alt:
            return False

        # Normalize for comparison
        source_normalized = source_alt.lower().strip()
        gen_normalized = generated_alt.lower().strip()

        # If alt texts are very similar (>80% overlap), use existing image
        if source_normalized == gen_normalized:
            return False

        # Check for key differences indicating new content
        # Extract key numbers and values
        source_numbers = set(re.findall(r'\d+(?:\.\d+)?', source_normalized))
        gen_numbers = set(re.findall(r'\d+(?:\.\d+)?', gen_normalized))

        # If numbers are different, might need new image
        if source_numbers != gen_numbers and len(gen_numbers) > 0:
            return True

        # Check for different subjects (molecular structures, shapes, etc.)
        subject_keywords = [
            'molecule', 'structure', 'graph', 'triangle', 'circle', 'square',
            'rectangle', 'equation', 'diagram', 'chart', 'plot', 'function'
        ]

        for keyword in subject_keywords:
            source_has = keyword in source_normalized
            gen_has = keyword in gen_normalized
            if source_has != gen_has:
                return True

        return False

    def _generate_images_for_question(
        self,
        generated: dict[str, Any],
        source: dict[str, Any],
    ) -> dict[str, Any]:
        """Generate new images for widgets that need them.

        Compares alt text between source and generated to determine
        if new images need to be created.
        """
        import copy
        result = copy.deepcopy(generated)

        gen_question = result.get("question", {})
        source_question = source.get("question", {})

        gen_widgets = gen_question.get("widgets", {})
        source_widgets = source_question.get("widgets", {})

        for widget_id, gen_widget in gen_widgets.items():
            if gen_widget.get("type") != "image":
                continue

            gen_options = gen_widget.get("options", {})
            gen_alt = gen_options.get("alt", "")

            # Get source alt for comparison
            source_widget = source_widgets.get(widget_id, {})
            source_options = source_widget.get("options", {})
            source_alt = source_options.get("alt", "")

            # Check current generated URL - detect hallucinated/fake URLs
            gen_bg = gen_options.get("backgroundImage", {})
            gen_url = gen_bg.get("url", "") if isinstance(gen_bg, dict) else ""

            # Get source URL for comparison
            source_bg = source_options.get("backgroundImage", {})
            source_url = source_bg.get("url", "") if isinstance(source_bg, dict) else ""

            # List of hallucinated URL patterns that AI commonly generates
            hallucinated_patterns = [
                "wikipedia.org", "wikimedia.org", "imgur.com", "i.imgur",
                "placeholder", "example.com", "stock", "shutterstock",
                "gettyimages", "unsplash", "pexels", "pixabay"
            ]

            # Force image generation if URL is hallucinated or alt text changed significantly
            url_is_hallucinated = any(pattern in gen_url.lower() for pattern in hallucinated_patterns)
            url_is_empty = not gen_url or gen_url.startswith("data:")
            url_is_already_local = "localhost" in gen_url

            # Also detect if AI returned a KA URL that's different from source (hallucinated)
            url_is_fake_ka = (
                "kastatic.org" in gen_url and
                gen_url != source_url
            )

            needs_new_image = (
                url_is_hallucinated or
                url_is_empty or
                url_is_fake_ka or
                (not url_is_already_local and self._should_generate_new_image(source_alt, gen_alt))
            )

            if needs_new_image:
                logger.info(f"Generating new image for widget {widget_id}")
                logger.info(f"  Reason: hallucinated={url_is_hallucinated}, empty={url_is_empty}")
                logger.info(f"  Source alt: {source_alt[:80]}...")
                logger.info(f"  Generated alt: {gen_alt[:80]}...")

                image_path = None

                # Try reference-based generation first if we have a source image
                if source_url and source_url.startswith(("http://", "https://", "web+graphie://")):
                    logger.info(f"  Using reference image: {source_url[:60]}...")
                    image_path = self.gemini.generate_image_from_reference(
                        source_image_url=source_url,
                        new_context=gen_alt,
                        style_instructions="Match the photographic/illustration style, lighting, and quality of the reference image",
                    )

                # Fall back to text-only generation if reference fails
                if not image_path:
                    logger.info("  Falling back to text-only generation...")
                    image_path = self.gemini.generate_educational_image(
                        description=gen_alt,
                        context="educational diagram for Khan Academy style question",
                        style="clean, simple, educational illustration with clear labels",
                    )

                if image_path:
                    # If it's already a full URL (GCS), use it directly
                    if image_path.startswith(("http://", "https://")):
                         image_url = image_path
                    else:
                        # Get just the filename
                        filename = os.path.basename(image_path)
                        # Create URL for serving
                        image_url = f"{GENERATED_IMAGES_BASE_URL}/{filename}"

                    # Update the widget with new image URL
                    gen_options["backgroundImage"] = {
                        "url": image_url,
                        "width": gen_options.get("backgroundImage", {}).get("width", 400),
                        "height": gen_options.get("backgroundImage", {}).get("height", 300),
                    }
                    logger.info(f"Generated image saved: {image_url}")
                else:
                    logger.warning(f"Failed to generate image for {widget_id}, keeping source")
                    # Fall back to source image
                    if source_options.get("backgroundImage"):
                        gen_options["backgroundImage"] = copy.deepcopy(
                            source_options["backgroundImage"]
                        )

        result["question"] = gen_question
        return result

    def generate_with_images(
        self,
        source_question: dict[str, Any],
        variation_type: str = "number_change",
        save_to_db: bool = True,
        generate_new_images: bool = True,
    ) -> Optional[dict[str, Any]]:
        """Generate a new question with optional image generation.

        Args:
            source_question: The source question to base generation on
            variation_type: Type of variation to apply
            save_to_db: Whether to save the result to database
            generate_new_images: If True, generate new images when content changes
        """
        source_id = str(source_question.get("_id", "unknown"))
        logger.info(f"Generating question with images from source: {source_id}")

        validation_feedback: list[str] = []
        attempt = 0

        while attempt < self.max_retries:
            attempt += 1
            logger.info(f"Generation attempt {attempt}/{self.max_retries}")

            # Build prompts
            system_prompt = get_system_prompt()

            # Generate using Gemini
            generated_json = self.gemini.generate_question_json(
                source_question,
                system_prompt,
                validation_feedback=validation_feedback if validation_feedback else None,
            )

            if not generated_json:
                logger.warning(f"Attempt {attempt}: Failed to generate valid JSON")
                validation_feedback.append("Failed to generate valid JSON. Ensure response is pure JSON.")
                continue

            # Post-process: Handle images
            if generate_new_images:
                # Generate NEW images using Gemini for both widget images and markdown images
                generated_json = self._generate_images_for_question(
                    generated_json, source_question
                )
                # Also handle markdown-embedded images
                generated_json = self._generate_new_markdown_images(
                    source_question, generated_json
                )
                # Generate images for radio choices (place value blocks, etc.)
                generated_json = self._generate_radio_choice_images(
                    source_question, generated_json
                )
            else:
                # Just copy image URLs from source
                generated_json = self._copy_image_urls_from_source(
                    source_question, generated_json
                )

            # Validate the generated question
            validation_result = self.validation_pipeline.validate(generated_json)

            if validation_result.is_valid:
                logger.info(f"Generated valid question after {attempt} attempt(s)")

                # Generate step-by-step solution for the generated question
                generated_json = self._add_solution_steps(generated_json)

                # Generate hints if source doesn't have any
                generated_json = self._generate_hints_if_missing(
                    generated_json, source_question
                )

                # Verify and fix answers using comprehensive verifier
                generated_json, answer_result = verify_and_fix_question(
                    generated_json, use_ai=True
                )

                if not answer_result.is_valid:
                    logger.warning(f"Answer verification issues: {answer_result.error_message}")
                    if answer_result.error_message:
                        validation_feedback.append(f"Answer error: {answer_result.error_message}")
                        continue
                else:
                    if answer_result.details.get("fixes_attempted"):
                        logger.info("Answers were auto-corrected")

                if save_to_db:
                    self._save_generated_question(
                        source_id=source_id,
                        generated_json=generated_json,
                        attempt_count=attempt,
                    )

                return generated_json
            else:
                logger.warning(f"Attempt {attempt}: Validation failed")
                logger.debug(f"Errors: {validation_result.errors}")
                validation_feedback = validation_result.errors

        logger.error(f"Failed to generate valid question after {self.max_retries} attempts")
        return None


    def generate_intelligent(
        self,
        source_question: dict[str, Any],
        save_to_db: bool = True,
    ) -> Optional[dict[str, Any]]:
        """Generate a question using intelligent mode with semantic understanding.

        This method:
        1. Extracts semantic constraints from source (gender, image dependencies, etc.)
        2. Determines the best generation strategy based on widget type
        3. Generates using LLM with strategy-specific prompts
        4. Validates coherence (text-image alignment, gender consistency, etc.)
        5. Applies automatic fixes for coherence issues
        6. Generates new images programmatically when appropriate

        Args:
            source_question: Source Perseus question
            save_to_db: Whether to save result to database

        Returns:
            Generated question dict or None if generation fails
        """
        source_id = str(source_question.get("_id", "unknown"))
        logger.info(f"[INTELLIGENT] Generating from source: {source_id}")

        # Step 1: Extract constraints
        constraints = self.constraint_extractor.extract(source_question)
        logger.info(f"  Strategy: {constraints.variation_strategy}")
        logger.info(f"  Widget type: {constraints.widget_type}")
        logger.info(f"  Image dependency: {constraints.image.dependency}")
        logger.info(f"  Gender: {constraints.image.gender}")

        validation_feedback: list[str] = []
        attempt = 0

        while attempt < self.max_retries:
            attempt += 1
            logger.info(f"[INTELLIGENT] Attempt {attempt}/{self.max_retries}")

            # Step 2: Build constraint-aware prompt
            system_prompt = self._build_intelligent_system_prompt(constraints)
            user_prompt = self._build_intelligent_user_prompt(source_question, constraints, validation_feedback)

            # Step 3: Generate with LLM
            generated_json = self.gemini.generate_question_json(
                source_question,
                system_prompt,
                validation_feedback=validation_feedback if validation_feedback else None,
            )

            if not generated_json:
                logger.warning(f"Attempt {attempt}: Failed to generate JSON")
                validation_feedback.append("Failed to generate valid JSON.")
                continue

            # Step 4: Apply smart post-processing
            result = self.smart_generator.generate(source_question, generated_json)

            if result.success and result.generated_data:
                generated_json = result.generated_data
                logger.info(f"  Smart generator strategy: {result.strategy_used}")
                for mod in result.modifications:
                    logger.info(f"  Modification: {mod}")
            elif result.coherence_result:
                # Log coherence issues
                for issue in result.coherence_result.issues:
                    logger.warning(f"  Coherence issue: [{issue.severity}] {issue.message}")

            # Step 5: Handle image generation based on strategy
            if constraints.variation_strategy == 'regenerate_place_value':
                generated_json = self._regenerate_place_value_images(
                    source_question, generated_json, constraints
                )
            elif constraints.image.dependency.value == 'required':
                # Preserve source images for image-dependent questions
                generated_json = self._copy_image_urls_from_source(source_question, generated_json)

            # Step 6: Generate radio choice images if needed
            generated_json = self._generate_radio_choice_images(source_question, generated_json)

            # Step 7: Final intelligent validation
            pipeline_result = self.intelligent_pipeline.validate(generated_json, source_question)

            if pipeline_result.is_valid or pipeline_result.overall_score > 0.7:
                logger.info(f"[INTELLIGENT] Generated valid question (score: {pipeline_result.overall_score:.2f})")

                # Use fixed data if available
                if pipeline_result.fixed_data:
                    generated_json = pipeline_result.fixed_data

                if save_to_db:
                    self._save_intelligent_question(
                        source_id=source_id,
                        generated_json=generated_json,
                        constraints=constraints,
                        pipeline_result=pipeline_result,
                        attempt_count=attempt,
                    )

                return generated_json
            else:
                # Collect errors for retry
                validation_feedback = pipeline_result.get_all_errors()
                logger.warning(f"Attempt {attempt}: Validation score {pipeline_result.overall_score:.2f}")
                for error in validation_feedback[:3]:  # Show first 3 errors
                    logger.warning(f"  {error}")

        logger.error(f"[INTELLIGENT] Failed after {self.max_retries} attempts")
        return None

    def _build_intelligent_system_prompt(self, constraints: QuestionConstraints) -> str:
        """Build system prompt based on extracted constraints."""
        base_prompt = get_system_prompt()

        # Add constraint-specific instructions
        additions = []

        if constraints.image.dependency.value == 'required':
            additions.append("""
CRITICAL IMAGE CONSTRAINT:
This question depends on an image. You MUST:
1. Keep all image references exactly as in the source
2. Not change any aspects that relate to what the image shows
""")
            if constraints.image.gender.value in ['male', 'female']:
                additions.append(f"""
GENDER CONSTRAINT:
The image shows a {constraints.image.gender.value} subject.
ALL gender references in your generated text MUST match '{constraints.image.gender.value}'.
Do NOT change 'female' to 'male' or vice versa.
""")

        if constraints.variation_strategy == 'numerical':
            additions.append("""
VARIATION STRATEGY: Numerical
- Change numbers, values, and quantities
- Keep the same mathematical operation/concept
- Ensure your new answer is mathematically correct
""")
        elif constraints.variation_strategy == 'contextual':
            additions.append("""
VARIATION STRATEGY: Contextual
- Keep the same question structure
- Change context/scenario while preserving the core concept
- Maintain scientific/factual accuracy
""")

        if constraints.must_preserve:
            additions.append(f"""
MUST PRESERVE these elements:
{chr(10).join('- ' + item for item in constraints.must_preserve)}
""")

        return base_prompt + '\n'.join(additions)

    def _build_intelligent_user_prompt(
        self,
        source: dict[str, Any],
        constraints: QuestionConstraints,
        feedback: list[str]
    ) -> str:
        """Build user prompt with constraint awareness."""
        import json

        prompt_parts = [
            "Generate a NEW question based on this source:",
            "",
            "```json",
            json.dumps(source, indent=2),
            "```",
            "",
            f"Widget type: {constraints.widget_type}",
            f"Topic: {constraints.topic}",
            f"Subject: {constraints.subject_area}",
            "",
        ]

        if constraints.image.dependency.value == 'required':
            prompt_parts.append("⚠️ This question has a REQUIRED image. Keep image references unchanged.")

        if constraints.image.gender.value in ['male', 'female']:
            prompt_parts.append(f"⚠️ Image shows {constraints.image.gender.value} - maintain gender consistency.")

        if feedback:
            prompt_parts.append("\nPrevious attempt had these issues:")
            for f in feedback:
                prompt_parts.append(f"- {f}")

        prompt_parts.append("\nReturn ONLY valid Perseus JSON.")

        return '\n'.join(prompt_parts)

    def _regenerate_place_value_images(
        self,
        source: dict[str, Any],
        generated: dict[str, Any],
        constraints: QuestionConstraints
    ) -> dict[str, Any]:
        """Regenerate place value block images for the question."""
        import copy
        result = copy.deepcopy(generated)

        # Extract the answer value
        answer_value = self._extract_numeric_answer(result)

        if answer_value and 10 <= answer_value <= 9999:
            logger.info(f"Generating place value image for {answer_value}")

            new_image = self.image_generator.generate_place_value_blocks(answer_value)

            if new_image:
                # Update content with new image
                content = result.get('question', {}).get('content', '')

                # Remove old images
                img_pattern = r'!\[[^\]]*\]\([^)]+\)'
                content = re.sub(img_pattern, '', content)

                # Add new image before widget
                new_markdown = f"![{new_image.alt_text}]({new_image.url})"
                widget_match = re.search(r'\[\[☃[^\]]+\]\]', content)
                if widget_match:
                    insert_pos = widget_match.start()
                    content = (
                        content[:insert_pos].strip() +
                        "\n\n" + new_markdown + "\n\n" +
                        content[insert_pos:]
                    )
                else:
                    content = content.strip() + "\n\n" + new_markdown

                result['question']['content'] = content.strip()
                result['question']['images'] = {
                    new_image.url: {
                        'url': new_image.url,
                        'width': new_image.width,
                        'height': new_image.height,
                        'alt': new_image.alt_text
                    }
                }

                logger.info(f"Generated place value image: {new_image.url}")

        return result

    def _save_intelligent_question(
        self,
        source_id: str,
        generated_json: dict[str, Any],
        constraints: QuestionConstraints,
        pipeline_result: Any,
        attempt_count: int,
    ) -> Optional[str]:
        """Save an intelligently generated question with metadata."""
        metadata = {
            "llm_model": config.gemini.model,
            "generation_mode": "intelligent",
            "validation_status": "valid" if pipeline_result.is_valid else "partial",
            "validation_score": pipeline_result.overall_score,
            "attempt_count": attempt_count,
            "generated_at": datetime.utcnow(),
            "constraints": {
                "widget_type": constraints.widget_type,
                "strategy": constraints.variation_strategy,
                "image_dependency": constraints.image.dependency.value,
                "gender": constraints.image.gender.value,
                "topic": constraints.topic,
            }
        }

        inserted_id = question_repo.insert_generated_question(
            source_question_id=source_id,
            perseus_json=generated_json,
            metadata=metadata,
        )

        if inserted_id:
            logger.info(f"[INTELLIGENT] Saved: {inserted_id}")
        else:
            logger.warning("[INTELLIGENT] Failed to save to database")

        return inserted_id


# Singleton instance
question_generator = QuestionGenerator()

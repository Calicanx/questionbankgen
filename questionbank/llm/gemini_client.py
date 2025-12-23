"""Google Gemini API client for question generation."""

import base64
import json
import logging
import os
import uuid
import requests
import sys
from io import BytesIO
from typing import Any, Optional

# Configure native library paths for macOS Homebrew installations
if sys.platform == 'darwin':
    import ctypes
    import ctypes.util

    # Pre-load cairo library for cairosvg
    cairo_paths = [
        '/opt/homebrew/opt/cairo/lib/libcairo.2.dylib',
        '/opt/homebrew/lib/libcairo.2.dylib',
        '/usr/local/lib/libcairo.2.dylib',
    ]
    for cairo_path in cairo_paths:
        if os.path.exists(cairo_path):
            try:
                ctypes.CDLL(cairo_path)
                break
            except OSError:
                pass

    # Set MAGICK_HOME for Wand/ImageMagick
    os.environ['MAGICK_HOME'] = '/opt/homebrew/opt/imagemagick'

    # Pre-load ImageMagick libraries for Wand (order matters: Core before Wand)
    magick_libs = [
        '/opt/homebrew/opt/imagemagick/lib/libMagickCore-7.Q16HDRI.10.dylib',
        '/opt/homebrew/opt/imagemagick/lib/libMagickWand-7.Q16HDRI.10.dylib',
    ]
    for magick_path in magick_libs:
        if os.path.exists(magick_path):
            try:
                ctypes.CDLL(magick_path, mode=ctypes.RTLD_GLOBAL)
            except OSError:
                pass

from PIL import Image

import google.generativeai as genai
from tenacity import retry, stop_after_attempt, wait_exponential

from questionbank.config import config

logger = logging.getLogger(__name__)

# Directory to store generated images
GENERATED_IMAGES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "generated_images")


class GeminiClient:
    """Client for Google Gemini API."""

    def __init__(self) -> None:
        self.api_key = config.gemini.api_key
        self.model_name = config.gemini.model
        self.image_model_name = config.gemini.image_model
        self.temperature = config.gemini.temperature

        if not self.api_key:
            raise ValueError("GEMINI_API_KEY is not set in environment")

        # Configure the API
        genai.configure(api_key=self.api_key)

        # Initialize the text model
        self.model = genai.GenerativeModel(
            model_name=self.model_name,
            generation_config={
                "temperature": self.temperature,
                "top_p": 0.95,
                "top_k": 40,
                "max_output_tokens": 8192,
            },
        )

        # Initialize the image generation model
        self.image_model = genai.GenerativeModel(
            model_name=self.image_model_name,
            generation_config={
                "temperature": 1.0,
                "top_p": 0.95,
                "top_k": 40,
                "response_modalities": ["IMAGE", "TEXT"],
            },
        )

        # Ensure generated images directory exists
        os.makedirs(GENERATED_IMAGES_DIR, exist_ok=True)

        logger.info(f"[GEMINI] Initialized with model: {self.model_name}")
        logger.info(f"[GEMINI] Image model: {self.image_model_name}")

    def generate_image(self, prompt: str, save_path: Optional[str] = None) -> Optional[str]:
        """Generate an image using Gemini's image generation model.

        Args:
            prompt: Description of the image to generate
            save_path: Optional path to save the image. If not provided, auto-generates.

        Returns:
            Path to the saved image file, or None if generation failed.
        """
        try:
            logger.info(f"[GEMINI] Generating image with prompt: {prompt[:100]}...")

            # Generate image using the image model
            response = self.image_model.generate_content(prompt)

            # Check if we got image data
            if not response.candidates:
                logger.warning("[GEMINI] No candidates in image response")
                return None

            # Extract image from response
            for part in response.candidates[0].content.parts:
                if hasattr(part, 'inline_data') and part.inline_data:
                    # Get image data
                    image_data = part.inline_data.data
                    mime_type = part.inline_data.mime_type

                    # Determine file extension
                    ext = ".png"
                    if "jpeg" in mime_type or "jpg" in mime_type:
                        ext = ".jpg"
                    elif "webp" in mime_type:
                        ext = ".webp"

                    # Generate save path if not provided
                    if not save_path:
                        filename = f"generated_{uuid.uuid4().hex[:8]}{ext}"
                        save_path = os.path.join(GENERATED_IMAGES_DIR, filename)

                    # Decode and save image
                    if isinstance(image_data, str):
                        image_bytes = base64.b64decode(image_data)
                    else:
                        image_bytes = image_data

                    with open(save_path, 'wb') as f:
                        f.write(image_bytes)

                    logger.info(f"[GEMINI] Image saved to: {save_path}")
                    return save_path

            # If no inline_data, check for text response (might contain URL or error)
            if response.text:
                logger.warning(f"[GEMINI] Got text instead of image: {response.text[:200]}")

            return None

        except Exception as e:
            logger.error(f"[GEMINI] Image generation error: {e}")
            return None

    def generate_educational_image(
        self,
        description: str,
        context: str = "educational math/science diagram",
        style: str = "clean, simple, educational illustration",
    ) -> Optional[str]:
        """Generate an educational image with appropriate styling.

        Args:
            description: What the image should show (e.g., "molecular structure of water H2O")
            context: Educational context for the image
            style: Visual style to use

        Returns:
            Path to saved image, or None if failed.
        """
        prompt = f"""Create a {style} for {context}.

The image should show: {description}

Requirements:
- Clear, simple visuals suitable for educational use
- High contrast for readability
- No text or labels (unless specifically requested)
- Clean white or light background
- Professional quality suitable for textbooks or online learning"""

        return self.generate_image(prompt)

    def generate_image_from_reference(
        self,
        source_image_url: str,
        new_context: str,
        style_instructions: str = "Match the style, composition, and quality of the reference image",
    ) -> Optional[str]:
        """Generate a new image based on a reference image and new context.

        This method:
        1. Downloads the source image
        2. Sends it to Gemini along with the new context
        3. Asks Gemini to generate a similar-style image for the new topic

        Args:
            source_image_url: URL of the reference image
            new_context: Description of what the new image should show
            style_instructions: Instructions for matching the style

        Returns:
            Path to saved image, or None if failed.
        """
        try:
            logger.info(f"[GEMINI] Generating image from reference: {source_image_url[:60]}...")
            logger.info(f"[GEMINI] New context: {new_context[:100]}...")

            # Download the source image with browser-like headers
            headers = {
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept': 'image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.9',
                'Accept-Encoding': 'gzip, deflate, br',
                'Referer': 'https://www.khanacademy.org/',
                'Origin': 'https://www.khanacademy.org',
                'Sec-Fetch-Dest': 'image',
                'Sec-Fetch-Mode': 'no-cors',
                'Sec-Fetch-Site': 'cross-site',
                'Connection': 'keep-alive',
            }

            # Handle different URL formats
            img_url = source_image_url
            if img_url.startswith("web+graphie://"):
                # web+graphie:// URLs need .svg extension for the actual image
                img_url = img_url.replace("web+graphie://", "https://") + ".svg"

            # Try multiple URL variations if needed
            url_variations = [img_url]
            if not img_url.endswith('.svg') and 'ka-perseus-graphie' in img_url:
                url_variations.append(img_url + '.svg')
                url_variations.append(img_url + '.png')

            response = None
            for url in url_variations:
                try:
                    response = requests.get(url, headers=headers, timeout=15)
                    if response.status_code == 200:
                        logger.info(f"[GEMINI] Successfully downloaded from: {url[:60]}...")
                        break
                except:
                    continue

            if not response or response.status_code != 200:
                raise requests.RequestException(f"All URL variations failed for {source_image_url[:60]}")

            # Load image - handle SVG files specially
            content_type = response.headers.get('Content-Type', '')
            image_data = BytesIO(response.content)

            if 'svg' in content_type or url.endswith('.svg'):
                # Convert SVG to PNG
                source_image = None

                # Method 1: Try Wand (ImageMagick) - most reliable
                try:
                    from wand.image import Image as WandImage
                    with WandImage(blob=response.content, format='svg') as img:
                        img.format = 'png'
                        png_data = BytesIO(img.make_blob())
                        source_image = Image.open(png_data)
                        logger.info("[GEMINI] Converted SVG to PNG using Wand/ImageMagick")
                except Exception as e:
                    logger.warning(f"[GEMINI] Wand conversion failed: {e}")

                # Method 2: Try svglib + reportlab (pure Python) as fallback
                if source_image is None:
                    try:
                        from svglib.svglib import svg2rlg
                        from reportlab.graphics import renderPM
                        import tempfile

                        with tempfile.NamedTemporaryFile(suffix='.svg', delete=False) as f:
                            f.write(response.content)
                            svg_path = f.name

                        drawing = svg2rlg(svg_path)
                        if drawing:
                            png_data = BytesIO()
                            renderPM.drawToFile(drawing, png_data, fmt='PNG')
                            png_data.seek(0)
                            source_image = Image.open(png_data)
                            logger.info("[GEMINI] Converted SVG to PNG using svglib")

                        os.unlink(svg_path)
                    except Exception as e:
                        logger.warning(f"[GEMINI] svglib conversion failed: {e}")

                # Method 3: Skip SVG reference, use text-only generation
                if source_image is None:
                    logger.warning("[GEMINI] Cannot convert SVG, skipping reference image")
                    return None  # This will trigger text-only fallback
            else:
                source_image = Image.open(image_data)

            # Build the prompt
            prompt = f"""Look at this reference image carefully. I need you to generate a NEW image that:

1. MATCHES THE STYLE: {style_instructions}
2. SHOWS NEW CONTENT: {new_context}

The new image should:
- Have similar visual quality and style as the reference
- Be appropriate for educational use (K-12 level)
- Be clear, professional, and engaging
- NOT copy the reference image, but create something NEW for the described topic

Generate the new image now."""

            # Send to Gemini with the image
            response = self.image_model.generate_content([prompt, source_image])

            # Extract and save the generated image
            if not response.candidates:
                logger.warning("[GEMINI] No candidates in response")
                return None

            for part in response.candidates[0].content.parts:
                if hasattr(part, 'inline_data') and part.inline_data:
                    image_data = part.inline_data.data
                    mime_type = part.inline_data.mime_type

                    ext = ".png"
                    if "jpeg" in mime_type or "jpg" in mime_type:
                        ext = ".jpg"

                    filename = f"generated_{uuid.uuid4().hex[:8]}{ext}"
                    save_path = os.path.join(GENERATED_IMAGES_DIR, filename)

                    if isinstance(image_data, str):
                        image_bytes = base64.b64decode(image_data)
                    else:
                        image_bytes = image_data

                    with open(save_path, 'wb') as f:
                        f.write(image_bytes)

                    logger.info(f"[GEMINI] Generated image saved: {save_path}")
                    return save_path

            # Try alternative method if inline_data not found
            if hasattr(response.candidates[0].content.parts[0], 'image'):
                img = response.candidates[0].content.parts[0].image
                filename = f"generated_{uuid.uuid4().hex[:8]}.png"
                save_path = os.path.join(GENERATED_IMAGES_DIR, filename)
                img.save(save_path)
                logger.info(f"[GEMINI] Generated image saved: {save_path}")
                return save_path

            logger.warning("[GEMINI] No image data in response")
            return None

        except requests.RequestException as e:
            logger.error(f"[GEMINI] Failed to download source image: {e}")
            return None
        except Exception as e:
            logger.error(f"[GEMINI] Image generation from reference failed: {e}")
            return None

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
    ) -> str:
        """Generate text using Gemini."""
        try:
            # Build the full prompt
            full_prompt = ""
            if system_prompt:
                full_prompt = f"{system_prompt}\n\n{prompt}"
            else:
                full_prompt = prompt

            # Generate response
            response = self.model.generate_content(full_prompt)

            if response.text:
                return response.text
            else:
                logger.warning("[GEMINI] Empty response received")
                return ""

        except Exception as e:
            logger.error(f"[GEMINI] Generation error: {e}")
            raise

    def generate_question_json(
        self,
        source_question: dict[str, Any],
        system_prompt: str,
        validation_feedback: Optional[list[str]] = None,
    ) -> Optional[dict[str, Any]]:
        """Generate a new question JSON from a source question."""
        try:
            # Build the prompt
            prompt = self._build_generation_prompt(source_question, validation_feedback)

            # Generate response
            response_text = self.generate(prompt, system_prompt)

            # Parse JSON from response
            return self._parse_json_response(response_text)

        except Exception as e:
            logger.error(f"[GEMINI] Question generation error: {e}")
            return None

    def _build_generation_prompt(
        self,
        source_question: dict[str, Any],
        validation_feedback: Optional[list[str]] = None,
    ) -> str:
        """Build the prompt for question generation."""
        # Extract just the Perseus JSON parts
        perseus_json = {
            "question": source_question.get("question", {}),
            "hints": source_question.get("hints", []),
            "answerArea": source_question.get("answerArea", {}),
            "itemDataVersion": source_question.get("itemDataVersion", {"major": 2, "minor": 0}),
        }

        prompt = f"""Given this source question in Perseus v2.0 JSON format:

```json
{json.dumps(perseus_json, indent=2)}
```

Generate a NEW question that:
1. Tests the same skill/concept as the source question
2. Uses DIFFERENT numbers, values, or context
3. Has a mathematically/factually correct answer
4. Follows the EXACT same widget structure and types
5. Maintains the same difficulty level

IMPORTANT:
- Return ONLY valid JSON (no markdown code blocks)
- Use the exact same widget IDs and structure
- Ensure all answers are correct
- Keep LaTeX syntax compatible with KaTeX
- Use Khan Academy color commands (\\blueD{{}}, \\redD{{}}, etc.) as in the source
"""

        if validation_feedback:
            prompt += "\n\nPREVIOUS VALIDATION ERRORS (fix these):\n"
            for error in validation_feedback:
                prompt += f"- {error}\n"

        prompt += "\n\nReturn the new question JSON:"

        return prompt

    def _parse_json_response(self, response_text: str) -> Optional[dict[str, Any]]:
        """Parse JSON from the response text."""
        if not response_text:
            return None

        # Clean up the response
        text = response_text.strip()

        # Remove markdown code blocks if present
        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]

        if text.endswith("```"):
            text = text[:-3]

        text = text.strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            logger.error(f"[GEMINI] JSON parse error: {e}")
            logger.debug(f"[GEMINI] Raw response: {text[:500]}")

            # Try to find JSON in the response
            json_match = self._extract_json_object(text)
            if json_match:
                try:
                    return json.loads(json_match)
                except json.JSONDecodeError:
                    pass

            return None

    def _extract_json_object(self, text: str) -> Optional[str]:
        """Try to extract a JSON object from text."""
        # Find the first { and last }
        start = text.find("{")
        end = text.rfind("}")

        if start != -1 and end != -1 and end > start:
            return text[start:end + 1]

        return None


# Singleton instance (lazy initialization)
_gemini_client: Optional[GeminiClient] = None


def get_gemini_client() -> GeminiClient:
    """Get or create the Gemini client instance."""
    global _gemini_client
    if _gemini_client is None:
        _gemini_client = GeminiClient()
    return _gemini_client

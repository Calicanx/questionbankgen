"""Google Gemini API client for question generation (UPDATED: google.genai)."""

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

# New SDK import
from google import genai
from google.genai import types

from tenacity import retry, stop_after_attempt, wait_exponential

from questionbank.config import config
from questionbank.utils.gcs import get_gcs_client

logger = logging.getLogger(__name__)

# Directory to store generated images
GENERATED_IMAGES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "generated_images")


class GeminiClient:
    """Client for Google Gemini API using the google.genai SDK.
    """

    def __init__(self) -> None:
        self.api_key = config.gemini.api_key
        self.model_name = config.gemini.model
        self.image_model_name = config.gemini.image_model
        self.temperature = config.gemini.temperature

        if not self.api_key:
            raise ValueError("GEMINI_API_KEY is not set")

        # Initialize the genai client
        # The google.genai library exposes a Client which we use to call models.generate_content
        self.client = genai.Client(api_key=self.api_key)

        # Ensure generated images directory exists
        os.makedirs(GENERATED_IMAGES_DIR, exist_ok=True)

        # Initialize GCS client
        self.gcs_client = get_gcs_client()

        logger.info(f"[GEMINI] Initialized with model: {self.model_name}")
        logger.info(f"[GEMINI] Image model: {self.image_model_name}")

    # ------------------------------------------------------------------
    # TEXT GENERATION
    # ------------------------------------------------------------------
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate text using Gemini via google.genai.

        Returns the textual response, or empty string on no text.
        """
        try:
            # Build contents list (system message first if provided)
            contents = []
            if system_prompt:
                contents.append(
                    types.Content(
                        role="system",
                        parts=[types.Part(text=system_prompt)],
                    )
                )

            contents.append(
                types.Content(
                    role="user",
                    parts=[types.Part(text=prompt)],
                )
            )

            response = self.client.models.generate_content(
                model=self.model_name,
                contents=contents,
                config=types.GenerateContentConfig(
                    temperature=self.temperature,
                    top_p=0.95,
                    top_k=40,
                    max_output_tokens=8192,
                ),
            )

            # The genai response exposes .text when a text candidate is returned
            if getattr(response, "text", None):
                return response.text

            # Fallback: try to collect text from candidates
            texts = []
            for c in getattr(response, "candidates", []) or []:
                try:
                    texts.append(c.content.text or "")
                except Exception:
                    pass

            if texts:
                return "\n".join(t for t in texts if t)

            logger.warning("[GEMINI] Empty response received")
            return ""

        except Exception as e:
            logger.error(f"[GEMINI] Generation error: {e}")
            raise

    # ------------------------------------------------------------------
    # IMAGE GENERATION (TEXT → IMAGE)
    # ------------------------------------------------------------------
    def generate_image(self, prompt: str, save_path: Optional[str] = None) -> Optional[str]:
        """Generate an image from text prompt and return either a GCS URL or local path.

        The implementation looks for `inline_data` in returned parts and uploads
        to GCS (via the existing get_gcs_client) or saves locally as a fallback.
        """
        try:
            logger.info(f"[GEMINI] Generating image with prompt: {prompt[:100]}...")

            response = self.client.models.generate_content(
                model=self.image_model_name,
                contents=[types.Content(role="user", parts=[types.Part(text=prompt)])],
                config=types.GenerateContentConfig(
                    temperature=1.0,
                    top_p=0.95,
                    top_k=40,
                    response_modalities=["IMAGE", "TEXT"],
                ),
            )

            if not getattr(response, "candidates", None):
                logger.warning("[GEMINI] No candidates in image response")
                return None

            # Extract image from response candidates
            candidate = response.candidates[0]
            for part in getattr(candidate.content, "parts", []) or []:
                inline = getattr(part, "inline_data", None)
                if inline and getattr(inline, "data", None):
                    image_data = inline.data
                    mime_type = inline.mime_type or "image/png"

                    # Determine extension
                    ext = ".png"
                    if "jpeg" in mime_type or "jpg" in mime_type:
                        ext = ".jpg"
                    elif "webp" in mime_type:
                        ext = ".webp"

                    if isinstance(image_data, str):
                        image_bytes = base64.b64decode(image_data)
                    else:
                        image_bytes = image_data

                    filename = f"generated_{uuid.uuid4().hex[:8]}{ext}"
                    blob_name = f"generated_images/{filename}"

                    # Try upload to GCS
                    try:
                        gcs_url = self.gcs_client.upload_bytes(image_bytes, blob_name, content_type=mime_type)
                        if gcs_url:
                            return gcs_url
                    except Exception as e:
                        logger.warning(f"[GEMINI] GCS upload failed: {e}")

                    # Fallback: save locally
                    if not save_path:
                        save_path = os.path.join(GENERATED_IMAGES_DIR, filename)

                    with open(save_path, "wb") as f:
                        f.write(image_bytes)

                    logger.info(f"[GEMINI] Image saved locally to: {save_path}")
                    return save_path

            # If we didn't find inline_data, check for textual response
            if getattr(response, "text", None):
                logger.warning(f"[GEMINI] Got text instead of image: {response.text[:200]}")

            logger.warning("[GEMINI] No image returned")
            return None

        except Exception as e:
            logger.error(f"[GEMINI] Image generation error: {e}")
            return None

    # ------------------------------------------------------------------
    # IMAGE FROM REFERENCE
    # ------------------------------------------------------------------
    def generate_image_from_reference(
        self,
        source_image_url: str,
        new_context: str,
        style_instructions: str = "Match the style, composition, and quality of the reference image",
    ) -> Optional[str]:
        """Generate a new image using a reference image and textual instruction.

        This mirrors the original behaviour: it downloads the source image (with
        some heuristics for graphie/svg URLs), attempts SVG → PNG conversions via
        Wand or svglib if needed, and then sends the reference image to the
        image model together with the prompt.
        """
        try:
            logger.info(f"[GEMINI] Generating image from reference: {source_image_url[:60]}...")
            logger.info(f"[GEMINI] New context: {new_context[:100]}...")

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

            img_url = source_image_url
            if img_url.startswith("web+graphie://"):
                img_url = img_url.replace("web+graphie://", "https://") + ".svg"

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
                except Exception:
                    continue

            if not response or response.status_code != 200:
                raise requests.RequestException(f"All URL variations failed for {source_image_url[:60]}")

            content_type = response.headers.get('Content-Type', '')
            image_data = BytesIO(response.content)

            source_image = None
            # Handle SVG specially
            if 'svg' in content_type or url.endswith('.svg'):
                # Try ImageMagick/Wand conversion
                try:
                    from wand.image import Image as WandImage
                    with WandImage(blob=response.content, format='svg') as img:
                        img.format = 'png'
                        png_data = BytesIO(img.make_blob())
                        source_image = Image.open(png_data)
                        logger.info("[GEMINI] Converted SVG to PNG using Wand/ImageMagick")
                except Exception as e:
                    logger.warning(f"[GEMINI] Wand conversion failed: {e}")

                # Fallback to svglib + reportlab
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
                            png_temp = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
                            renderPM.drawToFile(drawing, png_temp.name, fmt='PNG')
                            png_temp.close()
                            source_image = Image.open(png_temp.name)
                            logger.info("[GEMINI] Converted SVG to PNG using svglib")

                        try:
                            os.unlink(svg_path)
                        except Exception:
                            pass
                    except Exception as e:
                        logger.warning(f"[GEMINI] svglib conversion failed: {e}")

                if source_image is None:
                    logger.warning("[GEMINI] Cannot convert SVG, skipping reference image")
                    return None
            else:
                source_image = Image.open(image_data)

            # Build prompt
            prompt = f"""Look at this reference image carefully. I need you to generate a NEW image that:

1. MATCHES THE STYLE: {style_instructions}
2. SHOWS NEW CONTENT: {new_context}

The new image should:
- Have similar visual quality and style as the reference
- Be appropriate for educational use (K-12 level)
- Be clear, professional, and engaging
- NOT copy the reference image, but create something NEW for the described topic

Generate the new image now."""

            # Prepare contents. The google.genai SDK allows embedding binary image parts; we create
            # a parts list: textual prompt + the image as an image part. If the environment's
            # genai SDK doesn't support Part.from_image, the code below will still try to attach
            # raw bytes via types.Part with `image_bytes` passed as inline_data when possible.

            # Convert source_image to bytes PNG
            img_buf = BytesIO()
            source_image.save(img_buf, format='PNG')
            img_bytes = img_buf.getvalue()

            # Create parts: text part and image part
            text_part = types.Part(text=prompt)

            # Try best-effort to create an image part. Different genai versions expose different
            # helpers; try Part.from_image if available, otherwise set inline_data manually.
            image_part = None
            try:
                # This helper may exist in newer genai versions
                image_part = types.Part.from_image(img_bytes, mime_type='image/png')
            except Exception:
                # Fallback: construct a Part with inline_data structure if supported
                try:
                    inline = types.InlineData(data=base64.b64encode(img_bytes).decode('utf-8'), mime_type='image/png')
                    image_part = types.Part(inline_data=inline)
                except Exception:
                    # Last resort: we will send only text prompt (some models accept a separate image param)
                    image_part = None

            parts = [text_part]
            if image_part is not None:
                parts.append(image_part)

            contents = [types.Content(role='user', parts=parts)]

            response = self.client.models.generate_content(
                model=self.image_model_name,
                contents=contents,
                config=types.GenerateContentConfig(
                    response_modalities=['IMAGE'],
                    temperature=1.0,
                ),
            )

            if not getattr(response, 'candidates', None):
                logger.warning("[GEMINI] No candidates in response")
                return None

            candidate = response.candidates[0]
            for part in getattr(candidate.content, 'parts', []) or []:
                inline = getattr(part, 'inline_data', None)
                if inline and getattr(inline, 'data', None):
                    image_data = inline.data
                    mime_type = inline.mime_type or 'image/png'

                    ext = '.png'
                    if 'jpeg' in mime_type or 'jpg' in mime_type:
                        ext = '.jpg'

                    if isinstance(image_data, str):
                        image_bytes = base64.b64decode(image_data)
                    else:
                        image_bytes = image_data

                    filename = f"generated_{uuid.uuid4().hex[:8]}{ext}"
                    blob_name = f"generated_images/{filename}"

                    try:
                        gcs_url = self.gcs_client.upload_bytes(image_bytes, blob_name, content_type=mime_type)
                        if gcs_url:
                            return gcs_url
                    except Exception as e:
                        logger.warning(f"[GEMINI] GCS upload failed: {e}")

                    save_path = os.path.join(GENERATED_IMAGES_DIR, filename)
                    with open(save_path, 'wb') as f:
                        f.write(image_bytes)

                    logger.info(f"[GEMINI] Generated image saved: {save_path}")
                    return save_path

            # Alternative: some responses put an `image` object on parts[0]
            try:
                if hasattr(candidate.content.parts[0], 'image'):
                    img_obj = candidate.content.parts[0].image
                    filename = f"generated_{uuid.uuid4().hex[:8]}.png"
                    img_byte_arr = BytesIO()
                    img_obj.save(img_byte_arr, format='PNG')
                    image_bytes = img_byte_arr.getvalue()

                    blob_name = f"generated_images/{filename}"
                    try:
                        gcs_url = self.gcs_client.upload_bytes(image_bytes, blob_name, content_type='image/png')
                        if gcs_url:
                            return gcs_url
                    except Exception as e:
                        logger.warning(f"[GEMINI] GCS upload failed: {e}")

                    save_path = os.path.join(GENERATED_IMAGES_DIR, filename)
                    with open(save_path, 'wb') as f:
                        f.write(image_bytes)
                    logger.info(f"[GEMINI] Generated image saved: {save_path}")
                    return save_path
            except Exception:
                pass

            logger.warning('[GEMINI] No image data in response')
            return None

        except requests.RequestException as e:
            logger.error(f"[GEMINI] Failed to download source image: {e}")
            return None
        except Exception as e:
            logger.error(f"[GEMINI] Image generation from reference failed: {e}")
            return None

    # ------------------------------------------------------------------
    # QUESTION JSON GENERATION
    # ------------------------------------------------------------------
    def generate_question_json(
        self,
        source_question: dict[str, Any],
        system_prompt: str,
        validation_feedback: Optional[list[str]] = None,
        prompt: Optional[str] = None,
    ) -> Optional[dict[str, Any]]:
        """Generate a new question JSON from a source question."""
        try:
            if not prompt:
                prompt = self._build_generation_prompt(source_question, validation_feedback)

            response_text = self.generate(prompt, system_prompt)
            return self._parse_json_response(response_text)

        except Exception as e:
            logger.error(f"[GEMINI] Question generation error: {e}")
            return None

    # ------------------------------------------------------------------
    # PROMPT BUILDING & JSON PARSING
    # ------------------------------------------------------------------
    def _build_generation_prompt(
        self,
        source_question: dict[str, Any],
        validation_feedback: Optional[list[str]] = None,
    ) -> str:
        """Build the prompt for question generation."""
        perseus_json = {
            "question": source_question.get("question", {}),
            "hints": source_question.get("hints", []),
            "answerArea": source_question.get("answerArea", {}),
            "itemDataVersion": source_question.get("itemDataVersion", {"major": 2, "minor": 0}),
        }

        prompt = f"""Given this source question in Perseus v2.0 JSON format:
            SOURCE QUESTION (FOR CONCEPT ONLY — DO NOT COPY):

            ```json
            {json.dumps(perseus_json, indent=2)}
            ```

            Generate a NEW question that:
            1. Tests the same skill/concept as the source question
            2. Uses DIFFERENT numbers, values, and MOST IMPORTANTLY:
            - A COMPLETELY DIFFERENT REAL-WORLD content but relevant CONTEXT/SCENARIO to reference question
            3. Maintains the same difficulty level
            4. Ensure the words don't match at 60%

            IMPORTANT:
            - Return ONLY valid JSON (no markdown code blocks)
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
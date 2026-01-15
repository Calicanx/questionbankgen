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

from tenacity import retry, stop_after_attempt, wait_exponential
from PIL import Image
from google import genai
from google.genai import types

from questionbank.config import config
from questionbank.utils.gcs import get_gcs_client

logger = logging.getLogger(__name__)

GENERATED_IMAGES_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "generated_images",
)


class GeminiClient:
    """Client for Google Gemini API (google.genai)."""

    def __init__(self) -> None:
        self.api_key = config.gemini.api_key
        self.model_name = config.gemini.model
        self.image_model_name = config.gemini.image_model
        self.temperature = config.gemini.temperature

        if not self.api_key:
            raise ValueError("GEMINI_API_KEY is not set")

        self.client = genai.Client(api_key=self.api_key)

        os.makedirs(GENERATED_IMAGES_DIR, exist_ok=True)
        self.gcs_client = get_gcs_client()

        logger.info(f"[GEMINI] Text model: {self.model_name}")
        logger.info(f"[GEMINI] Image model: {self.image_model_name}")

    # ------------------------------------------------------------------
    # TEXT GENERATION
    # ------------------------------------------------------------------
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(1, 2, 10))
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
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

        return response.text or ""

    # ------------------------------------------------------------------
    # IMAGE GENERATION (TEXT → IMAGE)
    # ------------------------------------------------------------------
    def generate_image(self, prompt: str) -> Optional[str]:
        logger.info(f"[GEMINI] Generating image: {prompt[:80]}...")

        response = self.client.models.generate_content(
            model=self.image_model_name,
            contents=[types.Content(
                role="user",
                parts=[types.Part(text=prompt)],
            )],
            config=types.GenerateContentConfig(
                response_modalities=["IMAGE", "TEXT"],
                temperature=1.0,
            ),
        )

        for part in response.candidates[0].content.parts:
            if part.inline_data:
                image_bytes = base64.b64decode(part.inline_data.data)
                mime_type = part.inline_data.mime_type or "image/png"

                ext = ".png"
                if "jpeg" in mime_type:
                    ext = ".jpg"
                elif "webp" in mime_type:
                    ext = ".webp"

                filename = f"generated_{uuid.uuid4().hex[:8]}{ext}"
                blob_name = f"generated_images/{filename}"

                gcs_url = self.gcs_client.upload_bytes(
                    image_bytes,
                    blob_name,
                    content_type=mime_type,
                )
                if gcs_url:
                    return gcs_url

                save_path = os.path.join(GENERATED_IMAGES_DIR, filename)
                with open(save_path, "wb") as f:
                    f.write(image_bytes)

                return save_path

        logger.warning("[GEMINI] No image returned")
        return None

    # ------------------------------------------------------------------
    # IMAGE FROM REFERENCE
    # ------------------------------------------------------------------
    def generate_image_from_reference(
        self,
        source_image_url: str,
        new_context: str,
        style_instructions: str = "Match style and quality",
    ) -> Optional[str]:

        logger.info(f"[GEMINI] Reference image: {source_image_url}")

        r = requests.get(source_image_url, timeout=15)
        r.raise_for_status()

        source_image = Image.open(BytesIO(r.content))

        prompt = f"""
Using the provided reference image:

STYLE:
{style_instructions}

NEW CONTENT:
{new_context}

Rules:
- Educational quality
- Original (not a copy)
- Clean background
"""

        response = self.client.models.generate_content(
            model=self.image_model_name,
            contents=[
                types.Content(
                    role="user",
                    parts=[
                        types.Part(text=prompt),
                        types.Part.from_image(source_image),
                    ],
                )
            ],
            config=types.GenerateContentConfig(
                response_modalities=["IMAGE"],
                temperature=1.0,
            ),
        )

        for part in response.candidates[0].content.parts:
            if part.inline_data:
                image_bytes = base64.b64decode(part.inline_data.data)

                filename = f"generated_{uuid.uuid4().hex[:8]}.png"
                blob_name = f"generated_images/{filename}"

                gcs_url = self.gcs_client.upload_bytes(
                    image_bytes,
                    blob_name,
                    content_type="image/png",
                )
                if gcs_url:
                    return gcs_url

                save_path = os.path.join(GENERATED_IMAGES_DIR, filename)
                with open(save_path, "wb") as f:
                    f.write(image_bytes)

                return save_path

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

        if not prompt:
            prompt = self._build_generation_prompt(
                source_question,
                validation_feedback,
            )

        response_text = self.generate(prompt, system_prompt)
        return self._parse_json_response(response_text)

    # ------------------------------------------------------------------
    # PROMPT BUILDING & JSON PARSING
    # ------------------------------------------------------------------
    def _build_generation_prompt(
        self,
        source_question: dict[str, Any],
        validation_feedback: Optional[list[str]],
    ) -> str:

        perseus_json = {
            "question": source_question.get("question", {}),
            "hints": source_question.get("hints", []),
            "answerArea": source_question.get("answerArea", {}),
            "itemDataVersion": source_question.get(
                "itemDataVersion",
                {"major": 2, "minor": 0},
            ),
        }

        prompt = f"""
SOURCE QUESTION (Perseus v2):

{json.dumps(perseus_json, indent=2)}

TASK:
- Same skill
- Different context
- Same structure
- Correct math
- Valid KaTeX
- JSON only
"""

        if validation_feedback:
            prompt += "\nVALIDATION ERRORS:\n"
            for err in validation_feedback:
                prompt += f"- {err}\n"

        return prompt

    def _parse_json_response(self, text: str) -> Optional[dict[str, Any]]:
        if not text:
            return None

        text = text.strip().strip("```").strip("json").strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start, end = text.find("{"), text.rfind("}")
            if start != -1 and end != -1:
                try:
                    return json.loads(text[start : end + 1])
                except Exception:
                    pass

        logger.error("[GEMINI] Failed to parse JSON")
        return None


# ------------------------------------------------------------------
# SINGLETON
# ------------------------------------------------------------------
_gemini_client: Optional[GeminiClient] = None


def get_gemini_client() -> GeminiClient:
    global _gemini_client
    if _gemini_client is None:
        _gemini_client = GeminiClient()
    return _gemini_client

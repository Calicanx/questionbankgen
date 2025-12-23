"""Programmatic and AI image generation for question elements."""

import os
import logging
import hashlib
import base64
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Try to import Google GenAI for Imagen 3
try:
    from google import genai
    from google.genai import types
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False
    logger.warning("google-genai not available - AI image generation disabled")

# Try to import PIL for image generation
try:
    from PIL import Image, ImageDraw, ImageFont
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    logger.warning("PIL not available - image generation disabled")


@dataclass
class GeneratedImage:
    """A programmatically generated image."""
    filepath: str
    url: str
    width: int
    height: int
    alt_text: str


class ImageGenerator:
    """Generates images programmatically for question elements."""

    # Colors for place value blocks
    HUNDRED_COLOR = (66, 133, 244)  # Blue
    TEN_COLOR = (52, 168, 83)       # Green
    ONE_COLOR = (251, 188, 4)       # Yellow
    BACKGROUND_COLOR = (255, 255, 255)  # White
    OUTLINE_COLOR = (100, 100, 100)  # Gray

    def __init__(self, output_dir: Optional[str] = None):
        """Initialize with output directory for generated images."""
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            # Default to project's generated_images directory
            self.output_dir = Path(__file__).parent.parent / 'generated_images'

        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Image generator output dir: {self.output_dir}")

    def generate_place_value_blocks(
        self,
        number: int,
        include_labels: bool = False
    ) -> Optional[GeneratedImage]:
        """
        Generate an image showing place value blocks for a number.

        Args:
            number: The number to represent (0-999)
            include_labels: Whether to include text labels

        Returns:
            GeneratedImage with filepath and metadata
        """
        if not HAS_PIL:
            logger.warning("Cannot generate image - PIL not available")
            return None

        if number < 0 or number > 999:
            logger.warning(f"Number {number} out of range for place value blocks")
            return None

        hundreds = number // 100
        tens = (number % 100) // 10
        ones = number % 10

        # Calculate image dimensions
        block_size = 20
        padding = 15
        spacing = 10

        # Width calculation
        hundreds_width = hundreds * (block_size * 2 + spacing) if hundreds else 0
        tens_width = tens * (block_size + spacing) if tens else 0
        ones_width = ones * (block_size + spacing) if ones else 0

        total_width = max(200, padding * 2 + hundreds_width + tens_width + ones_width + spacing * 2)
        total_height = 120 if not include_labels else 150

        # Create image
        img = Image.new('RGB', (total_width, total_height), self.BACKGROUND_COLOR)
        draw = ImageDraw.Draw(img)

        x_offset = padding
        y_center = total_height // 2 - 10

        # Draw hundreds (10x10 flats)
        flat_size = block_size * 2
        for i in range(hundreds):
            x = x_offset + i * (flat_size + spacing)
            y = y_center - flat_size // 2

            # Draw flat (10x10 grid appearance)
            draw.rectangle(
                [x, y, x + flat_size, y + flat_size],
                fill=self.HUNDRED_COLOR,
                outline=self.OUTLINE_COLOR,
                width=2
            )
            # Add grid lines
            for j in range(1, 10):
                line_pos = j * (flat_size // 10)
                draw.line([(x + line_pos, y), (x + line_pos, y + flat_size)], fill=self.OUTLINE_COLOR)
                draw.line([(x, y + line_pos), (x + flat_size, y + line_pos)], fill=self.OUTLINE_COLOR)

        if hundreds:
            x_offset += hundreds * (flat_size + spacing) + spacing

        # Draw tens (rods)
        rod_width = block_size // 2
        rod_height = block_size * 2
        for i in range(tens):
            x = x_offset + i * (rod_width + spacing // 2)
            y = y_center - rod_height // 2

            draw.rectangle(
                [x, y, x + rod_width, y + rod_height],
                fill=self.TEN_COLOR,
                outline=self.OUTLINE_COLOR,
                width=1
            )
            # Add segment lines
            for j in range(1, 10):
                line_y = y + j * (rod_height // 10)
                draw.line([(x, line_y), (x + rod_width, line_y)], fill=self.OUTLINE_COLOR)

        if tens:
            x_offset += tens * (rod_width + spacing // 2) + spacing

        # Draw ones (unit cubes)
        cube_size = block_size // 2
        for i in range(ones):
            x = x_offset + i * (cube_size + spacing // 3)
            y = y_center - cube_size // 2

            draw.rectangle(
                [x, y, x + cube_size, y + cube_size],
                fill=self.ONE_COLOR,
                outline=self.OUTLINE_COLOR,
                width=1
            )

        # Add labels if requested
        if include_labels:
            try:
                font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 12)
            except:
                font = ImageFont.load_default()

            label_y = total_height - 25
            if hundreds:
                draw.text((padding, label_y), f"{hundreds} hundreds", fill=(0, 0, 0), font=font)
            if tens:
                draw.text((padding + hundreds_width + spacing, label_y), f"{tens} tens", fill=(0, 0, 0), font=font)
            if ones:
                draw.text((padding + hundreds_width + tens_width + spacing * 2, label_y), f"{ones} ones", fill=(0, 0, 0), font=font)

        # Generate filename based on content
        filename = f"place_value_{number}.png"
        filepath = self.output_dir / filename

        # Save image
        img.save(filepath, 'PNG')

        # Generate alt text
        alt_parts = []
        if hundreds:
            alt_parts.append(f"{hundreds} hundred-cube flat{'s' if hundreds > 1 else ''}")
        if tens:
            alt_parts.append(f"{tens} ten-cube rod{'s' if tens > 1 else ''}")
        if ones:
            alt_parts.append(f"{ones} unit cube{'s' if ones > 1 else ''}")

        alt_text = ", ".join(alt_parts) if alt_parts else "empty place value model"

        # Generate URL (relative to static serving)
        url = f"/static/generated_images/{filename}"

        return GeneratedImage(
            filepath=str(filepath),
            url=url,
            width=total_width,
            height=total_height,
            alt_text=alt_text
        )

    def generate_number_line(
        self,
        min_val: float,
        max_val: float,
        marked_points: list[float],
        labels: Optional[list[str]] = None,
        show_fractions: bool = False
    ) -> Optional[GeneratedImage]:
        """
        Generate a number line image with marked points.

        Args:
            min_val: Minimum value on number line
            max_val: Maximum value on number line
            marked_points: Points to mark on the line
            labels: Optional labels for marked points (A, B, C, etc.)
            show_fractions: Show values as fractions

        Returns:
            GeneratedImage with filepath and metadata
        """
        if not HAS_PIL:
            return None

        width = 500
        height = 100
        padding = 40
        line_y = height // 2

        img = Image.new('RGB', (width, height), self.BACKGROUND_COLOR)
        draw = ImageDraw.Draw(img)

        # Draw main line
        line_start = padding
        line_end = width - padding
        draw.line([(line_start, line_y), (line_end, line_y)], fill=(0, 0, 0), width=2)

        # Draw endpoints and ticks
        range_val = max_val - min_val
        tick_count = 10

        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 10)
            small_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 8)
        except:
            font = ImageFont.load_default()
            small_font = font

        for i in range(tick_count + 1):
            x = line_start + (line_end - line_start) * i / tick_count
            tick_val = min_val + range_val * i / tick_count

            # Draw tick
            draw.line([(x, line_y - 5), (x, line_y + 5)], fill=(0, 0, 0), width=1)

            # Draw label for endpoints and middle
            if i == 0 or i == tick_count or i == tick_count // 2:
                label = f"{tick_val:.1f}" if tick_val != int(tick_val) else str(int(tick_val))
                draw.text((x - 10, line_y + 10), label, fill=(0, 0, 0), font=small_font)

        # Draw marked points
        point_labels = labels or [chr(65 + i) for i in range(len(marked_points))]  # A, B, C...
        colors = [(66, 133, 244), (234, 67, 53), (52, 168, 83), (251, 188, 4)]  # Blue, Red, Green, Yellow

        for i, point in enumerate(marked_points):
            if min_val <= point <= max_val:
                x = line_start + (line_end - line_start) * (point - min_val) / range_val
                color = colors[i % len(colors)]

                # Draw point
                draw.ellipse([(x - 6, line_y - 6), (x + 6, line_y + 6)], fill=color, outline=(0, 0, 0))

                # Draw label
                if i < len(point_labels):
                    draw.text((x - 3, line_y - 20), point_labels[i], fill=color, font=font)

        # Generate filename
        points_hash = hashlib.md5(str(marked_points).encode()).hexdigest()[:8]
        filename = f"number_line_{int(min_val)}_{int(max_val)}_{points_hash}.png"
        filepath = self.output_dir / filename

        img.save(filepath, 'PNG')

        # Alt text
        alt_text = f"Number line from {min_val} to {max_val} with points marked"

        return GeneratedImage(
            filepath=str(filepath),
            url=f"/static/generated_images/{filename}",
            width=width,
            height=height,
            alt_text=alt_text
        )

    def generate_simple_shape(
        self,
        shape: str,  # 'rectangle', 'triangle', 'circle'
        dimensions: dict,  # {'width': 100, 'height': 50} or {'radius': 30}
        color: tuple = (66, 133, 244),
        show_dimensions: bool = True
    ) -> Optional[GeneratedImage]:
        """Generate a simple geometric shape."""
        if not HAS_PIL:
            return None

        padding = 40
        width = 250
        height = 200

        img = Image.new('RGB', (width, height), self.BACKGROUND_COLOR)
        draw = ImageDraw.Draw(img)

        center_x = width // 2
        center_y = height // 2

        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 12)
        except:
            font = ImageFont.load_default()

        if shape == 'rectangle':
            w = dimensions.get('width', 80)
            h = dimensions.get('height', 50)
            scale = min((width - padding * 2) / w, (height - padding * 2) / h, 1.5)
            w_scaled = int(w * scale)
            h_scaled = int(h * scale)

            x1 = center_x - w_scaled // 2
            y1 = center_y - h_scaled // 2
            x2 = center_x + w_scaled // 2
            y2 = center_y + h_scaled // 2

            draw.rectangle([x1, y1, x2, y2], fill=color, outline=(0, 0, 0), width=2)

            if show_dimensions:
                draw.text((center_x - 15, y2 + 5), f"{w} cm", fill=(0, 0, 0), font=font)
                draw.text((x2 + 5, center_y - 5), f"{h} cm", fill=(0, 0, 0), font=font)

        elif shape == 'circle':
            r = dimensions.get('radius', 40)
            scale = min((width - padding * 2) / (r * 2), (height - padding * 2) / (r * 2), 1.5)
            r_scaled = int(r * scale)

            draw.ellipse(
                [center_x - r_scaled, center_y - r_scaled, center_x + r_scaled, center_y + r_scaled],
                fill=color,
                outline=(0, 0, 0),
                width=2
            )

            if show_dimensions:
                draw.line([(center_x, center_y), (center_x + r_scaled, center_y)], fill=(0, 0, 0), width=1)
                draw.text((center_x + r_scaled // 2 - 10, center_y + 5), f"r={r}", fill=(0, 0, 0), font=font)

        elif shape == 'triangle':
            base = dimensions.get('base', 80)
            h = dimensions.get('height', 60)
            scale = min((width - padding * 2) / base, (height - padding * 2) / h, 1.5)

            b_scaled = int(base * scale)
            h_scaled = int(h * scale)

            points = [
                (center_x, center_y - h_scaled // 2),  # top
                (center_x - b_scaled // 2, center_y + h_scaled // 2),  # bottom left
                (center_x + b_scaled // 2, center_y + h_scaled // 2),  # bottom right
            ]

            draw.polygon(points, fill=color, outline=(0, 0, 0))

            if show_dimensions:
                draw.text((center_x - 15, center_y + h_scaled // 2 + 5), f"{base} cm", fill=(0, 0, 0), font=font)

        # Generate filename
        dims_str = "_".join(f"{k}{v}" for k, v in dimensions.items())
        filename = f"{shape}_{dims_str}.png"
        filepath = self.output_dir / filename

        img.save(filepath, 'PNG')

        return GeneratedImage(
            filepath=str(filepath),
            url=f"/static/generated_images/{filename}",
            width=width,
            height=height,
            alt_text=f"A {shape} with dimensions {dimensions}"
        )


class GeminiImageGenerator:
    """AI-powered image generation using Google Imagen 4.0."""

    MODEL_NAME = "imagen-4.0-generate-001"

    # Educational image prompt templates
    PROMPT_TEMPLATES = {
        "place_value_blocks": (
            "Educational math illustration showing place value blocks on a clean white background. "
            "Show {hundreds} blue 10x10 grid flats (hundreds), {tens} green vertical rods (tens), "
            "and {ones} yellow small unit cubes (ones). The blocks should be clearly separated "
            "and arranged left to right. Simple, clean, child-friendly educational style. "
            "No text or numbers on the image."
        ),
        "number_line": (
            "Clean educational number line illustration on white background. "
            "Horizontal line from {min_val} to {max_val} with evenly spaced tick marks. "
            "Points marked at positions {points} with colored dots. "
            "Simple, clear math education style suitable for elementary students."
        ),
        "geometric_shape": (
            "Clean educational geometry illustration on white background. "
            "A {shape} with {dimensions}. Show dimension labels clearly. "
            "Simple line drawing style suitable for math education. "
            "Professional, clean appearance."
        ),
        "fraction_visual": (
            "Educational fraction illustration on white background. "
            "A {shape} divided into {denominator} equal parts with {numerator} parts shaded in blue. "
            "Clean, simple style for elementary math education."
        ),
        "bar_graph": (
            "Simple educational bar graph on white background. "
            "Showing {num_bars} vertical bars with values {values}. "
            "Labels: {labels}. Clean grid lines, professional education style."
        ),
        "pie_chart": (
            "Clean educational pie chart on white background. "
            "Showing {num_slices} slices with percentages {percentages}. "
            "Different colors for each slice, labels visible. Education style."
        ),
        "custom": "{prompt}"
    }

    def __init__(self, output_dir: Optional[str] = None):
        """Initialize with output directory and API client."""
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            self.output_dir = Path(__file__).parent.parent / 'generated_images'

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize Gemini client
        self.client = None
        if HAS_GENAI:
            api_key = os.environ.get("GEMINI_API_KEY")
            if api_key:
                self.client = genai.Client(api_key=api_key)
                logger.info("Gemini Imagen 3 client initialized")
            else:
                logger.warning("GEMINI_API_KEY not set - AI image generation unavailable")

    def is_available(self) -> bool:
        """Check if AI image generation is available."""
        return self.client is not None

    def generate_image(
        self,
        prompt: str,
        filename: str,
        aspect_ratio: str = "1:1",
        alt_text: Optional[str] = None
    ) -> Optional[GeneratedImage]:
        """
        Generate an image using Imagen 3.

        Args:
            prompt: Text description of the image to generate
            filename: Name for the output file (without extension)
            aspect_ratio: Image aspect ratio ("1:1", "3:4", "4:3", "9:16", "16:9")
            alt_text: Alt text for accessibility

        Returns:
            GeneratedImage with filepath and metadata, or None on failure
        """
        if not self.client:
            logger.error("Gemini client not initialized")
            return None

        try:
            response = self.client.models.generate_images(
                model=self.MODEL_NAME,
                prompt=prompt,
                config=types.GenerateImagesConfig(
                    number_of_images=1,
                    aspect_ratio=aspect_ratio,
                    safety_filter_level="BLOCK_LOW_AND_ABOVE",
                )
            )

            if not response.generated_images:
                logger.error("No images generated")
                return None

            # Get the first generated image
            gen_image = response.generated_images[0]

            # Save the image
            filepath = self.output_dir / f"{filename}.png"

            # The image data is in gen_image.image.image_bytes
            if hasattr(gen_image.image, 'image_bytes'):
                image_data = gen_image.image.image_bytes
            elif hasattr(gen_image.image, '_pil_image'):
                # Save PIL image directly
                gen_image.image._pil_image.save(filepath, 'PNG')
                img = gen_image.image._pil_image
                return GeneratedImage(
                    filepath=str(filepath),
                    url=f"/static/generated_images/{filename}.png",
                    width=img.width,
                    height=img.height,
                    alt_text=alt_text or prompt[:100]
                )
            else:
                # Try to save using the show() method's underlying data
                gen_image.image.save(filepath)
                # Load to get dimensions
                img = Image.open(filepath)
                return GeneratedImage(
                    filepath=str(filepath),
                    url=f"/static/generated_images/{filename}.png",
                    width=img.width,
                    height=img.height,
                    alt_text=alt_text or prompt[:100]
                )

            # Write bytes to file
            with open(filepath, 'wb') as f:
                f.write(image_data)

            # Load to get dimensions
            img = Image.open(filepath)

            return GeneratedImage(
                filepath=str(filepath),
                url=f"/static/generated_images/{filename}.png",
                width=img.width,
                height=img.height,
                alt_text=alt_text or prompt[:100]
            )

        except Exception as e:
            logger.error(f"Image generation failed: {e}")
            return None

    def generate_place_value_blocks(
        self,
        number: int,
        include_labels: bool = False
    ) -> Optional[GeneratedImage]:
        """Generate place value blocks image using AI."""
        if number < 0 or number > 999:
            logger.warning(f"Number {number} out of range")
            return None

        hundreds = number // 100
        tens = (number % 100) // 10
        ones = number % 10

        prompt = self.PROMPT_TEMPLATES["place_value_blocks"].format(
            hundreds=hundreds,
            tens=tens,
            ones=ones
        )

        if include_labels:
            prompt += " Include small labels showing the place values."

        # Generate unique filename
        filename = f"ai_place_value_{number}"

        alt_parts = []
        if hundreds:
            alt_parts.append(f"{hundreds} hundred-cube flat{'s' if hundreds > 1 else ''}")
        if tens:
            alt_parts.append(f"{tens} ten-cube rod{'s' if tens > 1 else ''}")
        if ones:
            alt_parts.append(f"{ones} unit cube{'s' if ones > 1 else ''}")
        alt_text = ", ".join(alt_parts) if alt_parts else "empty place value model"

        return self.generate_image(prompt, filename, "4:3", alt_text)

    def generate_number_line(
        self,
        min_val: float,
        max_val: float,
        marked_points: list[float],
        labels: Optional[list[str]] = None
    ) -> Optional[GeneratedImage]:
        """Generate a number line image using AI."""
        points_str = ", ".join(str(p) for p in marked_points)

        prompt = self.PROMPT_TEMPLATES["number_line"].format(
            min_val=min_val,
            max_val=max_val,
            points=points_str
        )

        if labels:
            prompt += f" Label the points as {', '.join(labels)}."

        # Generate unique filename
        points_hash = hashlib.md5(str(marked_points).encode()).hexdigest()[:8]
        filename = f"ai_number_line_{int(min_val)}_{int(max_val)}_{points_hash}"

        alt_text = f"Number line from {min_val} to {max_val} with points at {points_str}"

        return self.generate_image(prompt, filename, "16:9", alt_text)

    def generate_geometric_shape(
        self,
        shape: str,
        dimensions: dict,
        color: str = "blue"
    ) -> Optional[GeneratedImage]:
        """Generate a geometric shape image using AI."""
        dims_str = ", ".join(f"{k}={v}" for k, v in dimensions.items())

        prompt = self.PROMPT_TEMPLATES["geometric_shape"].format(
            shape=shape,
            dimensions=dims_str
        )
        prompt += f" The shape should be {color}."

        # Generate unique filename
        dims_hash = hashlib.md5(str(dimensions).encode()).hexdigest()[:8]
        filename = f"ai_{shape}_{dims_hash}"

        alt_text = f"A {color} {shape} with {dims_str}"

        return self.generate_image(prompt, filename, "1:1", alt_text)

    def generate_fraction_visual(
        self,
        numerator: int,
        denominator: int,
        shape: str = "circle"
    ) -> Optional[GeneratedImage]:
        """Generate a fraction visualization using AI."""
        prompt = self.PROMPT_TEMPLATES["fraction_visual"].format(
            shape=shape,
            numerator=numerator,
            denominator=denominator
        )

        filename = f"ai_fraction_{numerator}_{denominator}_{shape}"
        alt_text = f"A {shape} showing {numerator}/{denominator}"

        return self.generate_image(prompt, filename, "1:1", alt_text)

    def generate_bar_graph(
        self,
        values: list[float],
        labels: list[str],
        title: Optional[str] = None
    ) -> Optional[GeneratedImage]:
        """Generate a bar graph using AI."""
        prompt = self.PROMPT_TEMPLATES["bar_graph"].format(
            num_bars=len(values),
            values=", ".join(str(v) for v in values),
            labels=", ".join(labels)
        )

        if title:
            prompt += f" Title: '{title}'."

        values_hash = hashlib.md5(str(values).encode()).hexdigest()[:8]
        filename = f"ai_bar_graph_{values_hash}"
        alt_text = f"Bar graph showing {', '.join(f'{l}: {v}' for l, v in zip(labels, values))}"

        return self.generate_image(prompt, filename, "4:3", alt_text)

    def generate_custom(
        self,
        prompt: str,
        filename: str,
        aspect_ratio: str = "1:1"
    ) -> Optional[GeneratedImage]:
        """Generate a custom educational image."""
        # Add educational context to prompt
        full_prompt = (
            f"Educational illustration for a math or science question. "
            f"Clean, professional style on white background. {prompt}"
        )

        return self.generate_image(full_prompt, filename, aspect_ratio, prompt[:100])


# Singleton instances
image_generator = ImageGenerator()
gemini_image_generator = GeminiImageGenerator()

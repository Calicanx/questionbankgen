"""Programmatic Image Generator for Educational Math Diagrams.

Generates mathematically accurate images for place value blocks,
counting objects, and other educational diagrams that AI image
models cannot reliably produce.
"""

import os
from pathlib import Path
from typing import Optional, Tuple
import hashlib
import uuid

from PIL import Image, ImageDraw

# Output directory for generated images
GENERATED_IMAGES_DIR = Path(__file__).parent.parent / "generated_images"
GENERATED_IMAGES_DIR.mkdir(exist_ok=True)

# Khan Academy style colors - orange/warm tones to match their place value blocks
COLORS = {
    "orange": (255, 165, 79),        # Orange for blocks (Khan Academy style)
    "orange_dark": (210, 120, 40),   # Darker orange for grid lines
    "orange_light": (255, 200, 140), # Lighter orange for highlight
    "purple": (148, 103, 189),       # Purple alternative
    "purple_dark": (118, 73, 159),   # Darker purple for grid lines
    "blue": (55, 126, 184),
    "grid_line": (160, 100, 30),     # Grid line color (dark orange)
    "background": (255, 255, 255),   # White background
    "white": (255, 255, 255),
}


def generate_place_value_blocks(
    number: int,
    output_path: Optional[str] = None,
    block_color: Tuple[int, int, int] = None,  # Default to orange
) -> str:
    """Generate a place value blocks image for a given number.

    Creates an accurate representation with:
    - Hundreds blocks: Flat 10x10 grids with visible unit squares
    - Tens rods: Vertical bars with 10 visible segments
    - Ones cubes: Small single unit cubes

    Args:
        number: The number to represent (0-9999)
        output_path: Optional path to save the image
        block_color: RGB tuple for block color

    Returns:
        Path to the generated image file
    """
    # Default to orange (Khan Academy style)
    if block_color is None:
        block_color = COLORS["orange"]

    # Calculate place values
    thousands = number // 1000
    hundreds = (number % 1000) // 100
    tens = (number % 100) // 10
    ones = number % 10

    # Image dimensions - scale based on content
    # Each hundred block is about 100x100 pixels
    # Tens rods are 15x100, ones cubes are 15x15

    # Calculate required width
    hundreds_width = hundreds * 110 + (100 if hundreds > 0 else 0)  # Stack effect
    tens_width = tens * 18 + 20
    ones_width = ones * 18 + 20

    total_width = max(600, hundreds_width + tens_width + ones_width + 100)
    height = 350

    # Create image
    img = Image.new("RGB", (total_width, height), COLORS["background"])
    draw = ImageDraw.Draw(img)

    # Starting positions
    x_offset = 40
    y_base = 250  # Bottom baseline for blocks

    # Draw hundreds blocks (10x10 grids, stacked with offset)
    block_size = 100
    unit_size = block_size // 10  # 10x10 grid

    for h in range(hundreds):
        # Offset each block slightly for stacking effect
        x = x_offset + h * 15
        y = y_base - block_size + h * 5

        _draw_hundred_block(draw, x, y, block_size, unit_size, block_color)

    if hundreds > 0:
        x_offset += hundreds * 15 + block_size + 30

    # Draw tens rods (vertical bars with 10 segments)
    rod_width = 15
    rod_height = 100
    segment_height = rod_height // 10

    for t in range(tens):
        x = x_offset + t * 18
        y = y_base - rod_height

        _draw_ten_rod(draw, x, y, rod_width, rod_height, segment_height, block_color)

    if tens > 0:
        x_offset += tens * 18 + 25

    # Draw ones cubes (small single units)
    cube_size = 15

    # Arrange ones in rows of 3 (like Khan Academy style)
    ones_per_row = 3
    for o in range(ones):
        row = o // ones_per_row
        col = o % ones_per_row

        x = x_offset + col * 18
        y = y_base - cube_size - row * 18

        _draw_one_cube(draw, x, y, cube_size, block_color)

    # Generate output path if not provided
    if output_path is None:
        unique_id = hashlib.md5(f"pv_{number}_{uuid.uuid4()}".encode()).hexdigest()[:8]
        output_path = str(GENERATED_IMAGES_DIR / f"placevalue_{unique_id}.png")

    # Save image
    img.save(output_path, "PNG")
    return output_path


def _draw_hundred_block(
    draw: ImageDraw.Draw,
    x: int,
    y: int,
    size: int,
    unit_size: int,
    color: Tuple[int, int, int],
) -> None:
    """Draw a hundreds block (10x10 grid)."""
    # Fill the block
    draw.rectangle([x, y, x + size, y + size], fill=color)

    # Draw 10x10 grid lines
    grid_color = COLORS["grid_line"]

    # Vertical lines
    for i in range(11):
        line_x = x + i * unit_size
        draw.line([(line_x, y), (line_x, y + size)], fill=grid_color, width=1)

    # Horizontal lines
    for i in range(11):
        line_y = y + i * unit_size
        draw.line([(x, line_y), (x + size, line_y)], fill=grid_color, width=1)

    # Draw border
    draw.rectangle([x, y, x + size, y + size], outline=COLORS["orange_dark"], width=2)


def _draw_ten_rod(
    draw: ImageDraw.Draw,
    x: int,
    y: int,
    width: int,
    height: int,
    segment_height: int,
    color: Tuple[int, int, int],
) -> None:
    """Draw a tens rod (vertical bar with 10 segments)."""
    # Fill the rod
    draw.rectangle([x, y, x + width, y + height], fill=color)

    # Draw segment lines
    grid_color = COLORS["grid_line"]
    for i in range(11):
        line_y = y + i * segment_height
        draw.line([(x, line_y), (x + width, line_y)], fill=grid_color, width=1)

    # Draw border
    draw.rectangle([x, y, x + width, y + height], outline=COLORS["orange_dark"], width=2)


def _draw_one_cube(
    draw: ImageDraw.Draw,
    x: int,
    y: int,
    size: int,
    color: Tuple[int, int, int],
) -> None:
    """Draw a ones cube (small unit block)."""
    # Fill the cube
    draw.rectangle([x, y, x + size, y + size], fill=color)

    # Draw border
    draw.rectangle([x, y, x + size, y + size], outline=COLORS["orange_dark"], width=2)


def generate_counting_objects(
    count: int,
    shape: str = "circle",
    output_path: Optional[str] = None,
) -> str:
    """Generate a counting objects diagram.

    Args:
        count: Number of objects to draw
        shape: Type of shape ("circle", "star", "square")
        output_path: Optional path to save the image

    Returns:
        Path to the generated image file
    """
    # Arrange in rows of 5 or 10 for easy counting
    objects_per_row = 5 if count <= 25 else 10
    rows = (count + objects_per_row - 1) // objects_per_row

    # Calculate dimensions
    obj_size = 30
    spacing = 40
    margin = 30

    width = objects_per_row * spacing + margin * 2
    height = rows * spacing + margin * 2

    # Create image
    img = Image.new("RGB", (width, height), COLORS["white"])
    draw = ImageDraw.Draw(img)

    # Draw objects
    color = COLORS["blue"]
    for i in range(count):
        row = i // objects_per_row
        col = i % objects_per_row

        cx = margin + col * spacing + spacing // 2
        cy = margin + row * spacing + spacing // 2

        if shape == "circle":
            r = obj_size // 2
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color)
        elif shape == "square":
            r = obj_size // 2
            draw.rectangle([cx - r, cy - r, cx + r, cy + r], fill=color)
        else:  # star or default to circle
            r = obj_size // 2
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color)

    # Generate output path if not provided
    if output_path is None:
        unique_id = hashlib.md5(f"count_{count}_{uuid.uuid4()}".encode()).hexdigest()[:8]
        output_path = str(GENERATED_IMAGES_DIR / f"counting_{unique_id}.png")

    img.save(output_path, "PNG")
    return output_path


def generate_fraction_visual(
    numerator: int,
    denominator: int,
    shape: str = "circle",
    output_path: Optional[str] = None,
) -> str:
    """Generate a fraction visualization.

    Args:
        numerator: Top number of fraction
        denominator: Bottom number of fraction
        shape: "circle" for pie chart, "rectangle" for bar
        output_path: Optional path to save the image

    Returns:
        Path to the generated image file
    """
    width = 300
    height = 300
    margin = 30

    img = Image.new("RGB", (width, height), COLORS["white"])
    draw = ImageDraw.Draw(img)

    if shape == "circle":
        # Pie chart style
        cx, cy = width // 2, height // 2
        radius = min(width, height) // 2 - margin

        # Draw full circle outline
        draw.ellipse(
            [cx - radius, cy - radius, cx + radius, cy + radius],
            outline=COLORS["purple_dark"],
            width=2,
        )

        # Draw filled sectors for numerator
        import math

        angle_per_part = 360 / denominator
        for i in range(denominator):
            start = -90 + i * angle_per_part  # Start from top
            end = start + angle_per_part

            if i < numerator:
                # Filled sector
                draw.pieslice(
                    [cx - radius, cy - radius, cx + radius, cy + radius],
                    start=start,
                    end=end,
                    fill=COLORS["purple"],
                    outline=COLORS["purple_dark"],
                )
            else:
                # Empty sector - just draw lines
                draw.pieslice(
                    [cx - radius, cy - radius, cx + radius, cy + radius],
                    start=start,
                    end=end,
                    fill=COLORS["white"],
                    outline=COLORS["purple_dark"],
                )
    else:
        # Rectangle bar style
        bar_width = width - margin * 2
        bar_height = 60
        y = (height - bar_height) // 2

        part_width = bar_width // denominator

        for i in range(denominator):
            x = margin + i * part_width

            if i < numerator:
                draw.rectangle(
                    [x, y, x + part_width, y + bar_height],
                    fill=COLORS["purple"],
                    outline=COLORS["purple_dark"],
                    width=2,
                )
            else:
                draw.rectangle(
                    [x, y, x + part_width, y + bar_height],
                    fill=COLORS["white"],
                    outline=COLORS["purple_dark"],
                    width=2,
                )

    # Generate output path if not provided
    if output_path is None:
        unique_id = hashlib.md5(f"frac_{numerator}_{denominator}_{uuid.uuid4()}".encode()).hexdigest()[:8]
        output_path = str(GENERATED_IMAGES_DIR / f"fraction_{unique_id}.png")

    img.save(output_path, "PNG")
    return output_path


if __name__ == "__main__":
    # Test the generators
    print("Testing place value blocks generator...")

    # Test with 693 (the original example)
    path = generate_place_value_blocks(693)
    print(f"Generated 693: {path}")

    # Test with 542
    path = generate_place_value_blocks(542)
    print(f"Generated 542: {path}")

    # Test with 125
    path = generate_place_value_blocks(125)
    print(f"Generated 125: {path}")

    print("\nTesting counting objects generator...")
    path = generate_counting_objects(15)
    print(f"Generated 15 objects: {path}")

    print("\nTesting fraction visual generator...")
    path = generate_fraction_visual(3, 4, "circle")
    print(f"Generated 3/4 circle: {path}")

    path = generate_fraction_visual(2, 5, "rectangle")
    print(f"Generated 2/5 rectangle: {path}")

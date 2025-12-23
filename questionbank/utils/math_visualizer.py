"""Math Visualizer - Programmatic generation of mathematically accurate graphs and diagrams.

Uses Matplotlib for precise, publication-quality math visualizations that AI image
generators cannot reliably produce.

This module handles:
- Function plots (f(x) = expressions)
- Coordinate point plotting
- Number lines
- Geometric shapes (triangles, circles, angles)
- Bar and pie charts
"""

import os
import hashlib
import uuid
from pathlib import Path
from typing import Optional, List, Tuple, Union
import logging

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server use
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

try:
    from sympy import symbols, sympify, lambdify
    from sympy.parsing.latex import parse_latex
    SYMPY_AVAILABLE = True
except ImportError:
    SYMPY_AVAILABLE = False

logger = logging.getLogger(__name__)

# Output directory for generated images
GENERATED_IMAGES_DIR = Path(__file__).parent.parent / "generated_images"
GENERATED_IMAGES_DIR.mkdir(exist_ok=True)

# Professional color palette
COLORS = {
    "primary": "#2563eb",      # Blue
    "secondary": "#7c3aed",    # Purple
    "accent": "#059669",       # Green
    "warning": "#d97706",      # Orange
    "error": "#dc2626",        # Red
    "grid": "#e2e8f0",         # Light gray
    "axis": "#64748b",         # Medium gray
    "background": "#ffffff",   # White
}


def generate_function_plot(
    expression: str,
    x_range: Tuple[float, float] = (-10, 10),
    title: Optional[str] = None,
    output_path: Optional[str] = None,
    show_grid: bool = True,
) -> Optional[str]:
    """Generate a plot of a mathematical function.

    Args:
        expression: Mathematical expression (e.g., "x**2 - 4", "sin(x)", "2*x + 3")
                   Can be Python syntax or LaTeX (if sympy available)
        x_range: Tuple of (x_min, x_max) for the plot domain
        title: Optional title for the plot
        output_path: Optional path to save the image
        show_grid: Whether to show grid lines

    Returns:
        Path to the generated image file, or None if failed
    """
    try:
        fig, ax = plt.subplots(figsize=(8, 6), dpi=100)

        x = np.linspace(x_range[0], x_range[1], 500)

        # Try to parse and evaluate the expression
        if SYMPY_AVAILABLE:
            try:
                # Try LaTeX first
                if '\\' in expression:
                    sym_expr = parse_latex(expression)
                else:
                    sym_expr = sympify(expression)
                x_sym = symbols('x')
                f = lambdify(x_sym, sym_expr, modules=['numpy'])
                y = f(x)
            except Exception:
                # Fall back to eval with numpy functions
                y = _safe_eval_expression(expression, x)
        else:
            y = _safe_eval_expression(expression, x)

        if y is None:
            logger.error(f"Could not evaluate expression: {expression}")
            return None

        # Handle infinities and NaN
        y = np.where(np.isfinite(y), y, np.nan)

        # Plot the function
        ax.plot(x, y, color=COLORS["primary"], linewidth=2.5, label=f"y = {expression}")

        # Style the plot
        ax.axhline(y=0, color=COLORS["axis"], linewidth=0.8)
        ax.axvline(x=0, color=COLORS["axis"], linewidth=0.8)

        if show_grid:
            ax.grid(True, color=COLORS["grid"], linestyle='-', linewidth=0.5)

        ax.set_xlabel('x', fontsize=12)
        ax.set_ylabel('y', fontsize=12)

        if title:
            ax.set_title(title, fontsize=14, fontweight='bold')

        ax.legend(loc='best', framealpha=0.9)

        # Set reasonable y limits
        valid_y = y[np.isfinite(y)]
        if len(valid_y) > 0:
            y_margin = (valid_y.max() - valid_y.min()) * 0.1
            ax.set_ylim(valid_y.min() - y_margin, valid_y.max() + y_margin)

        plt.tight_layout()

        # Generate output path if not provided
        if output_path is None:
            unique_id = hashlib.md5(f"func_{expression}_{uuid.uuid4()}".encode()).hexdigest()[:8]
            output_path = str(GENERATED_IMAGES_DIR / f"function_{unique_id}.png")

        plt.savefig(output_path, dpi=150, bbox_inches='tight',
                   facecolor=COLORS["background"], edgecolor='none')
        plt.close(fig)

        return output_path

    except Exception as e:
        logger.error(f"Error generating function plot: {e}")
        plt.close('all')
        return None


def generate_coordinate_plot(
    points: List[Tuple[float, float]],
    labels: Optional[List[str]] = None,
    connect_points: bool = False,
    x_range: Optional[Tuple[float, float]] = None,
    y_range: Optional[Tuple[float, float]] = None,
    title: Optional[str] = None,
    output_path: Optional[str] = None,
) -> Optional[str]:
    """Generate a coordinate plane with plotted points.

    Args:
        points: List of (x, y) coordinate tuples
        labels: Optional labels for each point
        connect_points: Whether to connect points with lines
        x_range: Optional (x_min, x_max) for axes
        y_range: Optional (y_min, y_max) for axes
        title: Optional title for the plot
        output_path: Optional path to save the image

    Returns:
        Path to the generated image file, or None if failed
    """
    try:
        fig, ax = plt.subplots(figsize=(8, 8), dpi=100)

        if not points:
            logger.error("No points provided for coordinate plot")
            return None

        xs = [p[0] for p in points]
        ys = [p[1] for p in points]

        # Auto-range if not specified
        if x_range is None:
            margin = max(1, (max(xs) - min(xs)) * 0.2) if len(xs) > 1 else 2
            x_range = (min(xs) - margin, max(xs) + margin)
        if y_range is None:
            margin = max(1, (max(ys) - min(ys)) * 0.2) if len(ys) > 1 else 2
            y_range = (min(ys) - margin, max(ys) + margin)

        # Set up the coordinate plane
        ax.set_xlim(x_range)
        ax.set_ylim(y_range)
        ax.set_aspect('equal', adjustable='box')

        # Draw axes
        ax.axhline(y=0, color=COLORS["axis"], linewidth=1.5)
        ax.axvline(x=0, color=COLORS["axis"], linewidth=1.5)

        # Grid
        ax.grid(True, color=COLORS["grid"], linestyle='-', linewidth=0.5)

        # Plot points
        ax.scatter(xs, ys, color=COLORS["primary"], s=100, zorder=5)

        # Connect points if requested
        if connect_points and len(points) > 1:
            ax.plot(xs, ys, color=COLORS["secondary"], linewidth=2, alpha=0.7)

        # Add labels
        if labels:
            for (x, y), label in zip(points, labels):
                ax.annotate(label, (x, y), textcoords="offset points",
                           xytext=(8, 8), fontsize=10, fontweight='bold',
                           color=COLORS["primary"])
        else:
            # Default labels with coordinates
            for i, (x, y) in enumerate(points):
                ax.annotate(f"({x}, {y})", (x, y), textcoords="offset points",
                           xytext=(8, 8), fontsize=9)

        ax.set_xlabel('x', fontsize=12)
        ax.set_ylabel('y', fontsize=12)

        if title:
            ax.set_title(title, fontsize=14, fontweight='bold')

        plt.tight_layout()

        # Generate output path if not provided
        if output_path is None:
            unique_id = hashlib.md5(f"coord_{str(points)}_{uuid.uuid4()}".encode()).hexdigest()[:8]
            output_path = str(GENERATED_IMAGES_DIR / f"coordinate_{unique_id}.png")

        plt.savefig(output_path, dpi=150, bbox_inches='tight',
                   facecolor=COLORS["background"], edgecolor='none')
        plt.close(fig)

        return output_path

    except Exception as e:
        logger.error(f"Error generating coordinate plot: {e}")
        plt.close('all')
        return None


def generate_number_line(
    value: Union[float, List[float]],
    range_start: float = -10,
    range_end: float = 10,
    tick_interval: float = 1,
    title: Optional[str] = None,
    output_path: Optional[str] = None,
) -> Optional[str]:
    """Generate a number line with marked values.

    Args:
        value: Single value or list of values to mark on the line
        range_start: Start of the number line
        range_end: End of the number line
        tick_interval: Interval between tick marks
        title: Optional title for the plot
        output_path: Optional path to save the image

    Returns:
        Path to the generated image file, or None if failed
    """
    try:
        fig, ax = plt.subplots(figsize=(10, 2), dpi=100)

        values = [value] if isinstance(value, (int, float)) else value

        # Draw the number line
        ax.axhline(y=0, color=COLORS["axis"], linewidth=2)

        # Add arrow heads
        ax.annotate('', xy=(range_end + 0.3, 0), xytext=(range_end, 0),
                   arrowprops=dict(arrowstyle='->', color=COLORS["axis"], lw=2))
        ax.annotate('', xy=(range_start - 0.3, 0), xytext=(range_start, 0),
                   arrowprops=dict(arrowstyle='->', color=COLORS["axis"], lw=2))

        # Draw tick marks
        ticks = np.arange(range_start, range_end + tick_interval, tick_interval)
        for tick in ticks:
            ax.plot([tick, tick], [-0.15, 0.15], color=COLORS["axis"], linewidth=1.5)
            ax.text(tick, -0.4, f"{int(tick) if tick == int(tick) else tick}",
                   ha='center', fontsize=10)

        # Mark the values
        for i, v in enumerate(values):
            color = [COLORS["primary"], COLORS["secondary"], COLORS["accent"]][i % 3]
            ax.plot(v, 0, 'o', color=color, markersize=15, zorder=5)
            ax.text(v, 0.35, f"{v}", ha='center', fontsize=12, fontweight='bold', color=color)

        # Style
        ax.set_xlim(range_start - 1, range_end + 1)
        ax.set_ylim(-0.8, 0.8)
        ax.axis('off')

        if title:
            ax.set_title(title, fontsize=14, fontweight='bold', pad=10)

        plt.tight_layout()

        # Generate output path if not provided
        if output_path is None:
            unique_id = hashlib.md5(f"numline_{str(values)}_{uuid.uuid4()}".encode()).hexdigest()[:8]
            output_path = str(GENERATED_IMAGES_DIR / f"numberline_{unique_id}.png")

        plt.savefig(output_path, dpi=150, bbox_inches='tight',
                   facecolor=COLORS["background"], edgecolor='none')
        plt.close(fig)

        return output_path

    except Exception as e:
        logger.error(f"Error generating number line: {e}")
        plt.close('all')
        return None


def generate_geometry(
    shape: str,
    params: dict,
    output_path: Optional[str] = None,
) -> Optional[str]:
    """Generate geometric shapes with measurements.

    Args:
        shape: Type of shape ("triangle", "rectangle", "circle", "angle")
        params: Shape-specific parameters:
            - triangle: {"vertices": [(x1,y1), (x2,y2), (x3,y3)], "labels": ["A", "B", "C"]}
            - rectangle: {"width": w, "height": h}
            - circle: {"radius": r, "center": (x, y)}
            - angle: {"vertex": (x, y), "angle_degrees": deg, "radius": r}
        output_path: Optional path to save the image

    Returns:
        Path to the generated image file, or None if failed
    """
    try:
        fig, ax = plt.subplots(figsize=(6, 6), dpi=100)

        if shape == "triangle":
            vertices = params.get("vertices", [(0, 0), (4, 0), (2, 3)])
            labels = params.get("labels", ["A", "B", "C"])

            # Draw triangle
            triangle = plt.Polygon(vertices, fill=False,
                                   edgecolor=COLORS["primary"], linewidth=2.5)
            ax.add_patch(triangle)

            # Add vertex labels
            for (x, y), label in zip(vertices, labels):
                offset = np.array([0.2, 0.2])
                ax.text(x + offset[0], y + offset[1], label,
                       fontsize=14, fontweight='bold', color=COLORS["primary"])

            # Add side length labels
            for i in range(3):
                x1, y1 = vertices[i]
                x2, y2 = vertices[(i + 1) % 3]
                mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
                length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                ax.text(mid_x, mid_y - 0.3, f"{length:.1f}",
                       fontsize=10, ha='center', color=COLORS["axis"])

        elif shape == "rectangle":
            width = params.get("width", 4)
            height = params.get("height", 3)

            rect = patches.Rectangle((0, 0), width, height,
                                     fill=False, edgecolor=COLORS["primary"], linewidth=2.5)
            ax.add_patch(rect)

            # Add dimension labels
            ax.text(width / 2, -0.3, f"{width}", ha='center', fontsize=12)
            ax.text(-0.3, height / 2, f"{height}", va='center', fontsize=12, rotation=90)

        elif shape == "circle":
            radius = params.get("radius", 2)
            center = params.get("center", (3, 3))

            circle = plt.Circle(center, radius, fill=False,
                               edgecolor=COLORS["primary"], linewidth=2.5)
            ax.add_patch(circle)

            # Draw radius line
            ax.plot([center[0], center[0] + radius], [center[1], center[1]],
                   color=COLORS["secondary"], linewidth=2, linestyle='--')
            ax.text(center[0] + radius/2, center[1] + 0.2, f"r = {radius}",
                   fontsize=12, color=COLORS["secondary"])

            # Mark center
            ax.plot(*center, 'o', color=COLORS["primary"], markersize=8)

        elif shape == "angle":
            vertex = params.get("vertex", (0, 0))
            angle_deg = params.get("angle_degrees", 45)
            radius = params.get("radius", 2)

            # Draw angle arms
            ax.plot([vertex[0], vertex[0] + radius], [vertex[1], vertex[1]],
                   color=COLORS["primary"], linewidth=2.5)

            end_x = vertex[0] + radius * np.cos(np.radians(angle_deg))
            end_y = vertex[1] + radius * np.sin(np.radians(angle_deg))
            ax.plot([vertex[0], end_x], [vertex[1], end_y],
                   color=COLORS["primary"], linewidth=2.5)

            # Draw angle arc
            arc = patches.Arc(vertex, 0.8, 0.8, angle=0,
                             theta1=0, theta2=angle_deg,
                             color=COLORS["secondary"], linewidth=2)
            ax.add_patch(arc)

            # Label angle
            label_x = vertex[0] + 0.6 * np.cos(np.radians(angle_deg / 2))
            label_y = vertex[1] + 0.6 * np.sin(np.radians(angle_deg / 2))
            ax.text(label_x, label_y, f"{angle_deg}°",
                   fontsize=12, fontweight='bold', color=COLORS["secondary"])

        ax.set_aspect('equal')
        ax.axis('off')

        # Auto-adjust limits
        ax.autoscale()
        margin = 0.5
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        ax.set_xlim(xlim[0] - margin, xlim[1] + margin)
        ax.set_ylim(ylim[0] - margin, ylim[1] + margin)

        plt.tight_layout()

        # Generate output path if not provided
        if output_path is None:
            unique_id = hashlib.md5(f"geo_{shape}_{str(params)}_{uuid.uuid4()}".encode()).hexdigest()[:8]
            output_path = str(GENERATED_IMAGES_DIR / f"geometry_{unique_id}.png")

        plt.savefig(output_path, dpi=150, bbox_inches='tight',
                   facecolor=COLORS["background"], edgecolor='none')
        plt.close(fig)

        return output_path

    except Exception as e:
        logger.error(f"Error generating geometry: {e}")
        plt.close('all')
        return None


def generate_bar_chart(
    data: dict,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    output_path: Optional[str] = None,
) -> Optional[str]:
    """Generate a bar chart.

    Args:
        data: Dictionary of {label: value} pairs
        title: Optional chart title
        xlabel: Optional x-axis label
        ylabel: Optional y-axis label
        output_path: Optional path to save the image

    Returns:
        Path to the generated image file, or None if failed
    """
    try:
        fig, ax = plt.subplots(figsize=(8, 6), dpi=100)

        labels = list(data.keys())
        values = list(data.values())

        bars = ax.bar(labels, values, color=COLORS["primary"], edgecolor=COLORS["axis"])

        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{value}', ha='center', va='bottom', fontsize=10)

        if title:
            ax.set_title(title, fontsize=14, fontweight='bold')
        if xlabel:
            ax.set_xlabel(xlabel, fontsize=12)
        if ylabel:
            ax.set_ylabel(ylabel, fontsize=12)

        ax.grid(True, axis='y', color=COLORS["grid"], linestyle='-', linewidth=0.5)

        plt.tight_layout()

        # Generate output path if not provided
        if output_path is None:
            unique_id = hashlib.md5(f"bar_{str(data)}_{uuid.uuid4()}".encode()).hexdigest()[:8]
            output_path = str(GENERATED_IMAGES_DIR / f"barchart_{unique_id}.png")

        plt.savefig(output_path, dpi=150, bbox_inches='tight',
                   facecolor=COLORS["background"], edgecolor='none')
        plt.close(fig)

        return output_path

    except Exception as e:
        logger.error(f"Error generating bar chart: {e}")
        plt.close('all')
        return None


def generate_pie_chart(
    data: dict,
    title: Optional[str] = None,
    output_path: Optional[str] = None,
) -> Optional[str]:
    """Generate a pie chart.

    Args:
        data: Dictionary of {label: value} pairs
        title: Optional chart title
        output_path: Optional path to save the image

    Returns:
        Path to the generated image file, or None if failed
    """
    try:
        fig, ax = plt.subplots(figsize=(8, 8), dpi=100)

        labels = list(data.keys())
        values = list(data.values())

        colors = [COLORS["primary"], COLORS["secondary"], COLORS["accent"],
                 COLORS["warning"], COLORS["error"]]

        wedges, texts, autotexts = ax.pie(values, labels=labels, autopct='%1.1f%%',
                                          colors=colors[:len(values)],
                                          textprops={'fontsize': 11})

        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')

        if title:
            ax.set_title(title, fontsize=14, fontweight='bold')

        ax.axis('equal')

        plt.tight_layout()

        # Generate output path if not provided
        if output_path is None:
            unique_id = hashlib.md5(f"pie_{str(data)}_{uuid.uuid4()}".encode()).hexdigest()[:8]
            output_path = str(GENERATED_IMAGES_DIR / f"piechart_{unique_id}.png")

        plt.savefig(output_path, dpi=150, bbox_inches='tight',
                   facecolor=COLORS["background"], edgecolor='none')
        plt.close(fig)

        return output_path

    except Exception as e:
        logger.error(f"Error generating pie chart: {e}")
        plt.close('all')
        return None


def _safe_eval_expression(expression: str, x: np.ndarray) -> Optional[np.ndarray]:
    """Safely evaluate a mathematical expression."""
    try:
        # Replace common math notation
        expr = expression.replace('^', '**')
        expr = expr.replace('sin', 'np.sin')
        expr = expr.replace('cos', 'np.cos')
        expr = expr.replace('tan', 'np.tan')
        expr = expr.replace('sqrt', 'np.sqrt')
        expr = expr.replace('abs', 'np.abs')
        expr = expr.replace('log', 'np.log')
        expr = expr.replace('exp', 'np.exp')

        # Only allow safe operations
        allowed_names = {'x': x, 'np': np}
        return eval(expr, {"__builtins__": {}}, allowed_names)
    except Exception:
        return None


if __name__ == "__main__":
    # Test the generators
    print("Testing math visualizer...")

    # Function plot
    path = generate_function_plot("x**2 - 4", title="Quadratic Function")
    print(f"Function plot: {path}")

    # Coordinate plot
    path = generate_coordinate_plot([(0, 0), (3, 4), (-2, 1)], connect_points=True)
    print(f"Coordinate plot: {path}")

    # Number line
    path = generate_number_line([2, -3, 5], range_start=-6, range_end=6)
    print(f"Number line: {path}")

    # Geometry - triangle
    path = generate_geometry("triangle", {"vertices": [(0, 0), (4, 0), (2, 3)]})
    print(f"Triangle: {path}")

    # Bar chart
    path = generate_bar_chart({"A": 25, "B": 40, "C": 35}, title="Test Scores")
    print(f"Bar chart: {path}")

    print("\nAll tests completed!")

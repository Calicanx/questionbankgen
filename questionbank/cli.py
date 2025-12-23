"""CLI interface for QuestionBank Generator."""

import json
import logging
import sys
from typing import Optional

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

console = Console()


@click.group()
@click.option("--debug", is_flag=True, help="Enable debug logging")
def cli(debug: bool) -> None:
    """QuestionBank Generator - AI-powered educational question generator."""
    if debug:
        logging.getLogger().setLevel(logging.DEBUG)


@cli.command()
@click.option("--source-id", help="Source question ID to generate from")
@click.option("--random", "use_random", is_flag=True, help="Use a random source question")
@click.option("--widget-type", help="Filter by widget type (e.g., radio, numeric-input)")
@click.option("--skill-prefix", help="Filter by skill prefix")
@click.option("--count", default=1, help="Number of questions to generate")
@click.option(
    "--variation",
    type=click.Choice(["number_change", "context_change", "structure_change"]),
    default="number_change",
    help="Type of variation to generate",
)
@click.option("--no-save", is_flag=True, help="Don't save to database")
@click.option("--output", "-o", type=click.Path(), help="Output JSON file")
def generate(
    source_id: Optional[str],
    use_random: bool,
    widget_type: Optional[str],
    skill_prefix: Optional[str],
    count: int,
    variation: str,
    no_save: bool,
    output: Optional[str],
) -> None:
    """Generate new questions from source questions."""
    from questionbank.config import config
    from questionbank.core.generator import QuestionGenerator

    # Validate configuration
    errors = config.validate_required()
    if errors:
        for error in errors:
            console.print(f"[red]Error:[/red] {error}")
        sys.exit(1)

    if not source_id and not use_random:
        console.print("[red]Error:[/red] Specify --source-id or --random")
        sys.exit(1)

    generator = QuestionGenerator()
    results = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        if source_id:
            task = progress.add_task(f"Generating from {source_id}...", total=count)

            for i in range(count):
                result = generator.generate_from_id(
                    question_id=source_id,
                    variation_type=variation,
                    save_to_db=not no_save,
                )
                if result:
                    results.append(result)
                progress.update(task, advance=1)

        elif use_random:
            task = progress.add_task("Generating from random sources...", total=count)

            for i in range(count):
                result = generator.generate_random(
                    widget_type=widget_type,
                    skill_prefix=skill_prefix,
                    variation_type=variation,
                    save_to_db=not no_save,
                )
                if result:
                    results.append(result)
                progress.update(task, advance=1)

    # Show results
    console.print()
    console.print(Panel(f"Generated [green]{len(results)}[/green] / {count} questions"))

    # Save to file if requested
    if output and results:
        with open(output, "w") as f:
            json.dump(results, f, indent=2, default=str)
        console.print(f"[green]Saved to:[/green] {output}")

    # Show sample of first result
    if results:
        first = results[0]
        content = first.get("question", {}).get("content", "")[:200]
        console.print(f"\n[dim]First question preview:[/dim]\n{content}...")


@cli.command()
@click.argument("question_id")
def validate(question_id: str) -> None:
    """Validate a question from the database."""
    from questionbank.config import config
    from questionbank.mongodb.repository import question_repo
    from questionbank.validation.pipeline import ValidationPipeline

    errors = config.validate_required()
    if errors:
        for error in errors:
            console.print(f"[red]Error:[/red] {error}")
        sys.exit(1)

    question = question_repo.get_question_by_id(question_id)

    if not question:
        console.print(f"[red]Question not found:[/red] {question_id}")
        sys.exit(1)

    pipeline = ValidationPipeline()
    result = pipeline.validate(question)

    if result.is_valid:
        console.print(Panel("[green]Question is valid![/green]"))
    else:
        console.print(Panel("[red]Validation failed[/red]"))

    # Show stage results
    table = Table(title="Validation Stages")
    table.add_column("Stage", style="cyan")
    table.add_column("Result", style="green")

    for stage, passed in result.stage_results.items():
        status = "[green]PASS[/green]" if passed else "[red]FAIL[/red]"
        table.add_row(stage.capitalize(), status)

    console.print(table)

    # Show errors
    if result.errors:
        console.print("\n[red]Errors:[/red]")
        for error in result.errors:
            console.print(f"  - {error}")

    # Show warnings
    if result.warnings:
        console.print("\n[yellow]Warnings:[/yellow]")
        for warning in result.warnings:
            console.print(f"  - {warning}")


@cli.command()
def stats() -> None:
    """Show database statistics."""
    from questionbank.config import config
    from questionbank.mongodb.repository import question_repo

    errors = config.validate_required()
    if errors:
        for error in errors:
            console.print(f"[red]Error:[/red] {error}")
        sys.exit(1)

    console.print(Panel("Database Statistics", style="bold"))

    # Collection counts
    source_count = question_repo.count_source()
    generated_count = question_repo.count_generated()

    console.print(f"[cyan]Source questions:[/cyan] {source_count:,}")
    console.print(f"[cyan]Generated questions:[/cyan] {generated_count:,}")

    # Widget type distribution
    console.print("\n[bold]Widget Type Distribution (Source):[/bold]")

    widget_stats = question_repo.get_widget_type_stats()

    table = Table()
    table.add_column("Widget Type", style="cyan")
    table.add_column("Count", style="green", justify="right")

    for widget_type, count in sorted(widget_stats.items(), key=lambda x: -x[1]):
        table.add_row(widget_type, f"{count:,}")

    console.print(table)


@cli.command()
def test_connection() -> None:
    """Test database and API connections."""
    from questionbank.config import config

    console.print(Panel("Testing Connections", style="bold"))

    # Check configuration
    errors = config.validate_required()
    if errors:
        console.print("[red]Configuration errors:[/red]")
        for error in errors:
            console.print(f"  - {error}")
        return

    console.print("[green]Configuration OK[/green]")

    # Test MongoDB
    console.print("\n[bold]MongoDB Connection:[/bold]")
    try:
        from questionbank.mongodb.connection import mongo_db

        if mongo_db.test_connection():
            console.print("[green]MongoDB connection successful[/green]")
        else:
            console.print("[red]MongoDB connection failed[/red]")
    except Exception as e:
        console.print(f"[red]MongoDB error:[/red] {e}")

    # Test Gemini API
    console.print("\n[bold]Gemini API:[/bold]")
    try:
        from questionbank.llm.gemini_client import get_gemini_client

        client = get_gemini_client()
        response = client.generate("Say 'Hello' in one word")
        if response:
            console.print(f"[green]Gemini API working[/green] - Response: {response[:50]}")
        else:
            console.print("[yellow]Gemini API returned empty response[/yellow]")
    except Exception as e:
        console.print(f"[red]Gemini API error:[/red] {e}")


@cli.command("generate-with-images")
@click.option("--source-id", help="Source question ID to generate from")
@click.option("--random", "use_random", is_flag=True, help="Use a random source question")
@click.option("--widget-type", help="Filter by widget type (e.g., image)")
@click.option("--count", default=1, help="Number of questions to generate")
@click.option("--no-save", is_flag=True, help="Don't save to database")
@click.option("--output", "-o", type=click.Path(), help="Output JSON file")
def generate_with_images(
    source_id: Optional[str],
    use_random: bool,
    widget_type: Optional[str],
    count: int,
    no_save: bool,
    output: Optional[str],
) -> None:
    """Generate questions with AI-generated images."""
    from questionbank.config import config
    from questionbank.core.generator import QuestionGenerator
    from questionbank.mongodb.repository import question_repo

    errors = config.validate_required()
    if errors:
        for error in errors:
            console.print(f"[red]Error:[/red] {error}")
        sys.exit(1)

    if not source_id and not use_random:
        console.print("[red]Error:[/red] Specify --source-id or --random")
        sys.exit(1)

    generator = QuestionGenerator()
    results = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Generating questions with images...", total=count)

        for i in range(count):
            if source_id:
                source = question_repo.get_question_by_id(source_id)
            else:
                source = question_repo.get_random_question(
                    widget_type=widget_type or "image",
                )

            if source:
                result = generator.generate_with_images(
                    source_question=source,
                    save_to_db=not no_save,
                    generate_new_images=True,
                )
                if result:
                    results.append(result)

            progress.update(task, advance=1)

    console.print()
    console.print(Panel(f"Generated [green]{len(results)}[/green] / {count} questions with images"))

    if output and results:
        with open(output, "w") as f:
            json.dump(results, f, indent=2, default=str)
        console.print(f"[green]Saved to:[/green] {output}")


@cli.command("test-image")
@click.argument("prompt")
@click.option("--output", "-o", type=click.Path(), help="Output file path")
def test_image(prompt: str, output: Optional[str]) -> None:
    """Test image generation with a prompt."""
    from questionbank.llm.gemini_client import get_gemini_client

    console.print(f"[bold]Generating image for:[/bold] {prompt}")

    try:
        client = get_gemini_client()
        image_path = client.generate_educational_image(
            description=prompt,
            context="educational diagram",
            style="clean, simple, educational illustration",
        )

        if image_path:
            console.print(f"[green]Image saved to:[/green] {image_path}")
        else:
            console.print("[red]Failed to generate image[/red]")
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")


@cli.command()
@click.argument("question_id")
def show(question_id: str) -> None:
    """Show a question from the database."""
    from questionbank.config import config
    from questionbank.mongodb.repository import question_repo

    errors = config.validate_required()
    if errors:
        for error in errors:
            console.print(f"[red]Error:[/red] {error}")
        sys.exit(1)

    question = question_repo.get_question_by_id(question_id)

    if not question:
        console.print(f"[red]Question not found:[/red] {question_id}")
        sys.exit(1)

    # Extract and display
    q = question.get("question", {})
    content = q.get("content", "")
    widgets = q.get("widgets", {})
    hints = question.get("hints", [])

    console.print(Panel(f"Question ID: {question_id}", style="bold"))
    console.print("\n[bold]Content:[/bold]")
    console.print(content)

    console.print(f"\n[bold]Widgets:[/bold] {len(widgets)}")
    for widget_id, widget in widgets.items():
        console.print(f"  - {widget_id}: {widget.get('type', 'unknown')}")

    console.print(f"\n[bold]Hints:[/bold] {len(hints)}")

    # Show full JSON
    console.print("\n[dim]Full JSON:[/dim]")
    console.print(json.dumps(question, indent=2, default=str)[:2000])


if __name__ == "__main__":
    cli()

"""Khan Academy LaTeX color command mappings."""

# Khan Academy color commands and their hex values
# These are expanded by the AthenaRenderer to \textcolor{#hex}{content}
KHAN_COLORS = {
    # Blue variants
    "blueA": "#ccfaff",
    "blueB": "#80f6ff",
    "blueC": "#63d9e8",
    "blueD": "#11accd",
    "blueE": "#0c7f99",

    # Teal variants
    "tealA": "#adfee8",
    "tealB": "#6bddc9",
    "tealC": "#44bbab",
    "tealD": "#159488",
    "tealE": "#0c6b61",

    # Green variants
    "greenA": "#b6ffb0",
    "greenB": "#78d375",
    "greenC": "#58b13c",
    "greenD": "#378d29",
    "greenE": "#266a1c",

    # Gold variants
    "goldA": "#fff9cc",
    "goldB": "#ffec80",
    "goldC": "#f8d541",
    "goldD": "#ffb800",
    "goldE": "#c69009",

    # Red variants
    "redA": "#ffdede",
    "redB": "#fc9999",
    "redC": "#f96666",
    "redD": "#e84d39",
    "redE": "#bc2612",

    # Maroon variants
    "maroonA": "#ffdedb",
    "maroonB": "#ff9fa1",
    "maroonC": "#ec5e5e",
    "maroonD": "#ca337c",
    "maroonE": "#9e034e",

    # Purple variants
    "purpleA": "#f6ddfc",
    "purpleB": "#e8b1f0",
    "purpleC": "#b955d0",
    "purpleD": "#9059a8",
    "purpleE": "#6b3f7d",

    # Pink variants
    "pinkA": "#ffe0e6",
    "pinkB": "#ffb3c2",
    "pinkC": "#ff668f",
    "pinkD": "#e8345c",
    "pinkE": "#b3002e",

    # Gray variants
    "grayA": "#f6f7f7",
    "grayB": "#dadfe0",
    "grayC": "#b5babf",
    "grayD": "#7b848e",
    "grayE": "#5c6268",

    # Kagreen (legacy)
    "kaGreen": "#71b307",
}

# All valid color command names
VALID_COLOR_COMMANDS = set(KHAN_COLORS.keys())

# Regex pattern to match Khan Academy color commands
# Matches: \blueD{content}, \redA{content}, etc.
COLOR_COMMAND_PATTERN = r"\\(" + "|".join(VALID_COLOR_COMMANDS) + r")\{([^}]*)\}"


def is_valid_color_command(command: str) -> bool:
    """Check if a color command name is valid."""
    return command in VALID_COLOR_COMMANDS


def get_color_hex(command: str) -> str | None:
    """Get the hex color value for a command."""
    return KHAN_COLORS.get(command)


def expand_color_command(command: str, content: str) -> str:
    """Expand a Khan Academy color command to standard LaTeX."""
    hex_color = KHAN_COLORS.get(command)
    if hex_color:
        return f"\\textcolor{{{hex_color}}}{{{content}}}"
    return f"\\{command}{{{content}}}"

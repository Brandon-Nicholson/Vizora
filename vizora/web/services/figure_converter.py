"""
Convert matplotlib figures to base64-encoded PNGs with dark theme styling.
"""

import io
import base64
from typing import Tuple
import matplotlib.pyplot as plt
import matplotlib

from vizora.web.models.responses import FigureData


# Dark theme colors
DARK_BG_PRIMARY = "#0a0a0f"
DARK_BG_SECONDARY = "#1a1a2e"
DARK_BG_TERTIARY = "#16213e"
DARK_TEXT = "#e0e0e0"
DARK_TEXT_TITLE = "#ffffff"
DARK_SPINE = "#4a4a6a"
DARK_GRID = "#2a2a4a"


def apply_dark_theme(fig: plt.Figure) -> None:
    """
    Apply dark theme styling to a matplotlib figure.

    Args:
        fig: The matplotlib figure to style.
    """
    # Set figure background
    fig.patch.set_facecolor(DARK_BG_PRIMARY)

    for ax in fig.axes:
        # Axes background
        ax.set_facecolor(DARK_BG_SECONDARY)

        # Tick colors
        ax.tick_params(colors=DARK_TEXT, which="both")

        # Label colors
        ax.xaxis.label.set_color(DARK_TEXT)
        ax.yaxis.label.set_color(DARK_TEXT)

        # Title color
        if ax.get_title():
            ax.title.set_color(DARK_TEXT_TITLE)

        # Spine colors
        for spine in ax.spines.values():
            spine.set_color(DARK_SPINE)

        # Grid styling
        ax.grid(True, color=DARK_GRID, alpha=0.3, linestyle="-", linewidth=0.5)

        # Legend styling
        legend = ax.get_legend()
        if legend:
            legend.get_frame().set_facecolor(DARK_BG_TERTIARY)
            legend.get_frame().set_edgecolor(DARK_SPINE)
            for text in legend.get_texts():
                text.set_color(DARK_TEXT)


def figure_to_base64(fig: plt.Figure, dpi: int = 150) -> str:
    """
    Convert a matplotlib figure to a base64-encoded PNG data URL.

    Args:
        fig: The matplotlib figure to convert.
        dpi: Resolution in dots per inch.

    Returns:
        Data URL string (data:image/png;base64,...)
    """
    # Apply dark theme
    apply_dark_theme(fig)

    # Save to buffer
    buf = io.BytesIO()
    fig.savefig(
        buf,
        format="png",
        dpi=dpi,
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),
        edgecolor="none",
        pad_inches=0.2
    )
    buf.seek(0)

    # Encode to base64
    img_base64 = base64.b64encode(buf.read()).decode("utf-8")

    # Close the figure to free memory
    plt.close(fig)

    return f"data:image/png;base64,{img_base64}"


def convert_figures(figures: list[Tuple[str, str, plt.Figure]]) -> list[FigureData]:
    """
    Convert a list of matplotlib figures to FigureData objects.

    Args:
        figures: List of (fig_type, fig_name, figure) tuples from ExecutionContext.

    Returns:
        List of FigureData objects with base64-encoded images.
    """
    result = []

    for i, (fig_type, fig_name, fig) in enumerate(figures):
        fig_id = f"{i+1:02d}_{fig_type}_{fig_name}"

        base64_png = figure_to_base64(fig)

        result.append(FigureData(
            id=fig_id,
            type=fig_type,
            name=fig_name,
            base64_png=base64_png
        ))

    return result

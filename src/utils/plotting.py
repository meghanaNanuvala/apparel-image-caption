"""One chart style for every figure in the project (notebooks and saved figures)."""

import textwrap

import matplotlib as mpl
import matplotlib.pyplot as plt
from PIL import Image

from utils import DATA_DIR

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

# Categorical colours, always used in this order. A single-series chart uses SERIES[0].
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
BLUE = SERIES[0]


def apply_style() -> None:
    mpl.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "figure.dpi": 110,
            "savefig.dpi": 160,
            "savefig.bbox": "tight",
            "font.family": "sans-serif",
            "font.size": 10,
            "text.color": INK,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.titlepad": 10,
            "axes.labelcolor": INK_SECONDARY,
            "axes.edgecolor": AXIS,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "grid.linestyle": "-",
            "xtick.color": AXIS,
            "ytick.color": AXIS,
            "xtick.labelcolor": INK_SECONDARY,
            "ytick.labelcolor": INK_SECONDARY,
            "legend.frameon": False,
        }
    )


def ranked_bars(ax, labels, values, value_format="{:,.0f}", color=BLUE) -> None:
    """Horizontal bars, largest on top, each labelled with its value at the tip."""
    labels, values = list(labels), list(values)
    positions = range(len(labels))
    ax.barh(positions, values, height=0.62, color=color)
    ax.set_yticks(positions, labels)
    ax.invert_yaxis()
    ax.tick_params(axis="y", length=0)
    ax.set_xticks([])
    ax.spines["bottom"].set_visible(False)
    pad = max(values) * 0.012
    for position, value in zip(positions, values):
        ax.text(value + pad, position, value_format.format(value), va="center", ha="left",
                color=INK_SECONDARY, fontsize=9)
    ax.set_xlim(0, max(values) * 1.13)


def histogram(ax, values, bins, color=BLUE, **kwargs) -> None:
    """Histogram with a thin surface-coloured gap between columns and a light y-grid."""
    ax.hist(values, bins=bins, color=color, edgecolor=SURFACE, linewidth=1.0, **kwargs)
    ax.grid(axis="y")
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda value, _: f"{value:,.0f}"))


def reference_line(ax, x, label, side="right") -> None:
    """A vertical reference line, labelled in headroom added above the data."""
    ax.set_ylim(top=ax.get_ylim()[1] * 1.14)
    ax.axvline(x, color=INK, linewidth=1.0)
    offset, align = (6, "left") if side == "right" else (-6, "right")
    ax.annotate(label, xy=(x, 0.97), xycoords=("data", "axes fraction"),
                xytext=(offset, 0), textcoords="offset points",
                ha=align, va="top", color=INK, fontsize=9)


def image_grid(rows, captions, ncols=4, wrap=34, image_height=3.6, caption_lines=7, fontsize=8,
               title=None):
    """Show product images in a grid with a wrapped caption under each one.

    rows: dicts with an "image" path relative to data/.  captions: one string per row.
    """
    rows, captions = list(rows), list(captions)
    nrows = -(-len(rows) // ncols)
    text_height = 0.17 * caption_lines
    title_height = 0.5 if title else 0.0
    fig_height = nrows * (image_height + text_height) + title_height
    fig, axes = plt.subplots(
        nrows * 2, ncols,
        figsize=(ncols * 2.7, fig_height),
        gridspec_kw={"height_ratios": [image_height, text_height] * nrows, "hspace": 0.06,
                     "wspace": 0.08},
        squeeze=False,
    )
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=1 - title_height / fig_height)
    if title:
        fig.suptitle(title, x=0.01, y=1 - 0.1 / fig_height, ha="left", va="top", fontsize=11,
                     fontweight="bold")
    for ax in axes.flat:
        ax.axis("off")
    for index, (row, caption) in enumerate(zip(rows, captions)):
        r, c = divmod(index, ncols)
        with Image.open(DATA_DIR / row["image"]) as img:
            axes[r * 2, c].imshow(img)
        axes[r * 2 + 1, c].text(0.5, 1.0, textwrap.fill(caption, wrap), ha="center", va="top",
                                fontsize=fontsize, color=INK, transform=axes[r * 2 + 1, c].transAxes)
    return fig


def caption_comparison(rows, fields, title=None, wrap=96, image_col=1.9, width=11.0):
    """One row per image: the picture on the left, several captions for it on the right.

    rows: dicts with an "image" path relative to data/ plus the text fields.
    fields: (key, label) pairs, shown top to bottom, e.g. ("reference", "Reference").
    """
    rows = list(rows)
    line, label_height, gap = 0.18, 0.17, 0.11  # inches
    wrapped = [[textwrap.fill(str(row[key]), wrap) for key, _ in fields] for row in rows]
    heights = [
        max(sum(label_height + line * (text.count("\n") + 1) + gap for text in texts), image_col * 1.4)
        for texts in wrapped
    ]
    title_height = 0.5 if title else 0.0
    fig_height = sum(heights) + title_height + 0.15 * len(rows)
    fig = plt.figure(figsize=(width, fig_height))
    grid = fig.add_gridspec(len(rows), 2, width_ratios=[image_col, width - image_col],
                            height_ratios=heights, hspace=0.12, wspace=0.03, left=0.01, right=0.99,
                            bottom=0.01, top=1 - title_height / fig_height)
    if title:
        fig.suptitle(title, x=0.01, y=1 - 0.1 / fig_height, ha="left", va="top", fontsize=11,
                     fontweight="bold")
    for index, (row, texts) in enumerate(zip(rows, wrapped)):
        image_ax, text_ax = fig.add_subplot(grid[index, 0]), fig.add_subplot(grid[index, 1])
        image_ax.axis("off")
        text_ax.axis("off")
        with Image.open(DATA_DIR / row["image"]) as img:
            image_ax.imshow(img)
        cursor = 0.0  # inches from the top of this row
        for (_, label), text in zip(fields, texts):
            text_ax.text(0.01, 1 - cursor / heights[index], label, transform=text_ax.transAxes,
                         va="top", fontsize=8, color=MUTED)
            cursor += label_height
            text_ax.text(0.01, 1 - cursor / heights[index], text, transform=text_ax.transAxes,
                         va="top", fontsize=9.5, color=INK, linespacing=1.25)
            cursor += line * (text.count("\n") + 1) + gap
    return fig

#!/usr/bin/env python3
"""
Generate publication-quality charts for the gemma-ws-edit project.

Reads CSV data from charts/data/ and saves PNG/SVG charts to assets/charts/.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from matplotlib.patches import Patch

# ---- Styling ----
mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.edgecolor": "#cccccc",
    "axes.linewidth": 0.8,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
})

PALETTE = {
    "baseline": "#4c72b0",   # muted blue
    "edited":   "#dd8452",   # muted orange
    "accent":   "#55a868",   # green
    "neutral":  "#8c8c8c",   # gray
}

CHART_DIR = os.path.join(os.path.dirname(__file__), "..", "assets", "charts")
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(CHART_DIR, exist_ok=True)


def save_fig(fig, name):
    """Save a figure as both PNG and SVG."""
    png_path = os.path.join(CHART_DIR, f"{name}.png")
    svg_path = os.path.join(CHART_DIR, f"{name}.svg")
    fig.savefig(png_path, facecolor="white")
    fig.savefig(svg_path, facecolor="white")
    plt.close(fig)
    print(f"Saved: {png_path}")


def chart_baseline_vs_edited():
    """Grouped bar chart: baseline vs edited across all three benchmark metrics."""
    df = pd.read_csv(os.path.join(DATA_DIR, "benchmark_summary.csv"))

    # Convert to percentage for readability
    df["baseline_pct"] = df["baseline"] * 100
    df["edited_pct"] = df["edited"] * 100

    labels = [
        "GSM8K (flexible)",
        "GSM8K (strict)",
        "MATH-500",
    ]
    baseline_vals = [
        df.loc[df["metric"] == "exact_match_flexible", "baseline_pct"].values[0],
        df.loc[df["metric"] == "exact_match_strict", "baseline_pct"].values[0],
        df.loc[df["metric"] == "exact_match", "baseline_pct"].values[0],
    ]
    edited_vals = [
        df.loc[df["metric"] == "exact_match_flexible", "edited_pct"].values[0],
        df.loc[df["metric"] == "exact_match_strict", "edited_pct"].values[0],
        df.loc[df["metric"] == "exact_match", "edited_pct"].values[0],
    ]

    x = range(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))

    bars1 = ax.bar([i - width / 2 for i in x], baseline_vals, width,
                   label="Baseline (original)", color=PALETTE["baseline"], edgecolor="white", linewidth=0.5)
    bars2 = ax.bar([i + width / 2 for i in x], edited_vals, width,
                   label="Edited (head amplification)", color=PALETTE["edited"], edgecolor="white", linewidth=0.5)

    # Value labels on bars
    for bars in (bars1, bars2):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f"{height:.2f}%",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Exact match accuracy (%)")
    ax.set_title("Baseline vs Weight-Edited Gemma 3 270M\n(GSM8K & MATH-500)")
    ax.set_ylim(0, max(max(baseline_vals), max(edited_vals)) * 1.35)
    ax.legend(frameon=False, loc="upper left")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)

    # Annotation for the GSM8K gain
    gain = edited_vals[0] - baseline_vals[0]
    ax.annotate(f"+{gain:.2f}pp",
                xy=(0 - width / 2, baseline_vals[0]),
                xytext=(0 - width / 2, max(baseline_vals[0], edited_vals[0]) + 0.05),
                ha="center", fontsize=9, color=PALETTE["accent"], fontweight="bold")

    plt.tight_layout()
    save_fig(fig, "baseline_vs_edited")


def chart_math500_by_subject():
    """Horizontal bar chart: MATH-500 accuracy by subject."""
    df = pd.read_csv(os.path.join(DATA_DIR, "math500_by_subject.csv"))
    df = df.sort_values("accuracy_percent", ascending=True)

    fig, ax = plt.subplots(figsize=(8, 5.5))

    colors = [PALETTE["accent"] if v > 0 else PALETTE["neutral"] for v in df["accuracy_percent"]]
    bars = ax.barh(df["subject"], df["accuracy_percent"], color=colors, edgecolor="white", linewidth=0.5)

    for bar in bars:
        width = bar.get_width()
        ax.annotate(f"{width:.2f}%",
                    xy=(width, bar.get_y() + bar.get_height() / 2),
                    xytext=(3, 0), textcoords="offset points",
                    ha="left", va="center", fontsize=9, fontweight="bold")

    ax.set_xlabel("Exact match accuracy (%)")
    ax.set_title("MATH-500 Accuracy by Subject\n(Gemma 3 270M, baseline, 500 samples)")
    ax.set_xlim(0, max(df["accuracy_percent"]) * 1.4 + 1)
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)

    # Legend
    legend_elements = [
        Patch(facecolor=PALETTE["accent"], label="Non-zero accuracy"),
        Patch(facecolor=PALETTE["neutral"], label="Zero accuracy"),
    ]
    ax.legend(handles=legend_elements, frameon=False, loc="lower right")

    plt.tight_layout()
    save_fig(fig, "math500_by_subject")


def chart_math500_by_level():
    """Bar chart: MATH-500 accuracy by difficulty level."""
    df = pd.read_csv(os.path.join(DATA_DIR, "math500_by_level.csv"))
    df = df.sort_values("difficulty_level")

    fig, ax = plt.subplots(figsize=(8, 5))

    colors = [PALETTE["accent"] if v > 0 else PALETTE["neutral"] for v in df["accuracy_percent"]]
    bars = ax.bar(df["difficulty_level"].astype(str), df["accuracy_percent"],
                  color=colors, edgecolor="white", linewidth=0.5)

    for bar in bars:
        height = bar.get_height()
        ax.annotate(f"{height:.2f}%",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xlabel("Difficulty level")
    ax.set_ylabel("Exact match accuracy (%)")
    ax.set_title("MATH-500 Accuracy by Difficulty Level\n(Gemma 3 270M, baseline, 500 samples)")
    ax.set_ylim(0, max(df["accuracy_percent"]) * 1.4 + 1)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)

    legend_elements = [
        Patch(facecolor=PALETTE["accent"], label="Non-zero accuracy"),
        Patch(facecolor=PALETTE["neutral"], label="Zero accuracy"),
    ]
    ax.legend(handles=legend_elements, frameon=False, loc="upper right")

    plt.tight_layout()
    save_fig(fig, "math500_by_level")


def chart_summary_dashboard():
    """Combined dashboard: baseline vs edited + subject breakdown."""
    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.2], hspace=0.35, wspace=0.25)

    # Top: Baseline vs Edited (all three)
    ax1 = fig.add_subplot(gs[0, :])
    df = pd.read_csv(os.path.join(DATA_DIR, "benchmark_summary.csv"))
    df["baseline_pct"] = df["baseline"] * 100
    df["edited_pct"] = df["edited"] * 100

    labels = ["GSM8K\n(flexible)", "GSM8K\n(strict)", "MATH-500"]
    baseline_vals = [
        df.loc[df["metric"] == "exact_match_flexible", "baseline_pct"].values[0],
        df.loc[df["metric"] == "exact_match_strict", "baseline_pct"].values[0],
        df.loc[df["metric"] == "exact_match", "baseline_pct"].values[0],
    ]
    edited_vals = [
        df.loc[df["metric"] == "exact_match_flexible", "edited_pct"].values[0],
        df.loc[df["metric"] == "exact_match_strict", "edited_pct"].values[0],
        df.loc[df["metric"] == "exact_match", "edited_pct"].values[0],
    ]

    x = range(len(labels))
    width = 0.35
    ax1.bar([i - width / 2 for i in x], baseline_vals, width, label="Baseline", color=PALETTE["baseline"], edgecolor="white")
    ax1.bar([i + width / 2 for i in x], edited_vals, width, label="Edited", color=PALETTE["edited"], edgecolor="white")
    for i, (b, e) in enumerate(zip(baseline_vals, edited_vals)):
        ax1.annotate(f"{b:.2f}%", (i - width / 2, b), xytext=(0, 3), textcoords="offset points",
                     ha="center", fontsize=8, fontweight="bold")
        ax1.annotate(f"{e:.2f}%", (i + width / 2, e), xytext=(0, 3), textcoords="offset points",
                     ha="center", fontsize=8, fontweight="bold")
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(labels)
    ax1.set_ylabel("Exact match (%)")
    ax1.set_title("Baseline vs Weight-Edited", fontweight="bold")
    ax1.set_ylim(0, max(max(baseline_vals), max(edited_vals)) * 1.4)
    ax1.legend(frameon=False, fontsize=9)
    ax1.grid(axis="y", linestyle="--", alpha=0.4)
    ax1.spines[["top", "right"]].set_visible(False)

    # Bottom-left: MATH-500 by subject
    ax2 = fig.add_subplot(gs[1, 0])
    df_subj = pd.read_csv(os.path.join(DATA_DIR, "math500_by_subject.csv")).sort_values("accuracy_percent", ascending=True)
    colors = [PALETTE["accent"] if v > 0 else PALETTE["neutral"] for v in df_subj["accuracy_percent"]]
    ax2.barh(df_subj["subject"], df_subj["accuracy_percent"], color=colors, edgecolor="white")
    for i, (_, row) in enumerate(df_subj.iterrows()):
        ax2.annotate(f"{row['accuracy_percent']:.2f}%", (row["accuracy_percent"], i),
                     xytext=(2, 0), textcoords="offset points", ha="left", va="center", fontsize=8)
    ax2.set_xlabel("Accuracy (%)")
    ax2.set_title("MATH-500 by Subject", fontweight="bold")
    ax2.set_xlim(0, max(df_subj["accuracy_percent"]) * 1.5 + 1)
    ax2.grid(axis="x", linestyle="--", alpha=0.4)
    ax2.spines[["top", "right"]].set_visible(False)

    # Bottom-right: MATH-500 by level
    ax3 = fig.add_subplot(gs[1, 1])
    df_lvl = pd.read_csv(os.path.join(DATA_DIR, "math500_by_level.csv")).sort_values("difficulty_level")
    colors = [PALETTE["accent"] if v > 0 else PALETTE["neutral"] for v in df_lvl["accuracy_percent"]]
    ax3.bar(df_lvl["difficulty_level"].astype(str), df_lvl["accuracy_percent"],
            color=colors, edgecolor="white")
    for i, (_, row) in enumerate(df_lvl.iterrows()):
        ax3.annotate(f"{row['accuracy_percent']:.2f}%", (i, row["accuracy_percent"]),
                     xytext=(0, 2), textcoords="offset points", ha="center", va="bottom", fontsize=8)
    ax3.set_xlabel("Difficulty level")
    ax3.set_ylabel("Accuracy (%)")
    ax3.set_title("MATH-500 by Difficulty", fontweight="bold")
    ax3.set_ylim(0, max(df_lvl["accuracy_percent"]) * 1.5 + 1)
    ax3.grid(axis="y", linestyle="--", alpha=0.4)
    ax3.spines[["top", "right"]].set_visible(False)

    fig.suptitle("gemma-ws-edit: Gemma 3 270M Weight-Space Editing Results", fontsize=15, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_fig(fig, "summary_dashboard")


if __name__ == "__main__":
    print("Generating charts for gemma-ws-edit...")
    chart_baseline_vs_edited()
    chart_math500_by_subject()
    chart_math500_by_level()
    chart_summary_dashboard()
    print(f"\nAll charts saved to: {os.path.abspath(CHART_DIR)}")
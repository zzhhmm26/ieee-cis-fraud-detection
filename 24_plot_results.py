"""Create traceable project figures from results/metrics.csv.

All panels use measured, single-split results. No uncertainty interval is shown
because repeated folds or seeds were not run; this limitation is stated in each
figure footer.
"""

from pathlib import Path

import matplotlib
import matplotlib as mpl

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Keep SVG text editable and enforce a readable publication-style minimum.
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['font.size'] = 7
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams.update({
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
})
plt.rcParams["axes.spines.right"] = False
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.linewidth"] = 0.8
plt.rcParams["legend.frameon"] = False


ROOT = Path(__file__).resolve().parent
METRICS_PATH = ROOT / "results" / "metrics.csv"
GROUP_IMPORTANCE_PATH = ROOT / "results" / "feature_group_importance.csv"
OUTPUT_DIR = ROOT / "results" / "figures"

BLUE = "#0F4D92"
BLUE_LIGHT = "#3775BA"
GREY = "#767676"
GREY_LIGHT = "#CFCECE"
GREEN = "#2E9E44"
ORANGE = "#E28E2C"


def save_figure(fig, stem):
    """Export editable vector files plus a high-resolution PNG preview."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_DIR / f"{stem}.svg", bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / f"{stem}.png", dpi=400, bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / f"{stem}.tiff", dpi=600, bbox_inches="tight")
    plt.close(fig)


def add_value_labels(ax, bars, values, fmt="{:.3f}"):
    upper = ax.get_ylim()[1]
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + upper * 0.012,
            fmt.format(value),
            ha="center",
            va="bottom",
            fontsize=6,
        )


def set_percent_axis(ax, label):
    ax.set_ylim(0, 0.72)
    ax.set_yticks(np.arange(0, 0.71, 0.1))
    ax.set_yticklabels([f"{tick:.0%}" for tick in ax.get_yticks()])
    ax.set_ylabel(label)
    ax.grid(axis="y", color="#E6E6E6", linewidth=0.7)
    ax.set_axisbelow(True)


def baseline_comparison(metrics):
    """Claim: XGBoost is the strongest fair V1 baseline on the late window."""
    order = [
        "logistic_regression_v1",
        "decision_tree_v1",
        "random_forest_v1",
        "xgb_v1_baseline",
    ]
    labels = ["Logistic\nregression", "Decision\ntree", "Random\nforest", "XGBoost"]
    subset = metrics.set_index("experiment").loc[order]
    x = np.arange(len(order))
    colors = [GREY_LIGHT, GREY, "#9FB8D1", BLUE]

    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.8), constrained_layout=True)
    for ax, column, title in zip(
        axes,
        ["pr_auc", "top_5_capture"],
        ["PR-AUC", "Fraud captured in top 5% review queue"],
    ):
        values = subset[column].to_numpy()
        bars = ax.bar(x, values, color=colors, edgecolor="#4D4D4D", linewidth=0.45)
        set_percent_axis(ax, title)
        ax.set_xticks(x, labels)
        add_value_labels(ax, bars, values)

    axes[0].text(-0.17, 1.04, "a", transform=axes[0].transAxes, fontweight="bold", fontsize=8)
    axes[1].text(-0.17, 1.04, "b", transform=axes[1].transAxes, fontweight="bold", fontsize=8)
    fig.suptitle("Fair baseline comparison on the late temporal validation window", fontsize=9, y=1.05)
    fig.text(
        0.5,
        -0.08,
        "All models use V1 features and the same chronological 80/20 split; single-split metrics, no confidence intervals.",
        ha="center",
        fontsize=6,
    )
    save_figure(fig, "01_baseline_comparison")


def feature_ablation(metrics):
    """Claim: C and D groups provide the major temporal-validation gains."""
    order = [
        "xgb_v1_baseline",
        "xgb_v2_frequency",
        "xgb_v3_C_group",
        "xgb_v4_C_M_group",
        "xgb_v5_C_D_300_late",
        "xgb_v5_C_D_414_late",
    ]
    labels = ["V1\nBase", "V2\n+Freq", "V3\n+C", "V4\n+M", "V5\n+D", "V5\n414 trees"]
    subset = metrics.set_index("experiment").loc[order]
    x = np.arange(len(order))

    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.8), constrained_layout=True)
    for ax, column, title in zip(
        axes,
        ["pr_auc", "top_5_capture"],
        ["PR-AUC", "Fraud captured in top 5% review queue"],
    ):
        values = subset[column].to_numpy()
        ax.plot(x, values, color=BLUE, marker="o", linewidth=1.8, markersize=5)
        ax.scatter(x[-1], values[-1], color=GREEN, s=30, zorder=3)
        set_percent_axis(ax, title)
        ax.set_xticks(x, labels)
        for xi, value in zip(x, values):
            ax.text(xi, value + 0.018, f"{value:.3f}", ha="center", fontsize=6)

    axes[0].text(-0.17, 1.04, "a", transform=axes[0].transAxes, fontweight="bold", fontsize=8)
    axes[1].text(-0.17, 1.04, "b", transform=axes[1].transAxes, fontweight="bold", fontsize=8)
    fig.suptitle("Temporal feature ablation and tree-count check", fontsize=9, y=1.05)
    fig.text(
        0.5,
        -0.08,
        "All points use the late chronological window (first 80% train, last 20% validation). Green = current main model.",
        ha="center",
        fontsize=6,
    )
    save_figure(fig, "02_feature_ablation")


def validation_strategy(metrics):
    """Claim: random splitting gives an optimistic estimate relative to temporal validation."""
    order = [
        "xgb_v5_C_D_300_early",
        "xgb_v5_C_D_300_late",
        "xgb_v5_C_D_random",
    ]
    labels = ["Early temporal\n60%→80%", "Late temporal\n80%→100%", "Stratified\nrandom 80/20"]
    subset = metrics.set_index("experiment").loc[order]
    x = np.arange(len(order))
    colors = [BLUE_LIGHT, BLUE, GREY_LIGHT]

    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.8), constrained_layout=True)
    for ax, column, title in zip(
        axes,
        ["pr_auc", "top_5_capture"],
        ["PR-AUC", "Fraud captured in top 5% review queue"],
    ):
        values = subset[column].to_numpy()
        bars = ax.bar(x, values, color=colors, edgecolor="#4D4D4D", linewidth=0.45)
        bars[-1].set_hatch("//")
        set_percent_axis(ax, title)
        ax.set_xticks(x, labels)
        add_value_labels(ax, bars, values)

    axes[0].text(-0.17, 1.04, "a", transform=axes[0].transAxes, fontweight="bold", fontsize=8)
    axes[1].text(-0.17, 1.04, "b", transform=axes[1].transAxes, fontweight="bold", fontsize=8)
    fig.suptitle("Validation design materially changes apparent model performance", fontsize=9, y=1.05)
    fig.text(
        0.5,
        -0.08,
        "All bars use V5 features and 300 trees. Hatched random split is a reference only, not used for model selection.",
        ha="center",
        fontsize=6,
    )
    save_figure(fig, "03_validation_strategy")


def review_capacity(metrics):
    """Claim: the final model increases fraud capture at fixed review capacity."""
    order = [
        "logistic_regression_v1",
        "random_forest_v1",
        "xgb_v1_baseline",
        "xgb_v5_C_D_414_late",
    ]
    labels = ["Logistic regression", "Random forest", "XGBoost V1", "XGBoost V5 (414 trees)"]
    colors = [GREY_LIGHT, GREY, BLUE_LIGHT, BLUE]
    widths = [1.1, 1.1, 1.3, 2.2]
    subset = metrics.set_index("experiment").loc[order]
    review_rates = np.array([0.01, 0.05, 0.10])

    fig, ax = plt.subplots(figsize=(3.55, 2.8), constrained_layout=True)
    for (_, row), label, color, width in zip(subset.iterrows(), labels, colors, widths):
        capture = row[["top_1_capture", "top_5_capture", "top_10_capture"]].to_numpy(dtype=float)
        ax.plot(review_rates, capture, marker="o", markersize=4.5, linewidth=width, color=color, label=label)

    ax.set_xlim(0.007, 0.103)
    ax.set_ylim(0, 0.75)
    ax.set_xticks(review_rates, ["1%", "5%", "10%"])
    ax.set_yticks(np.arange(0, 0.76, 0.15))
    ax.set_yticklabels([f"{tick:.0%}" for tick in ax.get_yticks()])
    ax.set_xlabel("Manual review budget")
    ax.set_ylabel("Fraud captured")
    ax.grid(axis="y", color="#E6E6E6", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", fontsize=6)
    ax.text(-0.18, 1.04, "a", transform=ax.transAxes, fontweight="bold", fontsize=8)
    fig.suptitle("Review capacity determines fraud captured", fontsize=9, y=1.04)
    fig.text(
        0.5,
        -0.07,
        "All models use the late chronological validation window; points are measured at fixed review budgets, not interpolated.",
        ha="center",
        fontsize=6,
    )
    save_figure(fig, "04_review_capacity")


def feature_group_reliance():
    """Claim: anonymous fields and missingness account for most V5 split gain."""
    groups = pd.read_csv(GROUP_IMPORTANCE_PATH)
    order = [
        "Business categorical features",
        "Anonymous C group",
        "Anonymous D values",
        "D missing indicators",
        "Frequency encoding",
        "Amount and time features",
    ]
    groups = groups.set_index("feature_group").loc[order].reset_index()
    colors = [GREY, BLUE, BLUE_LIGHT, "#8BCF8B", "#42949E", GREY_LIGHT]

    fig, ax = plt.subplots(figsize=(4.4, 3.1), constrained_layout=True)
    y = np.arange(len(groups))
    values = groups["importance_sum"].to_numpy()
    bars = ax.barh(y, values, color=colors, edgecolor="#4D4D4D", linewidth=0.45)
    ax.set_yticks(y, groups["feature_group"])
    ax.invert_yaxis()
    ax.set_xlim(0, 0.45)
    ax.set_xticks(np.arange(0, 0.46, 0.1))
    ax.set_xticklabels([f"{tick:.0%}" for tick in ax.get_xticks()])
    ax.set_xlabel("Share of XGBoost split-gain importance")
    ax.grid(axis="x", color="#E6E6E6", linewidth=0.7)
    ax.set_axisbelow(True)
    for bar, value, count in zip(bars, values, groups["feature_count"]):
        ax.text(value + 0.007, bar.get_y() + bar.get_height() / 2, f"{value:.1%}  (n={count})", va="center", fontsize=6)
    ax.text(-0.18, 1.04, "a", transform=ax.transAxes, fontweight="bold", fontsize=8)
    fig.suptitle("V5 depends substantially on anonymous fields and missingness", fontsize=9, y=1.03)
    fig.text(
        0.5,
        -0.08,
        "V5, 300 trees, late temporal window. Split-gain is a relative association measure, not causal importance.",
        ha="center",
        fontsize=6,
    )
    save_figure(fig, "05_feature_group_reliance")


def main():
    metrics = pd.read_csv(METRICS_PATH)
    required = {
        "experiment",
        "pr_auc",
        "top_5_capture",
        "validation_scheme",
    }
    missing = required.difference(metrics.columns)
    if missing:
        raise ValueError(f"metrics.csv is missing required columns: {sorted(missing)}")

    baseline_comparison(metrics)
    feature_ablation(metrics)
    validation_strategy(metrics)
    review_capacity(metrics)
    feature_group_reliance()

    print(f"Created 5 evidence-backed figure sets in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

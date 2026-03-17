#!/usr/bin/env python3
"""Create latency box plots from clean outputs after removing outliers."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
CLEAN_DIR = ROOT / "Clean output"
PLOT_DIR = ROOT / "analysis_plots"
OUTPUT_PNG = PLOT_DIR / "latency_boxplot_no_outliers.png"
OUTPUT_SVG = PLOT_DIR / "latency_boxplot_no_outliers.svg"
SUMMARY_CSV = PLOT_DIR / "latency_outlier_filter_summary.csv"

MODEL_DISPLAY_NAMES = {
    "claude-sonnet-4-6": "Claude Sonnet 4.6",
    "gemini-3.1-pro-preview": "Gemini 3.1 Pro Preview",
    "gpt-5.4": "GPT 5.4",
    "meta-llama/llama-4-scout-17b-16e-instruct": "Llama 4 Scout 17B",
}


def load_latency_rows(clean_dir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(clean_dir.glob("*.json")):
        with path.open("r", encoding="utf-8") as f:
            records = json.load(f)

        if not isinstance(records, list):
            continue

        for rec in records:
            if not isinstance(rec, dict):
                continue
            model = rec.get("model")
            elapsed_ms = rec.get("elapsed_ms")
            if model is None or elapsed_ms is None:
                continue

            try:
                latency = float(elapsed_ms)
            except (TypeError, ValueError):
                continue

            rows.append(
                {
                    "model": str(model),
                    "elapsed_ms": latency,
                    "source_file": path.name,
                }
            )
    return pd.DataFrame(rows)


def remove_outliers_per_model_iqr(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    kept_parts = []
    summary_rows = []

    for model, group in df.groupby("model", sort=False):
        q1 = group["elapsed_ms"].quantile(0.25)
        q3 = group["elapsed_ms"].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        keep_mask = (group["elapsed_ms"] >= lower) & (group["elapsed_ms"] <= upper)
        kept = group.loc[keep_mask].copy()
        removed_count = int((~keep_mask).sum())

        kept_parts.append(kept)
        summary_rows.append(
            {
                "model": model,
                "original_n": int(len(group)),
                "filtered_n": int(len(kept)),
                "removed_outliers_n": removed_count,
                "removed_pct": round((removed_count / len(group)) * 100, 2) if len(group) else 0.0,
                "iqr_lower_bound_ms": float(lower),
                "iqr_upper_bound_ms": float(upper),
            }
        )

    filtered = pd.concat(kept_parts, ignore_index=True) if kept_parts else df.iloc[0:0].copy()
    summary = pd.DataFrame(summary_rows).sort_values("model").reset_index(drop=True)
    return filtered, summary


def plot_latency_boxplot(df_filtered: pd.DataFrame) -> None:
    if df_filtered.empty:
        raise ValueError("No data available after outlier filtering.")

    sns.set_theme(style="whitegrid")
    df_plot = df_filtered.copy()
    df_plot["model_display"] = df_plot["model"].map(MODEL_DISPLAY_NAMES).fillna(df_plot["model"])

    model_order = (
        df_plot.groupby("model_display")["elapsed_ms"]
        .median()
        .sort_values(ascending=True)
        .index.tolist()
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(
        data=df_plot,
        x="model_display",
        y="elapsed_ms",
        order=model_order,
        showfliers=False,
        color="#79a7c7",
        width=0.6,
        ax=ax,
    )

    ax.set_title("Model Latency")
    ax.set_xlabel("Model")
    ax.set_ylabel("Latency (ms)")
    ax.tick_params(axis="x", rotation=20)
    for label in ax.get_xticklabels():
        label.set_ha("right")

    fig.tight_layout()
    fig.savefig(OUTPUT_PNG, dpi=200, bbox_inches="tight")
    fig.savefig(OUTPUT_SVG, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    latency_df = load_latency_rows(CLEAN_DIR)
    if latency_df.empty:
        raise ValueError(f"No latency rows found in {CLEAN_DIR}")

    filtered_df, summary_df = remove_outliers_per_model_iqr(latency_df)
    summary_df.to_csv(SUMMARY_CSV, index=False)
    plot_latency_boxplot(filtered_df)

    print(f"Loaded rows: {len(latency_df)}")
    print(f"Rows after outlier filtering: {len(filtered_df)}")
    print(f"Saved: {OUTPUT_PNG.relative_to(ROOT)}")
    print(f"Saved: {OUTPUT_SVG.relative_to(ROOT)}")
    print(f"Saved: {SUMMARY_CSV.relative_to(ROOT)}")
    print("")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()

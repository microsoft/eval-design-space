import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.paths import REPO_ROOT, repo_path

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

CSV_PATH = "results/yin_sampling_downsampling_analysis_original.csv"
MODEL_LABELS = {
    "gpt-4o-mini": "gpt-4o-mini",
    "gpt-5-nano-2025-08-07": "gpt-5-nano",
    "mistral-small-24b": "mistral-small-24b",
    "meta-llama-3.1-8b-instruct-turbo": "meta-llama-3.1-8b",
    "meta-llama-3.3-70b-instruct-turbo": "meta-llama-3.3-70b",
    "mistral-7b-v0.3": "mistral-7b",
}


def mean_selection_rates(data: pd.DataFrame, fixed_column: str, fixed_value: str,
                         comparison_column: str, comparison_values: list[str]) -> pd.DataFrame:
    """Return mean Women's selection rates by matched display model and module value."""
    subset = data[
        (data["gender"] == "W")
        & (data[fixed_column] == fixed_value)
        & (data[comparison_column].isin(comparison_values))
        & (data["model"].isin(MODEL_LABELS))
    ].copy()
    if subset.empty:
        raise ValueError(
            f"No rows found with {fixed_column}={fixed_value} and "
            f"{comparison_column} in {comparison_values}."
        )

    subset["display_model"] = subset["model"].map(MODEL_LABELS)
    return subset.pivot_table(
        index="display_model",
        columns=comparison_column,
        values="selection_rate",
        aggfunc="mean",
    )


df = pd.read_csv(repo_path(CSV_PATH))
models = list(MODEL_LABELS.values())

top_rates = mean_selection_rates(
    df, "name_bundle", "gaeb", "job_bundle", ["armstrong", "yin"]
).reindex(models)
top_green_data = top_rates["armstrong"].tolist()
top_orange_data = top_rates["yin"].tolist()

bottom_rates = mean_selection_rates(
    df, "job_bundle", "armstrong", "name_bundle", ["gaeb", "yin"]
).reindex(models)
bottom_blue_data = bottom_rates["gaeb"].tolist()
bottom_gold_data = bottom_rates["yin"].tolist()

y = np.arange(len(models))
bar_width = 0.35

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.5, 15))
plt.subplots_adjust(hspace=0.3)

ax1.barh(y - bar_width/2, top_green_data, bar_width, color='#82b09e')
ax1.barh(y + bar_width/2, top_orange_data, bar_width, color='#d99b78')

ax1.set_yticks(y)
ax1.set_yticklabels(models, fontsize=27)
ax1.set_xlim(0.3, 0.6)
ax1.set_xticks([0.3, 0.4, 0.5, 0.6])
ax1.set_xticklabels([0.3, 0.4, 0.5, 0.6], fontsize=18)
ax1.set_xlabel('Selection Rate for Women', fontsize=20)
ax1.axvline(x=0.5, color='gray', linestyle='--', linewidth=1)
ax1.invert_yaxis()

ax2.barh(y - bar_width/2, bottom_blue_data, bar_width, color='#6c85d9')
ax2.barh(y + bar_width/2, bottom_gold_data, bar_width, color='#dcb768')

ax2.set_yticks(y)
ax2.set_yticklabels(models, fontsize=27)
ax2.set_xlim(0.3, 0.6)
ax2.set_xticks([0.3, 0.4, 0.5, 0.6])
ax2.set_xticklabels([0.3, 0.4, 0.5, 0.6], fontsize=18)
ax2.set_xlabel('Selection Rate for Women', fontsize=20)
ax2.axvline(x=0.5, color='gray', linestyle='--', linewidth=1)
ax2.invert_yaxis()

line = Line2D([0.05, 0.95], [0.52, 0.52], transform=fig.transFigure, 
              color="black", linewidth=3)

out_dir = "results/figs/yin/swapped"
os.makedirs(repo_path(out_dir), exist_ok=True)
out_path = os.path.join(repo_path(out_dir), "swapped_module_results.png")
fig.savefig(repo_path(out_path), dpi=150, bbox_inches="tight")
print(f"Created plot: {out_path}")
plt.close(fig)

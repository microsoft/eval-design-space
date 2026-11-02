import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.paths import REPO_ROOT, repo_path

import argparse
import ast
import os
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np


# Match styling from plot_relative.py
TITLE_FONT_SIZE = 24
LABEL_FONT_SIZE = 22
LEGEND_FONT_SIZE = 15
TICK_FONT_SIZE = 20
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Inter-Regular', 'DejaVu Sans', 'Arial', 'Liberation Sans'],
    'axes.titlesize': TITLE_FONT_SIZE,
    'axes.labelsize': LABEL_FONT_SIZE,
    'legend.fontsize': LEGEND_FONT_SIZE,
    'legend.title_fontsize': LEGEND_FONT_SIZE,
    'xtick.labelsize': TICK_FONT_SIZE,
    'ytick.labelsize': TICK_FONT_SIZE,
})


PAPER_COLORS: Dict[str, str] = {
    'armstrong': 'tab:blue',
    'wang': 'tab:red',
    'yin': 'tab:pink',
}


def ensure_dirs(*paths: str) -> None:
    for p in paths:
        os.makedirs(repo_path(p), exist_ok=True)


def parse_result_value(value: object) -> Optional[float]:
    if value is None:
        return None
    try:
        parsed = ast.literal_eval(str(value))
    except (SyntaxError, ValueError):
        parsed = value

    if isinstance(parsed, (list, tuple)):
        if not parsed:
            return None
        parsed = parsed[0]

    try:
        return float(parsed)
    except (TypeError, ValueError):
        return None


def filter_random_state(df, pd):
    random_state_column = next(
        (column for column in ('Random_State', 'random_state', 'Random State') if column in df.columns),
        None,
    )
    if random_state_column is None:
        return df
    return df[pd.to_numeric(df[random_state_column], errors='coerce') == 42.0].copy()


def dedupe_by_name_job_model_metric(df):
    required = ["Names", "Jobs", "Model", "Metric"]
    if not all(col in df.columns for col in required):
        return df.drop_duplicates().copy()
    return (
        df.sort_values(["Names", "Jobs", "Metric"])
          .drop_duplicates(subset=required, keep="first")
          .set_index(required)
          .reset_index()
    )


def get_disagreement_percentages(
    armstrong_path: str,
    wang_path: str,
    yin_path: str,
) -> List[float]:
    import pandas as pd

    armstrong_df = filter_random_state(pd.read_csv(repo_path(armstrong_path)), pd)
    armstrong_df = armstrong_df[armstrong_df['Names'] != 'seshadri']

    armstrong_df = armstrong_df[
        armstrong_df['Sensitive_Attribute_Vector'].eq("['Woman']")
        & armstrong_df['Metric'].eq('regression_coefficients')
    ].copy()
    armstrong_df = dedupe_by_name_job_model_metric(armstrong_df)
    armstrong_values = armstrong_df['Result_Vector'].map(parse_result_value).dropna()
    
    wang_df = filter_random_state(pd.read_csv(repo_path(wang_path)), pd)
    wang_df = wang_df[wang_df['Names'] != 'seshadri']
    wang_df = wang_df[
        wang_df['Sensitive_Attribute_Vector'].eq("['Woman']")
        & wang_df['Metric'].eq('score_difference')
    ].copy()
    wang_df = dedupe_by_name_job_model_metric(wang_df)
    wang_values = -wang_df['Result_Vector'].map(parse_result_value).dropna()


    yin_df = pd.read_csv(repo_path(yin_path))
    yin_df = yin_df[yin_df['gender'].eq('W')].copy()
    yin_df = yin_df[yin_df['name_bundle'] != 'seshadri'].copy()
    yin_dedupe_columns = [
        column for column in ('gender', 'job', 'model', 'rank', 'name_bundle', 'job_bundle')
        if column in yin_df.columns
    ]
    if yin_dedupe_columns:
        yin_df = yin_df.drop_duplicates(subset=yin_dedupe_columns, keep='first')
    yin_df = yin_df.drop_duplicates()
    yin_values = pd.to_numeric(yin_df['selection_rate'], errors='coerce').dropna()

    values_by_paper = {
        'yin': (yin_values, 0.5),
        'wang': (wang_values, 0.0),
        'armstrong': (armstrong_values, 0.0),
    }
    disagreement_percentages = []
    for paper, (values, threshold) in values_by_paper.items():
        if values.empty:
            raise ValueError(f'No valid values found for {paper}.')
        disagreement_percentages.append((values < threshold).mean() * 100)

    return disagreement_percentages

def plot_male_preference_bar(
    out_path: str,
    armstrong_path: str,
    wang_path: str,
    yin_path: str,
    title: str = 'Male Preference Share',
) -> None:
    papers: List[str] = ['yin', 'wang', 'armstrong']


    yin_disagreement, wang_disagreement, armstrong_disagreement = get_disagreement_percentages(
        armstrong_path,
        wang_path,
        yin_path,
    )

    percentages: List[float] = [yin_disagreement, wang_disagreement, armstrong_disagreement]
    labels = ['Yin', 'Wang', 'Armstrong']
    colors = [PAPER_COLORS[p] for p in papers]

    x = np.arange(len(papers))

    # Square figure
    fig, ax = plt.subplots(figsize=(5, 5))

    bars = ax.bar(
        x,
        percentages,
        color=colors,
        edgecolor='black',
        linewidth=1.2,
        width=0.62,
        alpha=0.40,
    )

    for bar, pct in zip(bars, percentages):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.6,
            f'{pct:.0f}%',
            ha='center',
            va='bottom',
            fontsize=26,
            # fontweight='bold',
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Percentage')
    ax.set_xlabel('Experimental Framework')
    ax.set_title(title)

    ax.set_ylim(0, max(percentages) + 8)
    ax.grid(axis='y', alpha=0.25)

    fig.tight_layout()
    ensure_dirs(os.path.dirname(out_path) or '.')
    fig.savefig(repo_path(out_path), dpi=180)
    plt.close(fig)


def main() -> None:
    cwd = str(REPO_ROOT)
    default_results_dir = os.path.join(repo_path(cwd), 'results')

    parser = argparse.ArgumentParser(description='Plot male-preference percentage bars for Yin, Wang, and Armstrong.')
    parser.add_argument('--output', type=str, default=os.path.join(repo_path(default_results_dir), 'figs', 'relative', 'male_preference_bar.png'))
    parser.add_argument('--armstrong-input', type=str, default=os.path.join(repo_path(default_results_dir), 'armstrong_sampling_downsampling_analysis_original.csv'))
    parser.add_argument('--wang-input', type=str, default=os.path.join(repo_path(default_results_dir), 'wang_sampling_downsampling_analysis_original.csv'))
    parser.add_argument('--yin-input', type=str, default=os.path.join(repo_path(default_results_dir), 'yin_sampling_downsampling_analysis_original.csv'))
    parser.add_argument('--title', type=str, default='Disagreement Rate')
    args = parser.parse_args()

    plot_male_preference_bar(
        out_path=args.output,
        armstrong_path=args.armstrong_input,
        wang_path=args.wang_input,
        yin_path=args.yin_input,
        title=args.title,
    )
    print(f'Saved: {args.output}')


if __name__ == '__main__':
    main()

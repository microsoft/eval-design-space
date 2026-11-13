from utils.paths import REPO_ROOT, repo_path
from utils.plot_references import (
    PUBLISHED_REFERENCES, load_published_reference, load_reproduced_reference,
    REPRODUCED_SETTING_LABEL, reported_setting_label, published_setting_label,
)

import argparse
import os
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from utils.noise_floor import compute_default_framework_stats
import seaborn as sns
import ast

from utils.variables import colors, metric_to_plot, linestyles

# Font size defaults (make titles, labels, legend and ticks larger)
TITLE_FONT_SIZE = 40
LABEL_FONT_SIZE = 28
LEGEND_FONT_SIZE = 20
TICK_FONT_SIZE = 25

model_to_plotname= {
    "meta-llama-3.1-8b-instruct-turbo": "Llama-3.1-8B-Instruct-Turbo",
    "mistral-small-24b": "Mistral-Small-24B",
    "meta-llama-3.3-70b-instruct-turbo": "Llama-3.3-70B-Instruct-Turbo",
    "gpt-5-nano-2025-08-07": "GPT-5 Nano",
    "gpt-4o-mini": "GPT-4o Mini",
    "mistral-7b-v0.2": "Mistral-7B-v0.2",
    "mistral-7b-v0.3": "Mistral-7B-v0.3",
    
}

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica', 'Inter', 'DejaVu Sans', 'Arial', 'Liberation Sans'],
    'axes.titlesize': TITLE_FONT_SIZE,
    'axes.labelsize': LABEL_FONT_SIZE,
    'legend.fontsize': LEGEND_FONT_SIZE,
    'legend.title_fontsize': LEGEND_FONT_SIZE,
    'xtick.labelsize': TICK_FONT_SIZE,
    'ytick.labelsize': TICK_FONT_SIZE
})

REF_LINE_BROWN = "#8B4513"
REF_LINE_STYLES = ['-.', ':']
REPORTED_SETTING_MODELS = ["gpt-3.5-turbo", "gpt-4"]
DEFAULT_PERFORMANCE_RANKING_FILE = os.path.join(repo_path(str(REPO_ROOT)), "yin", "data", "output", "performance_ranking.csv")


# ----------------------------
# Helpers
# ----------------------------
def ensure_dirs(*paths: str) -> None:
    for p in paths:
        os.makedirs(repo_path(p), exist_ok=True)


def add_pval_suffix_to_path(path: str, p_value_threshold: Optional[float]) -> str:
    """Add p-value threshold suffix to file path if threshold is provided."""
    if p_value_threshold is not None:
        base, ext = os.path.splitext(path)
        return f"{base}_pval_{p_value_threshold}{ext}"
    return path


def parse_p_values(x: Any) -> Optional[List[float]]:
    """
    Parse P_Value_Vector which may be stored as string like "[0.123]" or as list.
    """
    if pd.isna(x):
        return None
    if isinstance(x, list):
        try:
            return [float(v) for v in x]
        except Exception:
            return None
    
    # Try to parse as string representation of list
    try:
        val = ast.literal_eval(str(x))
        if isinstance(val, list):
            return [float(v) for v in val]
        return [float(val)]
    except Exception:
        # Try stripping brackets manually
        s = str(x).strip()
        if s.startswith("[") and s.endswith("]"):
            s = s[1:-1].strip()
        try:
            return [float(s)]
        except Exception:
            return None


def clean_values(values: List[float], remove_zeros: bool = False, remove_outliers: bool = False) -> List[float]:
    """Clean values by removing zeros and/or outliers.
    
    Handles multi-indexed dataframes and various pandas types (Index, Series, scalars).
    """
    if not values:
        return []
    
    # First, extract actual scalar values from any pandas types (Index, Series, etc.)
    extracted_values = []
    for v in values:
        # Handle pandas Index objects
        if isinstance(v, pd.Index):
            extracted_values.extend(v.tolist())
        # Handle pandas Series
        elif isinstance(v, pd.Series):
            extracted_values.extend(v.tolist())
        # Handle numpy arrays
        elif isinstance(v, np.ndarray):
            extracted_values.extend(v.tolist())
        # Handle scalar values (including pandas scalars)
        else:
            # Use .item() for pandas scalars, otherwise just use the value
            if hasattr(v, 'item'):
                try:
                    extracted_values.append(v.item())
                except (ValueError, AttributeError):
                    extracted_values.append(v)
            else:
                extracted_values.append(v)
    
    # Now convert to numeric using pandas (handles strings, mixed types, etc.)
    series = pd.Series(extracted_values)
    numeric_series = pd.to_numeric(series, errors='coerce')
    
    # Get only finite, non-null values
    cleaned = numeric_series[numeric_series.notna() & np.isfinite(numeric_series)].tolist()
    
    if remove_zeros:
        cleaned = [v for v in cleaned if v != 0]
    if remove_outliers:
        cleaned = [v for v in cleaned if 0 <= v <= 1]
    return cleaned


def benjamini_hochberg(pvals: List[float], q: float) -> List[bool]:
    """Return a list of booleans indicating which p-values are rejected by BH.

    True == rejected, False == not rejected (i.e. accepted).
    """
    m = len(pvals)
    if m == 0:
        return []
    arr = np.asarray(pvals, dtype=float)
    order = np.argsort(arr)
    sorted_p = arr[order]
    thresholds = np.array([(i + 1) / m * q for i in range(m)], dtype=float)
    k = -1
    for i in range(m):
        if sorted_p[i] <= thresholds[i]:
            k = i
    rejected_sorted = np.zeros(m, dtype=bool)
    if k >= 0:
        rejected_sorted[: k + 1] = True
    # unsort
    rejected = np.empty(m, dtype=bool)
    rejected[order] = rejected_sorted
    return rejected.tolist()


def load_reported_setting_vlines_from_published_references(
    source: str = PUBLISHED_REFERENCES,
    models: Optional[List[str]] = None,
) -> List[Tuple[float, str, Any]]:
    requested_models = list(REPORTED_SETTING_MODELS) if models is None else models
    model_aliases = {"gpt-3.5-turbo": "GPT-3.5", "gpt-4": "GPT-4"}
    color_map = plt.cm.tab10(np.linspace(0, 1, max(2, len(requested_models))))
    out: List[Tuple[float, str, Any]] = []
    for model, color in zip(requested_models, color_map):
        csv_model = model_aliases.get(model.strip().lower(), model)
        value = load_published_reference("yin", "selection_rate", source, model=csv_model)
        out.append((value, reported_setting_label(model), color))
        print(f"Reported setting ({model}) from {source}: selection_rate = {value:.6g}")
    return out


def load_reported_setting_vlines_from_performance_ranking(
    ranking_file: str,
    models: Optional[List[str]] = None,
    metric_col: str = "selection_rate",
    divide_by: float = 4.0,
) -> List[Tuple[float, str, Any]]:
    """Load model-specific reported-setting vlines from Yin performance_ranking.csv.

    For each requested model, this computes:
      sum(selection_rate over demo rows ending with '_W') / divide_by
    and returns lines labeled as Reported Setting (model).
    """
    requested_models = models or list(REPORTED_SETTING_MODELS)
    out: List[Tuple[float, str, Any]] = []

    if not os.path.exists(repo_path(ranking_file)):
        print(f"Warning: reported-settings file not found: {ranking_file}")
        return out

    try:
        rank_df = pd.read_csv(repo_path(ranking_file))
    except Exception as exc:
        print(f"Warning: failed to read {ranking_file}: {exc}")
        return out

    needed = {"model", metric_col}
    if not needed.issubset(rank_df.columns):
        print(
            f"Warning: {ranking_file} missing required columns {sorted(list(needed))}. "
            f"Found: {rank_df.columns.tolist()}"
        )
        return out

    if divide_by == 0:
        print("Warning: divide_by is 0; cannot compute reported settings.")
        return out

    color_map = plt.cm.tab10(np.linspace(0, 1, max(2, len(requested_models))))
    model_series = rank_df["model"].astype(str).str.strip().str.lower()

    for model, color in zip(requested_models, color_map):
        model_key = str(model).strip().lower()
        sub = rank_df[model_series.eq(model_key)].copy()
        if sub.empty:
            print(f"Skipping reported setting for {model}: no rows in {ranking_file}.")
            continue

        if "demo" in sub.columns:
            sub_w = sub[sub["demo"].astype(str).str.upper().str.endswith("_W")].copy()
            if not sub_w.empty:
                sub = sub_w

        vals = pd.to_numeric(sub[metric_col], errors="coerce")
        vals = vals[vals.notna() & np.isfinite(vals)]
        if vals.empty:
            print(f"Skipping reported setting for {model}: no valid {metric_col} values.")
            continue

        reported_value = float(vals.sum()) / float(divide_by)
        out.append((reported_value, reported_setting_label(model), color))
        print(f"Reported setting ({model}) from {ranking_file}: {metric_col}_sum/{divide_by:g} = {reported_value:.6g}")

    return out


# ----------------------------
# Plotting functions
# ----------------------------
def plot_global_density(
    clean_vals: List[float],
    metric: str,
    out_path: str,
    split_by: str = "all",
    vlines: Optional[List[Tuple[float, str, str]]] = None,
    sig_values: Optional[List[Tuple[float, float]]] = None,
    p_value_threshold: Optional[float] = None,
    bw_threshold: Optional[float] = None,
    color: Optional[str] = "purple",
    noise_floor: Optional[Tuple[float, float]] = None,
) -> None:
    """Plot a single global density plot for all values (Armstrong-style)."""

    # Expect `clean_vals` to already be cleaned by caller
    # If no explicit noise_floor is provided, derive it from a reproducible
    # sample of up to 25 values.
    if noise_floor is None:
        # Compute noise-floor mean/std by sampling up to 25 values
        noise_floor_results = compute_default_framework_stats()
        nf_n, nf_mean, nf_std = noise_floor_results['yin']['n'], noise_floor_results['yin']['mean'], noise_floor_results['yin']['std']
        print(f"Noise floor (sampled n={nf_n}): mean={nf_mean}, std={nf_std}")

        noise_floor = (nf_mean, nf_std)
    if len(clean_vals) > 1 and np.std(clean_vals) > 1e-9:
        if len(vlines)==3: plt.figure(figsize=(13, 6))     
        else: plt.figure(figsize=(7, 6))
        kde = gaussian_kde(clean_vals)
        if bw_threshold is not None:
            kde.set_bandwidth(bw_method=bw_threshold)

        # Use fixed centered range like Armstrong implementation
        x_min_centered = 0.35
        x_max_centered = 0.65
        x_grid = np.linspace(x_min_centered, x_max_centered, 500)
        y_grid = kde(x_grid)

        threshold = 0.5
        below_mask = x_grid <= threshold
        above_mask = x_grid > threshold
        below_count = sum(value < threshold for value in clean_vals)
        above_count = len(clean_vals) - below_count
        below_pct = 100.0 * below_count / len(clean_vals)
        above_pct = 100.0 * above_count / len(clean_vals)

        # Plot density line and fill below/above threshold with color-blind-friendly colors
        men_color = "#648FFF"
        women_color = "#FFB000"
        if below_mask.any():
            plt.plot(x_grid[below_mask], y_grid[below_mask], color=men_color, lw=2)
            plt.fill_between(x_grid[below_mask], y_grid[below_mask], color=men_color, alpha=0.5, label="Preference for Men\nAcross Settings")
        if above_mask.any():
            plt.plot(x_grid[above_mask], y_grid[above_mask], color=women_color, lw=2)
            plt.fill_between(x_grid[above_mask], y_grid[above_mask], color=women_color, alpha=0.3, label="Preference for Women\nAcross Settings")

        # Add vertical line at threshold
        plt.axvline(x=threshold, color="gray", linestyle="--", linewidth=2, alpha=0.9)

        # Add vertical lines if provided
        if vlines:
            for idx, (x_val, label, line_color) in enumerate(vlines):
                plt.axvline(
                    x=x_val,
                    color=line_color,
                    linestyle=REF_LINE_STYLES[idx % len(REF_LINE_STYLES)],
                    linewidth=2,
                    alpha=0.9,
                    label=label,
                )

        # Draw noise floor shaded region if provided (mean, std)
        if noise_floor is not None:
            try:
                nf_mean, nf_std = float(noise_floor[0]), float(noise_floor[1])
                nf_low = nf_mean - nf_std
                nf_high = nf_mean + nf_std
                plt.axvspan(nf_low, nf_high, color=REF_LINE_BROWN, alpha=0.18, label="Noise Floor")
            except Exception:
                pass

        ax = plt.gca()
        ax.set_xlim((0.35, 0.65))

        # Place annotations near top
        axis_y = 0.13
        left_x = 0.35
        right_x = 0.68
        fontsize = 25
        plt.text(left_x, axis_y, f"{below_pct:.0f}%", ha="left", va="top", transform=ax.transAxes, fontsize=fontsize)
        plt.text(right_x, axis_y, f"{above_pct:.0f}%", ha="right", va="top", transform=ax.transAxes, fontsize=fontsize)

        # plt.title('Yin et. al')
        if split_by != "all":
            if split_by.startswith("model: "):
                model_name = split_by.split(': ')[1].strip()
                plot_name = model_to_plotname.get(model_name, model_name)
                plt.title(f'{plot_name}')
            if split_by.startswith("job: "):
                job = split_by.split(': ')[1].strip()
                plt.title(f'{job}'.capitalize())
            if split_by.startswith("name: "):
                name = split_by.split(': ')[1].strip()
                plt.title(f'{name}'.capitalize())
        else: 
            plt.title(f'Yin et. al', pad=16)

        plt.xlabel(metric_to_plot.get(metric, metric))
        plt.ylabel("Probability Density")
        plt.grid(axis='y', alpha=0.3)
        if len(vlines)==3:
            
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        final_path = add_pval_suffix_to_path(out_path, p_value_threshold)
        if len(vlines)==3:
            final_path = final_path.replace('.png', '_with_reported_settings.png')
        plt.savefig(repo_path(final_path))
        print(f"Created plot: {final_path}")
        plt.close()
    else:
        # Fallback to histogram
        plt.figure(figsize=(10, 6))
        plt.hist(clean_vals, bins=100, alpha=0.6, color="purple", edgecolor='black')

        threshold = 0.5
        plt.axvline(x=threshold, color="gray", linestyle="--", linewidth=2, alpha=0.9)

        # Compute proportions for annotation
        total_count = len(clean_vals)
        if total_count > 0:
            below_count = sum(1 for v in clean_vals if v < threshold)
            above_count = sum(1 for v in clean_vals if v >= threshold)
            below_pct = 100.0 * below_count / total_count
            above_pct = 100.0 * above_count / total_count
            ax = plt.gca()
            ax.set_xlim((min(clean_vals), max(clean_vals)))
            plt.text(0.01, 0.95, f"Below 0.5: {below_pct:.1f}% ({below_count}/{total_count})", transform=ax.transAxes, fontsize=12, va='top')
            plt.text(0.99, 0.95, f"At or above 0.5: {above_pct:.1f}% ({above_count}/{total_count})", transform=ax.transAxes, fontsize=12, va='top', ha='right')

        # Add vertical lines if provided
        if vlines:
            for idx, (x_val, label, line_color) in enumerate(vlines):
                plt.axvline(
                    x=x_val,
                    color=line_color,
                    linestyle=REF_LINE_STYLES[idx % len(REF_LINE_STYLES)],
                    linewidth=2,
                    alpha=0.9,
                    label=label,
                )

        # Draw noise floor shaded region if provided (mean, std)
        if noise_floor is not None:
            try:
                nf_mean, nf_std = float(noise_floor[0]), float(noise_floor[1])
                nf_low = nf_mean - nf_std
                nf_high = nf_mean + nf_std
                plt.axvspan(nf_low, nf_high, color=REF_LINE_BROWN, alpha=0.18, label="Noise Floor")
            except Exception:
                pass

        plt.title(f'Distribution of {metric_to_plot.get(metric, metric)} (All Data)\nGender: W')
        plt.xlabel(metric_to_plot.get(metric, metric))
        plt.ylabel("Frequency")
        if vlines:
            plt.legend(loc='best')
        plt.tight_layout()
        hist_path = out_path.replace('density', 'histogram')
        final_path = add_pval_suffix_to_path(hist_path, p_value_threshold)
        plt.savefig(repo_path(final_path))
        print(f"Created plot: {final_path}")
        plt.close()


def plot_global_histogram(clean_vals: List[float], metric: str, out_path: str, yin_values: Optional[List[float]] = None, vlines: Optional[List[Tuple[float, str, str]]] = None, sig_values: Optional[List[Tuple[float, float]]] = None, p_value_threshold: Optional[float] = None) -> None:
    # Calculate the same limits as the density plot
    x_min, x_max = min(clean_vals), max(clean_vals)
    x_range = x_max - x_min
    limit_min = x_min - 0.1 * x_range
    limit_max = x_max + 0.1 * x_range

    plt.figure(figsize=(10, 6))
    plt.hist(clean_vals, bins=100, alpha=0.6, color="purple", edgecolor='black')
    
    # Add vertical lines if provided
    if vlines:
        threshold = 0.5 if metric == "selection_rate" else 0.0
        plt.axvline(x=threshold, color='gray', linestyle='--', linewidth=2, alpha=0.9)
        for idx, (x_val, label, line_color) in enumerate(vlines):
            plt.axvline(
                x=x_val,
                color=line_color,
                linestyle=REF_LINE_STYLES[idx % len(REF_LINE_STYLES)],
                linewidth=2,
                alpha=0.9,
                label=label,
            )
    
    # Plot significant values if provided
    if sig_values:
        for x_val, _ in sig_values:
            plt.axvline(x=x_val, color='green', linestyle=':', linewidth=1, alpha=0.5)
    
    # Explicitly set the x-axis limits
    plt.xlim(limit_min, limit_max)
    
    plt.title(f'Distribution of {metric} (All Data)\nGender: W')
    plt.xlabel(metric)
    plt.ylabel("Frequency")
    plt.grid(axis='y', alpha=0.3)
    if vlines:
        plt.legend(loc='best')
    plt.tight_layout()
    final_path = add_pval_suffix_to_path(out_path, p_value_threshold)
    plt.savefig(repo_path(final_path))
    print(f"Created plot: {final_path}")
    plt.close()


def plot_split_density_cdf(
    split_dict: Dict[str, List[float]], 
    metric: str, 
    split_type: str,
    out_root: str,
    bls_values: Optional[Dict[str, float]] = None,
    yin_values: Optional[List[float]] = None,
    yin_model: Optional[str] = None,
    vlines: Optional[List[Tuple[float, str, str]]] = None,
    model_vlines: Optional[Dict[str, float]] = None,
    sig_values_dict: Optional[Dict[str, List[Tuple[float, float]]]] = None,
    p_value_threshold: Optional[float] = None,
    bw_threshold: Optional[float] = None,
) -> None:
    """Plot density, CDF, and histogram plots split by model/job/name."""
    if not split_dict:
        print(f"No data to plot for {split_type} split.")
        return
    
    # Initialize figures
    fig_den, ax_den = plt.subplots(figsize=(12, 7))
    fig_cdf, ax_cdf = plt.subplots(figsize=(12, 7))
    fig_hist, ax_hist = plt.subplots(figsize=(12, 7))
    
    # Use colormap
    n_items = len(split_dict)
    if n_items <= 10:
        colors_list = plt.cm.tab10(np.linspace(0, 1, n_items))
    else:
        colors_list = plt.cm.tab20(np.linspace(0, 1, n_items))
    
    has_plotted = False
    
    # Determine global x range for histogram bins
    all_clean_vals = []
    for values in split_dict.values():
        clean_vals = clean_values(values)
        if len(clean_vals) > 0:
            all_clean_vals.extend(clean_vals)
    
    # Set histogram bins based on all data
    if all_clean_vals:
        hist_bins = np.linspace(min(all_clean_vals), max(all_clean_vals), 30)
    else:
        hist_bins = 30
    
    for idx, (key, values) in enumerate(split_dict.items()):
        clean_vals = clean_values(values)
        
        if len(clean_vals) > 0:
            # Determine label
            if bls_values and key in bls_values:
                label_text = f"{key} (BLS: {bls_values[key]})"
            else:
                label_text = str(key)
            
            # Plot Histogram
            ax_hist.hist(clean_vals, 
                        bins=hist_bins if isinstance(hist_bins, np.ndarray) else 30,
                        alpha=0.5, 
                        color=colors_list[idx], 
                        edgecolor='black',
                        label=label_text,
                        density=False)  # Use frequency, not density
            
            if len(clean_vals) > 1 and np.std(clean_vals) > 1e-9:
                # Plot Density
                try:
                    kde = gaussian_kde(clean_vals)
                    if bw_threshold is not None:
                        kde.set_bandwidth(bw_method=bw_threshold)
                    x_min, x_max = min(clean_vals), max(clean_vals)
                    x_range = x_max - x_min
                    if x_range == 0:
                        x_range = 1.0
                    x_grid = np.linspace(x_min - 0.2 * x_range, x_max + 0.2 * x_range, 200)
                    
                    ax_den.plot(x_grid, kde(x_grid), 
                               color=colors_list[idx], 
                               lw=2, 
                               alpha=0.8, 
                               label=label_text)
                except np.linalg.LinAlgError:
                    print(f"Skipping density for {key}: Singular matrix (no variance).")
                    # Still plot histogram and CDF even if density fails
                
                # Plot CDF
                sorted_data = np.sort(clean_vals)
                yvals = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
                
                ax_cdf.step(sorted_data, yvals, 
                           where='post', 
                           color=colors_list[idx], 
                           lw=2, 
                           alpha=0.8, 
                           label=label_text)
                
                has_plotted = True
            else:
                print(f"Skipping density/CDF for {key}: Not enough variance (but histogram will be plotted).")
                # Still plot histogram even if variance is low
                has_plotted = True
    
    if has_plotted:
        # Add vertical lines to all plots if provided
        if vlines:
            threshold = 0.5 if metric == "selection_rate" else 0.0
            ax_hist.axvline(x=threshold, color='gray', linestyle='--', linewidth=2, alpha=0.9)
            ax_den.axvline(x=threshold, color='gray', linestyle='--', linewidth=2, alpha=0.9)
            ax_cdf.axvline(x=threshold, color='gray', linestyle='--', linewidth=2, alpha=0.9)
            for idx, (x_val, label, line_color) in enumerate(vlines):
                line_style = REF_LINE_STYLES[idx % len(REF_LINE_STYLES)]
                ax_hist.axvline(x=x_val, color=line_color, linestyle=line_style, linewidth=2, alpha=0.9, label=label)
                ax_den.axvline(x=x_val, color=line_color, linestyle=line_style, linewidth=2, alpha=0.9, label=label)
                ax_cdf.axvline(x=x_val, color=line_color, linestyle=line_style, linewidth=2, alpha=0.9, label=label)
        
        # Add model-specific vertical lines for model split
        if split_type == "model" and model_vlines:
            for model_name, x_val in model_vlines.items():
                if model_name in split_dict:
                    ax_hist.axvline(x=x_val, color='blue', linestyle='--', linewidth=1.5, alpha=0.7, label=model_name)
                    ax_den.axvline(x=x_val, color='blue', linestyle='--', linewidth=1.5, alpha=0.7, label=model_name)
                    ax_cdf.axvline(x=x_val, color='blue', linestyle='--', linewidth=1.5, alpha=0.7, label=model_name)
        
        # Add significant values if provided (just the values, no extra text)
        if sig_values_dict:
            for key, sig_vals in sig_values_dict.items():
                if key in split_dict:
                    for x_val, _ in sig_vals:
                        ax_hist.axvline(x=x_val, color='green', linestyle=':', linewidth=1, alpha=0.5)
                        ax_den.axvline(x=x_val, color='green', linestyle=':', linewidth=1, alpha=0.5)
                        ax_cdf.axvline(x=x_val, color='green', linestyle=':', linewidth=1, alpha=0.5)
        
        # Save Histogram Plot
        ax_hist.set_title(f'Comparison of {metric} Histogram by {split_type.capitalize()}\nGender: W')
        ax_hist.set_xlabel(metric)
        ax_hist.set_ylabel("Frequency")
        ax_hist.grid(True, linestyle="--", alpha=0.3, axis='y')
        ax_hist.legend(title=split_type.capitalize(), bbox_to_anchor=(1.05, 1), loc='upper left')
        fig_hist.tight_layout()
        
        hist_path = f"{out_root}/{metric}_combined_histogram_by_{split_type}.png"
        final_hist_path = add_pval_suffix_to_path(hist_path, p_value_threshold)
        print(f"Saving combined histogram plot to {final_hist_path}")
        fig_hist.savefig(repo_path(final_hist_path))
        print(f"Created plot: {final_hist_path}")
        
        # Save Density Plot
        ax_den.set_title(f'Comparison of {metric} Density by {split_type.capitalize()}\nGender: W')
        ax_den.set_xlabel(metric)
        ax_den.set_ylabel("Probability Density")
        ax_den.grid(True, linestyle="--", alpha=0.3)
        ax_den.legend(title=split_type.capitalize(), bbox_to_anchor=(1.05, 1), loc='upper left')
        ax_den.set_xlim((0.35, 0.650))
        
        fig_den.tight_layout()
        
        den_path = f"{out_root}/{metric}_combined_density_by_{split_type}.png"
        final_den_path = add_pval_suffix_to_path(den_path, p_value_threshold)
        print(f"Saving combined density plot to {final_den_path}")
        fig_den.savefig(repo_path(final_den_path))
        print(f"Created plot: {final_den_path}")
        
        # Save CDF Plot
        ax_cdf.set_title(f'Comparison of {metric} CDF by {split_type.capitalize()}\nGender: W')
        ax_cdf.set_xlabel(metric)
        ax_cdf.set_ylabel("Cumulative Probability")
        ax_cdf.grid(True, linestyle="--", alpha=0.3)
        ax_cdf.legend(title=split_type.capitalize(), bbox_to_anchor=(1.05, 1), loc='upper left')
        fig_cdf.tight_layout()
        
        cdf_path = f"{out_root}/{metric}_combined_cdf_by_{split_type}.png"
        final_cdf_path = add_pval_suffix_to_path(cdf_path, p_value_threshold)
        print(f"Saving combined CDF plot to {final_cdf_path}")
        fig_cdf.savefig(repo_path(final_cdf_path))
        print(f"Created plot: {final_cdf_path}")
    else:
        print(f"No {split_type}s had enough data to plot.")
    
    plt.close(fig_den)
    plt.close(fig_cdf)
    plt.close(fig_hist)


def plot_violin_plot(
    split_dict: Dict[str, List[float]],
    metric: str,
    split_type: str,
    out_root: str,
    sig_values_dict: Optional[Dict[str, List[Tuple[float, float]]]] = None,
    p_value_threshold: Optional[float] = None,
) -> None:
    """Create and save a violin plot for the distributions in `split_dict`.

    - X axis: keys of `split_dict` (labels)
    - Y axis: values (distribution)
    """
    if not split_dict:
        print(f"No data to plot violin for {split_type}.")
        return

    labels = list(split_dict.keys())
    data = [clean_values(split_dict[k]) for k in labels]
    # Filter out empty groups
    filtered = [(l, d) for l, d in zip(labels, data) if d]
    if not filtered:
        print(f"No valid numeric groups to plot violin for {split_type}.")
        return

    labels, data = zip(*filtered)
    positions = np.arange(1, len(data) + 1)

    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 0.6), 6))

    cmap = plt.cm.get_cmap('tab20' if len(labels) > 10 else 'tab10')
    colors_list = [cmap(i % cmap.N) for i in range(len(labels))]

    parts = ax.violinplot(data, positions=positions, showmeans=False, showmedians=True, widths=0.8)

    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors_list[i])
        pc.set_edgecolor('black')
        pc.set_alpha(0.7)

    # Style for median lines and extrema
    if 'cmedians' in parts:
        parts['cmedians'].set_edgecolor('black')
        parts['cmedians'].set_linewidth(1.0)

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_xlabel(metric_to_plot.get(metric, metric))
    ax.set_ylabel('Value')
    ax.set_title(f'{metric_to_plot.get(metric, metric)} Violin Plot by {split_type.capitalize()}\nGender: W')
    ax.grid(axis='y', alpha=0.3)

    # Overlay significant points if provided
    if sig_values_dict:
        for i, label in enumerate(labels):
            sig_vals = sig_values_dict.get(label, [])
            for sv in sig_vals:
                # sv may be (x, value) or just a numeric value
                val = sv[0] if isinstance(sv, (list, tuple)) else sv
                jitter = np.random.uniform(-0.04, 0.04)
                ax.scatter(positions[i] + jitter, val, color='red', s=36, zorder=5, edgecolor='black')

    fig.tight_layout()
    save_path = os.path.join(repo_path(out_root), f"{metric}_violin_by_{split_type}.png")
    final_path = add_pval_suffix_to_path(save_path, p_value_threshold)
    print(f"Saving violin plot to {final_path}")
    fig.savefig(repo_path(final_path), dpi=150)
    print(f"Created plot: {final_path}")
    plt.close(fig)


def plot_boxwhisker_plot(
    split_dict: Dict[str, List[float]],
    metric: str,
    split_type: str,
    out_root: str,
    sig_values_dict: Optional[Dict[str, List[Tuple[float, float]]]] = None,
    p_value_threshold: Optional[float] = None,
) -> None:
    """Create and save a box-and-whisker plot for the distributions in `split_dict`.

    - X axis: keys of `split_dict` (labels)
    - Y axis: values (distribution)
    """
    if not split_dict:
        print(f"No data to plot boxwhisker for {split_type}.")
        return

    labels = list(split_dict.keys())
    # make the labels more readable my replacing the '-' with '\n'

    data = [clean_values(split_dict[k]) for k in labels]
    # Filter out empty groups
    filtered = [(l, d) for l, d in zip(labels, data) if d]
    if not filtered:
        print(f"No valid numeric groups to plot boxwhisker for {split_type}.")
        return

    labels, data = zip(*filtered)
    positions = np.arange(1, len(data) + 1)

    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 0.6), 6))

    box = ax.boxplot(data, positions=positions, widths=0.6, patch_artist=True, manage_ticks=False)

    cmap = plt.cm.get_cmap('tab20' if len(labels) > 10 else 'tab10')
    colors_list = [cmap(i % cmap.N) for i in range(len(labels))]

    for i, patch in enumerate(box['boxes']):
        patch.set_facecolor(colors_list[i])
        patch.set_edgecolor('black')
        patch.set_alpha(0.7)

    for whisker in box['whiskers']:
        whisker.set_color('black')

    for cap in box['caps']:
        cap.set_color('black')

    for median in box['medians']:
        median.set_color('black')
        median.set_linewidth(1.2)

    # add gridlines
    ax.grid(axis='y', alpha=0.3)

    ax.set_xticks(positions)
    if split_type == "model":
        ax.set_xticklabels([label.replace('-', '\n') for label in labels], rotation=0, ha='right')
    else: 
        ax.set_xticklabels([label.replace('-', '\n') for label in labels], rotation=45, ha='right')
    ax.set_xlabel(metric_to_plot.get(metric, metric))
    ax.set_ylabel('Value')
    ax.set_ylim(0.35,0.65)
    ax.set_title(f'Yin et. al split by {split_type.capitalize()}\nGender: W')
    ax.grid(axis='y', alpha=0.3)

    # Overlay significant points if provided
    if sig_values_dict:
        for i, label in enumerate(labels):
            sig_vals = sig_values_dict.get(label, [])
            for sv in sig_vals:
                val = sv[0] if isinstance(sv, (list, tuple)) else sv
                jitter = np.random.uniform(-0.04, 0.04)
                ax.scatter(positions[i] + jitter, val, color='red', s=36, zorder=5, edgecolor='black')

    fig.tight_layout()
    save_path = os.path.join(repo_path(out_root), f"{metric}_boxwhisker_by_{split_type}.png")
    final_path = add_pval_suffix_to_path(save_path, p_value_threshold)
    print(f"Saving box-and-whisker plot to {final_path}")
    fig.savefig(repo_path(final_path), dpi=150)
    print(f"Created plot: {final_path}")
    plt.close(fig)


def plot_split_density_cdf_with_bundle(
    split_dict: Dict[str, List[float]], 
    metric: str, 
    split_type: str,
    out_root: str,
    job_bundle: str,
    bls_values: Optional[Dict[str, float]] = None,
    yin_values: Optional[List[float]] = None,
    yin_model: Optional[str] = None,
    vlines: Optional[List[Tuple[float, str, str]]] = None,
    model_vlines: Optional[Dict[str, float]] = None,
    sig_values_dict: Optional[Dict[str, List[Tuple[float, float]]]] = None,
    p_value_threshold: Optional[float] = None,
    bw_threshold: Optional[float] = None,
) -> None:
    """Plot density, CDF, and histogram plots split by job_name within a job_bundle.
    
    Similar to plot_split_density_cdf but includes job_bundle name in filenames.
    """
    if not split_dict:
        print(f"No data to plot for {split_type} split in job_bundle {job_bundle}.")
        return
    
    # Initialize figures
    fig_den, ax_den = plt.subplots(figsize=(12, 7))
    fig_cdf, ax_cdf = plt.subplots(figsize=(12, 7))
    fig_hist, ax_hist = plt.subplots(figsize=(12, 7))
    
    # Use colormap
    n_items = len(split_dict)
    if n_items <= 10:
        colors_list = plt.cm.tab10(np.linspace(0, 1, n_items))
    else:
        colors_list = plt.cm.tab20(np.linspace(0, 1, n_items))
    
    has_plotted = False
    
    # Determine global x range for histogram bins
    all_clean_vals = []
    for values in split_dict.values():
        clean_vals = clean_values(values)
        if len(clean_vals) > 0:
            all_clean_vals.extend(clean_vals)
    
    # Set histogram bins based on all data
    if all_clean_vals:
        hist_bins = np.linspace(min(all_clean_vals), max(all_clean_vals), 30)
    else:
        hist_bins = 30
    
    for idx, (key, values) in enumerate(split_dict.items()):
        clean_vals = clean_values(values)
        
        if len(clean_vals) > 0:
            # Determine label
            if bls_values and key in bls_values:
                label_text = f"{key} (BLS: {bls_values[key]})"
            else:
                label_text = str(key)
            
            # Plot Histogram
            ax_hist.hist(clean_vals, 
                        bins=hist_bins if isinstance(hist_bins, np.ndarray) else 30,
                        alpha=0.5, 
                        color=colors_list[idx], 
                        edgecolor='black',
                        label=label_text,
                        density=False)  # Use frequency, not density
            
            if len(clean_vals) > 1 and np.std(clean_vals) > 1e-9:
                # Plot Density
                try:
                    kde = gaussian_kde(clean_vals)
                    if bw_threshold is not None:
                        kde.set_bandwidth(bw_method=bw_threshold)
                    x_min, x_max = min(clean_vals), max(clean_vals)
                    x_range = x_max - x_min
                    if x_range == 0:
                        x_range = 1.0
                    x_grid = np.linspace(x_min - 0.2 * x_range, x_max + 0.2 * x_range, 200)
                    
                    ax_den.plot(x_grid, kde(x_grid), 
                               color=colors_list[idx], 
                               lw=2, 
                               alpha=0.8, 
                               label=label_text)
                except np.linalg.LinAlgError:
                    print(f"Skipping density for {key}: Singular matrix (no variance).")
                    # Still plot histogram and CDF even if density fails
                
                # Plot CDF
                sorted_data = np.sort(clean_vals)
                yvals = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
                
                ax_cdf.step(sorted_data, yvals, 
                           where='post', 
                           color=colors_list[idx], 
                           lw=2, 
                           alpha=0.8, 
                           label=label_text)
                
                has_plotted = True
            else:
                print(f"Skipping density/CDF for {key}: Not enough variance (but histogram will be plotted).")
                # Still plot histogram even if variance is low
                has_plotted = True
    
    if has_plotted:
        # Add vertical lines to all plots if provided
        if vlines:
            threshold = 0.5 if metric == "selection_rate" else 0.0
            ax_hist.axvline(x=threshold, color='gray', linestyle='--', linewidth=2, alpha=0.9)
            ax_den.axvline(x=threshold, color='gray', linestyle='--', linewidth=2, alpha=0.9)
            ax_cdf.axvline(x=threshold, color='gray', linestyle='--', linewidth=2, alpha=0.9)
            for idx, (x_val, label, line_color) in enumerate(vlines):
                line_style = REF_LINE_STYLES[idx % len(REF_LINE_STYLES)]
                ax_hist.axvline(x=x_val, color=line_color, linestyle=line_style, linewidth=2, alpha=0.9, label=label)
                ax_den.axvline(x=x_val, color=line_color, linestyle=line_style, linewidth=2, alpha=0.9, label=label)
                ax_cdf.axvline(x=x_val, color=line_color, linestyle=line_style, linewidth=2, alpha=0.9, label=label)
        
        # Add model-specific vertical lines for model split
        if split_type == "model" and model_vlines:
            for model_name, x_val in model_vlines.items():
                if model_name in split_dict:
                    ax_hist.axvline(x=x_val, color='blue', linestyle='--', linewidth=1.5, alpha=0.7, label=model_name)
                    ax_den.axvline(x=x_val, color='blue', linestyle='--', linewidth=1.5, alpha=0.7, label=model_name)
                    ax_cdf.axvline(x=x_val, color='blue', linestyle='--', linewidth=1.5, alpha=0.7, label=model_name)
        
        # Add significant values if provided (just the values, no extra text)
        if sig_values_dict:
            for key, sig_vals in sig_values_dict.items():
                if key in split_dict:
                    for x_val, _ in sig_vals:
                        ax_hist.axvline(x=x_val, color='green', linestyle=':', linewidth=1, alpha=0.5)
                        ax_den.axvline(x=x_val, color='green', linestyle=':', linewidth=1, alpha=0.5)
                        ax_cdf.axvline(x=x_val, color='green', linestyle=':', linewidth=1, alpha=0.5)
        
        # Save Histogram Plot (with job_bundle in filename)
        ax_hist.set_title(f'Comparison of {metric} Histogram by {split_type.capitalize()}\nJob Bundle: {job_bundle}\nGender: W')
        ax_hist.set_xlabel(metric)
        ax_hist.set_ylabel("Frequency")
        ax_hist.grid(True, linestyle="--", alpha=0.3, axis='y')
        ax_hist.legend(title=split_type.capitalize(), bbox_to_anchor=(1.05, 1), loc='upper left')
        fig_hist.tight_layout()
        
        hist_path = f"{out_root}/{metric}_combined_histogram_by_{split_type}_job_bundle_{job_bundle}.png"
        final_hist_path = add_pval_suffix_to_path(hist_path, p_value_threshold)
        print(f"Saving combined histogram plot to {final_hist_path}")
        fig_hist.savefig(repo_path(final_hist_path))
        
        # Save Density Plot (with job_bundle in filename)
        ax_den.set_title(f'Comparison of {metric_to_plot[metric]} Density by {split_type.capitalize()}\nJob Bundle: {job_bundle}\nGender: W')
        ax_den.set_xlabel(metric_to_plot[metric])
        ax_den.set_ylabel("Probability Density")
        ax_den.grid(True, linestyle="--", alpha=0.3)
        ax_den.legend(title=split_type.capitalize(), bbox_to_anchor=(1.05, 1), loc='upper left')
        ax_den.set_xlim((0.35, 0.650))
        
        fig_den.tight_layout()
        
        den_path = f"{out_root}/{metric}_combined_density_by_{split_type}_job_bundle_{job_bundle}.png"
        final_den_path = add_pval_suffix_to_path(den_path, p_value_threshold)
        print(f"Saving combined density plot to {final_den_path}")
        fig_den.savefig(repo_path(final_den_path))
        
        # Save CDF Plot (with job_bundle in filename)
        ax_cdf.set_title(f'Comparison of {metric} CDF by {split_type.capitalize()}\nJob Bundle: {job_bundle}\nGender: W')
        ax_cdf.set_xlabel(metric)
        ax_cdf.set_ylabel("Cumulative Probability")
        ax_cdf.grid(True, linestyle="--", alpha=0.3)
        ax_cdf.legend(title=split_type.capitalize(), bbox_to_anchor=(1.05, 1), loc='upper left')
        fig_cdf.tight_layout()
        
        cdf_path = f"{out_root}/{metric}_combined_cdf_by_{split_type}_job_bundle_{job_bundle}.png"
        final_cdf_path = add_pval_suffix_to_path(cdf_path, p_value_threshold)
        print(f"Saving combined CDF plot to {final_cdf_path}")
        fig_cdf.savefig(repo_path(final_cdf_path))
    else:
        print(f"No {split_type}s had enough data to plot for job_bundle {job_bundle}.")
    
    plt.close(fig_den)
    plt.close(fig_cdf)
    plt.close(fig_hist)


# ----------------------------
# Main
# ----------------------------
def main() -> None:
    cwd = str(REPO_ROOT)
    default_results_dir = os.path.join(repo_path(cwd), "results")
    default_yin_input = os.path.join(repo_path(default_results_dir), "yin_sampling_downsampling_analysis_original.csv")
    default_bls_csv_path = os.path.join(repo_path(cwd), "misc", "Matched_Jobs_BLS_Final.csv")

    parser = argparse.ArgumentParser(description="Run plotting for yin data format with split options.")
    parser.add_argument("--input_file", type=str, default=default_yin_input, help="Path to input CSV.")
    parser.add_argument("--reference_input", help="Baseline aggregate CSV (defaults to input_file).")
    parser.add_argument("--output_dir", type=str, default="results/figs/yin/all", help="Directory to save figures.")
    parser.add_argument("--split_by", type=str, default="all", 
                       choices=["all", "model", "job", "job_name", "name", "job_bundle_job"], 
                       help="How to split the data: 'all' (no split), 'model', 'job' (by job_bundle), 'job_name' (by job), 'name', or 'job_bundle_job' (one plot per job_bundle, split by job name).")
    parser.add_argument("--metric", type=str, default="selection_rate",
                       help="Metric to plot: 'all', 'selection_rate', 'disparate_impact_ratio', or 'selection_rate'.")
    parser.add_argument("--p_value_threshold", type=float, default=None,
                       help="If provided, only plot values where P_Value_Vector < threshold. Filename will include threshold.")

    parser.add_argument("--bw_threshold", type=float, default=0.1,
                       help="If provided, only plot the density graph with the provided threshold for bandwidth selection.")

    parser.add_argument(
        "--color",
        type=str,
        default="purple",
        help="Base color for density plots (default: purple).",
    )

    parser.add_argument("--noise_floor_mean", type=float, default=None,
                       help="Mean of the noise floor to display as shaded region.")
    parser.add_argument("--noise_floor_std", type=float, default=None,
                       help="Standard deviation of the noise floor to display as shaded region.")
    parser.add_argument(
        "--no_reported_settings",
        action="store_true",
        help="Hide reported-setting references while retaining reproduced references and the noise floor.",
    )
    parser.add_argument(
        "--reported_settings_file",
        type=str,
        default=None,
        help="Optional legacy performance_ranking.csv override; otherwise use configs/published_references.csv.",
    )
    parser.add_argument(
        "--reported_setting_models",
        type=str,
        nargs="*",
        default=list(REPORTED_SETTING_MODELS),
        help="Model names used for reported setting lines (default: gpt-3.5-turbo gpt-4; pass no names for the generic reference).",
    )
    parser.add_argument(
        "--reported_setting_divisor",
        type=float,
        default=4.0,
        help="Divisor for the legacy --reported_settings_file only; published reference values are used directly.",
    )
                       
    args = parser.parse_args()

    # BLS values for job legend (optional)
    bls_values = {
        'armstrong': 57.564,
        'rozado': 47.47,
        'wen': 57.88,
        'wang': 28.71,
        'karvonen': 67.98,
        'zollo': 48.77,
        'yin': 46.5
    }

    # Load BLS job-to-value mapping from CSV
    bls_job_values = {}
    bls_csv_path = default_bls_csv_path
    if os.path.exists(repo_path(bls_csv_path)):
        bls_df = pd.read_csv(repo_path(bls_csv_path))
        # Map 'User Job' (lower case) to 'BLS Value'
        for _, row in bls_df.iterrows():
            job_name = str(row['User Job']).strip().lower()
            bls_value = float(row['BLS Value'])
            bls_job_values[job_name] = bls_value
        print(f"Loaded {len(bls_job_values)} job BLS mappings.")
    else:
        print(f"Warning: BLS CSV file not found at {bls_csv_path}")


    # 1. Load Data
    if not os.path.exists(repo_path(args.input_file)):
        print(f"Error: File {args.input_file} not found.")
        return

    # 1. Load the data
    #    (We use index_col=[0, 1] so it correctly identifies the M/W and ID columns first)
    df = pd.read_csv(repo_path(args.input_file), index_col=[0, 1])
    df = df[df['name_bundle'] != 'seshadri']

    # 2. Flatten the table
    #    (This moves the Index levels into the dataframe)
    df = df.reset_index()


    # 3. Rename columns - handle variable number of columns
    #    Check if we have the standard 10 columns or more (with selection_rate, P_Value_Vector, etc.)
    num_cols = len(df.columns)
    if num_cols >= 10:
        # Standard columns
        base_cols = [
            'gender', 
            'top', 
            'top_og', 
            'selection_rate', 
            'disparate_impact_ratio', 
            'job', 
            'model', 
            'rank', 
            'name_bundle', 
            'job_bundle'
        ]
        # Add additional columns if they exist
        if num_cols > 10:
            additional_cols = df.columns[10:].tolist()
            df.columns = base_cols + additional_cols
        else:
            df.columns = base_cols
    else:
        # If fewer columns, try to map what we have
        print(f"Warning: Expected at least 10 columns, found {num_cols}. Attempting to map columns...")
        if 'gender' not in df.columns and num_cols > 0:
            df.columns = list(df.columns[:num_cols])
    
    # Parse P_Value_Vector if it exists
    if 'P_Value_Vector' in df.columns:
        df['P_Value_Vector'] = df['P_Value_Vector'].apply(parse_p_values)


    # 4. Filter for 'W'
    #    (Now works easily because 'gender' is a real column)
    if args.split_by == "job_name" or args.split_by == 'job_bundle_job':

        # For job_name split, filter for gender values ending with '_W', 
        # and then group by job, model, name_bundle, job_bundle, summing selection_rate and averaging disparate_impact_ratio
        filtered_df = df[df['gender'].str.endswith("_W")].copy()
        if not filtered_df.empty:
            group_cols = ['job', 'model', 'name_bundle', 'job_bundle']
            agg_dict = {
                'selection_rate': 'sum',
                'disparate_impact_ratio': 'mean',
                'gender': 'first',   # keep first gender value in each group
                'top': 'sum',
                'top_og': 'sum',
                'rank': 'first'      # or 'mean', depending on how you want to handle ranks
            }
            # Only use columns that exist in the DataFrame
            agg_dict = {k: v for k, v in agg_dict.items() if k in filtered_df.columns}
            working_df = filtered_df.groupby(group_cols, as_index=False).agg(agg_dict)
        else:
            working_df = filtered_df  # will be empty DataFrame
    else:
        # Otherwise, filter for gender == 'W'
        working_df = df[df['gender'] == 'W'].copy()

    
    # De-duplicate rows using the standard Yin identity columns.
    yin_dedupe_columns = [
        column for column in ('gender', 'job', 'model', 'rank', 'name_bundle', 'job_bundle')
        if column in working_df.columns
    ]
    if yin_dedupe_columns:
        working_df = working_df.drop_duplicates(subset=yin_dedupe_columns, keep='first')


# Compute a reproducible sampled noise-floor early so downstream plotting
# uses the same sampled mean/std when the user did not explicitly supply them.
    try:
        # Collect candidate metric columns (common Yin metrics)
        candidate_vals = []
        for col in ("selection_rate", "disparate_impact_ratio"):
            if col in working_df.columns:
                candidate_vals.extend(working_df[col].dropna().tolist())

        clean_all = clean_values(candidate_vals)
        if clean_all:
            noise_floor_results = compute_default_framework_stats()
            nf_n, nf_mean, nf_std = noise_floor_results['yin']['n'], noise_floor_results['yin']['mean'], noise_floor_results['yin']['std']
            print(f"Noise floor (sampled n={nf_n}): mean={nf_mean}, std={nf_std}")

            args.noise_floor_mean = float(nf_mean)
            args.noise_floor_std = float(nf_std)
            print(f"Computed noise-floor mean/std from data sample: {args.noise_floor_mean:.4f}, {args.noise_floor_std:.4f}")
        else:
            args.noise_floor_mean = None
            args.noise_floor_std = None
    except Exception as exc:
        print(f"Warning: failed to compute sampled noise floor: {exc}")
        args.noise_floor_mean = None
        args.noise_floor_std = None

    # 3. Define Metrics
    if args.metric == "all":
        metrics_to_plot = ["selection_rate", "disparate_impact_ratio"]
    else:
        metrics_to_plot = [args.metric]
    
    # Define vertical reference lines for selection_rate
    metric_vlines = {}
    model_vlines = None

    out_root = repo_path(args.output_dir)

    if "selection_rate" in metrics_to_plot:
        if args.no_reported_settings:
            reported_model_vlines = []
        elif args.reported_setting_models and args.reported_settings_file:
            reported_model_vlines = load_reported_setting_vlines_from_performance_ranking(
                ranking_file=args.reported_settings_file,
                models=args.reported_setting_models,
                metric_col="selection_rate",
                divide_by=args.reported_setting_divisor,
            )
        else:
            reported_model_vlines = load_reported_setting_vlines_from_published_references(
                models=args.reported_setting_models,
            )
        metric_vlines["selection_rate"] = [
            (load_reproduced_reference("yin", "selection_rate", args.reference_input or args.input_file), REPRODUCED_SETTING_LABEL, REF_LINE_BROWN),
        ] + ([] if args.no_reported_settings else reported_model_vlines if reported_model_vlines else [
            (load_published_reference("yin", "selection_rate"), published_setting_label("yin", "selection_rate"), REF_LINE_BROWN)
        ])
    if "disparate_impact_ratio" in metrics_to_plot: 
        metric_vlines["disparate_impact_ratio"] = [
            (load_reproduced_reference("yin", "disparate_impact_ratio", args.reference_input or args.input_file), REPRODUCED_SETTING_LABEL, REF_LINE_BROWN),
        ] + ([] if args.no_reported_settings else [
            (load_published_reference("yin", "disparate_impact_ratio"), published_setting_label("yin", "disparate_impact_ratio"), REF_LINE_BROWN)
        ])
    
    ensure_dirs(args.output_dir)

    print(f"Plotting metrics: {metrics_to_plot} (Split by: {args.split_by})")

    # Extract yin values for each metric (where name_bundle=='yin' and job_bundle=='yin')
    yin_values_dict = {}
    yin_model_dict = {}
    for metric in metrics_to_plot:
        if metric in working_df.columns:
            yin_subset = working_df[(working_df['name_bundle'] == 'yin') & (working_df['job_bundle'] == 'yin')]
            yin_vals = yin_subset[metric].dropna().tolist()
            yin_values_dict[metric] = clean_values(yin_vals) if yin_vals else []
            # Get the model name(s) for yin data (use the most common one if multiple)
            if not yin_subset.empty and 'model' in yin_subset.columns:
                yin_models = yin_subset['model'].value_counts()
                if len(yin_models) > 0:
                    yin_model_dict[metric] = yin_models.index[0]  # Use most common model
            if yin_values_dict[metric]:
                print(f"Found {len(yin_values_dict[metric])} yin values for {metric}: {yin_values_dict[metric]}")
                if metric in yin_model_dict:
                    print(f"Yin data belongs to model: {yin_model_dict[metric]}")
            else:
                print(f"No yin values found for {metric}")

    # 4. Loop over Metrics
    for metric in metrics_to_plot:
        vlines = metric_vlines.get(metric)
        if metric not in working_df.columns:
            print(f"Warning: Metric '{metric}' not found in data. Skipping.")
            continue

        # ----------------------------
        # Main plotting logic based on split_by
        # ----------------------------
        if args.split_by == "all":
            # Collect all values across all models
            print(f"Collecting all values for {metric}...")
            
            # Check if working_df is empty after filtering
            if working_df.empty:
                print(f"Warning: No data found after filtering (gender == 'W'). Skipping {metric}.")
                continue
            
            # Get the metric column and check for valid values
            metric_series = working_df[metric]
            non_null_count = metric_series.notna().sum()
            print(f"Found {non_null_count} non-null values out of {len(metric_series)} total rows for {metric}")
            
            # Filter by p-value threshold if provided
            all_collected_values = []
            sig_values = []  # List of (x_val, value) tuples for significant p-values
            
            if args.p_value_threshold is not None and 'P_Value_Vector' in working_df.columns:
                # Collect (value, p) rows, then apply Benjamini-Hochberg across the whole set
                rows: List[Tuple[float, Optional[float]]] = []
                pvals: List[float] = []
                p_index_map: Dict[int, int] = {}
                for i, (_, row) in enumerate(working_df.iterrows()):
                    metric_val = row[metric]
                    if pd.notna(metric_val):
                        p_vals = row.get('P_Value_Vector', None)
                        p_val = None
                        if p_vals is not None and isinstance(p_vals, list) and len(p_vals) > 0:
                            try:
                                p_val = float(p_vals[0])
                            except Exception:
                                p_val = None
                        rows.append((metric_val, p_val))
                        if p_val is not None:
                            p_index_map[len(rows) - 1] = len(pvals)
                            pvals.append(p_val)

                accepted_vals: List[float] = []
                accepted_sig_values: List[Tuple[float, float]] = []
                if pvals:
                    rejected = benjamini_hochberg(pvals, args.p_value_threshold)
                    for rowpos, (val, p) in enumerate(rows):
                        if p is None:
                            accepted_vals.append(val)
                            accepted_sig_values.append((val, val))
                        else:
                            pidx = p_index_map.get(rowpos)
                            if pidx is not None and not rejected[pidx]:
                                accepted_vals.append(val)
                                accepted_sig_values.append((val, val))
                else:
                    # no p-values present -> accept all
                    for val, _ in rows:
                        accepted_vals.append(val)

                all_collected_values = accepted_vals
                sig_values = accepted_sig_values
            else:
                # No p-value filtering, collect all values
                all_collected_values = metric_series.dropna().values.tolist()
            
            # Clean the values to ensure they're numeric and handle any edge cases
            clean_vals = clean_values(all_collected_values)
            clean_sig_vals = clean_values([v[0] for v in sig_values]) if sig_values else []
            clean_sig_values = [(v, v) for v in clean_sig_vals] if clean_sig_vals else None

            if clean_vals:
                yin_vals = yin_values_dict.get(metric, [])

                plot_global_density(
                    clean_vals, 
                    metric, 
                    f"{args.output_dir}/{metric}_global_density_all.png",
                    split_by=args.split_by,
                    vlines=vlines,
                    sig_values=clean_sig_values,
                    p_value_threshold=args.p_value_threshold,
                    bw_threshold=args.bw_threshold,
                    color=getattr(args, "color", "purple"),
                )
                # Save a version highlighting the noise floor
                nf_arg = (args.noise_floor_mean, args.noise_floor_std) if (args.noise_floor_mean is not None and args.noise_floor_std is not None) else None
                plot_global_density(
                    clean_vals,
                    metric,
                    f"{args.output_dir}/{metric}_global_density_all_with_noise_floor.png",
                    split_by=args.split_by,
                    vlines=vlines,
                    sig_values=clean_sig_values,
                    p_value_threshold=args.p_value_threshold,
                    bw_threshold=args.bw_threshold,
                    color=getattr(args, "color", "purple"),
                    noise_floor=nf_arg,
                )
            else:
                print(f"Warning: No valid numeric values found for metric {metric} after cleaning.")

        elif args.split_by == "model":
            # Split by model
            print(f"Generating plots split by model for {metric}...")
            if "model" not in working_df.columns:
                print("Error: 'model' column not found in data.")
                continue
            
            unique_models = working_df["model"].unique()
            split_dict = {}
            sig_values_dict = {}  # Dict mapping model_name -> list of (x_val, value) tuples
            
            for model_name in unique_models:
                model_subset = working_df[working_df["model"] == model_name]
                model_values = []
                model_sig_values = []
                
                # Collect values and p-values for this model then apply BH if requested
                rows: List[Tuple[float, Optional[float]]] = []
                pvals: List[float] = []
                p_index_map: Dict[int, int] = {}
                for i, (_, row) in enumerate(model_subset.iterrows()):
                    metric_val = row[metric]
                    if pd.notna(metric_val):
                        p_vals = row.get('P_Value_Vector', None)
                        p_val = None
                        if p_vals is not None and isinstance(p_vals, list) and len(p_vals) > 0:
                            try:
                                p_val = float(p_vals[0])
                            except Exception:
                                p_val = None
                        rows.append((metric_val, p_val))
                        if p_val is not None:
                            p_index_map[len(rows) - 1] = len(pvals)
                            pvals.append(p_val)

                if args.p_value_threshold is not None and 'P_Value_Vector' in model_subset.columns:
                    model_values = []
                    model_sig_values = []
                    if pvals:
                        rejected = benjamini_hochberg(pvals, args.p_value_threshold)
                        for rowpos, (val, p) in enumerate(rows):
                            if p is None:
                                model_values.append(val)
                                model_sig_values.append((val, val))
                            else:
                                pidx = p_index_map.get(rowpos)
                                if pidx is not None and not rejected[pidx]:
                                    model_values.append(val)
                                    model_sig_values.append((val, val))
                    else:
                        # no p-values -> include all
                        model_values = [val for (val, _) in rows]
                else:
                    model_values = [val for (val, _) in rows]
                
                if model_values:
                    split_dict[model_name] = model_values
                    if model_sig_values:
                        sig_values_dict[model_name] = model_sig_values

                    

                    os.makedirs(repo_path(f"{out_root}/{args.split_by}/{model_name}"), exist_ok=True)
                    if args.p_value_threshold is not None:
                        all_collected_values = model_sig_values.copy()
                    else:
                        all_collected_values = model_values.copy()
                    if args.p_value_threshold is not None: 
                        plot_global_density(
                            all_collected_values, 
                            metric, 
                            f"{out_root}/{args.split_by}/{model_name}/{metric}_combined_density_by_{args.split_by}.png",
                            split_by =f'model: {model_name}',
                            vlines=vlines,
                            sig_values=model_sig_values if args.p_value_threshold is not None else None,
                            p_value_threshold=args.p_value_threshold,
                            bw_threshold=args.bw_threshold
                        )
                    else: 
                        plot_global_density(
                            all_collected_values, 
                            metric, 
                            f"{out_root}/{args.split_by}/{model_name}/{metric}_combined_density_by_{args.split_by}.png",
                            split_by =f'model: {model_name}',
                            vlines=vlines,
                            sig_values=model_sig_values if args.p_value_threshold is not None else None,
                            p_value_threshold=args.p_value_threshold,
                            bw_threshold=args.bw_threshold
                        )
            
            yin_vals = yin_values_dict.get(metric, [])
            yin_model = yin_model_dict.get(metric, None)

            clean_sig_values_dict = {}
            for key, sig_vals in sig_values_dict.items():
                clean_vals_list = clean_values([v[0] for v in sig_vals])
                clean_sig_values_dict[key] = [(v, v) for v in clean_vals_list] if clean_vals_list else []
            plot_split_density_cdf(
                split_dict, metric, "model", args.output_dir, 
                yin_values=yin_vals, yin_model=yin_model,
                vlines=vlines, model_vlines=model_vlines,
                sig_values_dict=clean_sig_values_dict if clean_sig_values_dict else None,
                p_value_threshold=args.p_value_threshold,
                bw_threshold=args.bw_threshold,
            )

            # plot_violin_plot(
            #     split_dict,
            #     metric,
            #     args.split_by,
            #     args.output_dir,
            #     sig_values_dict=clean_sig_values_dict if clean_sig_values_dict else None,
            #     p_value_threshold=args.p_value_threshold,
            # )
            plot_boxwhisker_plot(
                split_dict,
                metric,
                args.split_by,
                args.output_dir,
                sig_values_dict=clean_sig_values_dict if clean_sig_values_dict else None,
                p_value_threshold=args.p_value_threshold,
            )   





        elif args.split_by == "job":
            # Split by job (use job_bundle if available, otherwise job)
            print(f"Generating plots split by job for {metric}...")
            job_col = "job_bundle" if "job_bundle" in working_df.columns else "job"
            if job_col not in working_df.columns:
                print(f"Error: '{job_col}' column not found in data.")
                continue
            
            unique_jobs = working_df[job_col].unique()
            split_dict = {}
            sig_values_dict = {}
            
            for job_val in unique_jobs:
                job_subset = working_df[working_df[job_col] == job_val]
                job_values = []
                job_sig_values = []
                
                # Collect values and p-values for this job then apply BH if requested
                rows: List[Tuple[float, Optional[float]]] = []
                pvals: List[float] = []
                p_index_map: Dict[int, int] = {}
                for i, (_, row) in enumerate(job_subset.iterrows()):
                    metric_val = row[metric]
                    if pd.notna(metric_val):
                        p_vals = row.get('P_Value_Vector', None)
                        p_val = None
                        if p_vals is not None and isinstance(p_vals, list) and len(p_vals) > 0:
                            try:
                                p_val = float(p_vals[0])
                            except Exception:
                                p_val = None
                        rows.append((metric_val, p_val))
                        if p_val is not None:
                            p_index_map[len(rows) - 1] = len(pvals)
                            pvals.append(p_val)

                if args.p_value_threshold is not None and 'P_Value_Vector' in job_subset.columns:
                    job_values = []
                    job_sig_values = []
                    if pvals:
                        rejected = benjamini_hochberg(pvals, args.p_value_threshold)
                        for rowpos, (val, p) in enumerate(rows):
                            if p is None:
                                job_values.append(val)
                                job_sig_values.append((val, val))
                            else:
                                pidx = p_index_map.get(rowpos)
                                if pidx is not None and not rejected[pidx]:
                                    job_values.append(val)
                                    job_sig_values.append((val, val))
                    else:
                        job_values = [val for (val, _) in rows]
                else:
                    job_values = [val for (val, _) in rows]
                
                if job_values:
                    split_dict[job_val] = job_values
                    if job_sig_values:
                        sig_values_dict[job_val] = job_sig_values

                    os.makedirs(repo_path(f"{out_root}/{args.split_by}/{job_val}"), exist_ok=True)
                    if args.p_value_threshold is not None:
                        all_collected_values = job_sig_values.copy()
                    else:
                        all_collected_values = job_values.copy()
                    if args.p_value_threshold is not None: 
                        plot_global_density(
                            all_collected_values, 
                            metric, 
                            f"{out_root}/{args.split_by}/{job_val}/{metric}_combined_density_by_{args.split_by}.png",
                            split_by =f'job: {job_val}',
                            vlines=vlines,
                            sig_values=job_sig_values if args.p_value_threshold is not None else None,
                            p_value_threshold=args.p_value_threshold,
                            bw_threshold=args.bw_threshold
                        )
                    else: 
                        plot_global_density(
                            all_collected_values, 
                            metric, 
                            f"{out_root}/{args.split_by}/{job_val}/{metric}_combined_density_by_{args.split_by}.png",
                            split_by =f'job: {job_val}',
                            vlines=vlines,
                            sig_values=job_sig_values if args.p_value_threshold is not None else None,
                            p_value_threshold=args.p_value_threshold,
                            bw_threshold=args.bw_threshold
                        )

                
            
            yin_vals = yin_values_dict.get(metric, [])
            yin_model = yin_model_dict.get(metric, None)
            clean_sig_values_dict = {}
            for key, sig_vals in sig_values_dict.items():
                clean_vals_list = clean_values([v[0] for v in sig_vals])
                clean_sig_values_dict[key] = [(v, v) for v in clean_vals_list] if clean_vals_list else []
            plot_split_density_cdf(
                split_dict, metric, "job", args.output_dir, bls_values, yin_vals, yin_model,
                vlines=vlines, sig_values_dict=clean_sig_values_dict if clean_sig_values_dict else None,
                p_value_threshold=args.p_value_threshold,
                bw_threshold=args.bw_threshold,
            )

            # plot_violin_plot(
            #     split_dict,
            #     metric,
            #     args.split_by,
            #     args.output_dir,
            #     sig_values_dict=clean_sig_values_dict if clean_sig_values_dict else None,
            #     p_value_threshold=args.p_value_threshold,
            # )
            plot_boxwhisker_plot(
                split_dict,
                metric,
                args.split_by,
                args.output_dir,
                sig_values_dict=clean_sig_values_dict if clean_sig_values_dict else None,
                p_value_threshold=args.p_value_threshold,
            )   

        elif args.split_by == "job_name":
            # Split by job_name (actual job names like "cashier", "administrative assistant", etc.)
            print(f"Generating plots split by job_name for {metric}...")
            if "job" not in working_df.columns:
                print("Error: 'job' column not found in data.")
                continue
            
            # Clean job names: strip whitespace and lowercase
            unique_job_names = working_df["job"].str.strip().str.lower().unique()
            split_dict = {}
            sig_values_dict = {}

            for job_name in unique_job_names:
                job_subset = working_df[working_df["job"].str.strip().str.lower() == job_name]
                job_values = []
                job_sig_values = []
                
                # Collect values and p-values for this job_name then apply BH if requested
                rows: List[Tuple[float, Optional[float]]] = []
                pvals: List[float] = []
                p_index_map: Dict[int, int] = {}
                for i, (_, row) in enumerate(job_subset.iterrows()):
                    metric_val = row[metric]
                    if pd.notna(metric_val):
                        p_vals = row.get('P_Value_Vector', None)
                        p_val = None
                        if p_vals is not None and isinstance(p_vals, list) and len(p_vals) > 0:
                            try:
                                p_val = float(p_vals[0])
                            except Exception:
                                p_val = None
                        rows.append((metric_val, p_val))
                        if p_val is not None:
                            p_index_map[len(rows) - 1] = len(pvals)
                            pvals.append(p_val)

                if args.p_value_threshold is not None and 'P_Value_Vector' in job_subset.columns:
                    job_values = []
                    job_sig_values = []
                    if pvals:
                        rejected = benjamini_hochberg(pvals, args.p_value_threshold)
                        for rowpos, (val, p) in enumerate(rows):
                            if p is None:
                                job_values.append(val)
                                job_sig_values.append((val, val))
                            else:
                                pidx = p_index_map.get(rowpos)
                                if pidx is not None and not rejected[pidx]:
                                    job_values.append(val)
                                    job_sig_values.append((val, val))
                    else:
                        job_values = [val for (val, _) in rows]
                else:
                    job_values = [val for (val, _) in rows]
                
                if job_values:
                    split_dict[job_name] = job_values
                    if job_sig_values:
                        sig_values_dict[job_name] = job_sig_values
            
            yin_vals = yin_values_dict.get(metric, [])
            yin_model = yin_model_dict.get(metric, None)
            clean_sig_values_dict = {}
            for key, sig_vals in sig_values_dict.items():
                clean_vals_list = clean_values([v[0] for v in sig_vals])
                clean_sig_values_dict[key] = [(v, v) for v in clean_vals_list] if clean_vals_list else []
            plot_split_density_cdf(
                split_dict, metric, "job_name", args.output_dir, yin_values=yin_vals, yin_model=yin_model,
                vlines=vlines, sig_values_dict=clean_sig_values_dict if clean_sig_values_dict else None,
                p_value_threshold=args.p_value_threshold,
                bw_threshold=args.bw_threshold,
            )

            

        elif args.split_by == "name":
            # Split by name
            print(f"Generating plots split by name for {metric}...")
            if "name_bundle" not in working_df.columns:
                print("Error: 'name_bundle' column not found in data.")
                continue
            
            unique_names = working_df["name_bundle"].unique()
            split_dict = {}
            sig_values_dict = {}
            
            for name_val in unique_names:
                name_subset = working_df[working_df["name_bundle"] == name_val]
                name_values = []
                name_sig_values = []
                
                for idx, row in name_subset.iterrows():
                    metric_val = row[metric]
                    if pd.notna(metric_val):
                        # Check p-value if threshold is set
                        if args.p_value_threshold is not None and 'P_Value_Vector' in row:
                            p_vals = row.get('P_Value_Vector', None)
                            is_sig = False
                            if p_vals is not None and isinstance(p_vals, list) and len(p_vals) > 0:
                                p_val = p_vals[0]
                                if pd.notna(p_val) and p_val < args.p_value_threshold:
                                    is_sig = True
                            
                            if is_sig:
                                name_values.append(metric_val)
                                name_sig_values.append((metric_val, metric_val))
                        else:
                            # No p-value filtering - include all values
                            name_values.append(metric_val)
                
                if name_values:
                    split_dict[name_val] = name_values
                    if name_sig_values:
                        sig_values_dict[name_val] = name_sig_values


                    os.makedirs(repo_path(f"{out_root}/{args.split_by}/{name_val}"), exist_ok=True)
                    if args.p_value_threshold is not None:
                        all_collected_values = name_sig_values.copy()
                    else:
                        all_collected_values = name_values.copy()
                    if args.p_value_threshold is not None: 
                        plot_global_density(
                            all_collected_values, 
                            metric, 
                            f"{out_root}/{args.split_by}/{name_val}/{metric}_combined_density_by_{args.split_by}.png",
                            split_by =f'name: {name_val}',
                            vlines=vlines,
                            sig_values=name_sig_values if args.p_value_threshold is not None else None,
                            p_value_threshold=args.p_value_threshold,
                            bw_threshold=args.bw_threshold
                        )
                    else: 
                        plot_global_density(
                            all_collected_values, 
                            metric, 
                            f"{out_root}/{args.split_by}/{name_val}/{metric}_combined_density_by_{args.split_by}.png",
                            split_by =f'name: {name_val}',
                            vlines=vlines,
                            sig_values=name_sig_values if args.p_value_threshold is not None else None,
                            p_value_threshold=args.p_value_threshold,
                            bw_threshold=args.bw_threshold
                        )
            
            yin_vals = yin_values_dict.get(metric, [])
            yin_model = yin_model_dict.get(metric, None)
            clean_sig_values_dict = {}
            for key, sig_vals in sig_values_dict.items():
                clean_vals_list = clean_values([v[0] for v in sig_vals])
                clean_sig_values_dict[key] = [(v, v) for v in clean_vals_list] if clean_vals_list else []
            plot_split_density_cdf(
                split_dict, metric, "name", args.output_dir, yin_values=yin_vals, yin_model=yin_model,
                vlines=vlines, sig_values_dict=clean_sig_values_dict if clean_sig_values_dict else None,
                p_value_threshold=args.p_value_threshold,
                bw_threshold=args.bw_threshold,
            )

            # plot_violin_plot(
            #     split_dict,
            #     metric,
            #     args.split_by,
            #     args.output_dir,
            #     sig_values_dict=clean_sig_values_dict if clean_sig_values_dict else None,
            #     p_value_threshold=args.p_value_threshold,
            # )
            plot_boxwhisker_plot(
                split_dict,
                metric,
                args.split_by,
                args.output_dir,
                sig_values_dict=clean_sig_values_dict if clean_sig_values_dict else None,
                p_value_threshold=args.p_value_threshold,
            )   

        elif args.split_by == "job_bundle_job":
            # Split by job_bundle, then by job name within each bundle
            # Creates one plot per job_bundle, with each plot showing different job names
            print(f"Generating plots split by job_bundle and job_name for {metric}...")
            if "job_bundle" not in working_df.columns:
                print("Error: 'job_bundle' column not found in data.")
                continue
            if "job" not in working_df.columns:
                print("Error: 'job' column not found in data.")
                continue
            
            unique_job_bundles = working_df["job_bundle"].unique()
            print(f"Found {len(unique_job_bundles)} unique job bundles: {unique_job_bundles}")
            
            # Track all job names and their values across all bundles for lollipop chart
            all_job_values = {}  # Dict mapping job_name -> list of metric values
            all_job_values_models = defaultdict(dict)  # Dict mapping job_name -> {model name: mean} (most common)
            
            for job_bundle in unique_job_bundles:
                print(f"\nProcessing job_bundle: '{job_bundle}'...")
                # Filter for this job bundle
                bundle_subset = working_df[working_df["job_bundle"] == job_bundle].copy()
                
                if bundle_subset.empty:
                    print(f"No data found for job_bundle '{job_bundle}'. Skipping.")
                    continue
                
                print(f"Found {len(bundle_subset)} rows for job_bundle '{job_bundle}'")
                
                # Get unique job names within this bundle (clean them)
                unique_job_names = bundle_subset["job"].str.strip().str.lower().unique()
                print(f"Found {len(unique_job_names)} unique job names in bundle '{job_bundle}': {unique_job_names}")
                split_dict = {}
                sig_values_dict = {}
                
                for job_name in unique_job_names:
                    job_subset = bundle_subset[bundle_subset["job"].str.strip().str.lower() == job_name]
                    job_values = []
                    job_sig_values = []
                    
                    for idx, row in job_subset.iterrows():
                        metric_val = row[metric]
                        if pd.notna(metric_val):
                            # Check p-value if threshold is set
                            if args.p_value_threshold is not None and 'P_Value_Vector' in row:
                                p_vals = row.get('P_Value_Vector', None)
                                is_sig = False
                                if p_vals is not None and isinstance(p_vals, list) and len(p_vals) > 0:
                                    p_val = p_vals[0]
                                    if pd.notna(p_val) and p_val < args.p_value_threshold:
                                        is_sig = True
                                
                                if is_sig:
                                    job_values.append(metric_val)
                                    job_sig_values.append((metric_val, metric_val))
                            else:
                                # No p-value filtering - include all values
                                job_values.append(metric_val)
                    
                    if job_values:
                        split_dict[job_name] = job_values
                        if job_sig_values:
                            sig_values_dict[job_name] = job_sig_values
                    
                    # Track job values across all bundles for lollipop chart
                    if job_name not in all_job_values:
                        all_job_values[job_name] = []
                    all_job_values[job_name].extend(job_values)

                    for model in job_subset['model'].unique():
                        model_subset = job_subset[job_subset['model'] == model]
                        model_metric_vals = model_subset[metric].dropna().tolist()
                        clean_model_vals = clean_values(model_metric_vals)
                        if clean_model_vals:
                            mean_val = np.mean(clean_model_vals)
                            all_job_values_models[job_name][model] = mean_val

                if split_dict:
                    print(f"Creating plots for job_bundle '{job_bundle}' with {len(split_dict)} job names...")
                    # Create output directory for this job bundle
                    bundle_output_dir = os.path.join(repo_path(args.output_dir), f"job_bundle_{job_bundle}")
                    ensure_dirs(bundle_output_dir)
                    print(f"Output directory: {bundle_output_dir}")
                    
                    yin_vals = yin_values_dict.get(metric, [])
                    yin_model = yin_model_dict.get(metric, None)
                    clean_sig_values_dict = {}
                    for key, sig_vals in sig_values_dict.items():
                        clean_vals_list = clean_values([v[0] for v in sig_vals])
                        clean_sig_values_dict[key] = [(v, v) for v in clean_vals_list] if clean_vals_list else []
                    
                    # Use a modified version of plot_split_density_cdf that includes job_bundle in filename
                    plot_split_density_cdf_with_bundle(
                        split_dict, metric, "job_name", bundle_output_dir, 
                        job_bundle=job_bundle,
                        yin_values=yin_vals, yin_model=yin_model,
                        vlines=vlines, sig_values_dict=clean_sig_values_dict if clean_sig_values_dict else None,
                        p_value_threshold=args.p_value_threshold,
                        bw_threshold=args.bw_threshold,
                    )
                    print(f"Completed plots for job_bundle '{job_bundle}'.\n")
                else:
                    print(f"No data to plot for job_bundle '{job_bundle}'.")
            
            # Create lollipop chart for all job names across all bundles
            if all_job_values:
                print(f"\nCreating lollipop chart for {metric} across all job bundles...")
                
                # Calculate average value per job name
                job_averages = {}
                for job_name, values in all_job_values.items():
                    clean_vals = clean_values(values)
                    if clean_vals:
                        job_averages[job_name] = np.mean(clean_vals)
                
                if job_averages:
                    # Sort by average value for better visualization
                    sorted_jobs = sorted(job_averages.items(), key=lambda x: x[1])
                    job_names = [j[0] for j in sorted_jobs]
                    job_avgs = [j[1] for j in sorted_jobs]

                    # Build a list of all models encountered for consistent coloring
                    all_models = set()
                    for jm in all_job_values_models.values():
                        all_models.update(jm.keys())
                    all_models = sorted(all_models)

                    # Choose a colormap and map each model to a color
                    if len(all_models) <= 10:
                        cmap = plt.cm.get_cmap('tab10')
                    else:
                        cmap = plt.cm.get_cmap('tab20')
                    model_colors = {m: cmap(i % cmap.N) for i, m in enumerate(all_models)}

                    # Create lollipop chart centered at 0.5
                    fig, ax = plt.subplots(figsize=(12, max(6, len(job_names) * 0.35)))

                    # Create baseline lines from 0.5 to each job average (use darkblue/purple as before)
                    y_pos = np.arange(len(job_names))
                    baseline_colors = ['darkblue' if avg < 0.5 else 'purple' for avg in job_avgs]
                    ax.hlines(y_pos, 0.5, job_avgs, colors=baseline_colors, linewidth=2.5)

                    # Plot one small colored dot per model for each job name (no jitter)
                    legend_handles = {}
                    for i, job_name in enumerate(job_names):
                        models_dict = all_job_values_models.get(job_name, {})
                        if not models_dict:
                            # Fallback: plot the job average as a larger marker with baseline color
                            ax.scatter(job_avgs[i], i, c=[baseline_colors[i]], s=100, zorder=4, edgecolors='black', linewidth=1.2)
                            continue

                        for model_name, mean_val in models_dict.items():
                            col = model_colors.get(model_name, (0.5, 0.5, 0.5))
                            sc = ax.scatter(mean_val, i, color=[col], s=40, zorder=4, edgecolors='black', linewidth=0.6)
                            if model_name not in legend_handles:
                                legend_handles[model_name] = sc

                    # Add vertical line at 0.5 (threshold)
                    ax.axvline(x=0.5, color='black', linestyle='--', linewidth=1.5, alpha=0.7)

                    ax.set_yticks(y_pos)
                    ax.set_yticklabels(job_names, fontsize=11)
                    ax.set_xlabel(f'{metric_to_plot.get(metric, metric)}', fontsize=12, fontweight='bold')
                    ax.set_ylabel('Job Name', fontsize=12, fontweight='bold')
                    ax.set_title(f'Average {metric_to_plot.get(metric, metric)} by Job Name\n(Across All Job Bundles, Gender: W)', fontsize=13, fontweight='bold')
                    ax.grid(True, axis='x', alpha=0.3, linestyle='--')

                    # Set x-axis limits to show good range around 0.5
                    min_val = min(job_avgs)
                    max_val = max(job_avgs)
                    padding = 0.02
                    ax.set_xlim(left=min_val - padding, right=max_val + padding)

                    # Add legend for models (if any)
                    if legend_handles:
                        ax.legend(legend_handles.values(), legend_handles.keys(), title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')

                    plt.tight_layout()
                    lollipop_path = os.path.join(repo_path(args.output_dir), f"{metric}_lollipop_all_job_bundles.png")
                    lollipop_path = add_pval_suffix_to_path(lollipop_path, args.p_value_threshold)
                    print(f"Saving lollipop chart to {lollipop_path}")
                    fig.savefig(repo_path(lollipop_path), dpi=150)
                    print(f"Created plot: {lollipop_path}")
                    plt.close(fig)

                    # Create scatter plot: BLS value (x-axis) vs. average metric value (y-axis)
                    print(f"\nCreating scatter plot: BLS value vs. average {metric}...")
                    scatter_bls_vals = []
                    scatter_metric_vals = []
                    scatter_job_labels = []

                    for job_name, avg_val in job_averages.items():
                        # Look up BLS value for this job
                        bls_val = bls_job_values.get(job_name)
                        if bls_val is not None:
                            scatter_bls_vals.append(bls_val)
                            scatter_metric_vals.append(avg_val)
                            scatter_job_labels.append(job_name)

                    if scatter_bls_vals:
                        fig_scatter, ax_scatter = plt.subplots(figsize=(10, 7))

                        # Plot scatter points
                        ax_scatter.scatter(scatter_bls_vals, scatter_metric_vals, s=100, alpha=0.6, edgecolors='black', linewidth=1)

                        # Fit a line of best fit and compute R²
                        from scipy.stats import linregress
                        slope, intercept, r_value, p_value, std_err = linregress(scatter_bls_vals, scatter_metric_vals)
                        r_squared = r_value ** 2
                        
                        # Plot regression line
                        x_line = np.array([min(scatter_bls_vals), max(scatter_bls_vals)])
                        y_line = slope * x_line + intercept
                        ax_scatter.plot(x_line, y_line, 'b-', linewidth=2, label=f'Linear Fit (R² = {r_squared:.3f})')

                        # Add job labels to each point (optional, with some offset for readability)
                        for i, (bls, metric_val, job_label) in enumerate(zip(scatter_bls_vals, scatter_metric_vals, scatter_job_labels)):
                            ax_scatter.annotate(job_label, (bls, metric_val), fontsize=8, alpha=0.7,
                                             xytext=(5, 5), textcoords='offset points')

                        # Add horizontal line at 0.5 for context (if metric is selection_rate)
                        if metric == 'selection_rate':
                            ax_scatter.axhline(y=0.5, color='red', linestyle='--', linewidth=1.5, alpha=0.5, label='No Gender Preference (0.5)')

                        ax_scatter.set_xlabel('BLS Value (%)', fontsize=12, fontweight='bold')
                        ax_scatter.set_ylabel(f'{metric_to_plot.get(metric, metric)}', fontsize=12, fontweight='bold')
                        ax_scatter.set_title(f'{metric_to_plot.get(metric, metric)} vs. BLS Value by Job\n(Across All Job Bundles, Gender: W)', fontsize=13, fontweight='bold')
                        ax_scatter.grid(True, alpha=0.3, linestyle='--')

                        # Always show legend now that we have the regression line
                        ax_scatter.legend(fontsize=10)

                        fig_scatter.tight_layout()
                        scatter_path = os.path.join(repo_path(args.output_dir), f"{metric}_scatter_bls_by_job.png")
                        scatter_path = add_pval_suffix_to_path(scatter_path, args.p_value_threshold)
                        print(f"Saving scatter plot to {scatter_path}")
                        fig_scatter.savefig(repo_path(scatter_path), dpi=150)
                        print(f"Created plot: {scatter_path}")
                        plt.close(fig_scatter)
                    else:
                        print(f"No job BLS values found for {metric}. Skipping scatter plot.")

if __name__ == "__main__":
    main()



'''


python plot_yin_all.py --split_by all --metric selection_rate --bw_threshold 0.1; 

python plot_yin_all.py --split_by job_bundle_job --input_file results/yin_sampling_downsampling_analysis_original_detailed.csv --p_value_threshold 0.05


python plot_yin_all.py --split_by all --metric selection_rate; 
python plot_yin_all.py --split_by model --metric selection_rate;  
python plot_yin_all.py --split_by name --metric selection_rate;  
python plot_yin_all.py --split_by job --metric selection_rate; 
python plot_yin_all.py --split_by job_bundle_job --input_file results/yin_sampling_downsampling_analysis_original_detailed.csv --metric selection_rate; 



'''
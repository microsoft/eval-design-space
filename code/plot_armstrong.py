from utils.paths import REPO_ROOT, repo_path
from utils.plot_references import (
    load_published_reference, load_reproduced_reference, REPRODUCED_SETTING_LABEL,
    published_setting_label,
)

import argparse
import ast
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import gaussian_kde
from utils.noise_floor import compute_default_framework_stats


from utils.variables import colors, metric_to_plot, linestyles, model_to_plotname

# Font size defaults (make titles, labels, legend and ticks larger) 🔧


TITLE_FONT_SIZE = 40
LABEL_FONT_SIZE = 28
LEGEND_FONT_SIZE = 20
TICK_FONT_SIZE = 25
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Inter', 'DejaVu Sans', 'Arial', 'Liberation Sans'],
    'axes.titlesize': TITLE_FONT_SIZE,
    'axes.labelsize': LABEL_FONT_SIZE,
    'legend.fontsize': LEGEND_FONT_SIZE,
    'legend.title_fontsize': LEGEND_FONT_SIZE,
    'xtick.labelsize': TICK_FONT_SIZE,
    'ytick.labelsize': TICK_FONT_SIZE
})

REF_LINE_BROWN = "#8B4513"
REF_LINE_STYLES = ['-.', ':']


# ----------------------------
# Parsing helpers
# ----------------------------
def _safe_literal_eval(x: Any) -> Any:
    """ast.literal_eval with guardrails."""
    if pd.isna(x):
        return None
    if isinstance(x, (list, dict, tuple, int, float)):
        return x
    s = str(x).strip()
    if s == "":
        return None
    try:
        return ast.literal_eval(s)
    except Exception:
        return None


def parse_float_list(x: Any) -> Optional[List[float]]:
    """
    Parses things like:
      "[np.float64(0.1), np.float64(0.2)]"
      "[0.1, 0.2]"
      "0.1"
    into a list[float] when possible.
    """
    if pd.isna(x):
        return None

    if isinstance(x, list):
        try:
            return [float(v) for v in x]
        except Exception:
            return None

    s = str(x)
    s = s.replace("np.float64(", "").replace(")", "")
    val = _safe_literal_eval(s)
    if val is None:
        # last-ditch: try float
        try:
            return [float(s)]
        except Exception:
            return None

    if isinstance(val, list):
        try:
            return [float(v) for v in val]
        except Exception:
            return None

    # scalar
    try:
        return [float(val)]
    except Exception:
        return None


def parse_str_list(x: Any) -> Optional[List[str]]:
    val = _safe_literal_eval(x)
    if val is None:
        return None
    if isinstance(val, list):
        # if the length fo the list is 1, return a string version of the element
        if len(val) == 1:  
            return str(val[0])
        else:
            return [str(v) for v in val]
    return None


def parse_p_values(x: Any) -> Optional[List[float]]:
    """
    Your P_Value_Vector appears to be stored like "[0.123]" (string).
    Parse into list[float]. If it's a scalar, return [scalar].
    """
    if pd.isna(x):
        return None
    if isinstance(x, list):
        try:
            return [float(v) for v in x]
        except Exception:
            return None

    val = _safe_literal_eval(x)
    if val is None:
        # try stripping brackets manually
        s = str(x).strip()
        if s.startswith("[") and s.endswith("]"):
            s = s[1:-1].strip()
        try:
            return [float(s)]
        except Exception:
            return None

    if isinstance(val, list):
        try:
            return [float(v) for v in val]
        except Exception:
            return None

    try:
        return [float(val)]
    except Exception:
        return None


# ----------------------------
# Domain helpers
# ----------------------------
def woman_index(groups: Optional[List[str]]) -> int:
    """Index of 'Woman' in Sensitive_Attribute_Vector; fallback 0."""
    if not groups:
        return 0
    try:
        return groups.index("Woman")
    except ValueError:
        return 0


def extract_job_name_from_attribute(attr: str) -> Optional[str]:
    """Extract job name from attribute like 'Woman_{job_name}'."""
    if not attr or not isinstance(attr, str):
        return None
    if attr.startswith("Woman_"):
        job_name = attr[6:]  # Remove "Woman_" prefix
        return job_name.strip().lower()
    return None


def get_job_specific_values_from_row(row: pd.Series) -> List[Tuple[str, float]]:
    """
    Extract job-specific values from a row where Sensitive_Attribute_Vector contains
    entries like 'Woman_{job_name}'. Returns a list of (job_name, value) tuples.
    """
    result = []
    
    groups = row.get("Sensitive_Attribute_Vector", None)
    rv = row.get("Result_Vector", None)
    
    if not isinstance(groups, list) or not isinstance(rv, list):
        return result
    
    for idx, attr in enumerate(groups):
        if isinstance(attr, str) and attr.startswith("Woman_"):
            job_name = extract_job_name_from_attribute(attr)
            if job_name and idx < len(rv):
                try:
                    value = float(rv[idx])
                    result.append((job_name, value))
                except (ValueError, TypeError):
                    continue
    
    return result


def ensure_dirs(*paths: str) -> None:
    for p in paths:
        os.makedirs(repo_path(p), exist_ok=True)


def add_pval_suffix_to_path(file_path: str, p_value_threshold: Optional[float]) -> str:
    """Add p-value threshold suffix to filename if threshold is provided."""
    if p_value_threshold is None:
        return file_path
    
    # Convert p-value to string format suitable for filename (e.g., 0.05 -> "pval0.05")
    pval_str = f"pval{p_value_threshold}".replace(".", "_")
    
    # Split path into directory, base filename, and extension
    dir_path = os.path.dirname(file_path)
    filename = os.path.basename(file_path)
    
    # Split filename into name and extension
    if "." in filename:
        name, ext = filename.rsplit(".", 1)
        new_filename = f"{name}_{pval_str}.{ext}"
    else:
        new_filename = f"{filename}_{pval_str}"
    
    # Reconstruct path
    if dir_path:
        return os.path.join(repo_path(dir_path), new_filename)
    else:
        return new_filename


def pad_ylim(values: List[float], frac: float = 0.05) -> Tuple[float, float]:
    if not values:
        return (0.0, 1.0)
    ymin, ymax = min(values), max(values)
    span = ymax - ymin
    if span == 0:
        span = 1.0
    return (ymin - frac * span, ymax + frac * span)


def label_for_cell(name: str, job: str) -> str:
    return f"name_{name}_job_{job}"


def clean_values(values: List[float], remove_zeros: bool = False, remove_outliers: bool = True) -> List[float]:
    """Clean values by removing zeros and/or outliers."""
    cleaned = values.copy()
    if remove_zeros:
        cleaned = [v for v in cleaned if v != 0]
    if remove_outliers:
        try: 
            cleaned = [v for v in cleaned if -5 <= v <= 5]
        except (TypeError, ValueError):
            try: 
                cleaned = [v[0] for v in cleaned if -5 <= v[0] <= 5]
            except: 
                breakpoint()  # If conversion fails, keep all values
    return cleaned


# ----------------------------
# Plotting functions
# ----------------------------
def plot_global_density(values: List[float], metric: str, out_path: str, split_by: str = "all", vlines: Optional[List[Tuple[float, str, str]]] = None, 
    sig_values: Optional[List[Tuple[float, float]]] = None, p_value_threshold: Optional[float] = None, bw_threshold: Optional[float] = None,
    noise_floor: Optional[Tuple[float, float]] = None) -> None:
    """Plot a single global density plot for all values."""
    # clean_vals = clean_values(values)
    clean_vals = values
    print(f"Plotting global density for {len(clean_vals)} cleaned values ({metric}, {out_path}).")
    if sig_values and p_value_threshold is not None:
        values = sig_values
    
    if len(clean_vals) > 1 and np.std(clean_vals) > 1e-9:
        plt.figure(figsize=(7, 6))
        kde = gaussian_kde(clean_vals)
        if bw_threshold is not None:
            kde.set_bandwidth(bw_method=bw_threshold)
        # Center plot at x=0 with fixed limits at -5 to 5
        x_min_centered = -4.5
        x_max_centered = 4.5
        x_grid = np.linspace(x_min_centered, x_max_centered, 500)
        y_grid = kde(x_grid)

        threshold = 0.0
        below_mask = x_grid <= threshold
        above_mask = x_grid > threshold
        vals_arr = np.asarray(clean_vals, dtype=float)
        below_pct = (vals_arr < threshold).mean() * 100.0
        above_pct = (vals_arr >= threshold).mean() * 100.0

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
            for idx, (x_val, label, _color) in enumerate(vlines):
                plt.axvline(
                    x=x_val,
                    color=REF_LINE_BROWN,
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
        # Set x-axis limits to fixed range -5 to 5
        ax.set_xlim((-4.5, 4.5))

        # Place annotations at the top-left and top-right
        axis_y = 0.13  # 5% up from the bottom of the plotting area
        left_x = 0.35  # below_pct moved right toward center
        right_x = 0.65  # above_pct moved left toward center

        fontsize = 25
        plt.text(
            left_x,
            axis_y,
            f"{below_pct:.0f}%",
            ha="left",
            va="top",
            transform=ax.transAxes,
            fontsize=fontsize,
        )
        plt.text(
            right_x,
            axis_y,
            f"{above_pct:.0f}%",
            ha="right",
            va="top",
            transform=ax.transAxes,
            fontsize=fontsize,
        )


        if p_value_threshold is not None: 
            plt.title(f'Density Plot for Armstrong et. al\n({split_by}) (n={len(sig_values)})**')
        
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
            plt.title(f'Armstrong et. al', pad=16)

        plt.xlabel(metric_to_plot[metric])
        plt.ylabel("Probability Density")
        plt.grid(axis='y', alpha=0.3)
        if vlines:
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.tight_layout()
        final_path = add_pval_suffix_to_path(out_path, p_value_threshold)
        plt.savefig(repo_path(final_path), bbox_inches='tight', pad_inches=0.2)
        plt.close()
    else:
        # Fallback to histogram
        plt.figure(figsize=(10, 6))
        counts, bins, patches = plt.hist(clean_vals, bins=100, alpha=0.6, color="purple", edgecolor='black')

        # Add vertical line at threshold (0.0) and compute proportions
        threshold = 0.0
        plt.axvline(x=threshold, color="gray", linestyle="--", linewidth=2, alpha=0.9)
        total_count = len(clean_vals)
        if total_count > 0:
            vals_arr = np.asarray(clean_vals, dtype=float)
            below_count = int((vals_arr < threshold).sum())
            above_count = int((vals_arr >= threshold).sum())
            below_pct = (vals_arr < threshold).mean() * 100.0
            above_pct = (vals_arr >= threshold).mean() * 100.0
            ax = plt.gca()
            ax.set_xlim((min(clean_vals), max(clean_vals)))
            plt.text(0.01, 0.95, f"Below 0.0: {below_pct:.1f}% ({below_count}/{total_count})", transform=ax.transAxes, fontsize=12, va='top')
            plt.text(0.99, 0.95, f"Above 0.0: {above_pct:.1f}% ({above_count}/{total_count})", transform=ax.transAxes, fontsize=12, va='top', ha='right')
        
        # Add vertical lines if provided
        if vlines:
            for idx, (x_val, label, _color) in enumerate(vlines):
                plt.axvline(
                    x=x_val,
                    color=REF_LINE_BROWN,
                    linestyle=REF_LINE_STYLES[idx % len(REF_LINE_STYLES)],
                    linewidth=2,
                    alpha=0.9,
                    label=label,
                )

        # Draw noise floor shaded region if provided (mean, std) on histogram fallback
        if noise_floor is not None:
            try:
                nf_mean, nf_std = float(noise_floor[0]), float(noise_floor[1])
                nf_low = nf_mean - nf_std
                nf_high = nf_mean + nf_std
                plt.axvspan(nf_low, nf_high, color=REF_LINE_BROWN, alpha=0.18, label="Noise Floor")
            except Exception:
                pass
        
        plt.title(f'Distribution of {metric_to_plot[metric]} (All Data)')
        plt.xlabel(metric_to_plot[metric])
        plt.ylabel("Frequency")
        if vlines:
            plt.legend(loc='best')
        plt.tight_layout()
        hist_path = out_path.replace('density', 'histogram')
        final_path = add_pval_suffix_to_path(hist_path, p_value_threshold)
        plt.savefig(repo_path(final_path), bbox_inches='tight', pad_inches=0.2)
        plt.close()


def plot_global_histogram(values: List[float], metric: str, out_path: str, vlines: Optional[List[Tuple[float, str, str]]] = None, sig_values: Optional[List[Tuple[float, float]]] = None, p_value_threshold: Optional[float] = None) -> None:
    clean_vals = clean_values(values)
    
    # Calculate the same limits as the density plot
    x_min, x_max = min(clean_vals), max(clean_vals)
    x_range = x_max - x_min
    limit_min = x_min - 0.1 * x_range
    limit_max = x_max + 0.1 * x_range

    plt.figure(figsize=(10, 6))
    counts, bins, patches = plt.hist(clean_vals, bins=100, alpha=0.6, color="purple", edgecolor='black')
    
    # Add vertical lines if provided
    if vlines:
        # x=0 threshold line styled to match Wang plots
        plt.axvline(x=0.0, color='gray', linestyle='--', linewidth=2, alpha=0.9)
        for idx, (x_val, label, _color) in enumerate(vlines):
            plt.axvline(
                x=x_val,
                color=REF_LINE_BROWN,
                linestyle=REF_LINE_STYLES[idx % len(REF_LINE_STYLES)],
                linewidth=2,
                alpha=0.9,
                label=label,
            )
    
    # Explicitly set the x-axis limits
    plt.xlim(limit_min, limit_max)

    plt.title(f'Distribution of {metric_to_plot[metric]} (All Data)')
    plt.xlabel(metric_to_plot[metric])
    plt.ylabel("Frequency")
    plt.grid(axis='y', alpha=0.3)
    if vlines:
        plt.legend(loc='best')
    plt.tight_layout()
    final_path = add_pval_suffix_to_path(out_path, p_value_threshold)
    plt.savefig(repo_path(final_path), bbox_inches='tight', pad_inches=0.2)
    plt.close()


def plot_split_density_cdf(
    split_dict: Dict[str, List[float]], 
    metric: str, 
    split_type: str,
    out_root: str,
    bls_values: Optional[Dict[str, float]] = None,
    vlines: Optional[List[Tuple[float, str, str]]] = None,
    model_vlines: Optional[Dict[str, float]] = None,
    sig_values_dict: Optional[Dict[str, List[Tuple[float, float]]]] = None,
    p_value_threshold: Optional[float] = None
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
            hist_result = ax_hist.hist(clean_vals, 
                        bins=hist_bins if isinstance(hist_bins, np.ndarray) else 30,
                        alpha=0.5, 
                        color=colors_list[idx], 
                        edgecolor='black',
                        label=label_text,
                        density=False)  # Use frequency, not density
            
            # Extract counts and bins from histogram result
            if isinstance(hist_result, tuple) and len(hist_result) >= 2:
                counts_hist = hist_result[0]
                bins_hist = hist_result[1]
            else:
                counts_hist = None
                bins_hist = hist_bins if isinstance(hist_bins, np.ndarray) else np.linspace(min(clean_vals), max(clean_vals), 30)
            
            if len(clean_vals) > 1 and np.std(clean_vals) > 1e-9:
                # Plot Density
                try:
                    kde = gaussian_kde(clean_vals)
                    # kde.set_bandwidth(bw_method=0.1)
                    x_min, x_max = min(clean_vals), max(clean_vals)
                    x_range = x_max - x_min
                    if x_range == 0:
                        x_range = 1.0
                    x_grid = np.linspace(x_min - 0.2 * x_range, x_max + 0.2 * x_range, 200)
                    y_grid = kde(x_grid)
                    
                    ax_den.plot(x_grid, y_grid, 
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
        # Calculate y_max for vertical line positioning
        # For histogram, estimate max count from data distribution
        if all_clean_vals:
            hist_counts, _ = np.histogram(all_clean_vals, bins=30)
            y_max_hist = max(hist_counts) if len(hist_counts) > 0 else 1.0
        else:
            y_max_hist = 1.0
        
        y_max_den = 0.0
        if all_clean_vals:
            try:
                kde_all = gaussian_kde(all_clean_vals)
                # kde.set_bandwidth(bw_method=0.1)
                x_test = np.linspace(min(all_clean_vals), max(all_clean_vals), 200)
                y_max_den = max(kde_all(x_test))
            except:
                y_max_den = 1.0
        
        # Add vertical lines if provided
        if vlines:
            ax_hist.axvline(x=0.0, color='gray', linestyle='--', linewidth=2, alpha=0.9)
            ax_den.axvline(x=0.0, color='gray', linestyle='--', linewidth=2, alpha=0.9)
            ax_cdf.axvline(x=0.0, color='gray', linestyle='--', linewidth=2, alpha=0.9)
            for idx, (x_val, label, _color) in enumerate(vlines):
                line_style = REF_LINE_STYLES[idx % len(REF_LINE_STYLES)]
                ax_hist.axvline(x=x_val, color=REF_LINE_BROWN, linestyle=line_style, linewidth=2, alpha=0.9, label=label)
                ax_den.axvline(x=x_val, color=REF_LINE_BROWN, linestyle=line_style, linewidth=2, alpha=0.9, label=label)
                ax_cdf.axvline(x=x_val, color=REF_LINE_BROWN, linestyle=line_style, linewidth=2, alpha=0.9, label=label)
        
        # Add model-specific vertical lines if provided
        if model_vlines and split_type == "model":
            for model_name, x_val in model_vlines.items():
                if model_name in split_dict:  # Only add line if model is in the plot
                    ax_hist.axvline(x=x_val, color='orange', linestyle='--', linewidth=1.5, alpha=0.7, label=model_name)
                    ax_den.axvline(x=x_val, color='orange', linestyle='--', linewidth=1.5, alpha=0.7, label=model_name)
                    ax_cdf.axvline(x=x_val, color='orange', linestyle='--', linewidth=1.5, alpha=0.7, label=model_name)
        
        # Save Histogram Plot
        ax_hist.set_title(f'Comparison of {metric_to_plot[metric]} Histogram by {split_type.capitalize()}')
        ax_hist.set_xlabel(metric_to_plot[metric])
        ax_hist.set_ylabel("Frequency")
        ax_hist.grid(True, linestyle="--", alpha=0.3, axis='y')
        # Combine data series legend with vertical line legend
        handles, labels = ax_hist.get_legend_handles_labels()
        ax_hist.legend(handles, labels, title=split_type.capitalize(), bbox_to_anchor=(1.05, 1), loc='upper left')
        fig_hist.tight_layout()
        
        hist_path = f"{out_root}/{metric_to_plot[metric]}_combined_histogram_by_{split_type}.png"
        hist_path = add_pval_suffix_to_path(hist_path, p_value_threshold)
        print(f"Saving combined histogram plot to {hist_path}")
        fig_hist.savefig(repo_path(hist_path), bbox_inches='tight', pad_inches=0.2)
        
        # Save Density Plot
        ax_den.set_title(f'Comparison of {metric_to_plot[metric]} Density by {split_type.capitalize()}')
        ax_den.set_xlabel(metric_to_plot[metric])
        ax_den.set_ylabel("Density")
        ax_den.grid(True, linestyle="--", alpha=0.3)
        # Combine data series legend with vertical line legend
        handles, labels = ax_den.get_legend_handles_labels()
        ax_den.legend(handles, labels, title=split_type.capitalize(), bbox_to_anchor=(1.05, 1), loc='upper left')
        fig_den.tight_layout()
        
        den_path = f"{out_root}/{metric_to_plot[metric]}_combined_density_by_{split_type}.png"
        den_path = add_pval_suffix_to_path(den_path, p_value_threshold)
        print(f"Saving combined density plot to {den_path}")
        fig_den.savefig(repo_path(den_path), bbox_inches='tight', pad_inches=0.2)
        
        # Save CDF Plot
        ax_cdf.set_title(f'Comparison of {metric_to_plot[metric]} CDF by {split_type.capitalize()}')
        ax_cdf.set_xlabel(metric_to_plot[metric])
        ax_cdf.set_ylabel("Cumulative Probability")
        ax_cdf.grid(True, linestyle="--", alpha=0.3)
        # Combine data series legend with vertical line legend
        handles, labels = ax_cdf.get_legend_handles_labels()
        ax_cdf.legend(handles, labels, title=split_type.capitalize(), bbox_to_anchor=(1.05, 1), loc='upper left')
        fig_cdf.tight_layout()

        cdf_path = f"{out_root}/{metric_to_plot[metric]}_combined_cdf_by_{split_type}.png"
        cdf_path = add_pval_suffix_to_path(cdf_path, p_value_threshold)
        print(f"Saving combined CDF plot to {cdf_path}")
        fig_cdf.savefig(repo_path(cdf_path), bbox_inches='tight', pad_inches=0.2)
    else:
        print(f"No {split_type}s had enough data to plot.")
    
    plt.close(fig_den)
    plt.close(fig_cdf)
    plt.close(fig_hist)


def benjamini_hochberg(pvals: List[float], q: float) -> List[bool]:
    """Return list of booleans indicating which p-values are rejected by BH at level q.

    True => rejected, False => not rejected. Expects pvals as a list of floats.
    """
    m = len(pvals)
    if m == 0:
        return []
    if q is None or q <= 0:
        return [False] * m

    # sort p-values with original indices
    idx = np.argsort(pvals)
    sorted_p = np.array(pvals)[idx]

    thresholds = (np.arange(1, m + 1) / float(m)) * q
    # Find largest k where p_k <= threshold_k
    compare = sorted_p <= thresholds
    if not compare.any():
        return [False] * m
    k = np.max(np.where(compare)[0])  # zero-based index
    # reject all with rank <= k
    rejected_sorted = np.zeros(m, dtype=bool)
    rejected_sorted[: k + 1] = True
    # map back to original order
    rejected = np.zeros(m, dtype=bool)
    rejected[idx] = rejected_sorted
    return rejected.tolist()


def get_reproduced_value_from_vlines(vlines: Optional[List[Tuple[float, str, str]]]) -> Optional[float]:
    """Return the numeric reproduced value from a vlines list.

    Looks for an entry whose label identifies the reproduced setting.
    Falls back to the first numeric entry if no explicit reproduced label is found.
    """
    if not vlines:
        return None
    for val, label, _ in vlines:
        if isinstance(label, str) and label.startswith("Reproduced"):
            try:
                return float(val)
            except Exception:
                continue
    # fallback: return first numeric value
    try:
        return float(vlines[0][0])
    except Exception:
        return None


def compute_relative_differences(values: List[float], reference: float) -> List[Optional[float]]:
    """Compute relative differences (value - reference) / |reference| for a list of values.

    If `reference` is zero, returns a list of None to indicate undefined relative differences.
    """
    if reference is None:
        return [None for _ in values]
    try:
        ref = float(reference)
    except Exception:
        return [None for _ in values]
    if ref == 0:
        return [None for _ in values]
    return [((float(v) - ref) / abs(ref)) if v is not None else None for v in values]


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
    
    if sig_values_dict:
        split_dict = sig_values_dict

    labels = list(split_dict.keys())
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

    ax.set_xticks(positions)
    if split_type == "model":
        ax.set_xticklabels([label.replace('-', '\n') for label in labels], rotation=0, ha='right')
    else: 
        ax.set_xticklabels([label.replace('-', '\n') for label in labels], rotation=45, ha='right')
    ax.set_xlabel(metric_to_plot.get(metric, metric))
    ax.set_ylim(-2,2)

    ax.set_ylabel('Value')
    if sig_values_dict and p_value_threshold is not None:
        ax.set_title(f'Armstrong et. al split by {split_type.capitalize()}\n(Gender: W, stat-sig)')
    else: ax.set_title(f'Armstrong et. al split by {split_type.capitalize()}\n(Gender: W)')
    ax.grid(axis='y', alpha=0.3)

    fig.tight_layout()
    save_path = os.path.join(repo_path(out_root), f"{metric}_boxwhisker_by_{split_type}.png")
    final_path = add_pval_suffix_to_path(save_path, p_value_threshold)
    print(f"Saving box-and-whisker plot to {final_path}")
    fig.savefig(repo_path(final_path), dpi=150, bbox_inches='tight', pad_inches=0.2)
    plt.close(fig)


# ----------------------------
# Main
# ----------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Unified plotting script for regression coefficients.")
    parser.add_argument("--sampling", type=str, default="downsampling", help="Sampling method to use.")
    parser.add_argument("--analysis", type=str, default="original", help="Analysis to perform.")
    parser.add_argument("--exp_framework", type=str, default="armstrong", help="Experiment framework to use.")
    parser.add_argument("--model", type=str, default="all", help="Model to use (or 'all').")
    parser.add_argument("--output_dir", help="Output directory; defaults to results/figs/FRAMEWORK/MODEL.")
    parser.add_argument("--metric", type=str, default="regression_coefficients", help="Metric to plot.")
    parser.add_argument("--reference_input", help="Baseline CSV (defaults to the unfiltered input CSV).")
    parser.add_argument("--split_by", type=str, default="all", 
                       choices=["all", "model", "job", "job_name", "name"], 
                       help="How to split the data: 'all' (no split), 'model', 'job', 'job_name', or 'name'.")
    parser.add_argument("--p_value_threshold", type=float, default=None,
                       help="If provided, annotate values with p-value below this threshold in plots.")
    parser.add_argument("--bw_threshold", type=float, default=0.1,
                       help="If provided, set bandwidth method for KDE (passed to gaussian_kde.set_bandwidth).")
    parser.add_argument("--color", type=bool, default=None,
                       help="If provided, create a plot for each unqiue value of the split_by argument.")
    

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

    # Load data
    in_path = f"results/{args.exp_framework}_sampling_{args.sampling}_analysis_{args.analysis}.csv"
    if not os.path.exists(repo_path(in_path)):
        print(f"Error: File {in_path} not found.")
        return
    
    df = pd.read_csv(repo_path(in_path))
    
    # filter out all the values where df['Names']== wang
    df = df[df['Names'] != 'seshadri']
    df = df[df['Random_State'] == 42.0]

    # Parse columns
    if "Result_Vector" in df.columns:
        df["Result_Vector"] = df["Result_Vector"].apply(parse_float_list)
        df["Std_Error_Vector"] = df["Std_Error_Vector"].apply(parse_float_list)
        df["Sensitive_Attribute_Vector"] = df["Sensitive_Attribute_Vector"].apply(parse_str_list)
        df["P_Value_Vector"] = df["P_Value_Vector"].apply(parse_p_values)
        data_format = "armstrong"
    else:
        # Assume yin format or other format
        data_format = "yin"
        print("Warning: Data format not recognized. Attempting to use direct column access.")

    # Filter by model if specified
    if args.model != "all" and "Model" in df.columns:
        working_df = df[df["Model"] == args.model].copy()
    else:
        working_df = df.copy()

    # Grid definition (10×7)
    row_labels = ["armstrong", "rozado", "wen", "wang", "gaeb", "lippens", "", "karvonen", "zollo", "yin"]
    col_labels = ["armstrong", "rozado", "wen", "wang", "karvonen", "zollo", "yin"]

    # Output dirs
    out_root = args.output_dir or f"results/figs/{args.exp_framework}/{args.model}"
    ensure_dirs(f"results/figs/{args.exp_framework}", out_root)

    # Build lookup table for grid logic
    if data_format == "armstrong":
        df_keyed = (
            working_df.sort_values(["Names", "Jobs", "Metric"])
              .drop_duplicates(subset=["Names", "Jobs", "Metric"], keep="first")
              .set_index(["Names", "Jobs", "Metric"])
        )

        def get_cell_row(name: str, job: str, metric: str) -> Optional[pd.Series]:
            try:
                return df_keyed.loc[(name, job, metric)]
            except KeyError:
                return None

        def get_woman_stats(row: Optional[pd.Series]) -> Tuple[float, Optional[float], Optional[float]]:
            """Returns (value_for_woman, stderr_for_woman, p_value_scalar_or_none)"""
            if row is None:
                return (0.0, None, None)

            groups = row.get("Sensitive_Attribute_Vector", None)
            idx = woman_index(groups)

            rv = row.get("Result_Vector", None)
            sv = row.get("Std_Error_Vector", None)
            pv = row.get("P_Value_Vector", None)

            val = 0.0
            err = None
            p = None

            if isinstance(rv, list) and len(rv) > idx:
                val = float(rv[idx])

            if isinstance(sv, list) and len(sv) > idx:
                err = float(sv[idx])

            if isinstance(pv, list) and len(pv) > 0:
                p = float(pv[0])

            return (val, err, p)
    else:
        # For yin format or other formats, define simpler accessors
        df_keyed = None
        
        def get_cell_row(name: str, job: str, metric: str) -> Optional[pd.Series]:
            return None  # Not supported for non-armstrong format
        
        def get_woman_stats(row: pd.Series) -> Tuple[float, Optional[float], Optional[float]]:
            metric_col = args.metric
            if metric_col in row:
                val = float(row[metric_col]) if pd.notna(row[metric_col]) else 0.0
            else:
                val = 0.0
            return (val, None, None)

    # Colors for job groups
    label_to_color = {
        "armstrong": "tab:blue",
        "rozado": "tab:orange",
        "wen": "tab:green",
        "wang": "tab:red",
        "karvonen": "tab:purple",
        "zollo": "tab:brown",
        "yin": "tab:pink",
        "lippens": "tab:gray",
        "seshadri": "tab:olive",
        "gaeb": "tab:cyan",
    }

    metric = args.metric

    # Define vertical reference lines for score_difference
    if metric == "score_difference":
        vlines = [
            # (0.0, "No Preference"),
            (load_reproduced_reference("armstrong", metric, args.reference_input or in_path), REPRODUCED_SETTING_LABEL, REF_LINE_BROWN),
            (load_published_reference("armstrong", metric), published_setting_label("armstrong", metric), REF_LINE_BROWN)
        ]
        model_vlines = {
        }
    elif metric == "regression_coefficients":
        vlines = [
            # (0.0, "No Preference"),
            (load_reproduced_reference("armstrong", metric, args.reference_input or in_path), REPRODUCED_SETTING_LABEL, REF_LINE_BROWN),
            (load_published_reference("armstrong", metric), published_setting_label("armstrong", metric), REF_LINE_BROWN)
        ]
        model_vlines = {
        }
        

    # Remove duplicate rows while avoiding list-valued columns (unhashable in pandas dedupe).
    # Keep one row per Names/Jobs/Model/Metric combination.
    df = df.drop_duplicates(subset=["Names", "Jobs", "Model", "Metric"], keep="first")

    # ----------------------------
    # Main plotting logic based on split_by
    # ----------------------------

    if args.split_by == "all":
        # Collect all values across all models
        print(f"Collecting all values for {metric}...")
        all_collected_values = []
        sig_values = []  # List of (x_val, value) tuples for significant p-values
        df = df[df["Sensitive_Attribute_Vector"] == 'Woman']
        df = df[df["Metric"] == metric]

        for _, row in df.iterrows():
            val, _, p = get_woman_stats(row)
            all_collected_values.append(val)
            # Collect p-values along with values for BH filtering later
            sig_values.append((val, p))
        
        if args.p_value_threshold is not None:
            # Apply Benjamini-Hochberg on collected p-values for the filtered df
            # sig_values currently stores tuples (val, p)
            vals_and_ps = sig_values
            ps = [p for (_, p) in vals_and_ps if p is not None]
            # For rows with missing p, treat as accepted (i.e., append to final sig list)
            accepted_vals_missing_p = [v for (v, p) in vals_and_ps if p is None]
            rejected_flags = benjamini_hochberg(ps, args.p_value_threshold) if ps else []

            # Map back: keep values where hypothesis is accepted (i.e., not rejected)
            accepted_vals = []
            pi = 0
            for (v, p) in vals_and_ps:
                if p is None:
                    accepted_vals.append(v)
                else:
                    # rejected_flags[pi] == True means rejected -> do NOT append
                    if not rejected_flags[pi]:
                        accepted_vals.append(v)
                    pi += 1

            sig_values = accepted_vals
            print(f"Total values accepted by BH (kept): {len(sig_values)}")

        working_subset = df
        print(f"Total collected values for {metric}: {len(all_collected_values)}")
        if all_collected_values:
            noise_floor_results = compute_default_framework_stats()
            nf_n, nf_mean, nf_std = noise_floor_results['armstrong']['n'], noise_floor_results['armstrong']['mean'], noise_floor_results['armstrong']['std']
            print(f"Noise floor (sampled n={nf_n}): mean={nf_mean}, std={nf_std}")
            
            sig_vals = sig_values if args.p_value_threshold is not None else None
            if args.p_value_threshold is not None:
                all_collected_values = sig_vals.copy()
            
            plot_global_density(
                all_collected_values, 
                metric, 
                f"{out_root}/{metric}_global_density_all.png",
                split_by='all',
                vlines=vlines,
                sig_values=sig_vals,
                p_value_threshold=args.p_value_threshold,
                bw_threshold=args.bw_threshold
            )
            # Also save a version that highlights the noise floor (mean, std)
            plot_global_density(
                all_collected_values,
                metric,
                f"{out_root}/{metric}_global_density_all_with_noise_floor.png",
                split_by='all',
                vlines=vlines,
                sig_values=sig_vals,
                p_value_threshold=args.p_value_threshold,
                bw_threshold=args.bw_threshold,
                noise_floor=(nf_mean, nf_std)
            )

            
    elif args.split_by == "model":
        # Split by model
        print(f"Generating plots split by model for {metric}...")
        if "Model" not in df.columns:
            print("Error: 'Model' column not found in data.")
            return

        df = df[df["Sensitive_Attribute_Vector"] == 'Woman']
        
        unique_models = df["Model"].unique()
        split_dict = {}
        sig_values_dict = {}  # Dict mapping model_name -> list of (x_val, value) tuples        
        for model_name in unique_models:
            model_subset = df[(df["Model"] == model_name) & (df["Metric"] == metric)]
            model_values = []
            model_sig_values = []
            for _, row in model_subset.iterrows():
                val, _, p = get_woman_stats(row)
                model_values.append(val)
                # collect for BH filtering
                model_sig_values.append((val, p))
            if model_values:
                split_dict[model_name] = model_values
                if model_sig_values:
                    # apply BH per-model if requested
                    if args.p_value_threshold is not None:
                        ps = [p for (_, p) in model_sig_values if p is not None]
                        rejected_flags = benjamini_hochberg(ps, args.p_value_threshold) if ps else []
                        accepted = []
                        pi = 0
                        for (v, p) in model_sig_values:
                            if p is None:
                                accepted.append(v)
                            else:
                                if not rejected_flags[pi]:
                                    accepted.append(v)
                                pi += 1
                        sig_values_dict[model_name] = accepted
                    else:
                        # previous behavior: include raw p-filtered values
                        sig_values_dict[model_name] = [v for (v, p) in model_sig_values if p is not None and p < (args.p_value_threshold or 1.0)]

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
                    sig_values=model_sig_values,
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
                    sig_values=model_sig_values,
                    p_value_threshold=args.p_value_threshold,
                    bw_threshold=args.bw_threshold
                )

        sig_vals = sig_values_dict if args.p_value_threshold is not None else None
        plot_split_density_cdf(split_dict, metric, "model", out_root, vlines=vlines, model_vlines=model_vlines, sig_values_dict=sig_vals, p_value_threshold=args.p_value_threshold)

        plot_boxwhisker_plot(
                split_dict,
                metric,
                args.split_by,
                out_root,
                sig_values_dict=sig_vals if sig_vals else None,
                p_value_threshold=args.p_value_threshold,
            )   

    elif args.split_by == "job":
        # Split by job
        print(f"Generating plots split by job for {metric}...")
        if "Jobs" not in df.columns:
            print("Error: 'Jobs' column not found in data.")
            return
        
        df = df[df["Sensitive_Attribute_Vector"] == 'Woman']

        unique_jobs = df["Jobs"].unique()
        split_dict = {}
        sig_values_dict = {}  # Dict mapping job_val -> list of (x_val, value) tuples


        
        for job_val in unique_jobs:
            job_subset = df[(df["Jobs"] == job_val) & (df["Metric"] == metric)]
            job_values = []
            job_sig_values = []
            for _, row in job_subset.iterrows():
                val, _, p = get_woman_stats(row)
                job_values.append(val)
                job_sig_values.append((val, p))
            if job_values:
                split_dict[job_val] = job_values
                if job_sig_values:
                    if args.p_value_threshold is not None:
                        ps = [p for (_, p) in job_sig_values if p is not None]
                        rejected_flags = benjamini_hochberg(ps, args.p_value_threshold) if ps else []
                        accepted = []
                        pi = 0
                        for (v, p) in job_sig_values:
                            if p is None:
                                accepted.append(v)
                            else:
                                if not rejected_flags[pi]:
                                    accepted.append(v)
                                pi += 1
                        sig_values_dict[job_val] = accepted
                    else:
                        sig_values_dict[job_val] = [v for (v, p) in job_sig_values if p is not None and p < (args.p_value_threshold or 1.0)]

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
        
        sig_vals = sig_values_dict if args.p_value_threshold is not None else None
        plot_split_density_cdf(split_dict, metric, "job", out_root, bls_values, vlines=vlines, sig_values_dict=sig_vals, p_value_threshold=args.p_value_threshold)
        plot_boxwhisker_plot(
                split_dict,
                metric,
                args.split_by,
                out_root,
                sig_values_dict=sig_vals if sig_vals else None,
                p_value_threshold=args.p_value_threshold,
            )   

    elif args.split_by == "job_name":
        # Split by job_name (extracted from Sensitive_Attribute_Vector like "Woman_{job_name}")
        print(f"Generating plots split by job_name for {metric}...")
        if "Sensitive_Attribute_Vector" not in df.columns:
            print("Error: 'Sensitive_Attribute_Vector' column not found in data.")
            return
        
        split_dict = {}
        
        # Filter to rows with the target metric
        metric_subset = df[df["Metric"] == metric]
        
        for _, row in metric_subset.iterrows():
            job_value_pairs = get_job_specific_values_from_row(row)
            for job_name, value in job_value_pairs:
                if job_name not in split_dict:
                    split_dict[job_name] = []
                split_dict[job_name].append(value)

        if split_dict:
            plot_split_density_cdf(split_dict, metric, "job_name", out_root)
        else:
            print("No job_name data found in Sensitive_Attribute_Vector.")

    elif args.split_by == "name":
        # Split by name
        print(f"Generating plots split by name for {metric}...")
        if "Names" not in df.columns:
            print("Error: 'Names' column not found in data.")
            return
        
        unique_names = df["Names"].unique()
        split_dict = {}
        sig_values_dict = {}  # Dict mapping name_val -> list of (x_val, value) tuples

        
        for name_val in unique_names:
            name_subset = df[(df["Names"] == name_val) & (df["Metric"] == metric)]
            name_values = []
            name_sig_values = []
            for _, row in name_subset.iterrows():
                val, _, p = get_woman_stats(row)
                name_values.append(val)
                name_sig_values.append((val, p))
            if name_values:
                split_dict[name_val] = name_values
                if name_sig_values:
                    if args.p_value_threshold is not None:
                        ps = [p for (_, p) in name_sig_values if p is not None]
                        rejected_flags = benjamini_hochberg(ps, args.p_value_threshold) if ps else []
                        accepted = []
                        pi = 0
                        for (v, p) in name_sig_values:
                            if p is None:
                                accepted.append(v)
                            else:
                                if not rejected_flags[pi]:
                                    accepted.append(v)
                                pi += 1
                        sig_values_dict[name_val] = accepted
                    else:
                        sig_values_dict[name_val] = [v for (v, p) in name_sig_values if p is not None and p < (args.p_value_threshold or 1.0)]

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
            
        


if __name__ == "__main__":
    main()

'''
# Plot all data together
python plot_armstrong.py --split_by all --metric regression_coefficients

# Split by model
python plot_armstrong.py --split_by model --metric regression_coefficients

# Split by job
python plot_armstrong.py --split_by job --metric regression_coefficients

# Split by name
python plot_armstrong.py --split_by name --metric regression_coefficients

# Split by job_name
python plot_armstrong.py --split_by job_name --metric score_difference


python plot_armstrong.py --split_by all --metric score_difference;
python plot_armstrong.py --split_by name --metric score_difference; 
python plot_armstrong.py --split_by job --metric score_difference;
python plot_armstrong.py --split_by name --metric score_difference;

python plot_armstrong.py --split_by all --metric regression_coefficients;
python plot_armstrong.py --split_by name --metric regression_coefficients; 
python plot_armstrong.py --split_by job --metric regression_coefficients;
python plot_armstrong.py --split_by name --metric regression_coefficients;




python plot_armstrong.py --split_by all --metric score_difference;
python plot_armstrong.py --split_by all --metric regression_coefficients;
python plot_armstrong.py --split_by name --metric score_difference; 
python plot_armstrong.py --split_by job --metric score_difference;
python plot_armstrong.py --split_by model --metric score_difference; 
python plot_armstrong.py --split_by model --metric regression_coefficients;
python plot_armstrong.py --split_by name --metric regression_coefficients; 
python plot_armstrong.py --split_by job --metric regression_coefficients;



python plot_armstrong.py --split_by all --metric score_difference; python plot_armstrong.py --split_by all --metric score_difference  --p_value_threshold 0.05;
python plot_armstrong.py --split_by model --metric score_difference; python plot_armstrong.py --split_by model --metric score_difference  --p_value_threshold 0.05; 
python plot_armstrong.py --split_by job --metric score_difference; python plot_armstrong.py --split_by job --metric score_difference  --p_value_threshold 0.05;
python plot_armstrong.py --split_by name --metric score_difference; python plot_armstrong.py --split_by job --metric score_difference  --p_value_threshold 0.05;
python plot_armstrong.py --split_by model --metric regression_coefficients; python plot_armstrong.py --split_by model --metric regression_coefficients  --p_value_threshold 0.05; 
python plot_armstrong.py --split_by name --metric regression_coefficients; python plot_armstrong.py --split_by name --metric regression_coefficients  --p_value_threshold 0.05; 
python plot_armstrong.py --split_by job --metric regression_coefficients; python plot_armstrong.py --split_by job --metric regression_coefficients  --p_value_threshold 0.05;
python plot_armstrong.py --split_by all --metric regression_coefficients; python plot_armstrong.py --split_by all --metric regression_coefficients  --p_value_threshold 0.05; 

'''
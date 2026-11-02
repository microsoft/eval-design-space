import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.paths import REPO_ROOT, repo_path
from utils.plot_references import (
    canonical_metric, load_published_reference, load_reproduced_reference, load_saved_wang_reference,
)

import argparse
import ast
import os
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde, ttest_1samp




TITLE_FONT_SIZE = 24
LABEL_FONT_SIZE = 22
LEGEND_FONT_SIZE = 15
TICK_FONT_SIZE = 22
VIOLIN_POINTS = 500
VIOLIN_BW_METHOD = 0.35 # 0.35
VIOLIN_TAIL_STD_MULTIPLIER = 2.0
plt.rcParams.update({
	'font.family': 'sans-serif',
	'font.sans-serif': ['Inter-Regular', 'DejaVu Sans', 'Arial', 'Liberation Sans'],
	'axes.titlesize': TITLE_FONT_SIZE,
	'axes.labelsize': LABEL_FONT_SIZE,
	'legend.fontsize': LEGEND_FONT_SIZE,
	'legend.title_fontsize': LEGEND_FONT_SIZE,
	'xtick.labelsize': TICK_FONT_SIZE,
	'ytick.labelsize': TICK_FONT_SIZE
})


PAPER_COLORS = {
    "armstrong": "tab:blue",
    "wang": "tab:red",
    "yin": "tab:pink",
}


# ---------------------------------------------------------------------------
# Wang per-model configuration
# ---------------------------------------------------------------------------

REPRODUCED_SETTING_MODELS = [
    'meta-llama-3.1-8b-instruct-turbo',
    'meta-llama-3.3-70b-instruct-turbo',
]


WANG_METRIC_LABELS = {
    'impact_ratio_rank': 'Impact Ratio (Rank, Protected Group)',
    'impact_ratio_score': 'Impact Ratio (Score, Protected Group)',
    'selection_rate_rank': 'Selection Rate (Rank, Protected Group)',
    'selection_rate_score': 'Selection Rate (Score, Protected Group)',
    'score_difference': 'Score Difference',
    'score_diff': 'Score Difference',
    'wang_score_diff': 'Score Difference',
    'impact_ratio_protect_rank': 'Impact Ratio (Rank, Protected Group)',
    'impact_ratio_protect_score': 'Impact Ratio (Score, Protected Group)',
    'selection_rate_protect_rank': 'Selection Rate (Rank, Protected Group)',
    'selection_rate_protect_score': 'Selection Rate (Score, Protected Group)',
}


WANG_MODEL_KEY_PREFIX = "wang::"


def wang_model_key(model: str) -> str:
    """Key used for a per-model Wang row."""
    return f"{WANG_MODEL_KEY_PREFIX}{model}".lower().strip()


def wang_model_label(model: str) -> str:
    if model == "meta-llama-3.1-8b-instruct-turbo":
        return "Wang et al.\nmeta-llama-3.1\n8b-instruct-turbo"
    if model == "meta-llama-3.3-70b-instruct-turbo":
        return "Wang et al.\nmeta-llama-3.3\n70b-instruct-turbo"
    return f"Wang et al.\n({model})"


def _safe_literal_eval(x: Any) -> Any:
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
    if pd.isna(x):
        return None
    if isinstance(x, list):
        vals = pd.to_numeric(pd.Series(x), errors="coerce")
        out = vals[vals.notna() & np.isfinite(vals)].astype(float).tolist()
        return out if out else None

    s = str(x).replace("np.float64(", "").replace(")", "")
    val = _safe_literal_eval(s)
    if val is None:
        return None
    if isinstance(val, list):
        vals = pd.to_numeric(pd.Series(val), errors="coerce")
        out = vals[vals.notna() & np.isfinite(vals)].astype(float).tolist()
        return out if out else None

    num = pd.to_numeric(pd.Series([val]), errors="coerce").iloc[0]
    if pd.isna(num) or not np.isfinite(num):
        return None
    return [float(num)]


def parse_str_list(x: Any) -> Optional[List[str]]:
    val = _safe_literal_eval(x)
    if isinstance(val, list):
        return [str(v) for v in val]
    if isinstance(val, str):
        return [val]
    return None


def clean_values(values: List[Any]) -> List[float]:
    if not values:
        return []
    vals = pd.to_numeric(pd.Series(values), errors="coerce")
    return vals[vals.notna() & np.isfinite(vals)].astype(float).tolist()


def min_max_normalize(values: List[float]) -> List[float]:
    """Normalize a numeric list to [0, 1]."""
    if not values:
        return []

    vmin = float(min(values))
    vmax = float(max(values))
    if vmax == vmin:
        return [0.5 for _ in values]

    denom = vmax - vmin
    return [float((v - vmin) / denom) for v in values]


def normalize_values_and_refs(
    values: List[float],
    reproduced_value: float,
    reported_value: float,
) -> Tuple[List[float], float, float]:
    """Z-score normalize values and apply the same transform to the reference points."""
    if not values:
        return ([], reproduced_value, reported_value)

    arr = np.array(values, dtype=float)
    mean = float(np.mean(arr))
    std = float(np.std(arr))
    if std == 0:
        return ([0.0 for _ in values], 0.0, 0.0)

    z_vals = list((arr - mean) / std)
    z_reproduced = (reproduced_value - mean) / std
    z_reported = (reported_value - mean) / std
    return (z_vals, float(z_reproduced), float(z_reported))


def ensure_dirs(*paths: str) -> None:
    for p in paths:
        os.makedirs(repo_path(p), exist_ok=True)


def zscore_differences(values: List[float], reproduced_value: float) -> List[float]:
    """Compute z(y_permuted) - z(y_reproduced)."""
    return [float(v - reproduced_value) for v in values]


def empirical_p_values(values: List[float], observed: float) -> Tuple[float, float, float]:
    """Return (two-sided, left-tail, right-tail) empirical p-values.

    Uses a +1 correction: (count + 1) / (n + 1).
    """
    if not values:
        return (float("nan"), float("nan"), float("nan"))

    arr = np.array(values, dtype=float)
    n = arr.size
    center = float(np.mean(arr))

    obs_dev = abs(float(observed) - center)
    dev = np.abs(arr - center)
    p_two_sided = float((np.sum(dev >= obs_dev) + 1) / (n + 1))
    p_left = float((np.sum(arr <= observed) + 1) / (n + 1))
    p_right = float((np.sum(arr >= observed) + 1) / (n + 1))
    return (p_two_sided, p_left, p_right)


def t_test_p_values(values: List[float], observed: float) -> Tuple[float, float, float]:
    """Return one-sample t-test p-values (two-sided, left-tail, right-tail)."""
    if not values or len(values) < 2:
        return (float("nan"), float("nan"), float("nan"))

    arr = np.array(values, dtype=float)
    if not np.isfinite(arr).all():
        return (float("nan"), float("nan"), float("nan"))

    two_sided = float(ttest_1samp(arr, popmean=observed, alternative="two-sided").pvalue)
    left_tail = float(ttest_1samp(arr, popmean=observed, alternative="less").pvalue)
    right_tail = float(ttest_1samp(arr, popmean=observed, alternative="greater").pvalue)
    return (two_sided, left_tail, right_tail)


def get_reference_values(
    paper: str, metric: str, input_file: Optional[str] = None,
    wang_results_dir: str = "wang/plots",
) -> Tuple[float, float]:
    if paper == "wang" and canonical_metric(metric) != "score_difference":
        raise ValueError(f"No saved Wang comparison reference for metric {metric!r}")
    reproduced = load_reproduced_reference(paper, metric, input_file)
    comparison = (
        load_saved_wang_reference(wang_results_dir) if paper == "wang"
        else load_published_reference(paper, metric)
    )
    return reproduced, comparison


def load_wang_model_reproduced_value(
    results_dir: str,
    model: str,
    metric_row: str = "Score Differences",
) -> Optional[float]:
    """Read one reproduced value for a model from statistical_results_<model>.csv."""
    model_filename = model.lower().replace('.', '_').replace('-', '_')
    results_path = os.path.join(repo_path(results_dir), f'statistical_results_{model_filename}.csv')
    if not os.path.exists(repo_path(results_path)):
        print(f"Skipping reproduced setting for {model}: {results_path} not found.")
        return None

    results_df = pd.read_csv(repo_path(results_path))
    if "Metric" not in results_df.columns or "Value" not in results_df.columns:
        print(f"Skipping reproduced setting for {model}: missing Metric/Value columns.")
        return None

    matches = results_df[results_df['Metric'].eq(metric_row)]
    if matches.empty:
        print(f"Skipping reproduced setting for {model}: '{metric_row}' not found.")
        return None

    value = pd.to_numeric(matches['Value'].iloc[0], errors='coerce')
    if pd.isna(value) or not np.isfinite(value):
        print(f"Skipping reproduced setting for {model}: invalid '{metric_row}' value.")
        return None
    return float(value)


def parse_model_value_overrides(pairs: Optional[List[str]]) -> Dict[str, float]:
    """Parse --wang_model_values 'GPT-4=0.21' 'GPT-4o=0.18' into a dict.

    Keys are normalized via `_normalize_model_name` so matching is robust to
    punctuation/case differences.
    """
    overrides: Dict[str, float] = {}
    for item in pairs or []:
        if "=" not in item:
            raise ValueError(f"--wang_model_values entry must look like Model=value, got: {item!r}")
        model, raw = item.split("=", 1)
        norm_key = _normalize_model_name(model.strip())
        if not norm_key:
            raise ValueError(f"--wang_model_values entry has empty model name: {item!r}")
        overrides[norm_key] = float(raw.strip())
    return overrides


def _normalize_model_name(name: Any) -> str:
    """Loose key for matching model names across files ('GPT-4o' -> 'gpt4o')."""
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


def load_wang_model_values_from_input(input_file: str, metric: str) -> Dict[str, float]:
    """Read each model's seed-42 original-bundle result, without averaging noise runs."""
    df = pd.read_csv(repo_path(input_file))
    rows = df[
        df["Names"].eq("wang") & df["Jobs"].eq("wang")
        & df["Sensitive_Attribute_Vector"].eq("['Woman']")
        & df["Metric"].map(canonical_metric).eq(canonical_metric(metric))
    ]
    return {
        _normalize_model_name(model): load_reproduced_reference("wang", metric, input_file, model)
        for model in rows["Model"].unique()
    }


def populate_wang_model_reference_points(
    metric: str,
    results_dir: Optional[str] = None,
    flip_sign: bool = False,
    metric_row: str = "Score Differences",
    overrides: Optional[Dict[str, float]] = None,
    wang_input: Optional[str] = None,
) -> Dict[str, Tuple[float, float]]:
    """Resolve per-model references from explicit overrides or saved results."""
    metric_key = metric.lower().strip()
    reported = load_saved_wang_reference(results_dir or "wang/plots")
    overrides = overrides or {}

    keys: List[str] = []
    resolved_refs: Dict[str, Tuple[float, float]] = {}
    used_override_keys: set = set()
    for model in REPRODUCED_SETTING_MODELS:
        key = wang_model_key(model)
        model_norm = _normalize_model_name(model)

        value: Optional[float] = overrides.get(model_norm)
        source = "--wang_model_values"
        if value is not None:
            used_override_keys.add(model_norm)
        if value is None and results_dir:
            value = load_wang_model_reproduced_value(results_dir, model, metric_row)
            source = "statistical_results CSV"
        if value is None and wang_input:
            value = load_reproduced_reference("wang", metric_key, wang_input, model)
            source = "wang input file (original bundle)"
        if value is None or not np.isfinite(value):
            raise ValueError(
                f"No reference for Wang model {model}; supply its saved result or --wang_model_values."
            )

        value = float(value)
        if flip_sign and metric_key in {"score_difference", "score_diff", "wang_score_diff"}:
            value = -value

        resolved_refs[key] = (value, float(reported))
        keys.append(key)
        print(f"  Wang row '{model}': reproduced={value:.6g}  [{source}]")

    unmatched_override_keys = sorted(k for k in overrides.keys() if k not in used_override_keys)
    if unmatched_override_keys:
        print(
            "  WARNING -- some --wang_model_values entries did not match any configured "
            "REPRODUCED_SETTING_MODELS row:\n"
            "    " + ", ".join(unmatched_override_keys)
        )

    print(f"\nWang per-model rows: {len(keys)} of {len(REPRODUCED_SETTING_MODELS)} registered.")

    return resolved_refs


def build_color_map(order: List[str]) -> Dict[str, Any]:
    """Base papers keep their fixed colors; Wang model rows get a tab10 ramp."""
    colors: Dict[str, Any] = dict(PAPER_COLORS)
    model_keys = [k for k in order if k.startswith(WANG_MODEL_KEY_PREFIX)]
    if model_keys:
        ramp = plt.cm.tab10(np.linspace(0, 1, max(len(model_keys), 2)))
        for key, color in zip(model_keys, ramp):
            colors[key] = color
    return colors


def extract_armstrong_woman_value(row: pd.Series) -> Optional[float]:
    groups = row.get("Sensitive_Attribute_Vector", None)
    values = row.get("Result_Vector", None)
    if not isinstance(values, list) or len(values) == 0:
        return None

    if isinstance(groups, list):
        lower_groups = [str(g).strip().lower() for g in groups]
        if "woman" in lower_groups:
            idx = lower_groups.index("woman")
            if idx < len(values):
                return float(values[idx])
            return None
        return None

    if isinstance(groups, str):
        if groups.strip().lower() == "woman":
            return float(values[0])
        return None

    # If group info is missing, fallback to first value.
    return float(values[0])


def load_armstrong_values(input_file: str, metric: str) -> List[float]:
    df = pd.read_csv(repo_path(input_file))
    if "Metric" not in df.columns or "Result_Vector" not in df.columns:
        raise ValueError("Armstrong file does not contain required columns.")

    if "Random_State" in df.columns:
        random_states = pd.to_numeric(df["Random_State"], errors="coerce")
        df = df[random_states == 42.0].copy()

    dedupe_cols = [c for c in ["Names", "Jobs", "Model", "Metric", "Sensitive_Attribute_Vector"] if c in df.columns]
    if dedupe_cols:
        df = df.drop_duplicates(subset=dedupe_cols, keep="first").copy()

    df = df[df["Metric"] == metric].copy()
    if df.empty:
        return []

    df["Result_Vector"] = df["Result_Vector"].apply(parse_float_list)
    if "Sensitive_Attribute_Vector" in df.columns:
        df["Sensitive_Attribute_Vector"] = df["Sensitive_Attribute_Vector"].apply(parse_str_list)

    values: List[float] = []
    for _, row in df.iterrows():
        v = extract_armstrong_woman_value(row)
        if v is not None:
            values.append(v)
    return clean_values(values)


def parse_result_vector(x: Any) -> Optional[List[float]]:
    if pd.isna(x):
        return None
    if isinstance(x, list):
        return clean_values(x)
    try:
        parsed = ast.literal_eval(str(x))
        if isinstance(parsed, list):
            return clean_values(parsed)
        return clean_values([parsed])
    except Exception:
        return None


def maybe_flip_wang_value(metric: Any, value: Any) -> Optional[float]:
    metric_str = str(metric).strip().lower()
    val = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(val) or not np.isfinite(val):
        return None
    fval = float(val)
    if metric_str in {"score_difference", "score_diff", "wang_score_diff"}:
        return -fval
    return fval


def load_wang_values_by_model(input_file: str, metric: str) -> Dict[str, List[float]]:
    """Load Wang permutation values grouped by model.

    Returns a dict keyed by normalized model name (e.g., 'gpt4o').
    """
    df = pd.read_csv(repo_path(input_file))
    metric_key = metric.strip().lower()

    if "Random_State" in df.columns:
        random_states = pd.to_numeric(df["Random_State"], errors="coerce")
        df = df[random_states == 42.0].copy()

    # Detailed format
    if {"model", "name_bundle", "job_bundle"}.issubset(df.columns) and metric_key in df.columns:
        out: Dict[str, List[float]] = {}
        for model_name, group in df.groupby("model"):
            vals = [maybe_flip_wang_value(metric_key, v) for v in group[metric_key].tolist()]
            cleaned = clean_values([v for v in vals if v is not None])
            if cleaned:
                out[_normalize_model_name(model_name)] = cleaned
        return out

    # Aggregate format
    required = {"Model", "Names", "Jobs", "Metric", "Result_Vector"}
    if required.issubset(set(df.columns)):
        sub = df[df["Metric"].astype(str).str.lower() == metric_key].copy()
        if sub.empty:
            return {}

        sub["parsed_result"] = sub["Result_Vector"].apply(parse_result_vector)
        sub = sub[sub["parsed_result"].notna()].copy()
        sub["value"] = sub["parsed_result"].apply(lambda x: x[0] if isinstance(x, list) and len(x) > 0 else np.nan)
        sub["value"] = sub.apply(lambda row: maybe_flip_wang_value(row.get("Metric"), row.get("value")), axis=1)
        sub = sub.dropna(subset=["value"])
        if sub.empty:
            return {}

        out: Dict[str, List[float]] = {}
        for model_name, group in sub.groupby("Model"):
            cleaned = clean_values(group["value"].tolist())
            if cleaned:
                out[_normalize_model_name(model_name)] = cleaned
        return out

    return {}


def load_wang_values(input_file: str, metric: str) -> List[float]:
    df = pd.read_csv(repo_path(input_file))
    metric_key = metric.strip().lower()

    if "Random_State" in df.columns:
        random_states = pd.to_numeric(df["Random_State"], errors="coerce")
        df = df[random_states == 42.0].copy()

    detailed_cols = [
        "impact_ratio_protect_rank",
        "impact_ratio_protect_score",
        "selection_rate_protect_rank",
        "selection_rate_protect_score",
    ]
    if all(c in df.columns for c in ["model", "name_bundle", "job_bundle"]) and any(c in df.columns for c in detailed_cols):
        if metric_key not in df.columns:
            return []
        vals = [maybe_flip_wang_value(metric_key, v) for v in df[metric_key].dropna().tolist()]
        return clean_values([v for v in vals if v is not None])

    required = {"Model", "Names", "Jobs", "Metric", "Result_Vector"}
    if required.issubset(set(df.columns)):
        df = df.copy()
        dedupe_cols = [c for c in ["Model", "Names", "Jobs", "Metric"] if c in df.columns]
        if dedupe_cols:
            df = df.drop_duplicates(subset=dedupe_cols, keep="first").copy()
        df = df[df["Metric"].astype(str).str.lower() == metric_key].copy()
        if df.empty:
            return []

        df["parsed_result"] = df["Result_Vector"].apply(parse_result_vector)
        df = df[df["parsed_result"].notna()].copy()
        df["value"] = df["parsed_result"].apply(lambda x: x[0] if isinstance(x, list) and len(x) > 0 else np.nan)
        df["value"] = df.apply(lambda row: maybe_flip_wang_value(row.get("Metric"), row.get("value")), axis=1)
        return clean_values(df["value"].tolist())

    raise ValueError(f"Unrecognized Wang input format in {input_file}")


def load_yin_values(input_file: str, metric: str) -> List[float]:
    # Matches the loading style used in plot_yin_all.py
    df = pd.read_csv(repo_path(input_file), index_col=[0, 1]).reset_index()
    num_cols = len(df.columns)

    if num_cols >= 10:
        base_cols = [
            "gender",
            "top",
            "top_og",
            "selection_rate",
            "disparate_impact_ratio",
            "job",
            "model",
            "rank",
            "name_bundle",
            "job_bundle",
        ]
        if num_cols > 10:
            df.columns = base_cols + df.columns[10:].tolist()
        else:
            df.columns = base_cols


    if metric not in df.columns:
        return []

    # Keep female rows (same convention as existing code)
    if "gender" in df.columns:
        mask = (df["gender"] == "W") | df["gender"].astype(str).str.endswith("_W")
        df = df[mask].copy()

    if "name_bundle" in df.columns:
        df = df[df["name_bundle"] != "wang"].copy()

    dedupe_columns = [
        column
        for column in ("gender", "job", "model", "rank", "name_bundle", "job_bundle")
        if column in df.columns
    ]
    if dedupe_columns:
        df = df.drop_duplicates(subset=dedupe_columns, keep="first").copy()

    return clean_values(df[metric].dropna().tolist())


def plot_relative_histogram(
    values_by_paper: Dict[str, List[float]],
    refs_by_paper: Dict[str, Tuple[float, float]],
    order: List[str],
    labels: List[str],
    out_path: str,
    title: str,
    num_bins: int = 100,
    xlim: Optional[Tuple[float, float]] = None,
) -> None:
    """Plot z-score differences as overlaid histograms with shared axis ranges."""
    color_map = build_color_map(order)

    rel_data_dict: Dict[str, List[float]] = {}
    normalized_refs_by_paper: Dict[str, Tuple[float, float]] = {}
    for paper in order:
        vals = values_by_paper.get(paper, [])
        reproduced, reported = refs_by_paper[paper]

        # Normalize CSV values before computing relative differences.
        norm_vals, norm_reproduced, norm_reported = normalize_values_and_refs(
            vals,
            reproduced,
            reported,
        )

        normalized_refs_by_paper[paper] = (norm_reproduced, norm_reported)
        rel_vals = zscore_differences(norm_vals, norm_reproduced)
        rel_data_dict[paper] = rel_vals

    if xlim is None:
        all_rel_vals = [x for vals in rel_data_dict.values() for x in vals]
        if all_rel_vals:
            xmin = min(float(min(all_rel_vals)), 0.0)
            xmax = max(float(max(all_rel_vals)), 0.0)
            pad = max((xmax - xmin) * 0.08, 0.2)
            xlim = (xmin - pad, xmax + pad)
        else:
            xlim = (-1.0, 1.0)

    fig, ax = plt.subplots(figsize=(12, 6))

    ymax_global = 0
    for paper, label in zip(order, labels):
        rel_vals = rel_data_dict.get(paper, [])
        if len(rel_vals) > 0:
            counts, bins, patches = ax.hist(
                rel_vals,
                bins=num_bins,
                color=color_map.get(paper, "tab:gray"),
                edgecolor="black",
                linewidth=0.5,
                alpha=0.6,
                label=label.replace("\n", " "),
            )
            ymax_global = max(ymax_global, counts.max() if len(counts) > 0 else 1.0)

    ax.axvline(0.0, color="black", linestyle="--", linewidth=2.0, zorder=5, label="Reproduced Implementation")
    ax.set_xlim(xlim)
    ax.set_ylim(0, (ymax_global if ymax_global else 1.0) * 1.1)
    ax.set_xlabel("$z(y_{permuted}) - z(y_{reproduced})$")
    ax.set_ylabel("Frequency")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.18)
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.legend(loc="best", ncol=2)

    fig.tight_layout()
    ensure_dirs(os.path.dirname(out_path) or ".")
    fig.savefig(repo_path(out_path), dpi=180)
    plt.close(fig)


def plot_relative_violin(
    values_by_paper: Dict[str, List[float]],
    refs_by_paper: Dict[str, Tuple[float, float]],
    order: List[str],
    labels: List[str],
    out_path: str,
    title: str,
    xlim: Optional[Tuple[float, float]] = None,
    full_width: bool = False,
    debug_refs_and_zscores: bool = False,
) -> None:
    color_map = build_color_map(order)
    y_positions = np.arange(1, len(order) + 1)

    rel_data: List[List[float]] = []
    normalized_refs_by_paper: Dict[str, Tuple[float, float]] = {}
    for paper in order:
        vals = values_by_paper.get(paper, [])
        reproduced, reported = refs_by_paper[paper]

        # Normalize CSV values before computing relative differences.
        norm_vals, norm_reproduced, norm_reported = normalize_values_and_refs(
            vals,
            reproduced,
            reported,
        )

        normalized_refs_by_paper[paper] = (norm_reproduced, norm_reported)
        rel_vals = zscore_differences(norm_vals, norm_reproduced)
        rel_data.append(rel_vals)

        if debug_refs_and_zscores:
            print(
                f"DEBUG [{paper}] refs_by_paper -> reproduced={reproduced:.10g}, reported={reported:.10g}"
            )
            print(
                f"DEBUG [{paper}] normalized refs -> z_reproduced={norm_reproduced:.10g}, z_reported={norm_reported:.10g}"
            )
            print(f"DEBUG [{paper}] z(y_permuted) list (n={len(norm_vals)}): {norm_vals}")
            print(f"DEBUG [{paper}] z(y_permuted)-z(y_reproduced) list (n={len(rel_vals)}): {rel_vals}")

    # Auto-fit the x range across every row, including rows whose reference point
    # shifts them far from zero. A hardcoded window silently clips those violins.
    if xlim is None:
        all_rel_vals = [x for vals in rel_data for x in vals]
        if all_rel_vals:
            xmin = float(min(all_rel_vals))
            xmax = float(max(all_rel_vals + [0.0]))
            xmin = min(xmin, 0.0)
            pad = max((xmax - xmin) * 0.08, 0.2)
            xlim = (xmin - pad, xmax + pad)
        else:
            xlim = (-1.0, 1.0)
    print(f"Violin x-limits: ({xlim[0]:.4g}, {xlim[1]:.4g})")

    # Grow the figure with the number of rows so 13 violins stay readable.
    many_rows = len(order) > 4
    fig_height = max(5.0, 0.85 * len(order) + 1.5)
    fig_width = (17.0 if many_rows else 14.0) if full_width else (11.0 if many_rows else 9.0)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    non_empty_positions = [pos for pos, vals in zip(y_positions, rel_data) if len(vals) > 0]
    non_empty_data = [vals for vals in rel_data if len(vals) > 0]
    non_empty_papers = [paper for paper, vals in zip(order, rel_data) if len(vals) > 0]

    if non_empty_data:
        half_width = 0.72 / 2.0
        for y_pos, vals, paper in zip(non_empty_positions, non_empty_data, non_empty_papers):
            arr = np.asarray(vals, dtype=float)
            arr = arr[np.isfinite(arr)]
            if arr.size == 0:
                continue

            # Degenerate case: all values identical.
            if np.allclose(arr, arr[0]):
                x0 = float(arr[0])
                ax.plot([x0, x0], [y_pos - half_width, y_pos + half_width], color="black", linewidth=1.0, alpha=0.5)
                continue

            arr_std = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
            spread = arr_std if np.isfinite(arr_std) and arr_std > 0 else max(abs(float(arr.mean())), 1.0) * 0.05
            x_min = float(np.min(arr)) - VIOLIN_TAIL_STD_MULTIPLIER * spread
            x_max = float(np.max(arr)) + VIOLIN_TAIL_STD_MULTIPLIER * spread
            x_eval = np.linspace(x_min, x_max, VIOLIN_POINTS)

            kde = gaussian_kde(arr, bw_method=VIOLIN_BW_METHOD)
            density = kde(x_eval)
            dmax = float(np.max(density))
            if dmax <= 0 or not np.isfinite(dmax):
                continue

            y_offset = (density / dmax) * half_width
            y_upper = y_pos + y_offset
            y_lower = y_pos - y_offset

            color = color_map.get(paper, "tab:gray")
            ax.fill_between(x_eval, y_lower, y_upper, facecolor=color, edgecolor="black", linewidth=1.0, alpha=0.40)

    for y, paper in zip(y_positions, order):
        ax.scatter(0.0, y, marker="x", s=120, color="black", linewidths=2.2, zorder=4)

    ax.axvline(0.0, color="gray", linestyle="--", linewidth=1.6, alpha=0.9)
    ax.grid(axis="x", alpha=0.25)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels)
    ax.tick_params(axis="y", labelsize=max(11, TICK_FONT_SIZE - 8) if many_rows else TICK_FONT_SIZE)
    ax.set_xlim(xlim)
    ax.set_xlabel("$z(y_{permuted}) - z(y_{reproduced})$")
    ax.set_ylabel("Experimental Framework")
    ax.set_title(title, fontsize=(TITLE_FONT_SIZE - 6) if many_rows else TITLE_FONT_SIZE, wrap=True)

    from matplotlib.lines import Line2D

    legend_items = [
        Line2D([0], [0], marker="x", color="black", linestyle="None", markersize=9, markeredgewidth=2, label="Our Reproduced\nImplementation"),
    ]
    ax.legend(handles=legend_items, loc="best")

    ax.set_xlim(xlim)

    if full_width:
        # Reserve only what's needed for y labels and let the x-axis occupy
        # most of the canvas width.
        fig.subplots_adjust(left=0.19 if many_rows else 0.16, right=0.995, top=0.92, bottom=0.12)
    else:
        fig.tight_layout()
    ensure_dirs(os.path.dirname(out_path) or ".")
    fig.savefig(repo_path(out_path), dpi=180)
    plt.close(fig)


def main() -> None:
    cwd = str(REPO_ROOT)
    default_results_dir = os.path.join(repo_path(cwd), "results")

    parser = argparse.ArgumentParser(
        description="Plot Armstrong, Wang, and Yin on a shared relative-performance axis."
    )

    parser.add_argument("--armstrong_input", type=str, default=os.path.join(repo_path(default_results_dir), "armstrong_sampling_downsampling_analysis_original.csv"))
    parser.add_argument("--wang_input", type=str, default=os.path.join(repo_path(default_results_dir), "wang_sampling_downsampling_analysis_original.csv"))
    parser.add_argument("--yin_input", type=str, default=os.path.join(repo_path(default_results_dir), "yin_sampling_downsampling_analysis_original.csv"))
    for framework in ("armstrong", "wang", "yin"):
        parser.add_argument(
            f"--{framework}_reference_input",
            help="Baseline aggregate CSV, if different from the distribution input.",
        )

    parser.add_argument("--armstrong_metric", type=str, default="regression_coefficients")
    parser.add_argument("--wang_metric", type=str, default="score_difference")
    parser.add_argument("--yin_metric", type=str, default="selection_rate")

    parser.add_argument(
        "--wang_model_results_dir",
        type=str,
        default=os.path.join(repo_path(cwd), "wang", "plots"),
        help="Directory holding statistical_results_<model>.csv files used as per-model Wang reference points.",
    )
    parser.add_argument(
        "--wang_model_metric_row",
        type=str,
        default="Score Differences",
        help="Value of the 'Metric' column to pull from each statistical_results_<model>.csv.",
    )
    parser.add_argument(
        "--wang_model_flip_sign",
        action="store_true",
        help="Negate the per-model reproduced values so they match the sign convention of the permuted Wang values.",
    )
    parser.add_argument(
        "--wang_model_values",
        type=str,
        nargs="*",
        default=None,
        help="Explicit per-model reproduced values, e.g. --wang_model_values 'GPT-4=0.21' 'GPT-4o=0.18'.",
    )
    parser.add_argument(
        "--no_wang_models",
        action="store_true",
        help="Plot only the original three rows (Armstrong, Wang, Yin).",
    )
    parser.add_argument("--xlim_min", type=float, default=None, help="Left x-limit. Omit both xlim args to auto-fit the data.")
    parser.add_argument("--xlim_max", type=float, default=None, help="Right x-limit. Omit both xlim args to auto-fit the data.")

    parser.add_argument("--output", type=str, default=os.path.join(repo_path(default_results_dir), "figs", "relative", "relative_violin.png"))
    parser.add_argument("--title", type=str, default="Relative Performance")
    parser.add_argument("--plot_type", type=str, choices=["violin", "histogram"], default="violin", help="Type of plot to generate")
    parser.add_argument("--num_bins", type=int, default=100, help="Number of bins for histogram (only used with --plot_type histogram)")
    parser.add_argument(
        "--full_width",
        action="store_true",
        help="Expand the horizontal plotting area so the x-axis uses nearly the full page width.",
    )
    parser.add_argument(
        "--debug_refs_and_zscores",
        action="store_true",
        help="Print per-row stored reference values and computed z-score lists used for violin plotting.",
    )
    args = parser.parse_args()

    # If the user activated --wang_model_values (even with an empty list),
    # save to a distinct filename so runs are clearly separated.
    if args.wang_model_values is not None:
        base, ext = os.path.splitext(args.output)
        args.output = f"{base}_wang_model_values{ext}"
        print(f"Output adjusted for --wang_model_values: {args.output}")

    arm_repro, arm_report = get_reference_values("armstrong", args.armstrong_metric, args.armstrong_reference_input or args.armstrong_input)
    wang_repro, wang_report = get_reference_values("wang", args.wang_metric, args.wang_reference_input or args.wang_input, args.wang_model_results_dir)
    yin_repro, yin_report = get_reference_values("yin", args.yin_metric, args.yin_reference_input or args.yin_input)

    armstrong_values = load_armstrong_values(args.armstrong_input, args.armstrong_metric)
    wang_values = load_wang_values(args.wang_input, args.wang_metric)
    wang_values_by_model = load_wang_values_by_model(args.wang_input, args.wang_metric)
    yin_values = load_yin_values(args.yin_input, args.yin_metric)

    order: List[str] = ["armstrong", "wang", "yin"]
    labels: List[str] = ["Armstrong\net al.", "Wang et al.", "Yin et al."]

    raw_values_by_paper: Dict[str, List[float]] = {
        "armstrong": armstrong_values,
        "wang": wang_values,
        "yin": yin_values,
    }
    refs_by_paper: Dict[str, Tuple[float, float]] = {
        "armstrong": (arm_repro, arm_report),
        "wang": (wang_repro, wang_report),
        "yin": (yin_repro, yin_report),
    }
    metric_by_paper: Dict[str, str] = {
        "armstrong": args.armstrong_metric,
        "wang": args.wang_metric,
        "yin": args.yin_metric,
    }

    # Add one Wang row per reported model. Same permutation distribution,
    # different reproduced reference point. These rows are always added.
    if not args.no_wang_models:
        print("\nResolving per-model Wang reference points:")
        wang_model_refs = populate_wang_model_reference_points(
            metric=args.wang_metric,
            results_dir=args.wang_model_results_dir,
            flip_sign=args.wang_model_flip_sign,
            metric_row=args.wang_model_metric_row,
            overrides=parse_model_value_overrides(args.wang_model_values),
            wang_input=args.wang_reference_input or args.wang_input,
        )
        for model in REPRODUCED_SETTING_MODELS:
            key = wang_model_key(model)
            order.append(key)
            labels.append(wang_model_label(model))
            model_values = wang_values_by_model.get(_normalize_model_name(model), wang_values)
            raw_values_by_paper[key] = model_values
            # Important: grab model-specific refs resolved from
            # REPRODUCED_SETTING_MODELS / --wang_model_values so each violin is
            # computed relative to that model's reproduced value.
            refs_by_paper[key] = wang_model_refs[key]
            metric_by_paper[key] = args.wang_metric

    values_by_paper = raw_values_by_paper

    for paper in order:
        print(f"{paper}: loaded {len(values_by_paper.get(paper, []))} values")

    metric_label = WANG_METRIC_LABELS.get(args.wang_metric.strip().lower(), args.wang_metric)
    print(f"\nWang metric: {metric_label}")

    print("\nEmpirical p-values on raw values (before any z-score normalization):")
    for paper in order:
        vals = raw_values_by_paper.get(paper, [])
        reproduced, _ = refs_by_paper[paper]
        p_two, p_left, p_right = empirical_p_values(vals, reproduced)
        if vals:
            print(
                f"- {paper} [{metric_by_paper[paper]}]: "
                f"reproduced={reproduced:.6g}, "
                f"mean_perm={np.mean(vals):.6g}, "
                f"p_two_sided={p_two:.4g}, p_left={p_left:.4g}, p_right={p_right:.4g}"
            )
        else:
            print(f"- {paper} [{metric_by_paper[paper]}]: no values, p-values unavailable")

    print("\nOne-sample t-test p-values on raw values (H0: mean permutation value = reproduced value):")
    for paper in order:
        vals = raw_values_by_paper.get(paper, [])
        reproduced, _ = refs_by_paper[paper]
        p_two, p_left, p_right = t_test_p_values(vals, reproduced)
        if vals and len(vals) >= 2:
            print(
                f"- {paper} [{metric_by_paper[paper]}]: "
                f"reproduced={reproduced:.6g}, "
                f"mean_perm={np.mean(vals):.6g}, "
                f"t_p_two_sided={p_two:.4g}, t_p_left={p_left:.4g}, t_p_right={p_right:.4g}"
            )
        else:
            print(f"- {paper} [{metric_by_paper[paper]}]: insufficient values for t-test")

    xlim: Optional[Tuple[float, float]] = None
    if args.xlim_min is not None and args.xlim_max is not None:
        xlim = (args.xlim_min, args.xlim_max)

    print(f"\nPlotting {len(order)} rows: {labels}")

    if args.plot_type == "histogram":
        plot_relative_histogram(
            values_by_paper, refs_by_paper, order, labels, args.output, args.title, args.num_bins, xlim
        )
    else:
        plot_relative_violin(
            values_by_paper,
            refs_by_paper,
            order,
            labels,
            args.output,
            args.title,
            xlim,
            args.full_width,
            args.debug_refs_and_zscores,
        )
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
    
"""Load plot references from experiment outputs and documented publication values."""

import ast
import warnings

import numpy as np
import pandas as pd

from utils.paths import repo_path


PUBLISHED_REFERENCES = "configs/published_references.csv"
REPRODUCED_SETTING_LABEL = "Reproduced Setting (GPT-4o-mini)"


def reported_setting_label(model: str) -> str:
    aliases = {"gpt-3.5-turbo": "GPT-3.5", "gpt-3.5": "GPT-3.5",
               "gpt-4": "GPT-4", "gpt-4o": "GPT-4o"}
    name = aliases.get(model.strip().lower(), model.strip())
    return f"Reported Setting ({name})"


def published_setting_label(framework: str, metric: str) -> str:
    df = pd.read_csv(repo_path(PUBLISHED_REFERENCES))
    rows = df[df["framework"].eq(framework) & df["metric"].eq(canonical_metric(metric))]
    if len(rows) > 1:
        rows = rows[rows["model"].fillna("").astype(str).str.strip().eq("")]
    if len(rows) != 1:
        raise ValueError(f"Expected one published {framework}/{metric} reference")
    model = rows.iloc[0].get("model")
    return reported_setting_label(str(model) if pd.notna(model) and str(model).strip()
                                  else "model unspecified")


def canonical_metric(metric: str) -> str:
    key = metric.strip().lower()
    return "score_difference" if key in {"score_diff", "wang_score_diff"} else key


def _scalar(value: object, source: str) -> float:
    if isinstance(value, str):
        value = ast.literal_eval(value)
    if isinstance(value, (list, tuple)):
        if len(value) != 1:
            raise ValueError(f"{source}: expected a single result, got {value!r}")
        value = value[0]
    if not isinstance(value, (int, float, np.number)) or not np.isfinite(value):
        raise ValueError(f"{source}: invalid reference value {value!r}")
    return float(value)


def load_reproduced_reference(
    framework: str,
    metric: str,
    input_file: str | None = None,
    model: str = "gpt-4o-mini",
) -> float:
    """Read an original-bundle result, not a mean over perturbations/noise runs.

    Prefer seed 42. For legacy archives without baseline seed labels, use the
    first seedless matching row and warn explicitly. Yin requires the aggregate
    women's rank-1 row. Conflicting labelled baselines are an error.
    """
    metric = canonical_metric(metric)
    source = repo_path(
        input_file or f"results/{framework}_sampling_downsampling_analysis_original.csv"
    )
    df = pd.read_csv(source)
    if framework == "yin":
        required = {"gender", "name_bundle", "job_bundle", "model", metric}
        if not required.issubset(df.columns):
            raise ValueError(f"{source}: baseline CSV is missing {sorted(required - set(df.columns))}")
        rows = df[
            df["gender"].eq("W") & df["name_bundle"].eq("yin")
            & df["job_bundle"].eq("yin")
            & df["model"].astype(str).str.strip().str.casefold().eq(model.strip().casefold())
        ]
        if "job" in rows:
            rows = rows[rows["job"].eq("ALL")]
        if "rank" in rows:
            rows = rows[pd.to_numeric(rows["rank"], errors="coerce").eq(1)]
        value_column = metric
    elif framework in {"armstrong", "wang"}:
        required = {"Names", "Jobs", "Model", "Metric", "Sensitive_Attribute_Vector", "Result_Vector"}
        if not required.issubset(df.columns):
            raise ValueError(
                f"{source}: baseline requires an aggregate CSV with {sorted(required)}; "
                "use the reference_input option if plotting detailed data."
            )
        rows = df[
            df["Names"].eq(framework) & df["Jobs"].eq(framework)
            & df["Model"].astype(str).str.strip().str.casefold().eq(model.strip().casefold())
            & df["Metric"].map(canonical_metric).eq(metric)
            & df["Sensitive_Attribute_Vector"].eq("['Woman']")
        ]
        value_column = "Result_Vector"
    else:
        raise ValueError(f"Unknown reference framework: {framework}")
    if rows.empty:
        raise ValueError(f"{source}: no original-bundle {framework}/{model}/{metric} reference")

    seed_column = next(
        (column for column in ("Random_State", "random_state", "Random State") if column in rows),
        None,
    )
    seeds = pd.to_numeric(rows[seed_column], errors="raise") if seed_column else None
    if seeds is not None and seeds.eq(42).any():
        rows = rows[seeds.eq(42)]
        values = [_scalar(value, source) for value in rows[value_column]]
        if not np.allclose(values, values[0], rtol=0, atol=1e-12):
            raise ValueError(f"{source}: conflicting seed-42 references for {framework}/{model}/{metric}")
    elif seeds is None or seeds.isna().any():
        if seeds is not None:
            rows = rows[seeds.isna()]
        warnings.warn(
            f"{source}: no seed-42 label for {framework}/{model}/{metric}; "
            f"using the first of {len(rows)} seedless original-bundle rows.",
            stacklevel=2,
        )
    else:
        raise ValueError(f"{source}: no seed-42 or seedless baseline for {framework}/{model}/{metric}")

    value = _scalar(rows[value_column].iloc[0], source)
    # Wang aggregate results store privilege - protect; plots use protect - privilege.
    if framework == "wang" and metric == "score_difference":
        value = -value
    print(f"Reproduced reference {framework}/{model}/{metric}: {value:.12g} [{source}]")
    return value


def load_saved_wang_reference(
    results_dir: str = "wang/plots", model: str = "gpt-4o",
    metric_row: str = "Score Differences",
) -> float:
    filename = model.lower().replace(".", "_").replace("-", "_")
    source = repo_path(f"{results_dir}/statistical_results_{filename}.csv")
    df = pd.read_csv(source)
    rows = df[df["Metric"].eq(metric_row)]
    if len(rows) != 1:
        raise ValueError(f"{source}: expected one {metric_row!r} row, found {len(rows)}")
    value = _scalar(rows["Value"].iloc[0], source)
    print(f"Saved Wang reference {model}/{metric_row}: {value:.12g} [{source}]")
    return value


def load_published_reference(
    framework: str, metric: str, source: str = PUBLISHED_REFERENCES,
    model: str | None = None,
) -> float:
    df = pd.read_csv(repo_path(source))
    rows = df[df["framework"].eq(framework) & df["metric"].eq(canonical_metric(metric))]
    if model is not None:
        if "model" not in rows:
            raise ValueError(f"{source}: published references have no model column")
        model_names = rows["model"].fillna("").astype(str).str.strip().str.casefold()
        rows = rows[model_names.eq(model.strip().casefold())]
    elif len(rows) > 1 and "model" in rows:
        rows = rows[rows["model"].fillna("").astype(str).str.strip().eq("")]
    if len(rows) != 1:
        suffix = f" for model {model}" if model is not None else ""
        raise ValueError(f"{source}: expected one published {framework}/{metric} reference{suffix}")
    row = rows.iloc[0]
    return _scalar(row["value"], source)

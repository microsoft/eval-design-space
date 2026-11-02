#!/usr/bin/env python3
"""Meta-regression analysis for selection rates.

Filters to rows where df['gender']=='W', one-hot encodes name/job/model (limited
to provided sets), evaluates predictive performance with k-fold cross-validation
on the full filtered dataset, and fits a final model on the full dataset to
print coefficients and generate coefficient plots.
"""
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.paths import REPO_ROOT, repo_path

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import statsmodels.api as sm
from statistics import NormalDist
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.inspection import permutation_importance
from sklearn.base import BaseEstimator, RegressorMixin



# Match styling from plot_relative.py
TITLE_FONT_SIZE = 25
LABEL_FONT_SIZE = 13
LEGEND_FONT_SIZE = 15
TICK_FONT_SIZE = 13
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


NAMES = ['armstrong', 'rozado', 'wen', 'wang', 'karvonen', 'zollo', 'yin', 'gaeb', 'lippens']
JOBS = ['armstrong', 'rozado', 'wen', 'wang', 'karvonen', 'zollo', 'yin']
YIN_MODELS = [
    # 'gpt-4o-2024-11-20',
    'gpt-4o-mini',
    # 'mistral-7b-v0.2',
    'mistral-7b-v0.3',
    'meta-llama-3.1-8b-instruct-turbo',
    'meta-llama-3.3-70b-instruct-turbo',
    'mistral-small-24b',
    'gpt-5-nano-2025-08-07',
]

ARMSTRONG_MODELS = [
    'gpt-4o-2024-11-20',
    'mistral-7b-v0.3',
    'meta-llama-3.1-8b-instruct-turbo',
    'meta-llama-3.3-70b-instruct-turbo',
    'mistral-small-24b',
    'gpt-5-nano-2025-08-07',
]

WANG_MODELS = [
    # 'gpt-4o-2024-11-20', ## ? 
    'meta-llama-3.1-8b-instruct-turbo',
    'meta-llama-3.3-70b-instruct-turbo',
    'mistral-small-24b',
    'gpt-4o-mini', ## ? 
    'gpt-5-nano-2025-08-07',
    'Mistral-7B-Instruct-v0.3',
    'Meta-Llama-3.1-8B-Instruct-Turbo'
]
       

EXCLUDED_NAME_BUNDLES = {'seshadri'}
EXCLUDED_MODEL_PATTERNS = ('job-resume', 'job_resume')



# Name of the synthetic control feature added for permutation importance only.
RANDOM_FEATURE = 'random_num'


def ensure_cols(df, col_prefix, values):
    """Create one-hot columns with prefix for each value in values.

    Returns list of created column names.
    """
    cols = []
    for v in values:
        cname = f"{col_prefix}_{v}"
        df[cname] = (df[col_prefix.replace('name_', 'name_bundle').replace('job_', 'job_bundle').replace('model_', 'model')] == v).astype(int)
        cols.append(cname)
    return cols


def build_design_matrix(df, exp_framework='yin', include_interactions=False):
    # create dummies limited to the provided lists
    name_cols = []
    if exp_framework == 'yin':
        framework_to_name = {'name': 'name_bundle', 'job': 'job_bundle', 'model': 'model'}
    elif exp_framework == 'wang':
        framework_to_name = {'name': 'Names', 'job': 'Jobs', 'model': 'Model'}
    else:
        framework_to_name = {'name': 'Names', 'job': 'Jobs', 'model': 'Model'}

    for n in NAMES:
        cname = f"name_{n}"
        df[cname] = (df[framework_to_name['name']] == n).astype(int)
        name_cols.append(cname)

    job_cols = []
    for j in JOBS:
        cname = f"job_{j}"
        df[cname] = (df[framework_to_name['job']] == j).astype(int)
        job_cols.append(cname)

    model_cols = []
    if exp_framework == 'yin':
        MODELS = YIN_MODELS
    elif exp_framework == 'wang':
        MODELS = WANG_MODELS
    else:
        MODELS = ARMSTRONG_MODELS
    for m in MODELS:
        cname = f"model_{m}"
        df[cname] = (df[framework_to_name['model']] == m).astype(int)
        model_cols.append(cname)

    # Choose reference (to be dropped) for name and job groups based on exp_framework
    # if the corresponding category exists; otherwise drop the first
    ref_name_val = exp_framework if exp_framework in NAMES else name_cols[0].replace('name_', '')
    ref_job_val = exp_framework if exp_framework in JOBS else job_cols[0].replace('job_', '')
    drop_name = f"name_{ref_name_val}"
    drop_job = f"job_{ref_job_val}"
    # For models, drop the first model as reference
    drop_model = model_cols[0]

    main_cols = name_cols + job_cols + model_cols
    X_cols = [c for c in main_cols if c not in (drop_name, drop_job, drop_model)]

    interaction_cols = []
    if include_interactions:
        kept_name_cols = [c for c in name_cols if c != drop_name]
        kept_job_cols = [c for c in job_cols if c != drop_job]
        kept_model_cols = [c for c in model_cols if c != drop_model]

        # Two-way interactions
        for ncol in kept_name_cols:
            for jcol in kept_job_cols:
                iname = f"int_{ncol}__X__{jcol}"
                df[iname] = df[ncol] * df[jcol]
                interaction_cols.append(iname)

        for ncol in kept_name_cols:
            for mcol in kept_model_cols:
                iname = f"int_{ncol}__X__{mcol}"
                df[iname] = df[ncol] * df[mcol]
                interaction_cols.append(iname)

        for jcol in kept_job_cols:
            for mcol in kept_model_cols:
                iname = f"int_{jcol}__X__{mcol}"
                df[iname] = df[jcol] * df[mcol]
                interaction_cols.append(iname)

        X_cols = X_cols + interaction_cols

    all_cols = main_cols + interaction_cols

    X = df[X_cols].astype(float)
    # add intercept
    X = sm.add_constant(X)
    return X, X_cols, all_cols, (drop_name, drop_job, drop_model)


def gaussian_rank_transform(series):
    """Transform a numeric series to Gaussian scores via rank-normalization."""
    s = pd.Series(series).astype(float)
    n = len(s)
    if n == 0:
        return s
    ranks = s.rank(method='average')
    u = (ranks - 0.5) / n
    inv_cdf = np.vectorize(NormalDist().inv_cdf)
    return pd.Series(inv_cdf(u), index=s.index)


def apply_ablation_to_target(y, ablation='none', truncate_q=0.01, ref_series=None, exp_framework='yin'):
    """Apply target-space ablations before model fitting."""
    y_out = pd.Series(y).astype(float).copy()

    if ablation in ('logistic', 'directionality'):
        # Predict binary direction: Yin targets are rates, so threshold at 0.5;
        # Armstrong/Wang targets are signed quantities, so threshold at 0.
        threshold = 0.5 if exp_framework == 'yin' else 0.0
        y_out = (y_out > threshold).astype(int)
    elif ablation == 'copula':
        y_out = gaussian_rank_transform(y_out)
    elif ablation == 'truncate':
        ref = pd.Series(ref_series).astype(float) if ref_series is not None else y_out
        lo = ref.quantile(truncate_q)
        hi = ref.quantile(1.0 - truncate_q)
        y_out = y_out.clip(lower=lo, upper=hi)
    elif ablation == 'standardize':
        ref = pd.Series(ref_series).astype(float) if ref_series is not None else y_out
        mu = ref.mean()
        sigma = ref.std(ddof=0)
        if sigma > 0:
            y_out = (y_out - mu) / sigma
        else:
            y_out = y_out - mu

    return y_out


def fit_and_report(X_fit, y_fit, X_cols, all_cols, drops, df_fit=None, exp_framework='yin', ablation='none', quantile=0.5):
    # Fit model based on ablation choice
    if ablation == 'quantile':
        model = sm.QuantReg(y_fit, X_fit).fit(q=quantile)
    elif ablation == 'logistic':
        model = sm.Logit(y_fit, X_fit).fit(disp=0)
    else:
        model = sm.OLS(y_fit, X_fit).fit()

    coefs = model.params
    pvals = model.pvalues

    # Prepare full coef/pval table including dropped reference levels (coef=0)
    # include const as well
    full_index = ['const'] + all_cols
    coef_series = pd.Series(index=full_index, dtype=float)
    pval_series = pd.Series(index=full_index, dtype=float)
    for idx in full_index:
        if idx in coefs.index:
            coef_series.loc[idx] = coefs.loc[idx]
            pval_series.loc[idx] = pvals.loc[idx]
        else:
            # dropped reference columns or missing params -> coef 0
            coef_series.loc[idx] = 0.0
            pval_series.loc[idx] = np.nan

    print("\nRegression coefficients:")
    print(coef_series)
    print("\nP-values:")
    print(pval_series)
    print(f"\nTotal estimated coefficients (including intercept): {len(model.params)}")
    print(f"Total coefficients shown (including dropped refs and intercept): {len(coef_series)}")
    
    if exp_framework == 'yin':
        framework_to_name = {'name': 'name_bundle', 'job': 'job_bundle', 'model': 'model'}
    else:
        framework_to_name = {'name': 'Names', 'job': 'Jobs', 'model': 'Model'}


    pred_fit = model.predict(X_fit)

    def manual_row_sum(row):
        s = coef_series['const'] if 'const' in coef_series.index else 0.0
        name_col = f"name_{row[framework_to_name['name']]}"
        if name_col in coef_series.index:
            s += coef_series[name_col]
        job_col = f"job_{row[framework_to_name['job']] }"
        if job_col in coef_series.index:
            s += coef_series[job_col]
        model_col = f"model_{row[framework_to_name['model']] }"
        if model_col in coef_series.index:
            s += coef_series[model_col]
        return s

    has_interactions = any(c.startswith('int_') for c in X_cols)
    if (df_fit is not None) and (ablation != 'logistic') and (not has_interactions):
        manual_vals = df_fit.apply(manual_row_sum, axis=1)
        diff = np.abs(manual_vals - pred_fit)
        print(f"\nMax abs difference between model.predict and manual sum on fit data: {diff.max():.6g}")
    else:
        print("\nManual intercept+main-effects check skipped for this ablation/specification.")

    return model, coef_series, pval_series, y_fit, pred_fit


def cross_validate(X, y, X_cols, df=None, color_col=None, n_splits=5, random_state=0, ablation='none', quantile=0.5, r2_mode='modified'):
    """Perform k-fold cross-validation and report R² and RMSE.

    If color_col is provided, also collect the corresponding values for each test fold
    (name_bundle/Names/job_bundle/Jobs/model) so that scatter points can be colored.
    Returns r2_mean, rmse_mean, all_y_test, all_pred_test, all_color_test
    """
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    r2_scores = []
    rmse_scores = []
    acc_scores = []
    
    print(f"\n{'='*60}")
    print(f"K-Fold Cross-Validation ({n_splits} folds)")
    print(f"{'='*60}")

    all_y_test = []
    all_pred_test = []
    all_color_test = []
    
    for fold, (train_idx, test_idx) in enumerate(kfold.split(X), 1):
        X_train_fold = X.iloc[train_idx]
        X_test_fold = X.iloc[test_idx]
        y_train_fold = y.iloc[train_idx]
        y_test_fold = y.iloc[test_idx]
        
        # Fit model on this fold
        if ablation == 'quantile':
            model = sm.QuantReg(y_train_fold, X_train_fold).fit(q=quantile)
        elif ablation == 'logistic':
            model = sm.Logit(y_train_fold, X_train_fold).fit(disp=0)
        else:
            model = sm.OLS(y_train_fold, X_train_fold).fit()
        pred = model.predict(X_test_fold)
        
        # Calculate metrics
        mse = mean_squared_error(y_test_fold, pred)

        if r2_mode == 'standard':
            # Standard R² uses test mean as baseline
            r2 = r2_score(y_test_fold, pred)
            print(f"\nStandard R^2 (using test mean): {r2:.6g}")
        else:
            # Modified out-of-sample R^2 using the training mean
            y_train_mean = y_train_fold.mean()
            num = np.sum((y_test_fold - pred) ** 2)
            den = np.sum((y_test_fold - y_train_mean) ** 2)
            r2 = 1.0 - (num / den) if den != 0 else np.nan
            print(f"\nModified out-of-sample R^2 (using train mean): {r2:.6g}")
        rmse = np.sqrt(mse)
        
        r2_scores.append(r2)
        rmse_scores.append(rmse)
        if ablation in ('logistic', 'directionality'):
            yhat = (np.asarray(pred) >= 0.5).astype(int)
            ytrue = np.asarray(y_test_fold).astype(int)
            acc = (yhat == ytrue).mean()
            acc_scores.append(acc)

        all_y_test.append(y_test_fold)
        all_pred_test.append(pred)
        if color_col is not None and df is not None:
            all_color_test.append(df.iloc[test_idx][color_col].reset_index(drop=True))
        else:
            all_color_test.append(pd.Series([None] * len(test_idx)))
        
        if ablation in ('logistic', 'directionality'):
            print(f"Fold {fold}: R² = {r2:.6g}, RMSE = {rmse:.6g}, Acc = {acc:.6g}")
        else:
            print(f"Fold {fold}: R² = {r2:.6g}, RMSE = {rmse:.6g}")
    
    # Report summary statistics
    r2_mean = np.mean(r2_scores)
    r2_std = np.std(r2_scores)
    rmse_mean = np.mean(rmse_scores)
    rmse_std = np.std(rmse_scores)
    
    print(f"{'='*60}")
    print(f"Cross-Validation Summary:")
    print(f"  R² = {r2_mean:.6g} (+/- {r2_std:.6g})")
    print(f"  RMSE = {rmse_mean:.6g} (+/- {rmse_std:.6g})")

    print(f"y_test range:  [{y_test_fold.min():.3f}, {y_test_fold.max():.3f}]")
    print(f"y_train mean:  {y_train_fold.mean():.3f}")
    print(f"pred range:    [{pred.min():.3f}, {pred.max():.3f}]")
    print(f"Train R²: {r2_score(y_train_fold, model.predict(X_train_fold)):.4f}")
    print(f"Test R²:  {r2:.4f}")

    if ablation in ('logistic', 'directionality') and len(acc_scores) > 0:
        acc_mean = np.mean(acc_scores)
        acc_std = np.std(acc_scores)
        print(f"  Accuracy = {acc_mean:.6g} (+/- {acc_std:.6g})")
    print(f"{'='*60}")
    return r2_mean, r2_std, rmse_mean, all_y_test, all_pred_test, all_color_test


def plot(all_y_test, all_pred_test, all_color_test, r2, rmse, args, ablation='none'):
    # Scatter plot with line of best fit, optionally colored by category
    fig, ax = plt.subplots(figsize=(9, 7))
    
    if args.color_by != 'none':
        # Collect all color values and create a color mapping
        all_cvals = []
        for color_fold in all_color_test:
            all_cvals.extend(color_fold.values)
        
        unique_cvals = sorted([c for c in set(all_cvals) if c is not None])
        color_palette = plt.cm.tab20(np.linspace(0, 1, max(20, len(unique_cvals))))
        color_map = {val: color_palette[i % len(color_palette)] for i, val in enumerate(unique_cvals)}
        
        # Track which values have been added to legend
        seen = set()
        fold_colors = plt.cm.tab10(np.linspace(0, 1, len(all_y_test)))
        all_slopes, all_intercepts = [], []
        x_min_global, x_max_global = np.inf, -np.inf
        for fold_i, (y_test_fold, pred_test_fold, color_fold) in enumerate(zip(all_y_test, all_pred_test, all_color_test)):
            for yval, pval, cval in zip(y_test_fold, pred_test_fold, color_fold):
                if cval is None or cval not in color_map:
                    ax.scatter(yval, pval, alpha=0.6, s=30, color='lightgray', label='_nolegend_')
                else:
                    display_cval = cval
                    if args.color_by == 'model' and isinstance(display_cval, str):
                        display_cval = display_cval.replace('-instruct-turbo', '').replace('instruct-turbo', '')
                    label = display_cval if display_cval not in seen else '_nolegend_'
                    ax.scatter(yval, pval, alpha=0.6, s=30, color=color_map[cval], label=label)
                    seen.add(display_cval)
            # Collect fold fit for average line
            if len(y_test_fold) > 1:
                z = np.polyfit(y_test_fold, pred_test_fold, 1)
                all_slopes.append(z[0])
                all_intercepts.append(z[1])
                x_min_global = min(x_min_global, float(y_test_fold.min()))
                x_max_global = max(x_max_global, float(y_test_fold.max()))
        # Plot the identity line y = x over the observed x range (diagonal)
        if x_min_global <= x_max_global:
            x_line = np.linspace(x_min_global, x_max_global, 100)
            ax.plot(x_line, x_line, color='black', linestyle='--', linewidth=2, label='y = x')
    else:
        fold_colors = plt.cm.tab10(np.linspace(0, 1, len(all_y_test)))
        all_slopes, all_intercepts = [], []
        x_min_global, x_max_global = np.inf, -np.inf
        for i, (y_test_fold, pred_test_fold) in enumerate(zip(all_y_test, all_pred_test)):
            color = fold_colors[i]
            ax.scatter(y_test_fold, pred_test_fold, alpha=0.6, s=30, color=color, label=f'Fold {i+1}')
            if len(y_test_fold) > 1:
                z = np.polyfit(y_test_fold, pred_test_fold, 1)
                all_slopes.append(z[0])
                all_intercepts.append(z[1])
                x_min_global = min(x_min_global, float(y_test_fold.min()))
                x_max_global = max(x_max_global, float(y_test_fold.max()))
        # Plot the identity line y = x over the observed x range (diagonal)
        if x_min_global <= x_max_global:
            x_line = np.linspace(x_min_global, x_max_global, 100)
            ax.plot(x_line, x_line, color='black', linestyle='--', linewidth=2, label=f'y = x \n(R² = {r2:.3f})')
    
    ax.set_xlabel('Observed')
    ax.set_ylabel('Predicted')
    color_by_label = f"colored by {args.color_by}" if args.color_by != 'none' else "by fold"
    ax.set_title('Observed vs Predicted Selection Rate')

    # ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.legend(loc='upper left')
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    os.makedirs(repo_path(f'results/meta/{args.exp_framework}'), exist_ok=True)
    plot_file = f"results/meta/{args.exp_framework}/meta_regression_plot_intercept_colorby_{args.color_by}_modelspec_{args.model_spec}_ablation_{ablation}.png"
    plt.savefig(repo_path(plot_file), dpi=100, bbox_inches='tight')
    print(f"Plot saved as '{plot_file}'")
    plt.close()


def framework_column_map(exp_framework):
    """Raw (pre-dummy) column names holding each dimension, per framework."""
    if exp_framework == 'yin':
        return {'name': 'name_bundle', 'job': 'job_bundle', 'model': 'model'}
    return {'name': 'Names', 'job': 'Jobs', 'model': 'Model'}


def framework_models(exp_framework):
    """Model level set used for a given framework."""
    if exp_framework == 'yin':
        return YIN_MODELS
    if exp_framework == 'wang':
        return WANG_MODELS
    return ARMSTRONG_MODELS


def ensure_cols(df, col_prefix, values):
    """Create one-hot columns with prefix for each value in values.

    Returns list of created column names.
    """
    cols = []
    for v in values:
        cname = f"{col_prefix}_{v}"
        df[cname] = (df[col_prefix.replace('name_', 'name_bundle').replace('job_', 'job_bundle').replace('model_', 'model')] == v).astype(int)
        cols.append(cname)
    return cols


def build_design_matrix(df, exp_framework='yin', include_interactions=False):
    # create dummies limited to the provided lists
    name_cols = []
    if exp_framework == 'yin':
        framework_to_name = {'name': 'name_bundle', 'job': 'job_bundle', 'model': 'model'}
    elif exp_framework == 'wang':
        framework_to_name = {'name': 'Names', 'job': 'Jobs', 'model': 'Model'}
    else:
        framework_to_name = {'name': 'Names', 'job': 'Jobs', 'model': 'Model'}

    for n in NAMES:
        cname = f"name_{n}"
        df[cname] = (df[framework_to_name['name']] == n).astype(int)
        name_cols.append(cname)

    job_cols = []
    for j in JOBS:
        cname = f"job_{j}"
        df[cname] = (df[framework_to_name['job']] == j).astype(int)
        job_cols.append(cname)

    model_cols = []
    if exp_framework == 'yin':
        MODELS = YIN_MODELS
    elif exp_framework == 'wang':
        MODELS = WANG_MODELS
    else:
        MODELS = ARMSTRONG_MODELS
    for m in MODELS:
        cname = f"model_{m}"
        df[cname] = (df[framework_to_name['model']] == m).astype(int)
        model_cols.append(cname)

    # Choose reference (to be dropped) for name and job groups based on exp_framework
    # if the corresponding category exists; otherwise drop the first
    ref_name_val = exp_framework if exp_framework in NAMES else name_cols[0].replace('name_', '')
    ref_job_val = exp_framework if exp_framework in JOBS else job_cols[0].replace('job_', '')
    drop_name = f"name_{ref_name_val}"
    drop_job = f"job_{ref_job_val}"
    # For models, drop the first model as reference
    drop_model = model_cols[0]

    main_cols = name_cols + job_cols + model_cols
    X_cols = [c for c in main_cols if c not in (drop_name, drop_job, drop_model)]

    interaction_cols = []
    if include_interactions:
        kept_name_cols = [c for c in name_cols if c != drop_name]
        kept_job_cols = [c for c in job_cols if c != drop_job]
        kept_model_cols = [c for c in model_cols if c != drop_model]

        # Two-way interactions
        for ncol in kept_name_cols:
            for jcol in kept_job_cols:
                iname = f"int_{ncol}__X__{jcol}"
                df[iname] = df[ncol] * df[jcol]
                interaction_cols.append(iname)

        for ncol in kept_name_cols:
            for mcol in kept_model_cols:
                iname = f"int_{ncol}__X__{mcol}"
                df[iname] = df[ncol] * df[mcol]
                interaction_cols.append(iname)

        for jcol in kept_job_cols:
            for mcol in kept_model_cols:
                iname = f"int_{jcol}__X__{mcol}"
                df[iname] = df[jcol] * df[mcol]
                interaction_cols.append(iname)

        X_cols = X_cols + interaction_cols

    all_cols = main_cols + interaction_cols

    X = df[X_cols].astype(float)
    # add intercept
    X = sm.add_constant(X)
    return X, X_cols, all_cols, (drop_name, drop_job, drop_model)




def permutation_feature_importance(df, y, args, drops, ablation='none'):
    """Permutation importance of each evaluation dimension (name/job/model).

    Uses the same folds as ``cross_validate``: the model is fit on the training
    split, then each dimension is shuffled in the held-out split and the drop in
    R² is recorded. A synthetic ``random_num`` predictor is included as a
    reference for what an uninformative dimension looks like.

    Returns a DataFrame of per-dimension importances.
    """
    cols = framework_column_map(args.exp_framework)
    # Level sets mirror build_design_matrix, so values outside these lists stay all-zero.
    levels = {
        cols['name']: NAMES,
        cols['job']: JOBS,
        cols['model']: framework_models(args.exp_framework),
    }
    drop_name, drop_job, drop_model = drops
    refs = {
        cols['name']: drop_name[len('name_'):],
        cols['job']: drop_job[len('job_'):],
        cols['model']: drop_model[len('model_'):],
    }

    X_raw = df[[cols['job'], cols['name'], cols['model']]].astype(str).copy()
    rng = np.random.RandomState(args.random_state)
    X_raw[RANDOM_FEATURE] = rng.normal(size=len(X_raw))
    y = pd.Series(np.asarray(y, dtype=float), index=X_raw.index)

    # Plot/report order requested: job, name, model, random_num
    feature_cols = [cols['job'], cols['name'], cols['model'], RANDOM_FEATURE]
    display_names = ['job', 'name', 'model', RANDOM_FEATURE]
    feature_idx = [X_raw.columns.get_loc(c) for c in feature_cols]

    print(f"\n{'='*60}")
    print(f"Permutation Feature Importance ({args.n_splits} folds x {args.n_repeats} repeats)")
    print(f"{'='*60}")
    if ablation in ('logistic', 'directionality'):
        print("Note: target is binary for this ablation; importances are still reported as a drop in R².")

    kfold = KFold(n_splits=args.n_splits, shuffle=True, random_state=args.random_state)
    per_fold_importances = []
    baseline_scores = []

    for fold, (train_idx, test_idx) in enumerate(kfold.split(X_raw), 1):
        X_train_fold, X_test_fold = X_raw.iloc[train_idx], X_raw.iloc[test_idx]
        y_train_fold, y_test_fold = y.iloc[train_idx], y.iloc[test_idx]

        est = DimensionRegressor(
            levels=levels,
            refs=refs,
            numeric_cols=(RANDOM_FEATURE,),
            include_interactions=(args.model_spec == 'interactions'),
            ablation=ablation,
            quantile=args.quantile,
            r2_mode=args.r2_mode,
        ).fit(X_train_fold, y_train_fold)

        baseline = est.score(X_test_fold, y_test_fold)
        baseline_scores.append(baseline)

        # scoring=None -> uses est.score, i.e. the same R² convention as the CV above
        r = permutation_importance(
            est,
            X_test_fold,
            y_test_fold,
            n_repeats=args.n_repeats,
            random_state=args.random_state,
        )
        per_fold_importances.append(r.importances[feature_idx, :])

        print(f"\nFold {fold}: held-out reference R² = {baseline:.6g}")
        for disp, i in zip(display_names, feature_idx):
            print(f"  {disp:<12} decrease in R² = {r.importances_mean[i]:.6g} +/- {r.importances_std[i]:.6g}")

    # Pool repeats across folds
    importances = np.concatenate(per_fold_importances, axis=1)
    means = importances.mean(axis=1)
    stds = importances.std(axis=1)

    print(f"\n{'='*60}")
    print(f"Permutation importance summary (reference R² = {np.mean(baseline_scores):.6g})")
    for disp, m, s in zip(display_names, means, stds):
        print(f"  {disp:<12} decrease in R² = {m:.6g} (+/- {s:.6g})")
    print(f"{'='*60}")

    importance_df = pd.DataFrame({
        'dimension': display_names,
        'importance_mean': means,
        'importance_std': stds,
        'reference_r2': np.mean(baseline_scores),
    })

    os.makedirs(repo_path(f'results/meta/{args.exp_framework}'), exist_ok=True)
    importance_file = f"results/meta/{args.exp_framework}/permutation_importance_ablation_{ablation}_{args.model_spec}.csv"
    importance_df.to_csv(repo_path(importance_file), index=False)
    print(f"Permutation importances saved to '{importance_file}'")

    # Horizontal plot: decrease in R² on x, dimensions on y
    fig, ax = plt.subplots(figsize=(8, 5))
    y_pos = np.arange(len(display_names))
    dim_colors = {'name': 'C0', 'job': 'C1', 'model': 'C2', RANDOM_FEATURE: 'gray'}
    colors = [dim_colors.get(d, 'gray') for d in display_names]
    for m, y_p, s, color in zip(means, y_pos, stds, colors):
        ax.errorbar(m, y_p, xerr=s, fmt='o', color=color, ecolor='gray', capsize=3)
    ax.axvline(x=0, color='red', linestyle='--', linewidth=1.5, alpha=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([d[0].upper() + d[1:] for d in display_names])
    ax.invert_yaxis()
    ax.set_xlabel('Decrease in R²')
    ax.set_title('Permutation Importances (test set)')
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    importance_plot_file = f"results/meta/{args.exp_framework}/permutation_importance_plot_ablation_{ablation}_{args.model_spec}.png"
    plt.savefig(repo_path(importance_plot_file), dpi=150)
    print(f"Permutation importance plot saved as '{importance_plot_file}'")
    plt.close()

    return importance_df


class DimensionRegressor(BaseEstimator, RegressorMixin):
    """sklearn-compatible wrapper around the statsmodels fits used above.

    ``X`` is a DataFrame of *raw* dimension columns (name / job / model) plus any
    numeric control columns. The one-hot encoding, reference-level dropping and
    (optionally) two-way interactions are rebuilt inside fit/predict, so
    permuting a single raw column permutes an entire dimension at once. That is
    what allows ``permutation_importance`` to score whole dimensions rather than
    individual indicator columns.

    ``score`` mirrors the ``--r2-mode`` convention used in ``cross_validate``:
    'modified' compares against the training mean, 'standard' against the test
    mean.
    """

    def __init__(self, levels=None, refs=None, numeric_cols=(), include_interactions=False,
                 ablation='none', quantile=0.5, r2_mode='modified'):
        self.levels = levels
        self.refs = refs
        self.numeric_cols = numeric_cols
        self.include_interactions = include_interactions
        self.ablation = ablation
        self.quantile = quantile
        self.r2_mode = r2_mode

    def _design(self, X):
        refs = self.refs or {}
        dummies = {}
        per_dim = {}
        for col, values in self.levels.items():
            kept = []
            for v in values:
                if str(v) == str(refs.get(col)):
                    continue  # reference level, dropped
                cname = f"{col}_{v}"
                dummies[cname] = (X[col].astype(str).values == str(v)).astype(float)
                kept.append(cname)
            per_dim[col] = kept
        design = pd.DataFrame(dummies, index=X.index)

        if self.include_interactions:
            dim_names = list(self.levels.keys())
            for a in range(len(dim_names)):
                for b in range(a + 1, len(dim_names)):
                    for ca in per_dim[dim_names[a]]:
                        for cb in per_dim[dim_names[b]]:
                            design[f"int_{ca}__X__{cb}"] = design[ca].values * design[cb].values

        for col in self.numeric_cols:
            design[col] = np.asarray(X[col], dtype=float)

        return sm.add_constant(design, has_constant='add')

    def fit(self, X, y):
        design = self._design(X)
        y = pd.Series(np.asarray(y, dtype=float), index=design.index)
        if self.ablation == 'quantile':
            self.result_ = sm.QuantReg(y, design).fit(q=self.quantile)
        elif self.ablation == 'logistic':
            self.result_ = sm.Logit(y, design).fit(disp=0)
        else:
            self.result_ = sm.OLS(y, design).fit()
        self.design_columns_ = list(design.columns)
        self.train_mean_ = float(y.mean())
        return self

    def predict(self, X):
        design = self._design(X)[self.design_columns_]
        return np.asarray(self.result_.predict(design), dtype=float)

    def score(self, X, y):
        pred = self.predict(X)
        y = np.asarray(y, dtype=float)
        if self.r2_mode == 'standard':
            return r2_score(y, pred)
        num = np.sum((y - pred) ** 2)
        den = np.sum((y - self.train_mean_) ** 2)
        return 1.0 - (num / den) if den != 0 else np.nan


class DimensionRegressor(BaseEstimator, RegressorMixin):
    """sklearn-compatible wrapper around the statsmodels fits used above.

    ``X`` is a DataFrame of *raw* dimension columns (name / job / model) plus any
    numeric control columns. The one-hot encoding, reference-level dropping and
    (optionally) two-way interactions are rebuilt inside fit/predict, so
    permuting a single raw column permutes an entire dimension at once. That is
    what allows ``permutation_importance`` to score whole dimensions rather than
    individual indicator columns.

    ``score`` mirrors the ``--r2-mode`` convention used in ``cross_validate``:
    'modified' compares against the training mean, 'standard' against the test
    mean.
    """

    def __init__(self, levels=None, refs=None, numeric_cols=(), include_interactions=False,
                 ablation='none', quantile=0.5, r2_mode='modified'):
        self.levels = levels
        self.refs = refs
        self.numeric_cols = numeric_cols
        self.include_interactions = include_interactions
        self.ablation = ablation
        self.quantile = quantile
        self.r2_mode = r2_mode

    def _design(self, X):
        refs = self.refs or {}
        dummies = {}
        per_dim = {}
        for col, values in self.levels.items():
            kept = []
            for v in values:
                if str(v) == str(refs.get(col)):
                    continue  # reference level, dropped
                cname = f"{col}_{v}"
                dummies[cname] = (X[col].astype(str).values == str(v)).astype(float)
                kept.append(cname)
            per_dim[col] = kept
        design = pd.DataFrame(dummies, index=X.index)

        if self.include_interactions:
            dim_names = list(self.levels.keys())
            for a in range(len(dim_names)):
                for b in range(a + 1, len(dim_names)):
                    for ca in per_dim[dim_names[a]]:
                        for cb in per_dim[dim_names[b]]:
                            design[f"int_{ca}__X__{cb}"] = design[ca].values * design[cb].values

        for col in self.numeric_cols:
            design[col] = np.asarray(X[col], dtype=float)

        return sm.add_constant(design, has_constant='add')

    def fit(self, X, y):
        design = self._design(X)
        y = pd.Series(np.asarray(y, dtype=float), index=design.index)
        if self.ablation == 'quantile':
            self.result_ = sm.QuantReg(y, design).fit(q=self.quantile)
        elif self.ablation == 'logistic':
            self.result_ = sm.Logit(y, design).fit(disp=0)
        else:
            self.result_ = sm.OLS(y, design).fit()
        self.design_columns_ = list(design.columns)
        self.train_mean_ = float(y.mean())
        return self

    def predict(self, X):
        design = self._design(X)[self.design_columns_]
        return np.asarray(self.result_.predict(design), dtype=float)

    def score(self, X, y):
        pred = self.predict(X)
        y = np.asarray(y, dtype=float)
        if self.r2_mode == 'standard':
            return r2_score(y, pred)
        num = np.sum((y - pred) ** 2)
        den = np.sum((y - self.train_mean_) ** 2)
        return 1.0 - (num / den) if den != 0 else np.nan



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=None, help="Input CSV file path")
    parser.add_argument("--exp_framework", default="yin", choices=["yin", "wang", "armstrong"], help="Experimental framework (yin/wang/armstrong)")
    parser.add_argument("--metric", default="selection_rate") # regression_coefficients, score_difference
    parser.add_argument("--holdout", type=float, default=0.1, help="Deprecated; retained for CLI compatibility and ignored by the current full-dataset CV workflow")
    parser.add_argument("--random-state", type=int, default=0, help="Random seed")
    parser.add_argument("--n-splits", type=int, default=5, help="Number of CV folds (default 5)")
    parser.add_argument("--color-by", choices=["none","name","job","model"], default="none", help="Category to color scatter points by (name/job/model)")
    parser.add_argument("--model_spec", choices=["main", "interactions"], default="main", help="Use main effects only or add two-way interactions")
    parser.add_argument("--ablation", choices=["none", "logistic", "directionality", "copula", "truncate", "standardize", "quantile", "all"], default="none", help="Ablation to apply before/while fitting")
    parser.add_argument("--r2-mode", choices=["modified", "standard"], default="modified", help="R² mode: 'modified' uses training mean as baseline (out-of-sample), 'standard' uses test mean")
    parser.add_argument("--truncate-q", type=float, default=0.1, help="Two-sided quantile cut for --ablation truncate")
    parser.add_argument("--quantile", type=float, default=0.5, help="Target quantile for --ablation quantile")
    parser.add_argument("--feature_importance", action="store_true", default=False, help="Compute permutation feature importance after fitting")
    parser.add_argument("--n-repeats", type=int, default=30, help="Number of shuffles per dimension for --feature_importance (default 30)")
    parser.add_argument("--remove_reference_cllass", action="store_true", default=False, help="Remove reference-level coefficients from the coefficient plot (do not plot reference class)")

    args = parser.parse_args()

    include_interactions = args.model_spec == 'interactions'
    print(f"Model specification: {args.model_spec}")
    print(f"Ablation: {args.ablation}")

    default_inputs = {
        'yin': 'results/yin_sampling_downsampling_analysis_original.csv',
        'wang': 'results/wang_sampling_downsampling_analysis_original.csv',
        'armstrong': 'results/armstrong_sampling_downsampling_analysis_original.csv',
    }
    input_path = args.input if args.input is not None else default_inputs.get(args.exp_framework)
    if input_path is None:
        raise ValueError(f"No default input path configured for exp_framework='{args.exp_framework}'. Please provide --input.")
    print(f"Using input CSV: {input_path}")

    df = pd.read_csv(repo_path(input_path))

    # Exclusions requested for meta-regression analysis
    if 'name_bundle' in df.columns:
        df = df[~df['name_bundle'].astype(str).str.lower().isin(EXCLUDED_NAME_BUNDLES)].copy()
    if 'Names' in df.columns:
        df = df[~df['Names'].astype(str).str.lower().isin(EXCLUDED_NAME_BUNDLES)].copy()

    if 'model' in df.columns:
        df = df[
            ~df['model'].astype(str).str.lower().str.contains('|'.join(EXCLUDED_MODEL_PATTERNS), regex=True, na=False)
        ].copy()
    if 'Model' in df.columns:
        df = df[
            ~df['Model'].astype(str).str.lower().str.contains('|'.join(EXCLUDED_MODEL_PATTERNS), regex=True, na=False)
        ].copy()


    if args.exp_framework=='yin':
        yin_df = df[df['gender'].eq('W')].copy()
        yin_dedupe_columns = [
            column for column in ('gender', 'job', 'model', 'rank', 'name_bundle', 'job_bundle')
            if column in yin_df.columns
        ]
        if yin_dedupe_columns:
            yin_df = yin_df.drop_duplicates(subset=yin_dedupe_columns, keep='first')
        yin_values = pd.to_numeric(yin_df['selection_rate'], errors='coerce').dropna()

        if args.metric == 'selection_rate':
            yin_df = yin_df.loc[yin_values.index].copy()

        if yin_df.empty:
            raise ValueError("No rows with gender=='W' found in input CSV")

        y = pd.to_numeric(yin_df[args.metric], errors='coerce')
        valid_rows = y.notna()
        yin_df = yin_df.loc[valid_rows].copy()
        y = y.loc[valid_rows]

        X, X_cols, all_cols, drops = build_design_matrix(yin_df, args.exp_framework, include_interactions=include_interactions)
        df = yin_df

    elif args.exp_framework=='wang':
        df = df.drop_duplicates(subset=["Names", "Jobs", "Model", "Metric"], keep="first")
        df = df[df['Result_Vector'].notna()]
        df = df[(df['Sensitive_Attribute_Vector'] == "['Woman']") & (df['Metric'] == args.metric) & (df['Model'].isin(WANG_MODELS))].copy()
        if df.empty:
            raise ValueError("No rows with Sensitive_Attribute_Vector=='Woman' and Metric==args.metric found in input CSV")

        
        y = df['Result_Vector']
        new_y = []
        for i in y:
            # handle cases where the value is already numeric or a string like "[0.123]"
            if isinstance(i, (float, int)):
                new_y.append(float(i))
                continue
            if isinstance(i, str):
                s = i.strip()
                if s.startswith('[') and s.endswith(']'):
                    y_val = s[1:-1].strip()  # Remove brackets and whitespace
                    try:
                        y_val = float(y_val)
                        new_y.append(y_val)
                    except ValueError:
                        raise ValueError(f"Could not convert '{y_val}' to float in Result_Vector")
                    continue
            raise ValueError(f"Unexpected format in Result_Vector: '{i}'")

        # Wang values are stored with opposite sign convention; flip sign before regression.
        # convert new_y to pd.Series
        X, X_cols, all_cols, drops = build_design_matrix(df, args.exp_framework, include_interactions=include_interactions)
        y = -pd.Series(new_y, index=df.index)

    elif args.exp_framework=='armstrong':
        # drop rows where the value at df['Result_Vector'] is nan
        # drop the rows where Name == 'wang'
        # df = df[df['Names'] != 'wang']

        df = df.drop_duplicates(subset=["Names", "Jobs", "Model", "Metric"], keep="first")
        df = df[df["Sensitive_Attribute_Vector"] == "['Woman']"]
        df = df[df["Metric"] == args.metric] #378

        if df.empty:
            raise ValueError("No rows with Sensitive_AttributeVector=='Woman' and Metric==args.metric found in input CSV")


        y =  df['Result_Vector']
        new_y = []
        for i in y:
            # handle cases where the value is already numeric or a string like "[0.123]"
            if isinstance(i, (float, int)):
                new_y.append(float(i))
                continue
            if isinstance(i, str):
                s = i.strip()
                if s.startswith('[') and s.endswith(']'):
                    y_val = s[1:-1].strip()  # Remove brackets and whitespace
                    try:
                        y_val = float(y_val)
                        new_y.append(y_val)
                    except ValueError:
                        raise ValueError(f"Could not convert '{y_val}' to float in Result_Vector")
                    continue
            raise ValueError(f"Unexpected format in Result_Vector: '{i}'")

        # convert new_y to pd.Series
        X, X_cols, all_cols, drops = build_design_matrix(df, args.exp_framework, include_interactions=include_interactions)
        y = pd.Series(new_y, index=df.index)

    # Decide which column to use for coloring based on user argument
    color_col = None
    if args.color_by != 'none':
        if args.color_by == 'name':
            color_col = 'name_bundle' if args.exp_framework=='yin' else 'Names'
        elif args.color_by == 'job':
            color_col = 'job_bundle' if args.exp_framework=='yin' else 'Jobs'
        elif args.color_by == 'model':
            color_col = 'model' if args.exp_framework=='yin' else 'Model'

    if args.ablation == 'all':
        ablations_to_run = ['none', 'directionality', 'copula', 'truncate', 'standardize', 'quantile']
    else:
        ablations_to_run = [args.ablation]

    summary_rows = []


    for ablation_name in ablations_to_run:
        print(f"\n{'#'*70}")
        print(f"Running ablation: {ablation_name}")
        print(f"{'#'*70}")

        y_full_for_cv = apply_ablation_to_target(
            y,
            ablation=ablation_name,
            truncate_q=args.truncate_q,
            ref_series=y,
            exp_framework=args.exp_framework,
        )

        # Perform k-fold cross-validation on full dataset
        r2_mean, r2_std, rmse_mean, all_y_test, all_pred_test, all_color_test = cross_validate(
            X, y_full_for_cv, X_cols, df, color_col=color_col, n_splits=args.n_splits, random_state=args.random_state, ablation=ablation_name, quantile=args.quantile, r2_mode=args.r2_mode
        )

        model, coef_series, pval_series, y_fit, pred_fit = fit_and_report(
            X,
            y_full_for_cv,
            X_cols,
            all_cols,
            drops,
            df_fit=df,
            exp_framework=args.exp_framework,
            ablation=ablation_name,
            quantile=args.quantile,
        )

        fit_mse = mean_squared_error(y_fit, pred_fit)
        fit_r2 = r2_score(y_fit, pred_fit)
        fit_rmse = np.sqrt(fit_mse)
        print(f"\nFull-data fit MSE: {fit_mse:.6g}")
        print(f"Full-data fit RMSE: {fit_rmse:.6g}")
        print(f"Full-data fit R^2: {fit_r2:.6g}")
        if ablation_name in ('logistic', 'directionality'):
            yhat_fit = (np.asarray(pred_fit) >= 0.5).astype(int)
            ytrue_fit = np.asarray(y_fit).astype(int)
            fit_acc = (yhat_fit == ytrue_fit).mean()
            print(f"Full-data fit Accuracy: {fit_acc:.6g}")

        summary_rows.append({
            'ablation': ablation_name,
            'cv_r2_mean': r2_mean,
            'cv_r2_std': r2_std,
            'cv_rmse_mean': rmse_mean,
            'full_fit_r2': fit_r2,
        })

        # Ensure results directory exists
        os.makedirs(repo_path('results'), exist_ok=True)

        # Coefficient plot with 95% CIs from the model trained on the training set
        try:
            ci = model.conf_int(alpha=0.05)
        except Exception:
            ci = pd.DataFrame(index=model.params.index, columns=[0, 1])
            ci[:] = np.nan

        full_index = ['const'] + all_cols
        est = model.params.reindex(full_index).fillna(0.0)
        ci = ci.reindex(full_index).fillna(pd.DataFrame({0: est.values, 1: est.values}, index=full_index))
        lower = ci[0].astype(float)
        upper = ci[1].astype(float)

        # Exclude 'const' from the y-axis; show its value in the title
        const_val = est.get('const', np.nan)
        if args.model_spec == 'interactions':
            # Keep interaction terms in the model fit, but omit them from the plot to reduce clutter
            coef_index = [i for i in full_index if i != 'const' and not i.startswith('int_')]
        else:
            coef_index = [i for i in full_index if i != 'const']
        # Optionally remove the reference-level coefficients from the plot
        if getattr(args, 'remove_reference_cllass', False):
            # `drops` is a tuple (drop_name, drop_job, drop_model) containing names like 'name_yin'
            refs_to_remove = list(drops)
            coef_index = [c for c in coef_index if c not in refs_to_remove]
        est_no_const = est.reindex(coef_index)
        lower_no_const = lower.reindex(coef_index)
        upper_no_const = upper.reindex(coef_index)
        errors_lower_no_const = est_no_const - lower_no_const
        errors_upper_no_const = upper_no_const - est_no_const

        # Define colors for coefficient types
        color_map = {'name_': 'C0', 'job_': 'C1', 'model_': 'C2', 'int_': 'C3'}
        colors = []
        for coef_name in coef_index:
            if coef_name.startswith('int_'):
                colors.append(color_map['int_'])
            elif coef_name.startswith('name_'):
                colors.append(color_map['name_'])
            elif coef_name.startswith('job_'):
                colors.append(color_map['job_'])
            elif coef_name.startswith('model_'):
                colors.append(color_map['model_'])
            else:
                colors.append('gray')

        fig, ax = plt.subplots(figsize=(8, max(6, len(coef_index) * 0.22)))
        y_pos = np.arange(len(coef_index))
        
        # Plot each coefficient with its color
        for i, (coef_val, y_p, color, err_lower, err_upper) in enumerate(zip(est_no_const.values, y_pos, colors, errors_lower_no_const.values, errors_upper_no_const.values)):
            ax.errorbar(coef_val, y_p, xerr=[[err_lower], [err_upper]], fmt='o', color=color, ecolor='gray', capsize=3)
        
        # Add vertical line at x=0
        ax.axvline(x=0, color='red', linestyle='--', linewidth=1.5, alpha=0.5, label='Zero')

        display_coef_labels = []
        for coef_name in coef_index:
            if coef_name.startswith('name_'):
                label = coef_name[len('name_'):]
            elif coef_name.startswith('job_'):
                label = coef_name[len('job_'):]
            elif coef_name.startswith('model_'):
                label = coef_name[len('model_'):]
                label = label.replace('-instruct-turbo', '').replace('instruct-turbo', '')
            else:
                label = coef_name

            if label:
                label = label[0].upper() + label[1:]
            display_coef_labels.append(label)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(display_coef_labels)
        ax.invert_yaxis()
        ax.set_xlabel('Coefficient estimate')
        # ax.set_title(f"Regression Coefficients const={const_val:.4f}")
        ax.set_title(f"Regression Coefficients")
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        
        # Add legend for only the coefficient groups that are plotted
        from matplotlib.patches import Patch
        legend_elements = []
        if any(c.startswith('name_') for c in coef_index):
            legend_elements.append(Patch(facecolor='C0', label='Name'))
        if any(c.startswith('job_') for c in coef_index):
            legend_elements.append(Patch(facecolor='C1', label='Job'))
        if any(c.startswith('model_') for c in coef_index):
            legend_elements.append(Patch(facecolor='C2', label='Model'))
        if any(c.startswith('int_') for c in coef_index):
            legend_elements.append(Patch(facecolor='C3', label='Interaction'))
        if legend_elements:
            ax.legend(handles=legend_elements, loc='best')
        plt.tight_layout()
        os.makedirs(repo_path(f'results/meta/{args.exp_framework}'), exist_ok=True)
        coef_plot_file = f"results/meta/{args.exp_framework}/coef_ci_plot_intercept_modelspec_{args.model_spec}_ablation_{ablation_name}.png"
        plt.savefig(repo_path(coef_plot_file), dpi=150)
        print(f"Coefficient plot saved as '{coef_plot_file}'")
        plt.close()

        # Scatter plot of train vs test (with optional coloring)
        plot(all_y_test, all_pred_test, all_color_test, r2_mean, rmse_mean, args, ablation=ablation_name)

    if len(summary_rows) > 1:
        summary_df = pd.DataFrame(summary_rows)
        summary_file = f"results/meta/{args.exp_framework}/ablation_summary_{args.model_spec}.csv"
        summary_df.to_csv(repo_path(summary_file), index=False)
        print(f"\nAblation summary saved to '{summary_file}'")
        print(summary_df)

    if args.feature_importance:
        print("\nComputing permutation feature importance...")
        permutation_feature_importance(df, y, args, drops, ablation=args.ablation)

if __name__ == '__main__':
    main()



'''
ARMSTRONG
python meta_regression_analysis_intercept.py --exp_framework armstrong  --holdout 0.1 --random-state 0 --n-splits 5 --ablation all 

python meta_regression_analysis_intercept.py --input results/yin_sampling_downsampling_analysis_original.csv  --holdout 0.1 --random-state 0 --n-splits 5


YIN
python meta_regression_analysis_intercept.py --exp_framework yin  --holdout 0.1 --random-state 0 --n-splits 5 --ablation all 

python meta_regression_analysis_intercept.py --input results/yin_sampling_downsampling_analysis_original.csv  --holdout 0.1 --random-state 0 --n-splits 5


python meta_regression_analysis_intercept.py --input results/armstrong_sampling_downsampling_analysis_original.csv --holdout 0.1 --random-state 0 --n-splits 5 --exp_framework armstrong --metric regression_coefficients

python meta_regression_analysis_intercept.py --input results/yin_sampling_downsampling_analysis_original.csv  --holdout 0.1 --random-state 0 --n-splits 5


python meta_regression_analysis_intercept.py --input results/armstrong_sampling_downsampling_analysis_original.csv --holdout 0.1 --random-state 0 --n-splits 5 --exp_framework armstrong --metric regression_coefficients


python meta_regression_analysis_intercept.py --input results/yin_sampling_downsampling_analysis_original.csv  --holdout 0.1 --random-state 0 --n-splits 5

python meta_regression_analysis_intercept.py --input results/wang_sampling_downsampling_analysis_original.csv  --holdout 0.1 --random-state 0 --n-splits 5



python meta_regression_analysis_intercept.py --input results/armstrong_sampling_downsampling_analysis_original.csv --holdout 0.1 --random-state 0 --n-splits 5 --exp_framework armstrong --metric regression_coefficients


'''


'''

python meta_regression_analysis_intercept.py --input results/yin_sampling_downsampling_analysis_original.csv  --holdout 0.1 --random-state 0 --n-splits 5 --exp_framework yin --model_spec interactions

python meta_regression_analysis_intercept.py --input results/wang_sampling_downsampling_analysis_original.csv  --holdout 0.1 --random-state 0 --n-splits 5 --exp_framework wang --metric score_difference

python meta_regression_analysis_intercept.py --input results/armstrong_sampling_downsampling_analysis_original.csv  --holdout 0.1 --random-state 0 --n-splits 5 --exp_framework armstrong --metric regression_coefficients

'''
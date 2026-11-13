#!/usr/bin/env python3
"""Standalone CatBoost/Elastic Net CV runner ported from prediction-cleanup.

Uses that branch's canonical factorial-cell selection, tuning grids, and
training-mean R^2. 
"""

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import ElasticNet, LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
NAMES = ['armstrong', 'rozado', 'wen', 'wang', 'karvonen', 'zollo', 'yin', 'gaeb', 'lippens']
JOBS = ['armstrong', 'rozado', 'wen', 'wang', 'karvonen', 'zollo', 'yin']
MODELS = [
    'gpt-4o-mini', 'mistral-7b-v0.3', 'meta-llama-3.1-8b-instruct-turbo',
    'meta-llama-3.3-70b-instruct-turbo', 'mistral-small-24b', 'gpt-5-nano-2025-08-07',
]
MODEL_ALIASES = {model.lower(): model for model in MODELS}
MODEL_ALIASES.update({
    'mistral-7b-instruct-v0.3': 'mistral-7b-v0.3',
    'mistral-small-24b-instruct-2501': 'mistral-small-24b',
})
METRICS = {'wang': 'score_difference', 'armstrong': 'regression_coefficients', 'yin': 'selection_rate'}
SOURCE_TRIALS = {'wang': 1, 'armstrong': 10}
CV_MODELS = [
    'elasticnet_interactions', 'catboost_main', 'elasticnet_main',
    'catboost_interactions', 'ols_main', 'ols_interactions',
]


def repo_path(path):
    path = Path(path).expanduser()
    return path if path.is_absolute() else ROOT / path


def prepare_data(frame, framework, metric):
    """Match prediction-cleanup: seed 42, aliases, and last row per cell."""
    df = frame.copy()
    original_count = len(df)
    if framework == 'yin':
        name_col, job_col, model_col = 'name_bundle', 'job_bundle', 'model'
    else:
        name_col, job_col, model_col = 'Names', 'Jobs', 'Model'
    df = df[~df[name_col].astype(str).str.lower().isin({'seshadri'})].copy()
    df = df[
        ~df[model_col].astype(str).str.lower().str.contains(
            'job-resume|job_resume', regex=True, na=False)
    ].copy()
    after_exclusions = len(df)
    if framework == 'yin':
        unknown = df[name_col].astype(str).str.lower().eq('lippens') & df['gender'].eq('Unknown')
        df.loc[unknown, 'gender'] = 'W'
        df = df[df['gender'].eq('W')].copy()
        keys = [name_col, job_col, model_col]
    else:
        df = df[
            df['Result_Vector'].notna()
            & df['Sensitive_Attribute_Vector'].eq("['Woman']")
            & df['Metric'].eq(metric)
            & pd.to_numeric(df['Random_State'], errors='coerce').eq(42)
            & pd.to_numeric(df['Num_Trials'], errors='coerce').eq(SOURCE_TRIALS[framework])
        ].copy()
        keys = [name_col, job_col, model_col]
    after_selection = len(df)
    aliases = df[model_col].astype(str).str.lower().map(MODEL_ALIASES)
    if aliases.isna().any():
        raise ValueError(f"Unrecognized model identifiers: {df.loc[aliases.isna(), model_col].unique().tolist()}")
    df[model_col] = aliases
    df = df.drop_duplicates(subset=keys, keep='last')
    canonical_count = len(df)
    if framework == 'yin':
        values = pd.to_numeric(df[metric], errors='coerce')
    else:
        values = pd.to_numeric(
            df['Result_Vector'].astype(str).str.strip().str.strip('[]'), errors='raise')
    finite = np.isfinite(values)
    if framework != 'yin' and not finite.all():
        raise ValueError(f"{framework}: non-finite targets in selected source rows")
    df = df.loc[finite].copy()
    df['_target'] = values.loc[finite] * (-1 if framework == 'wang' else 1)
    factors = df[keys].rename(columns={name_col: 'name', job_col: 'job', model_col: 'model'})
    if not factors['name'].isin(NAMES).all() or not factors['job'].isin(JOBS).all():
        raise ValueError('Selected rows contain an unrecognized name or job bundle')
    if framework != 'yin':
        df = df.sort_values(keys)
    df = df.reset_index(drop=True)
    factors = df[keys].rename(columns={name_col: 'name', job_col: 'job', model_col: 'model'})
    expected = len(NAMES) * len(JOBS) * len(MODELS)
    print(
        f"{framework}: {original_count} input rows -> {after_exclusions} after exclusions"
        f" -> {after_selection} source rows -> {canonical_count} cells"
        f" -> {len(df)}/{expected} finite cells"
    )
    print("  Selection: last CSV row per canonical name/job/model; "
          + ("women only" if framework == 'yin' else f"women, seed=42, trials={SOURCE_TRIALS[framework]}, metric={metric}"))
    if framework != 'yin' and len(df) != expected:
        raise ValueError(f"{framework}: expected {expected} canonical cells, found {len(df)}")
    if len(df) != expected:
        print(f"  Warning: {expected - len(df)} missing cells; no imputation.")
    if df.empty:
        raise ValueError(f"{framework}: no usable observations")
    return factors.reset_index(drop=True), df['_target'].reset_index(drop=True)


def build_features(factors, framework, model_name):
    interactions = model_name.endswith('_interactions')
    if model_name.startswith('catboost'):
        features = factors.astype(str).copy()
        if interactions:
            for left, right in combinations(factors.columns, 2):
                features[f'{left}_{right}'] = features[left] + '__' + features[right]
        return features
    levels = {'name': NAMES, 'job': JOBS, 'model': MODELS}
    references = {'name': framework, 'job': framework, 'model': MODELS[0]}
    groups = {}
    columns = {'const': np.ones(len(factors))}
    for factor, values in levels.items():
        groups[factor] = {}
        for value in values:
            if value != references[factor]:
                key = f'{factor}_{value}'
                groups[factor][key] = factors[factor].eq(value).astype(float).to_numpy()
                columns[key] = groups[factor][key]
    if interactions:
        for left, right in combinations(groups, 2):
            for left_name, left_values in groups[left].items():
                for right_name, right_values in groups[right].items():
                    columns[f'{left_name}__X__{right_name}'] = left_values * right_values
    return pd.DataFrame(columns, index=factors.index)


class SklearnCatBoostRegressor(RegressorMixin, BaseEstimator):
    """Keep CatBoost 1.2.x compatible with scikit-learn estimator cloning."""

    def __init__(self, iterations=250, depth=6, learning_rate=0.1,
                 l2_leaf_reg=3.0, random_state=0, cat_features=()):
        self.iterations = iterations
        self.depth = depth
        self.learning_rate = learning_rate
        self.l2_leaf_reg = l2_leaf_reg
        self.random_state = random_state
        self.cat_features = cat_features

    def fit(self, X, y):
        from catboost import CatBoostRegressor

        self.model_ = CatBoostRegressor(
            iterations=self.iterations, depth=self.depth,
            learning_rate=self.learning_rate, l2_leaf_reg=self.l2_leaf_reg,
            random_seed=self.random_state, cat_features=list(self.cat_features),
            loss_function='RMSE', verbose=False, allow_writing_files=False,
            thread_count=1,
        )
        self.model_.fit(X, y)
        return self

    def predict(self, X):
        return self.model_.predict(X)


def build_tuned_predictor(predictor, random_state, inner_splits, cat_features=(), n_jobs=1):
    if predictor == 'elasticnet':
        estimator = Pipeline([
            ('scale', StandardScaler()),
            ('model', ElasticNet(max_iter=200000, tol=1e-3, random_state=random_state)),
        ])
        grid = {
            'model__alpha': [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0],
            'model__l1_ratio': [0.1, 0.5, 0.9],
        }
    elif predictor == 'catboost':
        estimator = SklearnCatBoostRegressor(random_state=random_state, cat_features=cat_features)
        grid = {'depth': [4, 6], 'learning_rate': [0.03, 0.1], 'l2_leaf_reg': [3.0]}
    else:
        raise ValueError(f'Unsupported tuned predictor: {predictor}')
    return GridSearchCV(
        estimator, grid, scoring='neg_root_mean_squared_error',
        cv=KFold(n_splits=inner_splits, shuffle=True, random_state=random_state),
        n_jobs=n_jobs, refit=True, error_score='raise',
    )


def evaluate(features, target, model_name, n_splits=5, inner_splits=3,
             random_state=0, r2_mode='modified', n_jobs=1):
    if not 2 <= n_splits <= len(target) // 2:
        raise ValueError('Outer CV requires at least two folds and two test observations per fold')
    if not 2 <= inner_splits <= len(target) - int(np.ceil(len(target) / n_splits)):
        raise ValueError('Inner CV folds exceed the available outer-training observations')
    r2_values, rmse_values, parameters = [], [], []
    predictor = model_name.split('_', 1)[0]
    for fold, (train, test) in enumerate(
        KFold(n_splits=n_splits, shuffle=True, random_state=random_state).split(features), 1
    ):
        if predictor == 'ols':
            model = LinearRegression(fit_intercept=False)
        else:
            model = build_tuned_predictor(
                predictor, random_state + fold, inner_splits,
                tuple(features.columns) if predictor == 'catboost' else (), n_jobs,
            )
        model.fit(features.iloc[train], target.iloc[train])
        prediction = model.predict(features.iloc[test])
        observed = target.iloc[test].to_numpy()
        if r2_mode == 'standard':
            r2 = r2_score(observed, prediction)
        else:
            denominator = np.sum((observed - target.iloc[train].mean()) ** 2)
            r2 = 1 - np.sum((observed - prediction) ** 2) / denominator if denominator else np.nan
        rmse = np.sqrt(mean_squared_error(observed, prediction))
        r2_values.append(r2)
        rmse_values.append(rmse)
        parameters.append(model.best_params_ if predictor != 'ols' else {})
        print(f"  {model_name} fold {fold}: R^2={r2:.6g}, RMSE={rmse:.6g}")
    return {
        'predictor': model_name, 'feature_spec': model_name.split('_', 1)[1],
        'n_observations': len(target), 'n_design_columns': features.shape[1],
        'cv_r2_mean': np.mean(r2_values), 'cv_r2_std': np.std(r2_values),
        'cv_rmse_mean': np.mean(rmse_values), 'cv_rmse_std': np.std(rmse_values),
        'fold_r2': json.dumps(r2_values), 'fold_rmse': json.dumps(rmse_values),
        'selected_params': json.dumps(parameters, sort_keys=True),
        'n_splits': n_splits, 'inner_splits': inner_splits,
        'random_state': random_state, 'r2_mode': r2_mode,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exp_framework', '--exp-framework', choices=list(METRICS), required=True)
    parser.add_argument('--input', help='Input CSV (relative paths are anchored to this repo)')
    parser.add_argument('--metric', help='Defaults to the framework native metric')
    parser.add_argument('--cv-models', nargs='+', choices=CV_MODELS,
                        default=['elasticnet_interactions', 'catboost_main'])
    parser.add_argument('--n-splits', type=int, default=5)
    parser.add_argument('--inner-splits', type=int, default=3)
    parser.add_argument('--random-state', type=int, default=0)
    parser.add_argument('--r2-mode', choices=['modified', 'standard'], default='modified')
    parser.add_argument('--n-jobs', type=int, default=1, help='Parallel grid-search workers (default: 1)')
    parser.add_argument('--output', help='Summary CSV; defaults to results/meta/FRAMEWORK/predictor_comparison.csv')
    args = parser.parse_args(argv)
    if any(model.startswith('catboost') for model in args.cv_models):
        try:
            import catboost  # noqa: F401
        except ImportError:
            parser.error('CatBoost is missing; install with: python -m pip install catboost==1.2.8')
    framework = args.exp_framework
    metric = args.metric or METRICS[framework]
    input_path = repo_path(args.input or Path('results') / f'{framework}_sampling_downsampling_analysis_original.csv')
    output_path = repo_path(args.output or Path('results') / 'meta' / framework / 'predictor_comparison.csv')
    if output_path.resolve() == input_path.resolve():
        parser.error('--output must not overwrite the input CSV')
    print(f"Using input CSV: {input_path}")
    factors, target = prepare_data(pd.read_csv(input_path), framework, metric)
    rows = []
    for model_name in dict.fromkeys(args.cv_models):
        features = build_features(factors, framework, model_name)
        row = evaluate(
            features, target, model_name, args.n_splits, args.inner_splits,
            args.random_state, args.r2_mode, args.n_jobs,
        )
        row.update(framework=framework, metric=metric, input_csv=str(input_path))
        rows.append(row)
        print(f"{model_name}: R^2 mean={row['cv_r2_mean']:.6g} +/- {row['cv_r2_std']:.6g}")
    summary = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path, index=False)
    print(f"Saved results: {output_path}")


if __name__ == '__main__':
    main()

#!/usr/bin/env bash
set -euo pipefail

# Run everything from the EvalDims directory
BASEDIR="$(cd "$(dirname "$0")" && pwd)"
cd "$BASEDIR/" || { echo "EvalDims directory not found at $BASEDIR"; exit 1; }

# 1a. Plot relative (no wang models)
python ./utils/plot_relative.py --no_wang_models > /dev/null 2>&1
echo "Created: $BASEDIR/results/figs/relative/relative_violin.png"

# 1b. Plot relative (include the reproduced wang models)
python ./utils/plot_relative.py --wang_model_values > /dev/null 2>&1
echo "Created: $BASEDIR/results/figs/relative/relative_violin_wang_model_values.png"

# 2. Male preference bar
python ./utils/plot_male_preference_bar.py > /dev/null 2>&1
echo "Created: $BASEDIR/results/figs/relative/male_preference_bar.png"

# 3-4. Swapped module results
python ./utils/swapped_module_results.py > /dev/null 2>&1
echo "Created: $BASEDIR/results/figs/yin/swapped/swapped_module_results.png"

# 5-6. Yin meta-regression 
python ./utils/meta_regression_analysis_intercept.py --exp_framework yin --metric selection_rate --model_spec interactions --holdout 0.1 --n-splits 5 > /dev/null 2>&1
echo "Created: $BASEDIR/results/meta/yin/coef_ci_plot_intercept_modelspec_interactions_ablation_none.png"
echo "Created: $BASEDIR/results/meta/yin/meta_regression_plot_intercept_colorby_none_modelspec_interactions_ablation_none.png"

# 7-9. Density Plots with Noise Floor and on GPT-4o-mini
python ./plot_armstrong.py > /dev/null 2>&1
echo "Created: $BASEDIR/results/figs/armstrong/all/regression_coefficients_global_density_all_with_noise_floor.png"
python ./plot_yin_all.py > /dev/null 2>&1
echo "Created: $BASEDIR/results/figs/yin/all/selection_rate_global_density_all_with_noise_floor.png"
python ./plot_wang.py > /dev/null 2>&1
echo "Created: $BASEDIR/results/figs/wang/all/score_difference_global_density_all.png"


# 7-9. Density Plots with All Reproduced Model Values 
python plot_yin_all.py --reported_setting_models gpt-3.5-turbo gpt-4 > /dev/null 2>&1
echo "Created: $BASEDIR/results/figs/yin/all/selection_rate_global_density_all_with_noise_floor_with_reported_settings.png"
python plot_wang.py --reproduced_settings --full_width > /dev/null 2>&1
echo "Created: $BASEDIR/results/figs/wang/all/score_difference_global_density_all_with_noise_floor_reported_settings.png"

# 10-16. Armstrong meta-regression 
python ./utils/meta_regression_analysis_intercept.py --exp_framework armstrong --metric regression_coefficient > /dev/null 2>&1
echo "Created: $BASEDIR/results/meta/armstrong/ablation_summary_main.csv"
python ./utils/meta_regression_analysis_intercept.py --exp_framework armstrong --metric regression_coefficient --model_spec interactions > /dev/null 2>&1
echo "Created: $BASEDIR/results/meta/armstrong/meta_regression_plot_intercept_colorby_none_modelspec_interactions_ablation_none.png"

# 17-23. Wang meta-regression
python ./utils/meta_regression_analysis_intercept.py --exp_framework wang --metric score_difference > /dev/null 2>&1
echo "Created: $BASEDIR/results/meta/wang/ablation_summary_main.csv"
python ./utils/meta_regression_analysis_intercept.py --exp_framework wang --metric score_difference --model_spec interactions > /dev/null 2>&1
echo "Created: $BASEDIR/results/meta/wang/meta_regression_plot_intercept_colorby_none_modelspec_interactions_ablation_none.png"

# 24-30. Yin meta-regression ablations + interactions
python ./utils/meta_regression_analysis_intercept.py --exp_framework yin --metric selection_rate --ablation all > /dev/null 2>&1
echo "Created: $BASEDIR/results/meta/yin/ablation_summary_main.csv"

# 31. Noise Floor Results 
python ./utils/noise_floor.py > /dev/null 2>&1
echo "Created $BASEDIR/results/noise_floor_results.csv"

# 32. Permutation Importance 
python utils/meta_regression_analysis_intercept.py --feature_importance --model_spec interactions > /dev/null 2>&1 
echo "Created $BASEDIR/results/meta/yin/permutation_importance_plot_ablation_none_interactions.png" # fig
echo "Created $BASEDIR/results/meta/yin/permutation_importance_ablation_none_interactions.csv" # csv file

python utils/meta_regression_analysis_intercept.py --feature_importance > /dev/null 2>&1 
echo "Created $BASEDIR/results/meta/yin/permutation_importance_plot_ablation_none_main.png" # fig
echo "Created $BASEDIR/results/meta/yin/permutation_importance_ablation_none_main.csv" # csv file

# Elastic Net and Catboost for Yin, Armstrong, Wang
 python ./utils/meta_regression_predictors.py --exp_framework yin --metric selection_rate --cv-models catboost_interactions elasticnet_interactions
 python ./utils/meta_regression_predictors.py --exp_framework wang  --cv-models catboost_interactions elasticnet_interactions
 python ./utils/meta_regression_predictors.py --exp_framework armstrong  --cv-models catboost_interactions elasticnet_interactions


echo "All commands completed."

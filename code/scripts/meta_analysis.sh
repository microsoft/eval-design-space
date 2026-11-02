#!/bin/bash

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)" || exit 1
cd "$PROJECT_DIR" || exit 1

python utils/meta_regression_analysis_intercept.py --input results/yin_sampling_downsampling_analysis_original.csv  --holdout 0.1 --random-state 0 --n-splits 5 --exp_framework yin --model_spec interactions

python utils/meta_regression_analysis_intercept.py --input results/wang_sampling_downsampling_analysis_original.csv  --holdout 0.1 --random-state 0 --n-splits 5 --exp_framework wang --metric score_difference --model_spec interactions

python utils/meta_regression_analysis_intercept.py --input results/armstrong_sampling_downsampling_analysis_original.csv  --holdout 0.1 --random-state 0 --n-splits 5 --exp_framework armstrong --metric regression_coefficients --model_spec interactions

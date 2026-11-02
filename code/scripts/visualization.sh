#!/bin/bash

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)" || exit 1
cd "$PROJECT_DIR" || exit 1

python utils/plot_male_preference_bar.py

python utils/plot_relative.py

python plot_armstrong.py --split_by all --metric regression_coefficients

python plot_yin_all.py --split_by all --metric selection_rate --bw_threshold 0.1;
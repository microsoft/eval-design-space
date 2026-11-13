#!/bin/bash

set -euo pipefail

SESSION="generate_results"
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

# Each variant has its own output location so parallel runs cannot overwrite it.
# --holdout is deprecated; the two Yin regression runs are retained separately.
# interactions is a model specification, not an ablation.

commands=(
    "python utils/plot_relative.py --no_wang_models --output results/figs/relative/relative_violin_no_wang_models.png"
    "python utils/plot_male_preference_bar.py"
    "python utils/swapped_module_results.py"
    "python utils/meta_regression_analysis_intercept.py --exp_framework yin --metric selection_rate --model_spec interactions
    "python plot_armstrong.py --output_dir results/figs/armstrong/default"
    "python plot_yin_all.py --output_dir results/figs/yin/default"
    "python plot_wang.py --output_dir results/figs/wang/default"
    "python plot_armstrong.py --metric regression_coefficients --split_by all --output_dir results/figs/armstrong/regression_coefficients_all"
    "python plot_yin_all.py --metric selection_rate --split_by all --reported_setting_models gpt-3.5-turbo gpt-4 --output_dir results/figs/yin/reported_settings"
    "python utils/noise_floor.py"
    "python utils/plot_relative.py --output results/figs/relative/relative_violin.png"
    "python plot_wang.py --metric score_difference --split_by all --reproduced_settings --full_width --output_dir results/figs/wang/reproduced_settings_full_width"
    "python utils/meta_regression_predictors.py --exp_framework yin"
    "python utils/meta_regression_predictors.py --exp_framework armstrong"
    "python utils/meta_regression_predictors.py --exp_framework wang"
    "python utils/feature_importance.py --exp_framework yin --metric selection_rate --model_spec main --ablation none --feature_importance"
    "python utils/feature_importance.py --exp_framework yin --metric selection_rate --model_spec interactions --ablation none
    "python utils/feature_importance.py --exp_framework wang --metric score_difference --model_spec main --ablation all"
    "python utils/feature_importance.py --exp_framework wang --metric score_difference --model_spec interactions --ablation all"
    "python utils/feature_importance.py --exp_framework armstrong --metric regression_coefficients --model_spec main --ablation all"
    "python utils/feature_importance.py --exp_framework armstrong --metric regression_coefficients --model_spec interactions --ablation all"
    # python main.py --name armstrong --job armstrong --model gpt-3.5-turbo-0125 --base_config_path configs/base_armstrong.yaml
    # "python main.py --name seshadri --job wen --model Mistral-7B-Instruct-v0.3 --base_config_path configs/base_wang.yaml"
    # "python main.py --name seshadri --job rozado --model Mistral-7B-Instruct-v0.3 --base_config_path configs/base_wang.yaml"
    # "python main.py --name lippens --job wang --model Mistral-7B-Instruct-v0.3 --base_config_path configs/base_wang.yaml"
    # "python main.py --name lippens --job rozado --model Mistral-7B-Instruct-v0.3 --base_config_path configs/base_wang.yaml"
    # "python main.py --name lippens --job zollo --model Mistral-7B-Instruct-v0.3 --base_config_path configs/base_wang.yaml"
    # "python main.py --name gaeb --job rozado --model Mistral-7B-Instruct-v0.3 --base_config_path configs/base_wang.yaml"
    # "python main.py --name gaeb --job wang --model Mistral-7B-Instruct-v0.3 --base_config_path configs/base_wang.yaml"
    # "python main.py --name yin --job rozado --model Mistral-7B-Instruct-v0.3 --base_config_path configs/base_wang.yaml"
    # "python main.py --name yin --job wang --model Mistral-7B-Instruct-v0.3 --base_config_path configs/base_wang.yaml"
    # "python main.py --name zollo --job armstrong --model Mistral-7B-Instruct-v0.3 --base_config_path configs/base_wang.yaml"
    # "python main.py --name zollo --job rozado --model Mistral-7B-Instruct-v0.3 --base_config_path configs/base_wang.yaml"
    # "python main.py --name karvonen --job rozado --model Mistral-7B-Instruct-v0.3 --base_config_path configs/base_wang.yaml"
    # "python main.py --name armstrong --job karvonen --model Mistral-7B-Instruct-v0.3 --base_config_path configs/base_wang.yaml"
    # "python main.py --name armstrong --job zollo --model Mistral-7B-Instruct-v0.3 --base_config_path configs/base_wang.yaml"
    # "python main.py --name armstrong --job yin --model Mistral-7B-Instruct-v0.3 --base_config_path configs/base_wang.yaml"
    # "python main.py --name armstrong --job rozado --model Mistral-7B-Instruct-v0.3 --base_config_path configs/base_wang.yaml"
    # "python main.py --name armstrong --job wang --model Mistral-7B-Instruct-v0.3 --base_config_path configs/base_wang.yaml"


    # "python main.py --name karvonen --job wang --model gpt-5-nano-2025-08-07 --base_config_path configs/base_armstrong.yaml"
    # "python main.py --name karvonen --job wen --model gpt-5-nano-2025-08-07 --base_config_path configs/base_armstrong.yaml"
    # "python main.py --name rozado --job rozado --model gpt-5-nano-2025-08-07 --base_config_path configs/base_armstrong.yaml"
    # "python main.py --name rozado --job wang --model gpt-5-nano-2025-08-07 --base_config_path configs/base_armstrong.yaml"
    # "python main.py --name wang --job armstrong --model gpt-4o-mini --base_config_path configs/base_armstrong.yaml"
    # "python main.py --name wang --job wen --model gpt-4o-mini --base_config_path configs/base_armstrong.yaml"
    # "python main.py --name wen --job karvonen --model gpt-5-nano-2025-08-07 --base_config_path configs/base_armstrong.yaml"
    # "python main.py --name wen --job wen --model gpt-5-nano-2025-08-07 --base_config_path configs/base_armstrong.yaml"
    # "python main.py --name yin --job yin --model gpt-5-nano-2025-08-07 --base_config_path configs/base_armstrong.yaml"
    # "python main.py --name zollo --job wen --model gpt-5-nano-2025-08-07 --base_config_path configs/base_armstrong.yaml"


    # "python main.py --name wang --job yin --model Meta-Llama-3.1-8B-Instruct --base_config_path configs/base_wang.yaml"
    # "python main.py --name wang --job zollo --model Meta-Llama-3.1-8B-Instruct --base_config_path configs/base_wang.yaml"
    # "python main.py --name wang --job karvonen --model Meta-Llama-3.1-8B-Instruct --base_config_path configs/base_wang.yaml"
    # "python main.py --name wang --job wang --model Meta-Llama-3.1-8B-Instruct --base_config_path configs/base_wang.yaml"
    # "python main.py --name wang --job wen --model Meta-Llama-3.1-8B-Instruct --base_config_path configs/base_wang.yaml"
    # "python main.py --name wang --job rozado --model Meta-Llama-3.1-8B-Instruct --base_config_path configs/base_wang.yaml"


    # "python main.py --name wang --job yin --model Mistral-Small-24B-Instruct --base_config_path configs/base_wang.yaml"
    # "python main.py --name wang --job zollo --model Mistral-Small-24B-Instruct --base_config_path configs/base_wang.yaml"
    # "python main.py --name wang --job karvonen --model Mistral-Small-24B-Instruct --base_config_path configs/base_wang.yaml"
    # "python main.py --name wang --job wang --model Mistral-Small-24B-Instruct --base_config_path configs/base_wang.yaml"
    # "python main.py --name wang --job wen --model Mistral-Small-24B-Instruct --base_config_path configs/base_wang.yaml"
    # "python main.py --name wang --job rozado --model Mistral-Small-24B-Instruct --base_config_path configs/base_wang.yaml"

    # "python main.py --name lippens --job karvonen --model gpt-4o-mini --base_config_path configs/base_wang.yaml"
    # "python main.py --name lippens --job rozado --model gpt-4o-mini --base_config_path configs/base_wang.yaml"
    # "python main.py --name lippens --job wen --model gpt-4o-mini --base_config_path configs/base_wang.yaml"
    # "python main.py --name lippens --job yin --model gpt-4o-mini --base_config_path configs/base_wang.yaml"
    # "python main.py --name lippens --job zollo --model gpt-4o-mini --base_config_path configs/base_wang.yaml"
    # "python main.py --name karvonen --job wen --model gpt-5-nano-2025-08-07 --base_config_path configs/base_armstrong.yaml"
    # "python main.py --name rozado --job rozado --model gpt-5-nano-2025-08-07 --base_config_path configs/base_armstrong.yaml"
    # "python main.py --name wen --job karvonen --model gpt-5-nano-2025-08-07 --base_config_path configs/base_armstrong.yaml"
    # "python main.py --name wen --job wen --model gpt-5-nano-2025-08-07 --base_config_path configs/base_armstrong.yaml"
    # "python main.py --name yin --job yin --model gpt-5-nano-2025-08-07 --base_config_path configs/base_armstrong.yaml"
    # "python main.py --name zollo --job zollo --model gpt-5-nano-2025-08-07 --base_config_path configs/base_armstrong.yaml"
    
)




total=${#commands[@]}
max_windows=7

if [ $total -lt $max_windows ]; then
    window_count=$total
else
    window_count=$max_windows
fi

declare -a window_cmds
for ((w=0; w<window_count; w++)); do
    window_cmds[$w]=""
done

for ((i=0; i<total; i++)); do
    w=$((i % window_count))
    if [ -z "${window_cmds[$w]}" ]; then
        window_cmds[$w]="${commands[$i]}"
    else
        window_cmds[$w]="${window_cmds[$w]} && ${commands[$i]}"
    fi
done

tmux new-session -d -s "$SESSION"
printf -v quoted_project_dir '%q' "$PROJECT_DIR"

for ((w=0; w<window_count; w++)); do
    cmd="${window_cmds[$w]}"
    win_name="win_$w"

    if [ $w -eq 0 ]; then
        tmux rename-window -t $SESSION:0 "$win_name"
    else
        tmux new-window -t $SESSION -n "$win_name"
    fi

    tmux send-keys -t "$SESSION:$win_name" "cd ${quoted_project_dir} && ${cmd}" C-m
done

echo "Launched $total commands across $window_count tmux windows."
echo "Attach with: tmux attach -t $SESSION"




# "python main.py --name armstrong --job armstrong --model Mistral-7B-Instruct-v0.2"
# "python main.py --name armstrong --job rozado --model Mistral-7B-Instruct-v0.2"
# "python main.py --name armstrong --job wang --model Mistral-7B-Instruct-v0.2"
# "python main.py --name armstrong --job wen --model Mistral-7B-Instruct-v0.2"
# "python main.py --name armstrong --job karvonen --model Mistral-7B-Instruct-v0.2"
# "python main.py --name armstrong --job zollo --model Mistral-7B-Instruct-v0.2"
# "python main.py --name armstrong --job yin --model Mistral-7B-Instruct-v0.2"

# "python main.py --name rozado --job armstrong --model Mistral-7B-Instruct-v0.2"
# "python main.py --name rozado --job rozado --model Mistral-7B-Instruct-v0.2"
# "python main.py --name rozado --job wang --model Mistral-7B-Instruct-v0.2"
# "python main.py --name rozado --job wen --model Mistral-7B-Instruct-v0.2"
# "python main.py --name rozado --job karvonen --model Mistral-7B-Instruct-v0.2"
# "python main.py --name rozado --job zollo --model Mistral-7B-Instruct-v0.2"
# "python main.py --name rozado --job yin --model Mistral-7B-Instruct-v0.2"

# "python main.py --name wang --job armstrong --model Mistral-7B-Instruct-v0.2"
# "python main.py --name wang --job rozado --model Mistral-7B-Instruct-v0.2"
# "python main.py --name wang --job wang --model Mistral-7B-Instruct-v0.2"
# "python main.py --name wang --job wen --model Mistral-7B-Instruct-v0.2"
# "python main.py --name wang --job karvonen --model Mistral-7B-Instruct-v0.2"
# "python main.py --name wang --job zollo --model Mistral-7B-Instruct-v0.2"
# "python main.py --name wang --job yin --model Mistral-7B-Instruct-v0.2"

# "python main.py --name wen --job armstrong --model Mistral-7B-Instruct-v0.2"
# "python main.py --name wen --job rozado --model Mistral-7B-Instruct-v0.2"
# "python main.py --name wen --job wang --model Mistral-7B-Instruct-v0.2"
# "python main.py --name wen --job wen --model Mistral-7B-Instruct-v0.2"
# "python main.py --name wen --job karvonen --model Mistral-7B-Instruct-v0.2"
# "python main.py --name wen --job zollo --model Mistral-7B-Instruct-v0.2"
# "python main.py --name wen --job yin --model Mistral-7B-Instruct-v0.2"

# "python main.py --name karvonen --job armstrong --model Mistral-7B-Instruct-v0.2"
# "python main.py --name karvonen --job rozado --model Mistral-7B-Instruct-v0.2"
# "python main.py --name karvonen --job wang --model Mistral-7B-Instruct-v0.2"
# "python main.py --name karvonen --job wen --model Mistral-7B-Instruct-v0.2"
# "python main.py --name karvonen --job karvonen --model Mistral-7B-Instruct-v0.2"
# "python main.py --name karvonen --job zollo --model Mistral-7B-Instruct-v0.2"
# "python main.py --name karvonen --job yin --model Mistral-7B-Instruct-v0.2"

# "python main.py --name zollo --job armstrong --model Mistral-7B-Instruct-v0.2"
# "python main.py --name zollo --job rozado --model Mistral-7B-Instruct-v0.2"
# "python main.py --name zollo --job wang --model Mistral-7B-Instruct-v0.2"
# "python main.py --name zollo --job wen --model Mistral-7B-Instruct-v0.2"
# "python main.py --name zollo --job karvonen --model Mistral-7B-Instruct-v0.2"
# "python main.py --name zollo --job zollo --model Mistral-7B-Instruct-v0.2"
# "python main.py --name zollo --job yin --model Mistral-7B-Instruct-v0.2"

# "python main.py --name yin --job armstrong --model Mistral-7B-Instruct-v0.2"
# "python main.py --name yin --job rozado --model Mistral-7B-Instruct-v0.2"
# "python main.py --name yin --job wang --model Mistral-7B-Instruct-v0.2"
# "python main.py --name yin --job wen --model Mistral-7B-Instruct-v0.2"
# "python main.py --name yin --job karvonen --model Mistral-7B-Instruct-v0.2"
# "python main.py --name yin --job zollo --model Mistral-7B-Instruct-v0.2"
# "python main.py --name yin --job yin --model Mistral-7B-Instruct-v0.2"

# "python main.py --name gaeb --job armstrong --model Mistral-7B-Instruct-v0.2"
# "python main.py --name gaeb --job rozado --model Mistral-7B-Instruct-v0.2"
# "python main.py --name gaeb --job wang --model Mistral-7B-Instruct-v0.2"
# "python main.py --name gaeb --job wen --model Mistral-7B-Instruct-v0.2"
# "python main.py --name gaeb --job karvonen --model Mistral-7B-Instruct-v0.2"
# "python main.py --name gaeb --job zollo --model Mistral-7B-Instruct-v0.2"
# "python main.py --name gaeb --job yin --model Mistral-7B-Instruct-v0.2"

# "python main.py --name lippens --job armstrong --model Mistral-7B-Instruct-v0.2"
# "python main.py --name lippens --job wang --model Mistral-7B-Instruct-v0.2"
# "python main.py --name lippens --job wen --model Mistral-7B-Instruct-v0.2"
# "python main.py --name lippens --job rozado --model Mistral-7B-Instruct-v0.2"
# "python main.py --name lippens --job zollo --model Mistral-7B-Instruct-v0.2"
# "python main.py --name lippens --job yin --model Mistral-7B-Instruct-v0.2"
# "python main.py --name lippens --job karvonen --model Mistral-7B-Instruct-v0.2"

# "python main.py --name seshadri --job armstrong --model Mistral-7B-Instruct-v0.2"
# "python main.py --name seshadri --job wang --model Mistral-7B-Instruct-v0.2"
# "python main.py --name seshadri --job wen --model Mistral-7B-Instruct-v0.2"
# "python main.py --name seshadri --job rozado --model Mistral-7B-Instruct-v0.2"
# "python main.py --name seshadri --job zollo --model Mistral-7B-Instruct-v0.2"
# "python main.py --name seshadri --job yin --model Mistral-7B-Instruct-v0.2"
# "python main.py --name seshadri --job karvonen --model Mistral-7B-Instruct-v0.2"

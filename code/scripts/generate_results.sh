#!/bin/bash

set -u

SESSION="gpt_5o_wang"
PROJECT_DIR="/nlp/scr/nmeist/EvalDims"


commands=(
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
        window_cmds[$w]="${window_cmds[$w]}; ${commands[$i]}"
    fi
done

tmux new-session -d -s $SESSION

for ((w=0; w<window_count; w++)); do
    cmd="${window_cmds[$w]}"
    win_name="win_$w"

    if [ $w -eq 0 ]; then
        tmux rename-window -t $SESSION:0 "$win_name"
    else
        tmux new-window -t $SESSION -n "$win_name"
    fi

    tmux send-keys -t $SESSION:$win_name "cd ${PROJECT_DIR} && ${cmd}" C-m
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


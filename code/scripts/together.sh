#!/usr/bin/env bash

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)" || exit 1
cd "$PROJECT_DIR" || exit 1

python together/together_main_batch.py # might need to edit this to create the batch_info json file and fill it with an empty list ? might need to edit the arg parses
python together/together_process_batch.py --author wang
python together/together_process_outputs.py
# make sure these are launched sequentuially 
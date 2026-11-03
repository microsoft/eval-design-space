from pathlib import Path

import json
import os

import pandas as pd
import random
from tqdm import tqdm
import re

import argparse
from utils.variables import MODELS, NAMES, JOBS
from utils.paths import repo_path



def process_armstrong():
    batch_outputs_dir = repo_path('output_data/armstrong/together/batch_outputs')

    for model_key in MODELS.keys():
        model = MODELS[model_key].split("/")[-1]
        for name in NAMES:
            for job in JOBS:
                try: 
                    output_filepath=f"{batch_outputs_dir}/{model}/name_{name}_job_{job}.jsonl"
                    with open(output_filepath, "r", encoding="utf-8") as f:
                        pass 
                except Exception as e:
                    breakpoint()
                try: 
                    gpt_4o_example_filepath = repo_path(f'output_data/armstrong/model_gpt-4o-2024-11-20_temp_1.0_numtrials_10_jobresumes_{job}_names_{name}_random_state_42.csv')
                    gpt_4o_example_data = pd.read_csv(gpt_4o_example_filepath)
                except Exception as e:
                    gpt_4o_example_filepath = repo_path(f'output_data/armstrong/model_gpt-5-nano-2025-08-07_temp_1.0_numtrials_10_jobresumes_{job}_names_{name}_random_state_42.csv')
                    gpt_4o_example_data = pd.read_csv(gpt_4o_example_filepath)
                # make a copy of the dataframe: will put the together outputs in the value column 
                df = gpt_4o_example_data.copy()

                    
                with open(output_filepath, "r", encoding="utf-8") as f:
                    for i, line in enumerate(filter(str.strip, f)):
                        obj = json.loads(line)
                        # get the content value
                        # remove the newline characters
                        model_response = obj["response"]["body"]["choices"][0]["message"]["content"]
                        if model_response is not None:
                            temp = re.findall(r'\d+', model_response)
                            value = list(map(int, temp))[0] if temp else None
                        else:
                            value = None    

                        df.loc[i, "value"] = value
                # save the dataframe
                # create the directory if it doesn't exist

                os.makedirs(repo_path(f"output_data/armstrong/together/processed_batch_outputs/{model}"), exist_ok=True)
                df.to_csv(repo_path(f"output_data/armstrong/together/processed_batch_outputs/{model}/name_{name}_job_{job}.csv"), index=False)

def process_yin(data_dir=None):
    data_dir = Path(repo_path(data_dir or 'output_data/yin'))
    batch_outputs_dir = data_dir / 'together' / 'batch_outputs'
    batch_inputs_dir = data_dir / 'together' / 'batch_inputs'
    processed_batch_outputs_dir = data_dir / 'together' / 'processed_batch_outputs'

    for model_key in MODELS.keys():
        model = MODELS[model_key].split("/")[-1]
        if not (batch_outputs_dir / model).is_dir():
            continue
        list_of_files = os.listdir(batch_outputs_dir / model)
        list_of_metadata_files = os.listdir(batch_inputs_dir)
        for name_bundle in NAMES:
            for job_bundle in JOBS:

                list_of_files_filtered = sorted(
                    file for file in list_of_files
                    if re.fullmatch(
                        re.escape(f"name_{name_bundle}_job_{job_bundle}") + r"(?:_part\d+)?\.jsonl",
                        file,
                    )
                )
                metadata_prefix = f"metadata_{name_bundle}_{job_bundle}_{model_key.replace('-', '_')}"
                list_of_files_filtered_metadata = sorted(
                    file for file in list_of_metadata_files
                    if re.fullmatch(re.escape(metadata_prefix) + r"(?:_part\d+)?\.json", file)
                )
                if not list_of_files_filtered:
                    continue

                metadata = {}
                # read in the metadata but have each custom id be a key and the value be the metadata
                for file in list_of_files_filtered_metadata:
                    with open(batch_inputs_dir / file, "r", encoding="utf-8") as f:
                        try:
                            data = json.load(f)
                            for obj in data:
                                if obj["custom_id"] in metadata and metadata[obj["custom_id"]] != obj:
                                    raise ValueError(f"Conflicting Yin metadata in {file}: {obj['custom_id']}")
                                metadata[obj["custom_id"]] = obj
                        except json.JSONDecodeError as e:
                            print(f"Error reading JSON from {file}: {e}")
                            continue

                missing_metadata = 0
                for file in list_of_files_filtered:
                    with open(batch_outputs_dir / model / file, "r", encoding="utf-8") as f:
                        for line in f:
                            if not line.strip():
                                continue
                            obj = json.loads(line)
                            custom_id = obj["custom_id"]
                            if custom_id not in metadata:
                                missing_metadata += 1
                                continue
                            records = {'choices': [{'message': {'content': obj['response']['body']['choices'][0]['message']['content']}}]}
                            records['context'] = {
                                'job': metadata[custom_id]["job"],
                                'default_order': metadata[custom_id]["default_order_names"],
                                'demo_order': metadata[custom_id]["demo_order"],
                                'candidate_ids': metadata[custom_id].get("candidate_ids"),
                            }
                            job = metadata[custom_id]["job"]
                            i = metadata[custom_id]["iteration"]
                            dir_out = (processed_batch_outputs_dir / model /
                                       f'temp_1.0_names_{name_bundle}_jobresumes_{job_bundle}' / job)
                            os.makedirs(dir_out, exist_ok=True)
                            fn_out = os.path.join(dir_out, f"run_{i}.json")
                            with open(fn_out, "w", encoding="utf-8") as f:
                                f.write(json.dumps(records))
                if missing_metadata:
                    print(f"Skipped {missing_metadata} Yin responses without matching metadata: "
                          f"{model}, {name_bundle}, {job_bundle}")


    return 


def _extract_numeric_score(model_response):
    if model_response is None:
        return None

    # Prefer explicit JSON-like score fields first
    score_match = re.search(r'"Score"\s*:\s*(\d+)', str(model_response), flags=re.IGNORECASE)
    if score_match:
        return int(score_match.group(1))

    # Fallback: first integer in response text
    temp = re.findall(r'\d+', str(model_response))
    return int(temp[0]) if temp else None


def process_wang():
    batch_outputs_dir = repo_path('output_data/wang/together/batch_outputs')
    batch_inputs_dir = repo_path('output_data/wang/together/batch_inputs')
    processed_batch_outputs_dir = repo_path('output_data/wang/together/processed_batch_outputs')

    NAMES = ['wang']


    for model_key in MODELS.keys():
        model = MODELS[model_key].split('/')[-1]
        model_batch_output_dir = f"{batch_outputs_dir}/{model}"
        if not os.path.exists(model_batch_output_dir):
            continue

        list_of_files = os.listdir(model_batch_output_dir)
        list_of_metadata_files = os.listdir(batch_inputs_dir)

        for name_bundle in NAMES:
            for job_bundle in JOBS:
                # batch outputs are split by parts; collect all matching files
                list_of_files_filtered = [
                    file for file in list_of_files
                    if file.startswith(f"name_{name_bundle}_job_{job_bundle}")
                ]
                list_of_files_filtered_metadata = [
                    file for file in list_of_metadata_files
                    if file.startswith(f"metadata_{name_bundle}_{job_bundle}")
                ]

                if not list_of_files_filtered:
                    continue

                metadata = {}
                for file in list_of_files_filtered_metadata:
                    with open(f"{batch_inputs_dir}/{file}", 'r', encoding='utf-8') as f:
                        try:
                            data = json.load(f)
                            for obj in data:
                                metadata[obj['custom_id']] = obj
                        except json.JSONDecodeError as e:
                            print(f"Error reading JSON from {file}: {e}")
                            continue

                if not metadata:
                    print(f"No metadata found for name={name_bundle}, job={job_bundle}, model={model}")
                    continue

                # Key rows by (occupation, resume_index, pair_index)
                # Values store role and privilege/protect score lists.
                rows = {}

                for file in list_of_files_filtered:
                    with open(f"{model_batch_output_dir}/{file}", 'r', encoding='utf-8') as f:
                        for line in f:
                            if not line.strip():
                                continue

                            obj = json.loads(line)
                            custom_id = obj.get('custom_id')
                            if custom_id not in metadata:
                                continue

                            meta = metadata[custom_id]
                            occupation = meta.get('occupation', 'Unknown')
                            resume_index = meta.get('resume_index', -1)
                            pair_index = meta.get('pair_index', 0)
                            role = meta.get('role', occupation)
                            label_type = meta.get('label_type')  # 'Privilege' or 'Protect'
                            label_value = meta.get('label_value')
                            group_name = meta.get('group_name')

                            model_response = obj['response']['body']['choices'][0]['message']['content']
                            value = _extract_numeric_score(model_response)

                            row_key = (occupation, resume_index, pair_index)
                            if row_key not in rows:
                                rows[row_key] = {
                                    'Occupation': occupation,
                                    'Role': role,
                                    'resume_index': resume_index,
                                    'pair_index': pair_index,
                                    'group_name': group_name,
                                    'Privilege_Scores': [],
                                    'Protect_Scores': [],
                                    'privilege_label': None,
                                    'protect_label': None,
                                }

                            if label_type == 'Privilege':
                                rows[row_key]['Privilege_Scores'].append(value)
                                rows[row_key]['privilege_label'] = label_value
                            elif label_type == 'Protect':
                                rows[row_key]['Protect_Scores'].append(value)
                                rows[row_key]['protect_label'] = label_value

                if not rows:
                    continue

                df = pd.DataFrame(list(rows.values()))

                # Keep only rows where both labels are present
                df = df[
                    (df['Privilege_Scores'].apply(lambda x: isinstance(x, list) and len(x) > 0)) &
                    (df['Protect_Scores'].apply(lambda x: isinstance(x, list) and len(x) > 0))
                ].copy()

                if len(df) == 0:
                    continue

                df['Privilege_Avg_Score'] = df['Privilege_Scores'].apply(
                    lambda x: float(pd.Series([v for v in x if v is not None]).mean()) if len([v for v in x if v is not None]) > 0 else None
                )
                df['Protect_Avg_Score'] = df['Protect_Scores'].apply(
                    lambda x: float(pd.Series([v for v in x if v is not None]).mean()) if len([v for v in x if v is not None]) > 0 else None
                )

                df = df.dropna(subset=['Privilege_Avg_Score', 'Protect_Avg_Score']).copy()
                if len(df) == 0:
                    continue

                ranks = df[['Privilege_Avg_Score', 'Protect_Avg_Score']].rank(axis=1, ascending=False)
                df['Privilege_Rank'] = ranks['Privilege_Avg_Score']
                df['Protect_Rank'] = ranks['Protect_Avg_Score']

                # Score difference: positive means privilege group scored higher
                df['Score_Diff'] = df['Privilege_Avg_Score'] - df['Protect_Avg_Score']

                # Keep shape close to Wang generated data and add provenance
                df['model'] = model_key
                df['name'] = name_bundle
                df['job'] = job_bundle

                os.makedirs(f"{processed_batch_outputs_dir}/{model}", exist_ok=True)
                out_fp = f"{processed_batch_outputs_dir}/{model}/name_{name_bundle}_job_{job_bundle}.csv"
                df.to_csv(out_fp, index=False)

    return


def __main__():

    parser = argparse.ArgumentParser()
    parser.add_argument("--author", type=str, default="wang")
    parser.add_argument("--data-dir", type=Path, help="Yin output_data directory")
    args = parser.parse_args()

    if args.author == 'armstrong':
        process_armstrong()
    elif args.author == 'yin':
        process_yin(args.data_dir)
    elif args.author == 'wang':
        process_wang()

    return 

if __name__ == "__main__":
    __main__()
import json
import re
import time
import json_repair
import pandas as pd
from tqdm import tqdm


def create_summary(group_name, label, occupation, row, template):
    """Generate a dynamic summary for scoring the applicant, excluding the group feature.
       The occupation parameter allows customization of the job position.
    """

    resume_info = row['Cleaned_Resume']
    # resume_info = resume_info[:int(len(resume_info) * proportion)]

    info = f"{group_name}: {label};" if group_name else f"{label}"
    
    summary = template.format(
        role=row['Role'],
        counterfactual_info=info,
        resume_info=resume_info
    )

    return summary


def invoke_retry(prompt, agent, parameters, string_input=False):
    import gc
    attempts = 0
    delay = 2  # Initial delay in seconds
    max_attempts = 5  # Maximum number of retry attempts

    while attempts < max_attempts:
        try:
            score_text = agent.invoke(prompt, **parameters)
            
            # Check if None was returned (API call failed)
            if score_text is None:
                print(f"Attempt {attempts + 1} returned None from API")
                time.sleep(delay)
                delay *= 2
                attempts += 1
                gc.collect()  # Cleanup after failed attempt
                continue
            
            #print(f"Prompt: {prompt}")
            # print(f"Score text: {score_text}")
            # print("=============================================================")
            if string_input:
                return score_text
            if not isinstance(score_text, str):
                raise Exception(f"Non-string API response type: {type(score_text).__name__}")

            # Fast/robust parse path to avoid expensive JSON repair on long outputs.
            m = re.search(r'"Score"\s*:\s*(\d+)', score_text)
            if m:
                result = int(m.group(1))
                del score_text  # Clear from memory
                gc.collect()
                return result

            try:
                score_json = json.loads(score_text)
                result = int(score_json['Score'])
                del score_text, score_json  # Clear from memory
                gc.collect()
                return result
            except Exception:
                # Only attempt json repair for reasonably small payloads.
                if len(score_text) <= 10000:
                    try:
                        repaired = json_repair.repair_json(score_text, skip_json_loads=True, return_objects=False)
                        score_json = json.loads(repaired)
                        result = int(score_json['Score'])
                        del score_text, repaired, score_json  # Clear from memory
                        gc.collect()
                        return result
                    except Exception:
                        pass

                # Final fallback: first integer in response.
                m_any = re.search(r'\b(\d{1,3})\b', score_text)
                if m_any:
                    result = int(m_any.group(1))
                    del score_text  # Clear from memory
                    gc.collect()
                    return result
                raise Exception("Could not extract numeric score from model output.")

        except Exception as e:
            print(f"Attempt {attempts + 1} failed: {e}")
            time.sleep(delay)
            delay *= 2  # Exponential increase of the delay
            attempts += 1
            gc.collect()  # Cleanup after each failed attempt

    print(f"Failed to complete API call after {max_attempts} attempts. Returning -1.")
    return -1


def calculate_avg_score(score_list):
    if isinstance(score_list, list) and score_list:
        valid_scores = [score for score in score_list if score is not None]
        if valid_scores:
            avg_score = sum(valid_scores) / len(valid_scores)
            return avg_score
    return None


def process_scores_multiple(df, num_run, parameters, privilege_label, protect_label, agent, group_name, occupation
                            , template, model, name, job):
    import sys
    import gc
    print(f"Processing {len(df)} entries with {num_run} runs each.", flush=True)
    print(f"privilege_label type: {type(privilege_label)}, value: {privilege_label}", flush=True)
    print(f"protect_label type: {type(protect_label)}, value: {protect_label}", flush=True)
    """ Process entries and compute scores concurrently, with progress updates. """
    scores = {key: [[] for _ in range(len(df))] for key in ['Privilege', 'Protect']}

    try:
        for run in tqdm(range(num_run), desc="Processing runs", unit="run"):
            print(f"Starting run {run + 1}/{num_run}", flush=True)
            for index, (idx, row) in tqdm(enumerate(df.iterrows()), total=len(df), desc="Processing entries", unit="entry"):
                print(f"  Processing entry {index + 1}/{len(df)}", flush=True)
                if isinstance(privilege_label, str):
                    # privilege_label is a single string
                    print(f"    Single label mode", flush=True)
                    for key, label in zip(['Privilege', 'Protect'], [privilege_label, protect_label]):
                        print(f"      Calling API for {key}: {label}", flush=True)
                        prompt_normal = create_summary(group_name, label, occupation, row, template)
                        result_normal = invoke_retry(prompt_normal, agent, parameters)
                        print(f"      Got result: {result_normal}", flush=True)
                        scores[key][index].append(result_normal)
                        del prompt_normal  # Cleanup large string immediately
                        gc.collect()

                else: 
                    # privilege_label is a list of names, iterate through pairs
                    print(f"    List label mode with {len(privilege_label)} pairs", flush=True)
                    for pair_idx, (priv_label, prot_label) in enumerate(zip(privilege_label, protect_label)):
                        print(f"      Processing pair {pair_idx + 1}: {priv_label} vs {prot_label}", flush=True)
                        for key, label in zip(['Privilege', 'Protect'], [priv_label, prot_label]):
                            print(f"        Calling API for {key}: {label}", flush=True)
                            prompt_normal = create_summary(group_name, label, occupation, row, template)
                            result_normal = invoke_retry(prompt_normal, agent, parameters)
                            print(f"        Got result: {result_normal}", flush=True)
                            scores[key][index].append(result_normal)
                            del prompt_normal  # Cleanup large string immediately
                            gc.collect()
        
        # Ensure all scores are lists and calculate average scores
        for category in ['Privilege', 'Protect']:
            # Ensure the scores are lists and check before assignment
            series_data = [lst if isinstance(lst, list) else [lst] for lst in scores[category]]
            df[f'{category}_Scores'] = series_data

            # Calculate the average score with additional debug info
            df[f'{category}_Avg_Score'] = df[f'{category}_Scores'].apply(calculate_avg_score)
            
            
            # Clear from memory after processing
            del series_data

        
        df['privilege_label'] = str(privilege_label)
        df['protect_label'] = str(protect_label)

        
        
        # Clear scores from memory
        del scores
        gc.collect()

        # Add ranks for each score within each row
        ranks = df[['Privilege_Avg_Score', 'Protect_Avg_Score']].rank(axis=1, ascending=False)

        df['Privilege_Rank'] = ranks['Privilege_Avg_Score']
        df['Protect_Rank'] = ranks['Protect_Avg_Score']
        df['model'] = model
        df['name'] = name
        # df['job'] = job
    
        
        del ranks
        gc.collect()
        
        return df

    except Exception as e:
        print(f"FATAL ERROR in process_scores_multiple: {type(e).__name__}: {e}", flush=True)
        import traceback
        traceback.print_exc()
        raise


import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.paths import REPO_ROOT, repo_path

import yaml
import pandas as pd
import os
import numpy as np
from scipy.stats import wasserstein_distance
import itertools
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import mixedlm
from utils.merge_config import merge_config
from scipy.stats import ttest_ind
import glob
from collections import Counter
from IPython.display import display, HTML
from tqdm import tqdm
import json
from utils.variables import MODELS
from utils.yin_ranking import parse_yin_demographic_ranking

columns = [
    "Author",
    "Names",
    "Jobs",
    "Num_Trials",
    "Model",
    "Temperature",
    "Metric",
    "Sensitive_Attribute_Vector",
    "Result_Vector",
    "Std_Error_Vector",
    "P_Value_Vector"
]

def downsample_data(df, config):
    if config['experimental_framework'] == 'armstrong':

        # first ensure that the 



        if config['Name']['Bundle_Name'] == 'wang':  
            # for each unique job in df['role'] 
            unique_jobs = df['role'].unique()
            sampled_rows = []
            for job in unique_jobs:
                # sample 1 value from each unique name
                job_rows = df[df['role'] == job]
                unique_names = job_rows['name'].unique()
                for name in unique_names:
                    name_rows = job_rows[job_rows['name'] == name]
                    sampled_row = name_rows.sample(n=1, random_state=42)
                    sampled_rows.append(sampled_row)
            df = pd.concat(sampled_rows, ignore_index=True)
            return df
        else: 
            return df



def compute_ttest(df, sensitive_attribute_vector):
    results, std, pvals = [], [], []
    for attr in sensitive_attribute_vector:
        if attr == 'Woman':
            group1 = df[df['gender'] == 'women']['value']
            group2 = df[df['gender'] == 'men']['value']
            ttest = ttest_ind(group1, group2)
            results.append(ttest.statistic)
            pvals.append(ttest.pvalue)
            # Standard error of difference in means, Welch's formula
            n1 = len(group1)
            n2 = len(group2)
            s1_sq = np.var(group1, ddof=1)
            s2_sq = np.var(group2, ddof=1)
            std_err = np.sqrt(s1_sq / n1 + s2_sq / n2)
            std.append(std_err)

    return results, std, pvals

def score_diff(df, sensitive_attribute_vector):
    
    # Initialize lists to store our three metrics
    results = [] # Will store the Mean Difference (e.g., -5.2 points)
    std = []     # Will store the Standard Error
    pvals = []   # Will store the p-value
    
    for attr in sensitive_attribute_vector:
        
        # 1. Define the two groups based on the attribute
        if attr == 'Woman' or attr.startswith('Woman_'):
            group_target = df[df['gender'] == 'women']['value']
            group_ref = df[df['gender'] == 'men']['value']
            
        elif attr == 'Asian':
            group_target = df[df['race'] == 'asian']['value']
            group_ref = df[df['race'] == 'white']['value']
            
        elif attr == 'Black':
            group_target = df[df['race'] == 'black']['value']
            group_ref = df[df['race'] == 'white']['value']
            
        elif attr == 'Hispanic':
            group_target = df[df['race'] == 'hispanic']['value']
            group_ref = df[df['race'] == 'white']['value']
            
        else:
            # Append None if attribute is not recognized
            results.append(None)
            std.append(None)
            pvals.append(None)
            continue

        # 2. Perform calculations if both groups have data
        if len(group_target) > 0 and len(group_ref) > 0:
            
            # --- A. Average Difference (The "Effect Size") ---
            # This answers "How many points difference?"
            diff = group_target.mean() - group_ref.mean()
            results.append(diff)
            
            # --- B. P-Value (Significance) ---
            # We use Welch's t-test (equal_var=False) to match your manual SE formula
            t_stat, p_val = ttest_ind(group_target, group_ref, equal_var=False, nan_policy='omit')
            pvals.append(p_val)

            # --- C. Standard Error (Precision) ---
            # Manual calculation using Welch's formula
            n1 = len(group_target)
            n2 = len(group_ref)
            v1 = group_target.var(ddof=1)
            v2 = group_ref.var(ddof=1)
            
            std_err = np.sqrt(v1/n1 + v2/n2)
            std.append(std_err)
            
        else:
            results.append(None)
            std.append(None)
            pvals.append(None)

    return results, std, pvals




def impact_ratio(df, threshold_value, sensitive_attribute_vector):
    # Select exactly the top (1-threshold_value) percent
    # threshold_value=0.75 means we want top 25%
    n_select = int(len(df) * (1 - threshold_value))
    # Sort by value descending and select top n_select
    df_sorted = df.sort_values('value', ascending=False)
    top_indices = df_sorted.head(n_select).index
    df['selected'] = 0
    df.loc[top_indices, 'selected'] = 1
    gender_rates = df.groupby('gender')['selected'].mean()
    race_rates = df.groupby('race')['selected'].mean()

    results = []
    try: 
        for attr in sensitive_attribute_vector:
            if attr == 'Woman':
                ir = float(gender_rates['women']) / float(gender_rates['men'])

            elif attr == 'Asian':
                ir = float(race_rates['asian']) /  float(race_rates['white'])

            elif attr == 'Black':
                ir = float(race_rates['black']) /  float(race_rates['white'])

            elif attr == 'Hispanic':   
                ir = float(race_rates['hispanic']) /  float(race_rates['white'])

            results.append(ir)
    except: 
        return None, None
    return results, None

    
def compute_wasserstein_distance(df, sensitive_attribute_vector):
    df = df.groupby(
        ["role", "prompt", "variable", "race", "gender", "raceBLS", "genderBLS", "name"],
        group_keys=False
    ).apply(lambda x: x.sample(n=10, random_state=42))

    df_filtered = df

    results = []
    for attr in sensitive_attribute_vector:
        if attr == 'Woman':
            # Compute Wasserstein distance between men and women
            values_men = df_filtered[df_filtered['gender'] == 'men']['value']
            values_women = df_filtered[df_filtered['gender'] == 'women']['value']
            wd = wasserstein_distance(values_men, values_women)

        elif attr == 'Asian':

            # Compute Wasserstein distance between white and asian
            values_white = df_filtered[df_filtered['race'] == 'white']['value']
            values_asian = df_filtered[df_filtered['race'] == 'asian']['value']
            wd = wasserstein_distance(values_white, values_asian)

        elif attr == 'Black':
            # Compute Wasserstein distance between white and black
            values_white = df_filtered[df_filtered['race'] == 'white']['value']
            values_black = df_filtered[df_filtered['race'] == 'black']['value']
            wd = wasserstein_distance(values_white, values_black)

        elif attr == 'Hispanic':
            # Compute Wasserstein distance between white and hispanic
            values_white = df_filtered[df_filtered['race'] == 'white']['value']
            values_hispanic = df_filtered[df_filtered['race'] == 'hispanic']['value']
            wd = wasserstein_distance(values_white, values_hispanic)

        results.append(wd)

    return results, None

def compute_regression_coefficients(df, sensitive_attribute_vector, analysis):
    # Drop index column if present
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])


    # Convert 'value' to numeric, coercing bad strings like "NULL" to NaN
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    # Define levels (to match R's factor levels)
    gender_levels = ["men", "women"]
    race_levels = ["white", "hispanic", "asian", "black"]


    def prepare_subset(df):

        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        subset=df

        subset = subset[
            subset["gender"].isin(gender_levels) &
            subset["race"].isin(race_levels)
        ].copy()
        

        subset = subset.dropna(subset=["raceBLS", "genderBLS", "gender", "race", "role", "name", "value"])


        # make sure calendar of gender only 

        # Factor-like columns (preserve level order like in R)
        subset["gender"] = pd.Categorical(subset["gender"], categories=gender_levels, ordered=True)
        subset["race"] = pd.Categorical(subset["race"], categories=race_levels, ordered=True)

        subset = subset.reset_index(drop=True)
        return subset


    def fit_lme(df_subset):
        # formula = "value ~ gender * race + raceBLS + genderBLS + role"
        # formula = "value ~ gender * race + raceBLS  + role" # LinAlg Error 
        # formula = "value ~ gender * race  + role"
        has_race = "race" in df_subset and df_subset["race"].nunique() > 2


        if analysis == 'original' and has_race:
            formula = "value ~ gender * race  + role"
        else:
            formula = "value ~ gender + role"

        df_subset = df_subset.reset_index(drop=True)
        print(len(df_subset))
        model = mixedlm(formula, data=df_subset, groups=df_subset["name"])

        # model = mixedlm(formula, data=df_subset, groups=df_subset["name"])
        result = model.fit(reml=True, method="nm", maxiter=5000)
        return result
        
    # Interview
    df_interview = prepare_subset(df)
    try: coefs = fit_lme(df_interview)
    except: return None, None, None
    # extract the parameter values and standard errors

    results, std, pvals = [], [], [] 

    for attr in sensitive_attribute_vector:
        if attr == 'Woman':
            results.append(coefs.params["gender[T.women]"])
            std.append(coefs.bse["gender[T.women]"])   
            pvals.append(coefs.pvalues["gender[T.women]"])

        # elif attr == 'Asian':
        #     results.append(coefs.params["gender[T.women]"])
        #     std.append(coefs.bse["gender[T.women]"])    
        #     pvals.append(coefs.pvalues["gender[T.women]"])
        # elif attr == 'Black':
        #     results.append(coefs.params["gender[T.women]"])
        #     std.append(coefs.bse["gender[T.women]"])   
        #     pvals.append(coefs.pvalues["gender[T.women]"]) 
        # elif attr == 'Hispanic':
        #     results.append(coefs.params["gender[T.women]"])
        #     std.append(coefs.bse["gender[T.women]"])   
        #     pvals.append(coefs.pvalues["gender[T.women]"])
    return results, std, pvals


def compute_wang_ras_impact_ratio(df, use_scores=False):
    if use_scores:
        privilege_col = 'Privilege_Avg_Score'
        protect_col = 'Protect_Avg_Score'
        privilege_selected = lambda privilege, protect: privilege >= protect
        protect_selected = lambda privilege, protect: protect >= privilege
    else:
        privilege_col = 'Privilege_Rank'
        protect_col = 'Protect_Rank'
        privilege_selected = lambda privilege, protect: privilege <= protect
        protect_selected = lambda privilege, protect: protect <= privilege

    if privilege_col not in df.columns or protect_col not in df.columns:
        return {
            'n_samples': 0,
            'selection_rate_privilege': np.nan,
            'selection_rate_protect': np.nan,
            'impact_ratio_privilege': np.nan,
            'impact_ratio_protect': np.nan,
        }

    privilege = pd.to_numeric(df[privilege_col], errors='coerce')
    protect = pd.to_numeric(df[protect_col], errors='coerce')
    valid_mask = privilege.notna() & protect.notna()

    if valid_mask.sum() == 0:
        return {
            'n_samples': 0,
            'selection_rate_privilege': np.nan,
            'selection_rate_protect': np.nan,
            'impact_ratio_privilege': np.nan,
            'impact_ratio_protect': np.nan,
        }

    privilege = privilege[valid_mask]
    protect = protect[valid_mask]
    n_samples = len(privilege)

    privilege_wins = int(privilege_selected(privilege, protect).sum())
    protect_wins = int(protect_selected(privilege, protect).sum())

    selection_rate_privilege = privilege_wins / n_samples
    selection_rate_protect = protect_wins / n_samples
    max_selection_rate = max(selection_rate_privilege, selection_rate_protect)

    if max_selection_rate > 0:
        impact_ratio_privilege = selection_rate_privilege / max_selection_rate
        impact_ratio_protect = selection_rate_protect / max_selection_rate
    else:
        impact_ratio_privilege = np.nan
        impact_ratio_protect = np.nan

    return {
        'n_samples': n_samples,
        'selection_rate_privilege': selection_rate_privilege,
        'selection_rate_protect': selection_rate_protect,
        'impact_ratio_privilege': impact_ratio_privilege,
        'impact_ratio_protect': impact_ratio_protect,
    }


def compute_wang_score_diff(df):
    """Compute the average score difference between the privilege and protected group.
    Uses Privilege_Avg_Score and Protect_Avg_Score columns.
    Returns a dict with 'score_diff' (privilege_mean - protect_mean), 'std_error', and 'pvalue'.
    """
    if 'Privilege_Avg_Score' not in df.columns or 'Protect_Avg_Score' not in df.columns:
        return {'score_diff': np.nan, 'std_error': np.nan, 'pvalue': np.nan}

    privilege = pd.to_numeric(df['Privilege_Avg_Score'], errors='coerce')
    protect = pd.to_numeric(df['Protect_Avg_Score'], errors='coerce')
    valid_mask = privilege.notna() & protect.notna()

    if valid_mask.sum() == 0:
        return {'score_diff': np.nan, 'std_error': np.nan, 'pvalue': np.nan}

    privilege = privilege[valid_mask]
    protect = protect[valid_mask]

    diff = privilege.mean() - protect.mean()

    # Welch's t-test (unpaired) for significance
    t_stat, pval = ttest_ind(privilege, protect, equal_var=False, nan_policy='omit')

    # Standard error of the difference in means (Welch's formula)
    n1, n2 = len(privilege), len(protect)
    std_err = np.sqrt(privilege.var(ddof=1) / n1 + protect.var(ddof=1) / n2)

    return {'score_diff': diff, 'std_error': std_err, 'pvalue': pval}


def compute_wang_metric(df, sensitive_attribute_vector, metric):
    metric = metric.lower()

    if metric in {'impact_ratio_rank', 'impact_ratio_ras_rank', 'impact_ratio'}:
        metric_results = compute_wang_ras_impact_ratio(df, use_scores=False)
        value_key = 'impact_ratio_protect'
    elif metric in {'impact_ratio_score', 'impact_ratio_ras_score'}:
        metric_results = compute_wang_ras_impact_ratio(df, use_scores=True)
        value_key = 'impact_ratio_protect'
    elif metric in {'selection_rate_rank', 'selection_rate_ras_rank'}:
        metric_results = compute_wang_ras_impact_ratio(df, use_scores=False)
        value_key = 'selection_rate_protect'
    elif metric in {'selection_rate_score', 'selection_rate_ras_score'}:
        metric_results = compute_wang_ras_impact_ratio(df, use_scores=True)
        value_key = 'selection_rate_protect'
    elif metric in {'score_difference', 'score_diff', 'wang_score_diff'}:
        sd_results = compute_wang_score_diff(df)
        results = []
        for attr in sensitive_attribute_vector:
            if attr == 'Woman':
                results.append(sd_results['score_diff'])
            else:
                results.append(None)
        std = [sd_results['std_error'] if attr == 'Woman' else None for attr in sensitive_attribute_vector]
        pvals = [sd_results['pvalue'] if attr == 'Woman' else None for attr in sensitive_attribute_vector]
        return results, std, pvals
    else:
        return None, None, None

    results = []
    for attr in sensitive_attribute_vector:
        if attr == 'Woman':
            results.append(metric_results[value_key])
        else:
            results.append(None)

    return results, None, None

def preprocess_rozado(config):

    '''
    # process the score+decision folder csvs to update the chosen_candidate_first_name column
    # open the csv file, read it into a pandas dataframe, and then update the chosen_candidate_first_name column
    import pandas as pd
    import os
    import time
    import random
    from pydantic import BaseModel


    # Define the main folder and the subdirectories
    base_folder = "experiment_name"
    models = ["gpt-3.5-turbo", "gpt-4o-mini" , "gpt-4o"]

    # Loop through each model folder
    for model in models:
        
        print(f"Processing model: {model}")

        model = model+'.csv'
        file_path = os.path.join(base_folder, model)

        # 1. Read the CSV file for each model
        df = pd.read_csv('score_task_format_'+file_path)

        # go through every row in the dataframe. extract the value at the user_prompt column 
        for index, row in df.iterrows():
            
            # 2. grab the response from the 'model_response' column
            model_response = row['model_response']

            try: 
                cv1_score, cv2_score = model_response.split(', ')
                cv1_score = cv1_score.split('CV 1: ')[1].split('\n')[0].strip()
                # if the cv1_score contains [ ], remove the brackets
                if cv1_score.startswith('[') and cv1_score.endswith(']'):
                    cv1_score = cv1_score[1:-1].strip()

                cv2_score = cv2_score.split('CV 2: ')[1].split('\n')[0].strip()
                if cv2_score.startswith('[') and cv2_score.endswith(']'):
                    cv2_score = cv2_score[1:-1].strip()

                # remove the final period from cv1_score and cv2_score if it exists
                if cv1_score.endswith('.'):
                    cv1_score = cv1_score[:-1].strip()
                if cv2_score.endswith('.'):
                    cv2_score = cv2_score[:-1].strip()

                if cv1_score.startswith('[') and cv1_score.endswith(']'):
                    cv1_score = cv1_score[1:-1].strip()

                if cv2_score.startswith('[') and cv2_score.endswith(']'):
                    cv2_score = cv2_score[1:-1].strip()
                
                # print(cv1_score)
                # print(cv2_score)
            except: 
                cv1_score = '0'
                cv2_score = '0'
                print(f"Error extracting CV scores from {model_response}, setting to 0")

            # compare the two CV scores. if cv1 is greater than cv2, then set the cv_chosen to the value at row['cv1']. else set it to the value at row['cv2']
            try: cv1_score = float(cv1_score)
            except ValueError:
                cv1_score = float(cv1_score[0])
            
            try:  cv2_score = float(cv2_score)
            except ValueError:
                cv2_score = float(cv2_score[0])

            if float(cv1_score) > float(cv2_score):
                cv_chosen = row['cv1']
            else:
                cv_chosen = row['cv2']

            # grab the name from the cv_chosen variable by splitting on "Name: "
            cv_chosen_name = cv_chosen.split('Name: ')[1]
            candidate_first_name = cv_chosen_name.split(" ")[0]
            candidate_first_name = candidate_first_name.strip()

            # store the value at response.choices[0].message.content in the value at row['model_response']
            df.at[index, 'chosen_candidate_first_name'] = candidate_first_name
            # 5. identify if this first name matches the value in male_full_name or female_full_name 
            if candidate_first_name in row['male_full_name']:
                df.at[index, 'chosen_gender'] = 'Male'
            else: 
                df.at[index, 'chosen_gender'] = 'Female'
            print(candidate_first_name)
            print(df.at[index, 'chosen_gender'])

        # Save the updated DataFrame back to the CSV file
        df.to_csv('score_task_format_'+file_path, index=False)
    '''
    return 

def analyze_armstrong(args, config, all_together=False):

    perturbation_results_filepath = 'results/{}_sampling_{}_analysis_{}.csv'.format(config['experimental_framework'], args.sampling, args.analysis)
    
    # read the perturbation_results_filepath 
    if not os.path.exists(repo_path(perturbation_results_filepath)):
        perturbation_results_df = pd.DataFrame(columns=columns)
        perturbation_results_df.to_csv(repo_path(perturbation_results_filepath), index=False)
    else:
        perturbation_results_df = pd.read_csv(repo_path(perturbation_results_filepath))


    # save data into results_df

    author = config['experimental_framework']
    model = config['Model']['Model_Name']
    temp = config['Model']['Temperature']
    num_trials = config['Prompt']['Trials_Per_Query']
    name_bundle = config['Name']['Bundle_Name']
    job_bundle = config['Job']['Bundle_Name']
    random_state = args.random_state

    # if the row exists with the same Author, Names, Jobs,Num_Trials, and Model value in the perturbation_results_df, and Sensitive_Attribute_Vector = regression_coefficients; check if the value at Result_Vector[0] exists. if it does, then skip the computation and break
    # 1. Filter the DataFrame for the matching row
    mask = (
        (perturbation_results_df['Author'] == author) &
        (perturbation_results_df['Names'] == name_bundle) &
        (perturbation_results_df['Jobs'] == job_bundle) &
        (perturbation_results_df['Num_Trials'] == num_trials) &
        (perturbation_results_df['Model'] == model) &
        (perturbation_results_df['Metric'] == 'regression_coefficients')
    )

    # 2. Check if the row exists
    if not perturbation_results_df.loc[mask].empty:
        # 3. Get the value of Result_Vector for that row (taking the first match)
        existing_result = perturbation_results_df.loc[mask, 'Result_Vector'].iloc[0]

        # 4. Check if the value "exists" (is not None, NaN, or empty depending on your data type)
        # Using pd.notna checks for NaN/None. If it's a list, check if len > 0.
        if pd.notna(existing_result):
            # Optional: specific check if it is a list/array and you want index 0
            # if len(existing_result) > 0:
            print(f"Row already exists for {author}, {name_bundle}, {job_bundle}, {num_trials}, {model}")
            return

    if author == 'rozado':
        preprocess_rozado(config)
        
    keys = columns[:6]
    values = [author, name_bundle, job_bundle, num_trials, model, temp, random_state]

    new_row_template = dict(zip(keys, values))

    metrics_to_compute = config['Results']['Metrics_to_Compute']
    if args.name == 'karvonen' or args.name == 'seshadri':
        sensitive_attribute_vector = ['Woman', 'Black']
    else:
        sensitive_attribute_vector = config['Results']['Sensitive_Attribute_Vector'] 

    # read in the data
    # data_filepath = "output_data/{}/model_{}_temp_{}_numtrials_{}_jobs_{}_names_{}_random_state_{}.csv".format(author, model, temp, num_trials, job_bundle, name_bundle, random_state)
    if all_together:
        model_fullname = MODELS[model]
        model_fullname = model_fullname.split('/')[-1]
        data_filepath = f"output_data/armstrong/together/processed_batch_outputs/{model_fullname}/name_{name_bundle}_job_{job_bundle}.csv"
    else:
        try: data_filepath = "output_data/{}/model_{}_temp_{}_numtrials_{}_jobresumes_{}_names_{}_random_state_{}.csv".format(author, model, temp, num_trials, job_bundle, name_bundle, random_state)
        except: breakpoint()
  
    # READ IN THE OUTPUT DATA AND COMPUTE THE METRICS 
    try: 
        df = pd.read_csv(repo_path(data_filepath))
    except: 
        print(f"Error reading {data_filepath}")
        return

    prompt_type='interview'

    df = df[
        (df["prompt"] == prompt_type) &
        (df["value"].notnull()) &
        (df["value"] > 0) 
    ].copy()

    if args.sampling == 'downsampling':
        df = downsample_data(df, config)
    
    new_rows_list = []
    pvals=None
    for metric in metrics_to_compute:
        print(f"Computing {metric}...")
        if metric == 'regression_coefficients':
            values, std, pvals = compute_regression_coefficients(df, sensitive_attribute_vector, args.analysis)
            # save a row with the coefficients
            print(data_filepath, pvals)
        
        elif metric == 'wasserstein_distance':
            values, std = compute_wasserstein_distance(df, sensitive_attribute_vector)
            # save a row with the distances

        elif metric == 'score_difference':
            values, std, pvals= score_diff(df, sensitive_attribute_vector)

        elif metric == 'ttest':
            values, std, pvals = compute_ttest(df, sensitive_attribute_vector)
            
        elif 'impact_ratio' in metric:
            values, std = impact_ratio(df, float(metric.split('_')[-1]), sensitive_attribute_vector)

        new_row_dict = new_row_template.copy()
        new_row_dict['Metric'] = metric
        new_row_dict['Sensitive_Attribute_Vector'] = sensitive_attribute_vector
        new_row_dict['Result_Vector'] = values
        new_row_dict['Std_Error_Vector'] = std
        new_row_dict['P_Value_Vector'] = pvals
        new_row_dict['Random_State'] = random_state
        new_rows_list.append(new_row_dict)

    # eventually 
    # Handle per_job case
    if args.per_job == True:
        # Get unique jobs from the dataframe
        unique_jobs = df['role'].unique() if 'role' in df.columns else []
        
        if len(unique_jobs) == 0:
            print("Warning: No unique jobs found in dataframe. Cannot compute per_job analysis.")
            return
        
        # Loop through each unique job
        for job in unique_jobs:
            # Filter dataframe to this job
            df_job = df[df['role'] == job].copy()
            
            if len(df_job) == 0:
                continue
            
            # Modify sensitive_attribute_vector to include job-specific attributes
            # Replace 'Woman' with 'Woman_{job}' in the sensitive attribute vector
            job_sensitive_attribute_vector = []
            for attr in sensitive_attribute_vector:
                if attr == 'Woman':
                    job_sensitive_attribute_vector.append(f'Woman_{job}')
                else:
                    job_sensitive_attribute_vector.append(attr)
            
            # Compute metrics for this job
            for metric in metrics_to_compute:
                print(f"Computing {metric} for job {job}...")
                if metric == 'regression_coefficients':
                    values, std, pvals = compute_regression_coefficients(df_job, job_sensitive_attribute_vector, args.analysis)
                    # save a row with the coefficients
                    print(data_filepath, pvals)
                
                elif metric == 'wasserstein_distance':
                    values, std = compute_wasserstein_distance(df_job, job_sensitive_attribute_vector)
                    # save a row with the distances

                elif metric == 'score_difference':
                    values, std, pvals = score_diff(df_job, job_sensitive_attribute_vector)

                elif metric == 'ttest':
                    values, std, pvals = compute_ttest(df_job, job_sensitive_attribute_vector)
                    
                elif 'impact_ratio' in metric:
                    values, std = impact_ratio(df_job, float(metric.split('_')[-1]), job_sensitive_attribute_vector)

                new_row_dict = new_row_template.copy()
                new_row_dict['Metric'] = metric
                new_row_dict['Sensitive_Attribute_Vector'] = job_sensitive_attribute_vector
                new_row_dict['Result_Vector'] = values
                new_row_dict['Std_Error_Vector'] = std
                new_row_dict['P_Value_Vector'] = pvals
                new_row_dict['Random_State'] = random_state
                new_rows_list.append(new_row_dict)
    else:
        # Original logic: compute metrics for all data
        for metric in metrics_to_compute:
            print(f"Computing {metric}...")
            if metric == 'regression_coefficients':
                values, std, pvals = compute_regression_coefficients(df, sensitive_attribute_vector, args.analysis)
                # save a row with the coefficients
                print(data_filepath, pvals)
            
            elif metric == 'wasserstein_distance':
                values, std = compute_wasserstein_distance(df, sensitive_attribute_vector)
                # save a row with the distances

            elif metric == 'score_difference':
                values, std = score_diff(df, sensitive_attribute_vector)

            elif metric == 'ttest':
                values, std, pvals = compute_ttest(df, sensitive_attribute_vector)
                
            elif 'impact_ratio' in metric:
                values, std = impact_ratio(df, float(metric.split('_')[-1]), sensitive_attribute_vector)

            new_row_dict = new_row_template.copy()
            new_row_dict['Metric'] = metric
            new_row_dict['Sensitive_Attribute_Vector'] = sensitive_attribute_vector
            new_row_dict['Result_Vector'] = values
            new_row_dict['Std_Error_Vector'] = std
            new_row_dict['P_Value_Vector'] = pvals
            new_row_dict['Random_State'] = random_state
            new_rows_list.append(new_row_dict)


    new_rows = pd.DataFrame(new_rows_list)

    # Drop all-NA columns from new_rows to avoid FutureWarning before concatenation
    new_rows = new_rows.dropna(axis=1, how='all')
    # if perturbation_results_filepath doesnt exist, create a csv with the right columns
    


    perturbation_results_df = pd.concat([perturbation_results_df, new_rows], ignore_index=True)
    # save the results to the filepath 
    perturbation_results_df.to_csv(repo_path(perturbation_results_filepath), index=False)
    return 


'''
def analyze_yin(args, config, all_together=False):

    author = config['experimental_framework']
    model = config['Model']['Model_Name']
    temp = config['Model']['Temperature']
    name_bundle = config['Name']['Bundle_Name']
    job_bundle = config['Job']['Bundle_Name']
    perturbation_results_filepath = 'results/{}_sampling_{}_analysis_{}.csv'.format(config['experimental_framework'], args.sampling, args.analysis)
    

    # Outputs
    fn_ranking_detailed = perturbation_results_filepath.replace('.csv', '_detailed.csv')
    fn_ranking_gender = perturbation_results_filepath


    # read the perturbation_results_filepath 
    if not os.path.exists(fn_ranking_gender):
        results_gender = pd.DataFrame(columns=columns)
        results_gender.to_csv(fn_ranking_gender, index=False)


    # read the perturbation_results_filepath 
    if not os.path.exists(fn_ranking_detailed):
        # Ensure 'columns' variable is defined in your previous scope
        results_detailed = pd.DataFrame(columns=columns) 
        results_detailed.to_csv(fn_ranking_detailed, index=False)

    if all_together:
        model_fullname = MODELS[model]
        model_fullname = model_fullname.split('/')[-1]
    # Inputs
    if all_together:
        fn_gpt4 = f"output_data/yin/together/processed_batch_outputs/{model_fullname}/temp_1.0_names_{name_bundle}_jobresumes_{job_bundle}/*/*.json"
    else:
        fn_gpt4 = f"output_data/{author}/{model.replace('_', '-')}/temp_{temp}_names_{name_bundle}_jobresumes_{job_bundle}/*/*.json"
    
    files = glob.glob(fn_gpt4)

    if all_together:
        jobs = os.listdir(f"output_data/{author}/together/processed_batch_outputs/{model_fullname}/temp_1.0_names_{name_bundle}_jobresumes_{job_bundle}")
    else:
        jobs = [f for f in os.listdir(f"output_data/{author}/{model.replace('_', '-')}/temp_{temp}_names_{name_bundle}_jobresumes_{job_bundle}") if not f.startswith('run_')]

    # --- Helper Function for Gender ---
    def get_gender(demo_str):
        """Parses 'Asian_W' to 'W', 'White_M' to 'M', etc."""
        s = str(demo_str).strip()
        if s.endswith('_W'):
            return 'W'
        elif s.endswith('_M'):
            return 'M'
        return 'Unknown'

    # --- Step 1: Pre-load and Parse Data (Optimization) ---
    # We read files once to extract the ranking order, rather than reading them 
    # repeatedly for every job/analysis loop.

    parsed_data = []
    exact_match_count = 0  # Track when gpt_race_order exactly matches demo_order
    missing_names_count = 0  # Track number of instances when names are not found
    example_sentence_with_missing_names = None  # Store one example sentence with missing names

    for fn in tqdm(files, desc="Parsing Files"):
        try:
            records = json.load(open(fn))
            sentence = records['choices'][0]['message']['content'].lower()
            context = records['context']
            
            real_order = context['default_order']
            real_order = [_.lower() for _ in real_order]
            demo_order = context['demo_order']
            
            # Determine GPT Order based on text position
            name2len = {}
            has_missing_names = False  # Track if any names are not found in this sentence
            for name in real_order:
                # Split sentence by name, take length of first part to find position
                name2len[name] = len(sentence.split(name.split(' ')[0])[0])
                # Check if the name (or its first part) is actually in the sentence
                name_first_part = name.split(' ')[0]
                if name_first_part not in sentence and name not in sentence:
                    has_missing_names = True
            
            # Count instances when names are not found
            if has_missing_names:
                missing_names_count += 1
                # Store the first example sentence with missing names
                if example_sentence_with_missing_names is None:
                    example_sentence_with_missing_names = records['choices'][0]['message']['content']
            
            name2len = dict(sorted(name2len.items(), key=lambda item: item[1]))
            gpt_order = list(name2len.keys())

            
            # Map back to demographics
            name2race = dict(zip(real_order, demo_order))
            gpt_race_order = [name2race.get(_) for _ in gpt_order]
            
            # Check if gpt_race_order exactly matches demo_order
            if gpt_race_order == demo_order:
                exact_match_count += 1
            
            # Store processed record
            parsed_data.append({
                'model': model,
                'job': context['job'],
                'demo_order': demo_order,         # The ground truth list of demos
                'gpt_race_order': gpt_race_order, # The re-ranked list of demos
                'natural_first': demo_order[0]    # The demographic that was naturally 1st
            })
            
        except Exception as e:
            # print(f"Error parsing {fn}: {e}")
            continue

    # Save exact match statistics to a new file
    if len(parsed_data) > 0:
        exact_match_filepath = 'results/{}_sampling_{}_analysis_{}_exact_matches.csv'.format(
            config['experimental_framework'], args.sampling, args.analysis
        )
        
        # Check if file exists, if not create with headers
        if not os.path.exists(exact_match_filepath):
            exact_match_df = pd.DataFrame(columns=['model', 'name_bundle', 'job_bundle', 'number_of_instances', 'total_instances'])
            exact_match_df.to_csv(exact_match_filepath, index=False)
        
        # Create new row with exact match statistics
        exact_match_row = pd.DataFrame([{
            'model': model,
            'name_bundle': name_bundle,
            'job_bundle': job_bundle,
            'number_of_instances': exact_match_count,
            'total_instances': len(parsed_data)
        }])
        
        # Append to file
        exact_match_row.to_csv(exact_match_filepath, mode='a', index=False, header=False)
        print(f"Saved exact match statistics to {exact_match_filepath}: {exact_match_count}/{len(parsed_data)} = {exact_match_count/len(parsed_data):.4f}")

    # Save missing names statistics to a file
    if len(parsed_data) > 0:
        missing_names_filepath = 'results/{}_sampling_{}_analysis_{}_missing_names.csv'.format(
            config['experimental_framework'], args.sampling, args.analysis
        )
        
        # Check if file exists, if not create with headers
        if not os.path.exists(missing_names_filepath):
            missing_names_df = pd.DataFrame(columns=['model', 'name_bundle', 'job_bundle', 'number_of_instances', 'total_instances', 'example_sentence'])
            missing_names_df.to_csv(missing_names_filepath, index=False)
        
        # Create new row with missing names statistics
        missing_names_row = pd.DataFrame([{
            'model': model,
            'name_bundle': name_bundle,
            'job_bundle': job_bundle,
            'number_of_instances': missing_names_count,
            'total_instances': len(parsed_data),
            'example_sentence': example_sentence_with_missing_names if example_sentence_with_missing_names is not None else ''
        }])
        
        # Append to file
        missing_names_row.to_csv(missing_names_filepath, mode='a', index=False, header=False)
        print(f"Saved missing names statistics to {missing_names_filepath}: {missing_names_count}/{len(parsed_data)} = {missing_names_count/len(parsed_data):.4f}")


    # --- Step 2: Generate Original Detailed CSV (By Job, Specific Demos) ---
    data_detailed = []

    model_data = parsed_data
    for N_top in range(1, 1+1):
        # Calculate overall top-1 agreement (just for logging purposes)
        topistop = sum(1 for d in model_data if d['gpt_race_order'][0] == d['natural_first'])
        print(f"Model: {model} | Top {N_top} | Natural Order Agreement: {topistop / len(model_data):.4f}")

        for job in jobs:
            # Filter data for this specific job
            job_data = [d for d in model_data if d['job'] == job]
            
            top_og = Counter()
            top_gpt = Counter()
            c = len(job_data)
            
            if c == 0: continue

            for record in job_data:
                top_og.update(record['demo_order'][:N_top])
                top_gpt.update(record['gpt_race_order'][:N_top])
            
            # Create Stats DataFrame
            print(f"Processing Job: {job}")
            df = pd.DataFrame(top_gpt.most_common(), columns=['demo', 'top'])
            df_og = pd.DataFrame(top_og.most_common(), columns=['demo', 'top_og'])            
            df = df.merge(df_og, on='demo', how='outer').fillna(0)

            df['selection_rate'] = df['top'] / c
            df['disparate_impact_ratio'] = df['selection_rate'] / df['selection_rate'].max()
            
            df['job'] = job
            df['model'] = model
            df['rank'] = N_top
            df['name_bundle'] = name_bundle
            df['job_bundle'] = job_bundle
            
            
            data_detailed.extend(df.to_dict(orient='records'))


    # Save Detailed CSV
    if data_detailed:  # Only write if there is data
        results_detailed = pd.DataFrame(data_detailed)
        
        # mode='a' appends to the file
        # header=False prevents writing the column names again in the middle of the file
        results_detailed.to_csv(fn_ranking_detailed, mode='a', index=False, header=False)
        print(f"Appended detailed ranking to {fn_ranking_detailed}")
    else:
        print("No new data to append.")

    # --- Step 3: Generate Gender Aggregate CSV (All Jobs, W vs M) ---
    data_gender = []

    print("\n--- Generating Gender Aggregate Analysis ---")

    model_data = parsed_data
    
    for N_top in range(1, 1+1):
        # No job loop here - we use all model_data
        gender_top_og = Counter()
        gender_top_gpt = Counter()
        c = len(model_data)
        
        for record in model_data:
            # Convert specific demos to gender
            genders_og = [get_gender(d) for d in record['demo_order'][:N_top]]
            genders_gpt = [get_gender(d) for d in record['gpt_race_order'][:N_top]]
            
            gender_top_og.update(genders_og)
            gender_top_gpt.update(genders_gpt)
        
        # Create Stats DataFrame
        df = pd.DataFrame(gender_top_gpt.most_common(), columns=['gender', 'top'])
        df_og = pd.DataFrame(gender_top_og.most_common(), columns=['gender', 'top_og'])
        
        df = df.merge(df_og, on='gender', how='outer').fillna(0)
        
        df['selection_rate'] = df['top'] / c
        df['disparate_impact_ratio'] = df['selection_rate'] / df['selection_rate'].max()
        
        df['job'] = 'ALL' # Aggregated
        df['model'] = model
        df['rank'] = N_top

        df['name_bundle'] = name_bundle
        df['job_bundle'] = job_bundle
        
        print(f"Aggregated Results (Top {N_top}):")
        display(HTML(df.sort_values(by='disparate_impact_ratio', ascending=True).reset_index(drop=1).to_html()))
        
        data_gender.extend(df.to_dict(orient='records'))
        # Save Detailed CSV
    if data_gender:  # Only write if there is data
        results_gender = pd.DataFrame(data_gender)
        
        # mode='a' appends to the file
        # header=False prevents writing the column names again in the middle of the file
        results_gender.to_csv(fn_ranking_gender, mode='a', index=False, header=False)
        print(f"Appended gender ranking to {fn_ranking_gender}")
    else:
        print("No new data to append.")
        

    return 

'''

def analyze_yin(args, config, all_together=False):

    author = config['experimental_framework']
    model = config['Model']['Model_Name']
    temp = config['Model']['Temperature']
    name_bundle = config['Name']['Bundle_Name']
    job_bundle = config['Job']['Bundle_Name']
    perturbation_results_filepath = 'results/{}_sampling_{}_analysis_{}.csv'.format(config['experimental_framework'], args.sampling, args.analysis)
    
    # Outputs
    fn_ranking_detailed = perturbation_results_filepath.replace('.csv', '_detailed.csv')
    fn_ranking_gender = perturbation_results_filepath


    ranking_columns = [
        'top', 'top_og', 'selection_rate', 'disparate_impact_ratio',
        'job', 'model', 'rank', 'name_bundle', 'job_bundle',
    ]
    os.makedirs(repo_path(os.path.dirname(fn_ranking_gender)), exist_ok=True)

    if not os.path.exists(repo_path(fn_ranking_gender)):
        results_gender = pd.DataFrame(columns=['gender'] + ranking_columns)
        results_gender.to_csv(repo_path(fn_ranking_gender), index=False)


    # read the perturbation_results_filepath 
    if not os.path.exists(repo_path(fn_ranking_detailed)):
        results_detailed = pd.DataFrame(columns=['demo'] + ranking_columns + ['random_state'])
        results_detailed.to_csv(repo_path(fn_ranking_detailed), index=False)

    if all_together:
        model_fullname = MODELS[model]
        model_fullname = model_fullname.split('/')[-1]

    # Inputs
    if all_together:
        fn_gpt4 = f"output_data/yin/together/processed_batch_outputs/{model_fullname}/temp_1.0_names_{name_bundle}_jobresumes_{job_bundle}/*/*.json"
    else:
        if int(args.random_state) != 42: 
            fn_gpt4 = f"output_data/{author}/{model.replace('_', '-')}/temp_{temp}_names_{name_bundle}_jobresumes_{job_bundle}_random_state_{args.random_state}/*/*.json"
        else: 
            fn_gpt4 = f"output_data/{author}/{model.replace('_', '-')}/temp_{temp}_names_{name_bundle}_jobresumes_{job_bundle}/*/*.json"
    files = glob.glob(repo_path(fn_gpt4))

    if all_together:
        jobs = os.listdir(repo_path(f"output_data/{author}/together/processed_batch_outputs/{model_fullname}/temp_1.0_names_{name_bundle}_jobresumes_{job_bundle}"))
    else:
        try: jobs = [f for f in os.listdir(repo_path(f"output_data/{author}/{model.replace('_', '-')}/temp_{temp}_names_{name_bundle}_jobresumes_{job_bundle}_random_state_{args.random_state}")) if not f.startswith('run_')]
        except: jobs = [f for f in os.listdir(repo_path(f"output_data/{author}/{model.replace('_', '-')}/temp_{temp}_names_{name_bundle}_jobresumes_{job_bundle}")) if not f.startswith('run_')]

    # --- Helper Function for Gender ---
    def get_gender(demo_str):
        """Parses 'Asian_W' to 'W', 'White_M' to 'M', etc."""
        s = str(demo_str).strip()
        if s.endswith('_W'):
            return 'W'
        elif s.endswith('_M'):
            return 'M'
        return 'Unknown'

    parsed_data = []
    exact_match_count = 0
    complete_rankings_count = 0
    parse_counts = Counter()
    missing_names_count = 0
    attempted_rankings_count = 0
    example_sentence_with_missing_names = None

    for fn in tqdm(files, desc="Parsing Files"):
        attempted_rankings_count += 1
        try:
            with open(repo_path(fn), encoding='utf-8') as source:
                records = json.load(source)
            sentence = records['choices'][0]['message']['content']
            context = records['context']
            real_order = context['default_order']
            demo_order = context['demo_order']
            gpt_race_order, parse_method = parse_yin_demographic_ranking(
                sentence, real_order, demo_order, context.get('candidate_ids'), top_only=True,
            )
            if gpt_race_order is None:
                parse_counts[parse_method] += 1
                missing_names_count += 1
                if example_sentence_with_missing_names is None:
                    example_sentence_with_missing_names = sentence
                continue
            parsed_data.append({
                'model': model,
                'job': context['job'],
                'demo_order': demo_order,
                'gpt_race_order': gpt_race_order,
                'natural_first': demo_order[0],
                'parse_method': parse_method,
            })
            parse_counts[parse_method] += 1
            if parse_method == 'complete_ranking':
                complete_rankings_count += 1
                if gpt_race_order == demo_order:
                    exact_match_count += 1
        except (OSError, ValueError, KeyError, IndexError, TypeError):
            parse_counts['invalid_record'] += 1
            missing_names_count += 1

    parse_stats_filepath = perturbation_results_filepath.replace('.csv', '_parsing.csv')
    parse_stats = pd.DataFrame([
        {'model': model, 'name_bundle': name_bundle, 'job_bundle': job_bundle,
         'status': status, 'number_of_instances': count,
         'total_instances': attempted_rankings_count}
        for status, count in sorted(parse_counts.items())
    ])
    if not parse_stats.empty:
        parse_stats.to_csv(
            repo_path(parse_stats_filepath), mode='a', index=False,
            header=not os.path.exists(repo_path(parse_stats_filepath)),
        )

    # Top-one recovery is not evidence of an exact full-ranking match.
    if complete_rankings_count > 0:
        exact_match_filepath = 'results/{}_sampling_{}_analysis_{}_exact_matches.csv'.format(
            config['experimental_framework'], args.sampling, args.analysis
        )
        
        # Check if file exists, if not create with headers
        if not os.path.exists(repo_path(exact_match_filepath)):
            exact_match_df = pd.DataFrame(columns=['model', 'name_bundle', 'job_bundle', 'number_of_instances', 'total_instances'])
            exact_match_df.to_csv(repo_path(exact_match_filepath), index=False)
        
        # Create new row with exact match statistics
        exact_match_row = pd.DataFrame([{
            'model': model,
            'name_bundle': name_bundle,
            'job_bundle': job_bundle,
            'number_of_instances': exact_match_count,
            'total_instances': complete_rankings_count
        }])
        
        # Append to file
        exact_match_row.to_csv(repo_path(exact_match_filepath), mode='a', index=False, header=False)
        print(f"Saved exact match statistics to {exact_match_filepath}: {exact_match_count}/{complete_rankings_count} = {exact_match_count/complete_rankings_count:.4f}")

    # Save missing names statistics to a file
    if attempted_rankings_count > 0:
        missing_names_filepath = 'results/{}_sampling_{}_analysis_{}_missing_names.csv'.format(
            config['experimental_framework'], args.sampling, args.analysis
        )
        
        # Check if file exists, if not create with headers
        if not os.path.exists(repo_path(missing_names_filepath)):
            missing_names_df = pd.DataFrame(columns=['model', 'name_bundle', 'job_bundle', 'number_of_instances', 'total_instances', 'example_sentence'])
            missing_names_df.to_csv(repo_path(missing_names_filepath), index=False)
        
        # Create new row with missing names statistics
        missing_names_row = pd.DataFrame([{
            'model': model,
            'name_bundle': name_bundle,
            'job_bundle': job_bundle,
            'number_of_instances': missing_names_count,
            'total_instances': attempted_rankings_count,
            'example_sentence': example_sentence_with_missing_names if example_sentence_with_missing_names is not None else ''
        }])
        
        # Append to file
        missing_names_row.to_csv(repo_path(missing_names_filepath), mode='a', index=False, header=False)
        print(f"Saved missing names statistics to {missing_names_filepath}: {missing_names_count}/{attempted_rankings_count} = {missing_names_count/attempted_rankings_count:.4f}")

    if not parsed_data:
        print("No unambiguous Yin rankings to analyze.")
        return

    # --- Step 2: Generate Original Detailed CSV (By Job, Specific Demos) ---
    data_detailed = []

    model_data = parsed_data
    for N_top in range(1, 1+1):
        # Calculate overall top-1 agreement (just for logging purposes)
        topistop = sum(1 for d in model_data if d['gpt_race_order'][0] == d['natural_first'])
        # print(f"Model: {model} | Top {N_top} | Natural Order Agreement: {topistop / len(model_data):.4f}")

        for job in jobs:
            # Filter data for this specific job
            job_data = [d for d in model_data if d['job'] == job]
            
            top_og = Counter()
            top_gpt = Counter()
            c = len(job_data)
            
            if c == 0: continue

            for record in job_data:
                top_og.update(record['demo_order'][:N_top])
                top_gpt.update(record['gpt_race_order'][:N_top])
            
            # Create Stats DataFrame
            print(f"Processing Job: {job}")
            df = pd.DataFrame(top_gpt.most_common(), columns=['demo', 'top'])
            df_og = pd.DataFrame(top_og.most_common(), columns=['demo', 'top_og'])            
            df = df.merge(df_og, on='demo', how='outer').fillna(0)

            df['selection_rate'] = df['top'] / c
            df['disparate_impact_ratio'] = df['selection_rate'] / df['selection_rate'].max()
            
            df['job'] = job
            df['model'] = model
            df['rank'] = N_top
            df['name_bundle'] = name_bundle
            df['job_bundle'] = job_bundle
            df['random_state'] = args.random_state
            
            
            data_detailed.extend(df.to_dict(orient='records'))


    # Save Detailed CSV
    if data_detailed:  # Only write if there is data
        results_detailed = pd.DataFrame(data_detailed)
        
        # mode='a' appends to the file
        # header=False prevents writing the column names again in the middle of the file
        results_detailed.to_csv(repo_path(fn_ranking_detailed), mode='a', index=False, header=False)
        print(f"Appended detailed ranking to {fn_ranking_detailed}")
    else:
        print("No new data to append.")

    # --- Step 3: Generate Gender Aggregate CSV (All Jobs, W vs M) ---
    data_gender = []

    print("\n--- Generating Gender Aggregate Analysis ---")

    model_data = parsed_data
    
    for N_top in range(1, 1+1):
        # No job loop here - we use all model_data
        gender_top_og = Counter()
        gender_top_gpt = Counter()
        c = len(model_data)
        
        for record in model_data:
            # Convert specific demos to gender
            genders_og = [get_gender(d) for d in record['demo_order'][:N_top]]
            genders_gpt = [get_gender(d) for d in record['gpt_race_order'][:N_top]]
            
            gender_top_og.update(genders_og)
            gender_top_gpt.update(genders_gpt)
        
        # Create Stats DataFrame
        df = pd.DataFrame(gender_top_gpt.most_common(), columns=['gender', 'top'])
        df_og = pd.DataFrame(gender_top_og.most_common(), columns=['gender', 'top_og'])
        
        df = df.merge(df_og, on='gender', how='outer').fillna(0)
        
        df['selection_rate'] = df['top'] / c
        df['disparate_impact_ratio'] = df['selection_rate'] / df['selection_rate'].max()
        
        df['job'] = 'ALL' # Aggregated
        df['model'] = model
        df['rank'] = N_top

        df['name_bundle'] = name_bundle
        df['job_bundle'] = job_bundle
        
        print(f"Aggregated Results (Top {N_top}):")
        display(HTML(df.sort_values(by='disparate_impact_ratio', ascending=True).reset_index(drop=1).to_html()))
        
        data_gender.extend(df.to_dict(orient='records'))
        # Save Detailed CSV
    if data_gender:  # Only write if there is data
        results_gender = pd.DataFrame(data_gender)
        
        # mode='a' appends to the file
        # header=False prevents writing the column names again in the middle of the file
        results_gender.to_csv(repo_path(fn_ranking_gender), mode='a', index=False, header=False)
        print(f"Appended gender ranking.")
    else:
        print("No new data to append.")
        

    return 


def analyze_wang(args, config, all_together=False):
    author = config['experimental_framework']
    model = config['Model']['Model_Name']
    temp = config['Model']['Temperature']
    num_trials = config['Prompt']['Trials_Per_Query']
    name_bundle = config['Name']['Bundle_Name']
    job_bundle = config['Job']['Bundle_Name']
    random_state = args.random_state


    perturbation_results_filepath = 'results/{}_sampling_{}_analysis_{}.csv'.format(
        author, args.sampling, args.analysis
    )
    detailed_results_filepath = perturbation_results_filepath.replace('.csv', '_detailed.csv')

    if not os.path.exists(repo_path(perturbation_results_filepath)):
        perturbation_results_df = pd.DataFrame(columns=columns)
        perturbation_results_df.to_csv(repo_path(perturbation_results_filepath), index=False)
    else:
        perturbation_results_df = pd.read_csv(repo_path(perturbation_results_filepath))

    detailed_columns = [
        'model',
        'name_bundle',
        'job_bundle',
        'occupation',
        'n_samples',
        'selection_rate_privilege_rank',
        'selection_rate_protect_rank',
        'impact_ratio_privilege_rank',
        'impact_ratio_protect_rank',
        'selection_rate_privilege_score',
        'selection_rate_protect_score',
        'impact_ratio_privilege_score',
        'impact_ratio_protect_score',
    ]
    if not os.path.exists(repo_path(detailed_results_filepath)):
        pd.DataFrame(columns=detailed_columns).to_csv(repo_path(detailed_results_filepath), index=False)

    existing_mask = (
        (perturbation_results_df['Author'] == author) &
        (perturbation_results_df['Names'] == name_bundle) &
        (perturbation_results_df['Jobs'] == job_bundle) &
        (perturbation_results_df['Num_Trials'] == num_trials) &
        (perturbation_results_df['Model'] == model) & 
        (perturbation_results_df['Random_State']== args.random_state)
    )
    if not perturbation_results_df.loc[existing_mask].empty:
        print(f"Row already exists for {author}, {name_bundle}, {job_bundle}, {num_trials}, {model}")
        return

    if all_together:
        model_fullname = MODELS.get(model, model)
        model_fullname = model_fullname.split('/')[-1]
        candidate_paths = [
            f"output_data/{author}/together/processed_batch_outputs/{model_fullname}/name_{name_bundle}_job_{job_bundle}.csv",
            f"output_data/{author}/together/processed_batch_outputs/{model}/name_{name_bundle}_job_{job_bundle}.csv",
            os.path.join(
                repo_path('output_data'),
                author,
                'together',
                'processed_batch_outputs',
                model_fullname,
                f'name_{name_bundle}_job_{job_bundle}.csv'
            ),
            os.path.join(
                repo_path('output_data'),
                author,
                'together',
                'processed_batch_outputs',
                model,
                f'name_{name_bundle}_job_{job_bundle}.csv'
            ),
        ]
    else:
        candidate_paths = [
            os.path.join(
                repo_path('output_data'),
                author,
                model,
                f'temp_{temp}_numtrials_{num_trials}_jobresumes_{job_bundle}_names_{name_bundle}_random_state_{random_state}.csv'
            ),
            os.path.join(repo_path('wang'), f'final_data_{model.replace("-", "")}.csv'),
            os.path.join(repo_path('wang'), 'final_data.csv'),
            'final_data.csv',
        ]
    data_filepath = next((path for path in candidate_paths if os.path.exists(repo_path(path))), None)

    if data_filepath is None:
        print(f"Error reading Wang data. Tried: {candidate_paths}")
        return

    try:
        df = pd.read_csv(repo_path(data_filepath))
    except Exception:
        print(f"Error reading {data_filepath}")
        return

    if 'Industry' in df.columns and 'Occupation' not in df.columns:
        df['Occupation'] = df['Industry']

    required_columns = ['Privilege_Rank', 'Protect_Rank', 'Privilege_Avg_Score', 'Protect_Avg_Score']
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        print(f"Missing required Wang columns: {missing_columns}")
        return

    metrics_to_compute = config['Results'].get('Metrics_to_Compute') or [
        'impact_ratio_rank',
        'impact_ratio_score',
    ]
    sensitive_attribute_vector = config['Results'].get('Sensitive_Attribute_Vector', ['Woman'])

    new_row_template = {
        'Author': author,
        'Names': name_bundle,
        'Jobs': job_bundle,
        'Num_Trials': num_trials,
        'Model': model,
        'Temperature': temp,
        'Random_State': random_state,
    }

    new_rows_list = []
    for metric in metrics_to_compute:
        values, std, pvals = compute_wang_metric(df, sensitive_attribute_vector, metric)
        if values is None:
            print(f"Skipping unsupported Wang metric: {metric}")
            continue

        new_row_dict = new_row_template.copy()
        new_row_dict['Metric'] = metric
        new_row_dict['Sensitive_Attribute_Vector'] = sensitive_attribute_vector
        new_row_dict['Result_Vector'] = values
        new_row_dict['Std_Error_Vector'] = std
        new_row_dict['P_Value_Vector'] = pvals
        new_rows_list.append(new_row_dict)

    occupation_column = 'Occupation' if 'Occupation' in df.columns else None
    detailed_rows = []
    if occupation_column is not None:
        for occupation, df_occ in df.groupby(occupation_column):
            rank_results = compute_wang_ras_impact_ratio(df_occ, use_scores=False)
            score_results = compute_wang_ras_impact_ratio(df_occ, use_scores=True)
            detailed_rows.append({
                'model': model,
                'name_bundle': name_bundle,
                'job_bundle': job_bundle,
                'occupation': occupation,
                'n_samples': rank_results['n_samples'],
                'selection_rate_privilege_rank': rank_results['selection_rate_privilege'],
                'selection_rate_protect_rank': rank_results['selection_rate_protect'],
                'impact_ratio_privilege_rank': rank_results['impact_ratio_privilege'],
                'impact_ratio_protect_rank': rank_results['impact_ratio_protect'],
                'selection_rate_privilege_score': score_results['selection_rate_privilege'],
                'selection_rate_protect_score': score_results['selection_rate_protect'],
                'impact_ratio_privilege_score': score_results['impact_ratio_privilege'],
                'impact_ratio_protect_score': score_results['impact_ratio_protect'],
            })

    if new_rows_list:
        new_rows = pd.DataFrame(new_rows_list).dropna(axis=1, how='all')
        perturbation_results_df = pd.concat([perturbation_results_df, new_rows], ignore_index=True)
        perturbation_results_df.to_csv(repo_path(perturbation_results_filepath), index=False)

    if detailed_rows:
        pd.DataFrame(detailed_rows).to_csv(repo_path(detailed_results_filepath), mode='a', index=False, header=False)

    return

def analyze(args, config, all_together=False):
    if config['experimental_framework'] == 'armstrong':
        analyze_armstrong(args, config, all_together)
    elif config['experimental_framework'] == 'yin':
        analyze_yin(args, config, all_together)
    elif config['experimental_framework'] == 'wang':
        analyze_wang(args, config, all_together)
    else:
        raise ValueError(f"experimental_framework {config['experimental_framework']} not supported")


'''

'''

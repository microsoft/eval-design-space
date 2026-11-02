from utils.paths import REPO_ROOT, repo_path

# read in the config file to figure out what to run
import os
import sys
import yaml
from utils import generate_results, analyze_results
import pandas as pd
import argparse
from utils.merge_config import merge_config
import gc
import tqdm


def main():
    # read in the config file path as an argument
    parser = argparse.ArgumentParser(description='Run perturbation experiments.')
    parser.add_argument('--base_config_path', type=str, default='configs/base_wang.yaml', help='Path to the config file.')
    # parser.add_argument('--overlay_config_path', type=str, help='Path to the config file.')
    parser.add_argument('--name', type=str, help='Name of the bundle.')
    parser.add_argument('--job', type=str, help='job of the bundle.')
    parser.add_argument('--model', type=str, default='gpt-4o-mini', help='model to use.')
    parser.add_argument('--save_job_name_values', action='store_true', help='Whether to save job and name values.')
    args = parser.parse_args() 
    
    config = yaml.load(open(repo_path(args.base_config_path)), Loader=yaml.FullLoader)
    config['Name']['Bundle_Name'] = args.name
    config['Job']['Bundle_Name'] = args.job
    config['Model']['Model_Name'] = args.model

    # config = merge_config(args.base_config_path, args.overlay_config_path)
    # if you need to generate new result, generate new results.
    if args.save_job_name_values:
        generate_results.save_job_names(config)
        return 

    generate_results.generate(config)



if __name__ == "__main__":
    main()


'''
python main.py --base_config_path configs/base_yin.yaml --name gaeb --job armstrong --model gpt-4o-mini --swap_jobdescription
python main.py --base_config_path configs/base_yin.yaml --name gaeb --job yin --model gpt-4o-mini --swap_jobdescription
python main.py --base_config_path configs/base_yin.yaml --name gaeb --job yin --model gpt-4o-mini --swap_jobdescription
'''

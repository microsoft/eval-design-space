from utils.paths import REPO_ROOT, repo_path

# read in the config file to figure out what to run
import os
import sys
import yaml
from utils import generate_results, analyze_results
import pandas as pd
import argparse

from utils.merge_config import merge_config
from pathlib import Path
import json
from together import Together
from utils.variables import MODELS, NAMES, JOBS, MODELS_GPT

def main():
    # read in the config file path as an argument
    parser = argparse.ArgumentParser(description='Run perturbation experiments.')
    parser.add_argument('--config_path', type=str, default='configs/base_wang.yaml', help='Path to the config file.')
   # parser.add_argument('--overlay_config_path', type=str, default='configs/armstrong/overlays/name_armstrong_job_armstrong.yaml', help='Path to the overlay config file.')
    analysis_mode = parser.add_mutually_exclusive_group()
    analysis_mode.add_argument('--all', action='store_true', help='Analyze all API-result configs.')
    analysis_mode.add_argument('--all_together', action='store_true', help='Analyze all Together batch results.')
    analysis_mode.add_argument('--model', type=str, help='Analyze this model instead of the config default.')
    parser.add_argument('--sampling', type=str, default='downsampling', help='Sampling method to use.') 
    parser.add_argument('--random_state', type=str, default='42', help='Random state for the experiment.') # for sampling 
    parser.add_argument('--analysis', type=str, default='original', help='Analysis to perform.') # original or modified metric 
    parser.add_argument('--author', type=str, default='wang', help='Author to analyze.') # all or specific author 
    parser.add_argument('--overlay_type', type=str, default='overlays', help='Type of overlay to use.') # overlays or additional_overlays 
    parser.add_argument('--name', type=str, help='Name of the bundle.')
    parser.add_argument('--job', type=str, help='job of the bundle.')
    parser.add_argument(
        '--per_job', 
        action='store_true', 
        default=False, 
        help='If set, check the list of things to relaunch and only let the code happen if it fits one of the conditions.'
    )


    # parser.add_argument('--random_state', type=int, default=42, help='Random state for the experiment.')
    args = parser.parse_args()
    
    # config_path = args.config_path

    base_config_path = args.config_path
    author = os.path.splitext(os.path.basename(repo_path(base_config_path)))[0].split('base_')[1].split('_')[0]

    if args.all or args.all_together:
        models = MODELS if args.all_together else MODELS_GPT
        for model in models:
            for name in NAMES:
                for job in JOBS:
                    with open(repo_path(base_config_path), encoding='utf-8') as config_file:
                        config = yaml.safe_load(config_file)
                    config['Model']['Model_Name'] = model
                    config['Name']['Bundle_Name'] = name
                    config['Job']['Bundle_Name'] = job
                    config['experimental_framework'] = author
                    print(f"Analyzing {model} {name} {job}")
                    analyze_results.analyze(args, config, all_together=args.all_together)
        return

    with open(repo_path(base_config_path), encoding='utf-8') as config_file:
        config = yaml.safe_load(config_file)
    if args.model is not None:
        config['Model']['Model_Name'] = args.model
    if args.name is not None:
        config['Name']['Bundle_Name'] = args.name
    if args.job is not None:
        config['Job']['Bundle_Name'] = args.job
    config['experimental_framework'] = author
    analyze_results.analyze(args, config)

if __name__ == "__main__":
    main()

'''

'''

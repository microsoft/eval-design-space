"""
Script to generate final_data.csv using GPT-4 (2023-11-06)
Based on pages/1_Injection.py with Streamlit dependencies removed.
"""
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.paths import REPO_ROOT, repo_path


import pandas as pd
import os
from dotenv import load_dotenv
from util.injection import process_scores_multiple
from util.prompt import PROMPT_TEMPLATE

sys.path.insert(0, os.path.join(repo_path(os.path.dirname(__file__)), '..'))
from utils.model import Model


class RunQueryAgent:
    """Adapter so existing pipeline can call `invoke()` using `Model.run_query()`."""

    def __init__(self, model: Model):
        self.model = model

    def invoke(self, prompt: str, **kwargs):
        return self.model.run_query(prompt)

# Load environment variables
load_dotenv(repo_path(".env"))

# Configuration
CONFIG = {
    # Model settings
    "model_name": "gpt-4o",  # GPT-4 Turbo

    # Generation parameters
    "temperature": 0.0,
    "max_tokens": 300,

    # Experiment settings
    "group_name": "Gender",
    "privilege_label": "Male",
    "protect_label": "Female",
    "num_run": 1,
    "sample_size": None,  # None means use all data
    "occupation": None,  # None means process all occupations

    # File paths
    "input_file": "wang/resume_subsampled.csv",
    "output_file": "wang/final_data_gpt4o.csv",

    # Prompt template
    "prompt_template": PROMPT_TEMPLATE,
}


def load_data(filepath: str, occupation: str = None, sample_size: int = None) -> pd.DataFrame:
    """Load and optionally filter/sample the resume data."""
    df = pd.read_csv(repo_path(filepath))

    if occupation:
        df = df[df["Occupation"] == occupation]

    if sample_size and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=42)

    return df


def process_occupation(df: pd.DataFrame, agent: RunQueryAgent, config: dict) -> pd.DataFrame:
    """Process a single occupation's data through the model."""
    parameters = {
        "temperature": config["temperature"],
        "max_tokens": config["max_tokens"]
    }

    processed_df = process_scores_multiple(
        df=df,
        num_run=config["num_run"],
        parameters=parameters,
        privilege_label=config["privilege_label"],
        protect_label=config["protect_label"],
        agent=agent,
        group_name=config["group_name"],
        occupation=df["Occupation"].iloc[0] if len(df) > 0 else "Unknown",
        template=config["prompt_template"]
    )

    return processed_df


def main():
    """Main function to generate final_data.csv"""
    print("=" * 60)
    print("Resume Scoring Generation Script - GPT-4o")
    print("=" * 60)

    # Initialize the OpenAI model
    print(f"\nInitializing OpenAI model...")
    print(f"  Model: {CONFIG['model_name']}")
    print(f"  Temperature: {CONFIG['temperature']}")
    print(f"  Max tokens: {CONFIG['max_tokens']}")

    base_model = Model(
        model_name=CONFIG["model_name"],
        temperature=CONFIG["temperature"],
        max_tokens=CONFIG["max_tokens"]
    )
    agent = RunQueryAgent(base_model)

    # Load data
    print(f"\nLoading data from: {CONFIG['input_file']}")
    df = load_data(
        CONFIG["input_file"],
        occupation=CONFIG["occupation"],
        sample_size=CONFIG["sample_size"]
    )
    print(f"  Loaded {len(df)} entries")

    if CONFIG["occupation"]:
        print(f"  Filtered to occupation: {CONFIG['occupation']}")

    # Get unique occupations
    occupations = df["Occupation"].unique()
    print(f"  Occupations: {list(occupations)}")

    # Process data
    print(f"\nProcessing settings:")
    print(f"  Group: {CONFIG['group_name']}")
    print(f"  Privilege label: {CONFIG['privilege_label']}")
    print(f"  Protect label: {CONFIG['protect_label']}")
    print(f"  Number of runs: {CONFIG['num_run']}")
    print(f"  Temperature: {CONFIG['temperature']}")
    print(f"  Max tokens: {CONFIG['max_tokens']}")

    all_results = []

    for occupation in occupations:
        print(f"\n{'='*40}")
        print(f"Processing occupation: {occupation}")
        print(f"{'='*40}")

        occupation_df = df[df["Occupation"] == occupation].copy()
        print(f"  Entries: {len(occupation_df)}")

        processed_df = process_occupation(occupation_df, agent, CONFIG)
        all_results.append(processed_df)

    # Combine all results
    final_df = pd.concat(all_results, ignore_index=True)

    # Save results
    print(f"\nSaving results to: {CONFIG['output_file']}")
    final_df.to_csv(repo_path(CONFIG["output_file"]), index=False)
    print(f"  Saved {len(final_df)} entries")

    # Print summary statistics
    print("\n" + "=" * 60)
    print("Summary Statistics")
    print("=" * 60)

    for occupation in final_df["Occupation"].unique():
        occ_df = final_df[final_df["Occupation"] == occupation]
        print(f"\n{occupation}:")
        print(f"  Privilege Avg Score: {occ_df['Privilege_Avg_Score'].mean():.2f}")
        print(f"  Protect Avg Score: {occ_df['Protect_Avg_Score'].mean():.2f}")
        print(f"  Neutral Avg Score: {occ_df['Neutral_Avg_Score'].mean():.2f}")

    print("\n" + "=" * 60)
    print("Generation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

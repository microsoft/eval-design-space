"""
Script to evaluate Score Difference by occupation.

This is a score-difference-only companion to evaluate_IR.py.

Score Difference Formula:
	score_difference = mean(Privilege_Avg_Score - Protect_Avg_Score)

Interpretation:
	- Positive: Privilege group (Male) scored higher on average
	- Negative: Protected group (Female) scored higher on average
	- Zero: no average difference
"""
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.paths import REPO_ROOT, repo_path


import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp, wilcoxon
from typing import Dict


# Configuration
CONFIG = {
	"input_file": "wang/final_data_gpt4o.csv",
	"occupations": ["HEALTHCARE", "FINANCE", "CONSTRUCTION"],
	"privilege_label": "Male",
	"protect_label": "Female",
}


def calculate_score_difference(df: pd.DataFrame) -> Dict[str, float]:
	"""
	Calculate score difference metrics for one subset of data.

	Uses row-level paired differences:
		diff_i = Privilege_Avg_Score_i - Protect_Avg_Score_i

	Returns:
		Dictionary with mean score difference and significance tests.
	"""
	n = len(df)

	if n == 0:
		return {
			"n_samples": 0,
			"n_valid_pairs": 0,
			"mean_privilege_score": np.nan,
			"mean_protect_score": np.nan,
			"mean_score_difference": np.nan,
			"std_score_difference": np.nan,
			"std_error": np.nan,
			"ci95_lower": np.nan,
			"ci95_upper": np.nan,
			"t_stat": np.nan,
			"t_pvalue": np.nan,
			"wilcoxon_stat": np.nan,
			"wilcoxon_pvalue": np.nan,
		}

	privilege_scores = pd.to_numeric(df["Privilege_Avg_Score"], errors="coerce")
	protect_scores = pd.to_numeric(df["Protect_Avg_Score"], errors="coerce")

	valid_mask = privilege_scores.notna() & protect_scores.notna()
	diffs = (privilege_scores[valid_mask] - protect_scores[valid_mask]).astype(float)

	n_valid = len(diffs)
	if n_valid == 0:
		return {
			"n_samples": n,
			"n_valid_pairs": 0,
			"mean_privilege_score": np.nan,
			"mean_protect_score": np.nan,
			"mean_score_difference": np.nan,
			"std_score_difference": np.nan,
			"std_error": np.nan,
			"ci95_lower": np.nan,
			"ci95_upper": np.nan,
			"t_stat": np.nan,
			"t_pvalue": np.nan,
			"wilcoxon_stat": np.nan,
			"wilcoxon_pvalue": np.nan,
		}

	mean_privilege = float(privilege_scores[valid_mask].mean())
	mean_protect = float(protect_scores[valid_mask].mean())
	mean_diff = float(diffs.mean())

	std_diff = float(diffs.std(ddof=1)) if n_valid > 1 else np.nan
	std_error = float(std_diff / np.sqrt(n_valid)) if n_valid > 1 else np.nan

	if n_valid > 1 and not np.isnan(std_error):
		ci95_lower = float(mean_diff - 1.96 * std_error)
		ci95_upper = float(mean_diff + 1.96 * std_error)
	else:
		ci95_lower, ci95_upper = np.nan, np.nan

	# One-sample t-test against zero difference
	if n_valid > 1:
		t_result = ttest_1samp(diffs, popmean=0.0, nan_policy="omit")
		t_stat = float(t_result.statistic)
		t_pvalue = float(t_result.pvalue)
	else:
		t_stat, t_pvalue = np.nan, np.nan

	# Paired non-parametric test (Wilcoxon signed-rank on paired scores)
	if n_valid > 1:
		try:
			w_stat, w_pvalue = wilcoxon(privilege_scores[valid_mask], protect_scores[valid_mask])
			w_stat, w_pvalue = float(w_stat), float(w_pvalue)
		except ValueError:
			# Happens if all paired differences are zero
			w_stat, w_pvalue = np.nan, np.nan
	else:
		w_stat, w_pvalue = np.nan, np.nan

	return {
		"n_samples": n,
		"n_valid_pairs": n_valid,
		"mean_privilege_score": mean_privilege,
		"mean_protect_score": mean_protect,
		"mean_score_difference": mean_diff,
		"std_score_difference": std_diff,
		"std_error": std_error,
		"ci95_lower": ci95_lower,
		"ci95_upper": ci95_upper,
		"t_stat": t_stat,
		"t_pvalue": t_pvalue,
		"wilcoxon_stat": w_stat,
		"wilcoxon_pvalue": w_pvalue,
	}


def evaluate_occupation(df: pd.DataFrame, occupation: str) -> Dict:
	"""Evaluate score difference for one occupation."""
	occ_df = df[df["Occupation"].str.upper() == occupation.upper()].copy()

	if len(occ_df) == 0:
		print(f"  Warning: No data found for occupation '{occupation}'")
		return None

	score_results = calculate_score_difference(occ_df)
	return {
		"Occupation": occupation,
		"score_difference": score_results,
	}


def print_score_results(results: Dict, occupation: str):
	"""Print score-difference results in a formatted way."""
	print(f"\n{'=' * 60}")
	print(f"Occupation: {occupation.upper()}")
	print(f"{'=' * 60}")

	s = results["score_difference"]

	print(f"\nSample Size: {s['n_samples']} (valid pairs: {s['n_valid_pairs']})")
	print("\n--- Score Difference (Privilege - Protect) ---")
	print(f"  Mean Score (Privilege):     {s['mean_privilege_score']:.4f}")
	print(f"  Mean Score (Protect):       {s['mean_protect_score']:.4f}")
	print(f"  Mean Score Difference:      {s['mean_score_difference']:.4f}")
	print(f"  Std. Dev. of Difference:    {s['std_score_difference']:.4f}")
	print(f"  Std. Error:                 {s['std_error']:.4f}")
	print(f"  95% CI:                     [{s['ci95_lower']:.4f}, {s['ci95_upper']:.4f}]")
	print(f"  t-test (H0: diff=0):        t={s['t_stat']:.4f}, p={s['t_pvalue']:.6f}")
	print(f"  Wilcoxon signed-rank:       W={s['wilcoxon_stat']:.4f}, p={s['wilcoxon_pvalue']:.6f}")


def print_summary_table(all_results: Dict):
	"""Print a summary table of score difference across occupations."""
	print(f"\n{'=' * 90}")
	print("SUMMARY: Score Difference (Privilege - Protect)")
	print(f"{'=' * 90}")

	summary_data = []
	for occ, results in all_results.items():
		if results is None:
			continue
		s = results["score_difference"]
		summary_data.append({
			"Occupation": occ.upper(),
			"N": s["n_samples"],
			"Valid_Pairs": s["n_valid_pairs"],
			"Mean_Privilege": s["mean_privilege_score"],
			"Mean_Protect": s["mean_protect_score"],
			"Score_Diff": s["mean_score_difference"],
			"CI95_Lower": s["ci95_lower"],
			"CI95_Upper": s["ci95_upper"],
			"pvalue_ttest": s["t_pvalue"],
			"pvalue_wilcoxon": s["wilcoxon_pvalue"],
		})

	summary_df = pd.DataFrame(summary_data)
	print(f"\n{summary_df.to_string(index=False)}")
	return summary_df


def main():
	"""Main function to evaluate score differences."""
	print("=" * 60)
	print("Score Difference Evaluation")
	print("GPT-4o Results")
	print("=" * 60)

	print(f"\nLoading data from: {CONFIG['input_file']}")
	try:
		df = pd.read_csv(repo_path(CONFIG["input_file"]))
	except FileNotFoundError:
		print(f"Error: File '{CONFIG['input_file']}' not found.")
		print("Please run generate.py first to create the data file.")
		return

	print(f"  Loaded {len(df)} entries")

	# Filter to GPT-4o model only
	if "model" in df.columns:
		df = df[df["model"].str.lower() == "gpt-4o"].copy()
		print(f"  Filtered to GPT-4o: {len(df)} entries")

	# Check required columns
	required_cols = ["Occupation", "Privilege_Avg_Score", "Protect_Avg_Score"]
	missing_cols = [col for col in required_cols if col not in df.columns]
	if missing_cols:
		print(f"Error: Missing required columns: {missing_cols}")
		print(f"Available columns: {list(df.columns)}")
		return

	# Get available occupations
	available_occupations = df["Occupation"].dropna().astype(str).str.upper().unique()
	print(f"  Available occupations: {list(available_occupations)}")

	# Filter to target occupations
	target_occupations = [occ for occ in CONFIG["occupations"] if occ.upper() in available_occupations]

	if not target_occupations:
		print("\nWarning: None of the target occupations found.")
		print(f"  Target: {CONFIG['occupations']}")
		print(f"  Available: {list(available_occupations)}")
		print("\nProcessing all available occupations instead...")
		target_occupations = list(available_occupations)

	# Evaluate each occupation
	all_results = {}
	for occupation in target_occupations:
		print(f"\nEvaluating: {occupation}")
		results = evaluate_occupation(df, occupation)
		if results:
			all_results[occupation] = results
			print_score_results(results, occupation)

	# Print summary table
	summary_df = print_summary_table(all_results)

	# Save summary to CSV
	output_file = "wang/score_difference_results.csv"
	summary_df.to_csv(repo_path(output_file), index=False)
	print(f"\nResults saved to: {output_file}")

	print("\n" + "=" * 60)
	print("Evaluation complete!")
	print("=" * 60)


if __name__ == "__main__":
	main()

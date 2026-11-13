"""Utilities to compute noise-floor / reproduction statistics for frameworks.

Provides a small CLI and functions to compute:
- Original reproduced value (random_state == 42 when available)
- Mean ± std, Mean 95% CI
- Disagreement rate relative to the reproduced direction (threshold 0 for
  armstrong/wang, 0.5 for yin; neutral is a separate direction)

Reads the project's CSV outputs under `results/` by default.
"""
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.paths import REPO_ROOT, repo_path


from typing import Dict, Any, Optional
import math
import re
import numpy as np
import os
import pandas as pd

DEFAULT_ARMSTRONG_CSV = os.path.join('results', 'armstrong_noise_floor.csv')
DEFAULT_WANG_CSV = os.path.join('results', 'wang_noise_floor.csv')
DEFAULT_YIN_CSV = os.path.join('results', 'yin_noise_floor.csv')


def _extract_number_from_bracketed(x: Any) -> float:
	"""Extract first numeric token from a possibly bracketed string like "[0.5]".

Falls back to NaN if nothing found.
	"""
	if pd.isna(x):
		return float('nan')
	s = str(x)
	m = re.search(r"-?\d+\.?\d*(?:[eE][-+]?\d+)?", s)
	return float(m.group(0)) if m else float('nan')


def sample_and_stats(values: Any, sample_size: int = 25, seed: int = 42):
	"""Sample up to `sample_size` values (without replacement) and return
	(sample_array, mean, std, n).
	Uses a reproducible RNG with `seed`.
	"""
	arr = np.asarray(values, dtype=float)
	if arr.size == 0:
		return arr, float('nan'), float('nan'), 0
	if arr.size > sample_size:
		rng = np.random.default_rng(seed)
		idx = rng.choice(arr.size, size=sample_size, replace=False)
		sample = arr[idx]
	else:
		sample = arr
	n = sample.size
	mean = float(np.mean(sample)) if n > 0 else float('nan')
	std = float(np.std(sample, ddof=1)) if n > 1 else 0.0
	return sample, mean, std, n


def compute_framework_stats(
	arm_csv: str,
	wang_csv: str,
	yin_csv: Optional[str] = None,
) -> Dict[str, Dict[str, Any]]:
		"""Compute statistics for `armstrong`, `wang`, and `yin` frameworks.

		Uses the per-framework sampling CSVs for armstrong and wang (which contain
		`Result_Vector` entries) and the yin CSV (which contains `selection_rate`).

		Returns a dict keyed by framework with keys:
		- original: reproduced value (random_state==42 when available)
		- mean, std, ci, n
		- disagreement_rate
		"""
		results: Dict[str, Dict[str, Any]] = {}

		for framework, samp_path in (('armstrong', arm_csv), ('wang', wang_csv)):
			samp = pd.read_csv(repo_path(samp_path))
			# select rows for this framework
			sel = samp[(samp.get('Names') == framework) & (samp.get('Jobs') == framework) & (samp.get('Model') == 'gpt-4o-mini')]
			# extract numeric from Result_Vector which may be like "[0.5]"
			vals = sel['Result_Vector'].apply(_extract_number_from_bracketed).dropna().astype(float).values
			# Sample up to 25 values and compute stats via helper
			sample_vals, mean, std, n = sample_and_stats(vals, sample_size=25, seed=42)
			ci = 1.96 * std / math.sqrt(n) if n > 0 else float('nan')

			# original: prefer row with Random_State == 42
			original = float('nan')
			if 'Random_State' in samp.columns:
				samp = samp.copy()
				samp['RS_num'] = pd.to_numeric(samp['Random_State'], errors='coerce')
				samp_match = samp[(samp.get('Names') == framework) & (samp.get('Jobs') == framework) & (samp['RS_num'] == 42)]
			else:
				samp_match = samp[(samp.get('Names') == framework) & (samp.get('Jobs') == framework)]

			if not samp_match.empty:
				original = _extract_number_from_bracketed(samp_match.iloc[0].get('Result_Vector'))
			elif n > 0:
				original = float(vals[0])

			if n > 0 and not np.isfinite(original):
				raise ValueError(f"{samp_path}: reproduced baseline must be finite to compute disagreement")
			dis_rate = float(np.mean(np.sign(sample_vals) != np.sign(original))) if n > 0 else float('nan')

			results[framework] = {
				'original': original,
				'mean': mean,
				'std': std,
				'ci': ci,
				'n': n,
				'disagreement_rate': dis_rate,
			}

		# yin: optionally evaluate selection_rate rows in separate CSV
		if yin_csv:
			ny = pd.read_csv(repo_path(yin_csv))
			# support both header styles (some files have name_bundle/job_bundle)
			if 'name_bundle' not in ny.columns or 'job_bundle' not in ny.columns:
				ny.columns = [c.lower() for c in ny.columns]
			
			nyt = ny[(ny.get('name_bundle') == 'yin') & (ny.get('job_bundle') == 'yin') & (ny.get('model') == 'gpt-4o-mini')  & (ny.get('gender') == 'W')]

			if not nyt.empty:
				sel_rates = pd.to_numeric(nyt.get('selection_rate'), errors='coerce').dropna().astype(float).values
				# Sample up to 25 selection rates and compute stats via helper
				sample_sel, mean_y, std_y, n_y = sample_and_stats(sel_rates, sample_size=25, seed=42)
				ci_y = 1.96 * std_y / math.sqrt(n_y) if n_y > 0 else float('nan')
				# original: prefer row with random_state == 42
				original_y = float('nan')
				if 'random_state' in nyt.columns:
					nyt = nyt.copy()
					nyt['RS_num'] = pd.to_numeric(nyt['random_state'], errors='coerce')
					orig_rows = nyt[nyt['RS_num'] == 42]
					if not orig_rows.empty:
						original_y = float(pd.to_numeric(orig_rows.iloc[0]['selection_rate'], errors='coerce'))
				if math.isnan(original_y) and n_y > 0:
					original_y = float(sel_rates[0])

				if n_y > 0 and not np.isfinite(original_y):
					raise ValueError(f"{yin_csv}: reproduced baseline must be finite to compute disagreement")
				dis_rate_y = float(np.mean(np.sign(sample_sel - 0.5) != np.sign(original_y - 0.5))) if n_y > 0 else float('nan')
				results['yin'] = {
					'original': original_y,
					'mean': mean_y,
					'std': std_y,
					'ci': ci_y,
					'n': n_y,
					'disagreement_rate': dis_rate_y,
				}
			else:
				results['yin'] = {'original': float('nan'), 'mean': float('nan'), 'std': float('nan'), 'ci': float('nan'), 'n': 0, 'disagreement_rate': float('nan')}
		else:
			# skip yin evaluation
			results['yin'] = {'original': float('nan'), 'mean': float('nan'), 'std': float('nan'), 'ci': float('nan'), 'n': 0, 'disagreement_rate': float('nan')}

		return results


def compute_default_framework_stats() -> Dict[str, Dict[str, Any]]:
	return compute_framework_stats(DEFAULT_ARMSTRONG_CSV, DEFAULT_WANG_CSV, DEFAULT_YIN_CSV)


def print_stats_table(stats: Dict[str, Dict[str, Any]]) -> None:
	for fw in ('yin', 'wang', 'armstrong'):
		r = stats.get(fw, {})
		print(f"{fw.upper()}")
		print(f"Original Reproduced Value: {r.get('original')}")
		print(f"Mean ± std: {r.get('mean')} ± {r.get('std')} (n={r.get('n')})")
		print(f"Mean, 95% CI: {r.get('mean')} ± {r.get('ci')}")
		print(f"Disagreement Rate (direction differs from reproduced baseline): {r.get('disagreement_rate')}")
		print()


if __name__ == '__main__':
	# Always evaluate all three frameworks by default using the project's
	# results CSVs.
	# get current directory 
	current_dir= str(REPO_ROOT)
	stats = compute_default_framework_stats()
	stats['wang']['original'] = -stats['wang']['original']
	stats['wang']['mean'] = -stats['wang']['mean']
	# Save results as CSV: one row with columns 'armstrong', 'wang', 'yin'.
	# Each cell contains the four metric values separated by ';':
	# Reproduced Value; Mean ± SD; Mean, 95% CI (SE); Disagreement Rate (Sign Flip Frequency)
	results_dir = os.path.join(repo_path(current_dir), 'results')
	os.makedirs(repo_path(results_dir), exist_ok=True)

	def _fmt_num(x: Any) -> str:
		try:
			if x is None:
				return 'NA'
			if isinstance(x, float) and math.isnan(x):
				return 'NA'
			# Use compact formatting
			return f"{float(x):.6g}"
		except Exception:
			return str(x)

	row = {}
	for fw in ('armstrong', 'wang', 'yin'):
		r = stats.get(fw, {})
		repr_val = _fmt_num(r.get('original'))
		mean_val = _fmt_num(r.get('mean'))
		std_val = _fmt_num(r.get('std'))
		ci_val = _fmt_num(r.get('ci'))
		dis_val = _fmt_num(r.get('disagreement_rate'))
		cell = (
			f"Reproduced Value: {repr_val}; "
			f"Mean ± SD: {mean_val} ± {std_val}; "
			f"Mean, 95% CI (SE): {mean_val} ± {ci_val}; "
			f"Disagreement Rate (Sign Flip Frequency): {dis_val}"
		)
		row[fw] = cell

	df_out = pd.DataFrame([row])
	out_path = os.path.join(repo_path(results_dir), 'noise_floor_results.csv')
	df_out.to_csv(repo_path(out_path), index=False)
	print(f"Saved noise-floor results to: {out_path}")

	print_stats_table(stats)

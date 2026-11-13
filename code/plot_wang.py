from utils.paths import REPO_ROOT, repo_path
from utils.plot_references import (
	load_reproduced_reference, load_saved_wang_reference, REPRODUCED_SETTING_LABEL,
	reported_setting_label,
)

import os
import ast
import argparse
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from utils.noise_floor import compute_default_framework_stats

from utils.variables import metric_to_plot, model_to_plotname



TITLE_FONT_SIZE = 40
LABEL_FONT_SIZE = 28
LEGEND_FONT_SIZE = 20
TICK_FONT_SIZE = 25
plt.rcParams.update({
	'font.family': 'sans-serif',
	'font.sans-serif': ['Inter-Regular', 'DejaVu Sans', 'Arial', 'Liberation Sans'],
	'axes.titlesize': TITLE_FONT_SIZE,
	'axes.labelsize': LABEL_FONT_SIZE,
	'legend.fontsize': LEGEND_FONT_SIZE,
	'legend.title_fontsize': LEGEND_FONT_SIZE,
	'xtick.labelsize': TICK_FONT_SIZE,
	'ytick.labelsize': TICK_FONT_SIZE
})

REF_LINE_BROWN = '#8B4513'
REF_LINE_STYLES = ['-.', ':']
REPRODUCED_SETTING_MODELS = [
	'GPT-4',
	'GPT-4o',
	'Claude-Haiku',
	'Claude-Sonnet',
	'Gemini-15-Flash',
	'Gemini-15-Pro',
	'GPT-3.5',
	'llama3-70b-Instruct',
	'llama3-8b-Instruct',
	'Mistral-Large',
]


WANG_METRIC_LABELS = {
	'impact_ratio_rank': 'Impact Ratio (Rank, Protected Group)',
	'impact_ratio_score': 'Impact Ratio (Score, Protected Group)',
	'selection_rate_rank': 'Selection Rate (Rank, Protected Group)',
	'selection_rate_score': 'Selection Rate (Score, Protected Group)',
	'score_difference': 'Score Difference',
	'score_diff': 'Score Difference',
	'wang_score_diff': 'Score Difference',
	'impact_ratio_protect_rank': 'Impact Ratio (Rank, Protected Group)',
	'impact_ratio_protect_score': 'Impact Ratio (Score, Protected Group)',
	'selection_rate_protect_rank': 'Selection Rate (Rank, Protected Group)',
	'selection_rate_protect_score': 'Selection Rate (Score, Protected Group)',
}


def pretty_metric(metric: str) -> str:
	return WANG_METRIC_LABELS.get(metric, metric_to_plot.get(metric, metric))


def ensure_dirs(*paths: str) -> None:
	for p in paths:
		os.makedirs(repo_path(p), exist_ok=True)


def safe_dirname(name: Any) -> str:
	"""Convert arbitrary group names into filesystem-safe folder names."""
	s = str(name).strip()
	s = re.sub(r'[\\/:*?"<>|]+', '_', s)
	s = re.sub(r'\s+', '_', s)
	return s if s else 'unknown'


def clean_values(values: List[Any], remove_zeros: bool = False, remove_outliers: bool = False) -> List[float]:
	if not values:
		return []

	series = pd.to_numeric(pd.Series(values), errors='coerce')
	cleaned = series[series.notna() & np.isfinite(series)].tolist()

	if remove_zeros:
		cleaned = [v for v in cleaned if v != 0]
	if remove_outliers:
		cleaned = [v for v in cleaned if 0 <= v <= 1]
	return cleaned


def parse_result_vector(x: Any) -> Optional[List[float]]:
	if pd.isna(x):
		return None
	if isinstance(x, list):
		return clean_values(x)
	try:
		parsed = ast.literal_eval(str(x))
		if isinstance(parsed, list):
			return clean_values(parsed)
		return clean_values([parsed])
	except Exception:
		return None


def maybe_flip_for_plot(metric: Any, value: Any) -> Any:
	"""Flip sign for score-difference style metrics so plots show protect - privilege."""
	metric_str = str(metric).strip().lower()
	if metric_str in {'score_difference', 'score_diff', 'wang_score_diff'}:
		val = pd.to_numeric(pd.Series([value]), errors='coerce').iloc[0]
		if pd.notna(val):
			return -float(val)
	return value


def load_reproduced_setting_vlines(results_dir: str) -> List[Tuple[float, str, str]]:
	"""Load one reproduced score-difference value for each evaluated model."""
	vlines = []
	colors = plt.cm.tab10(np.linspace(0, 1, len(REPRODUCED_SETTING_MODELS)))
	for model, color in zip(REPRODUCED_SETTING_MODELS, colors):
		model_filename = model.lower().replace('.', '_').replace('-', '_')
		results_path = os.path.join(repo_path(results_dir), f'statistical_results_{model_filename}.csv')
		if not os.path.exists(repo_path(results_path)):
			print(f"Skipping reproduced setting for {model}: {results_path} not found.")
			continue

		results_df = pd.read_csv(repo_path(results_path))
		matches = results_df[results_df['Metric'].eq('Score Differences')]
		if matches.empty:
			print(f"Skipping reproduced setting for {model}: Score Differences not found.")
			continue

		value = pd.to_numeric(matches['Value'].iloc[0], errors='coerce')
		if pd.isna(value):
			print(f"Skipping reproduced setting for {model}: invalid Score Differences value.")
			continue
		vlines.append((float(value), reported_setting_label(model), color))
	return vlines


def load_wang_data(input_file: str) -> pd.DataFrame:
	df = pd.read_csv(repo_path(input_file))

	# Detailed format from analyze_wang
	detailed_metric_cols = [
		'impact_ratio_protect_rank',
		'impact_ratio_protect_score',
		'selection_rate_protect_rank',
		'selection_rate_protect_score',
	]
	if all(col in df.columns for col in ['model', 'name_bundle', 'job_bundle']) and any(col in df.columns for col in detailed_metric_cols):
		available = [c for c in detailed_metric_cols if c in df.columns]
		long_df = df.melt(
			id_vars=[c for c in ['model', 'name_bundle', 'job_bundle', 'occupation', 'n_samples'] if c in df.columns],
			value_vars=available,
			var_name='metric',
			value_name='value'
		)
		long_df['value'] = pd.to_numeric(long_df['value'], errors='coerce')
		long_df = long_df.dropna(subset=['value']).copy()
		if 'occupation' not in long_df.columns:
			long_df['occupation'] = 'ALL'
		return long_df

	# Aggregate format from analyze_wang
	required = {'Model', 'Names', 'Jobs', 'Metric', 'Result_Vector', "Random_State"}
	if required.issubset(set(df.columns)):
		df = df.copy()
		df['parsed_result'] = df['Result_Vector'].apply(parse_result_vector)
		df = df[df['parsed_result'].notna()].copy()
		df['value'] = df['parsed_result'].apply(lambda x: x[0] if isinstance(x, list) and len(x) > 0 else np.nan)
		df['value'] = df.apply(lambda row: maybe_flip_for_plot(row.get('Metric'), row.get('value')), axis=1)
		df = df.dropna(subset=['value']).copy()
		df = df.rename(columns={
			'Model': 'model',
			'Names': 'name_bundle',
			'Jobs': 'job_bundle',
			'Metric': 'metric',
			'Random_State': 'random_state'
		})
		df['occupation'] = 'ALL'
		return df[['model', 'name_bundle', 'job_bundle', 'occupation', 'metric', 'value', 'random_state'] if 'random_state' in df.columns else ['model', 'name_bundle', 'job_bundle', 'occupation', 'metric', 'value']]

	raise ValueError(f"Unrecognized Wang input format in {input_file}")


def plot_global_density(
	values: List[float],
	metric: str,
	out_path: str,
	vlines: Optional[List[tuple]] = None,
	split_by: Optional[str] = None,
	bw_threshold: Optional[float] = None,
	noise_floor: Optional[Tuple[float, float]] = None,
	full_width: bool = False,
) -> None:

	vals = clean_values(values)
	# If caller did not supply a noise_floor, derive it from a reproducible
	# sample of up to 25 values using the project's helper.
	if noise_floor is None:
		# Compute noise-floor mean/std by sampling up to 25 values
		noise_floor_results = compute_default_framework_stats()
		nf_n, nf_mean, nf_std = noise_floor_results['wang']['n'], noise_floor_results['wang']['mean'], noise_floor_results['wang']['std']
		nf_mean = -1 * nf_mean
		print(f"Noise floor (sampled n={nf_n}): mean={nf_mean}, std={nf_std}")
		noise_floor = (nf_mean, nf_std)
	if not vals:
		return		
	# Full-width mode expands horizontal plotting area.
	# plt.figure(figsize=((14, 6) if full_width else (7, 6)))
	if full_width:
		plt.figure(figsize=(18, 8))     
	else:
		plt.figure(figsize=(7, 6))
		
	if len(vals) > 1 and np.std(vals) > 1e-9:
		kde = gaussian_kde(vals)
		if bw_threshold is not None:
			kde.set_bandwidth(bw_method=bw_threshold)
		x_min, x_max = min(vals), max(vals)
		pad = max((x_max - x_min) * 0.15, 1e-3)
		#. x_limit = max(abs(x_min), abs(x_max)) + pad
		x_limit = 1
		x_grid = np.linspace(-x_limit, x_limit, 300)
		y_grid = kde(x_grid)

		threshold = 0.0
		below_mask = x_grid <= threshold
		above_mask = x_grid > threshold
		# below_count = sum(value < threshold for value in vals)
		# above_count = len(vals) - below_count
		# below_pct = 100.0 * below_count / len(vals)
		# above_pct = 100.0 * above_count / len(vals)
		vals_arr = np.asarray(vals, dtype=float)
		below_pct = (vals_arr < threshold).mean() * 100.0
		above_pct = (vals_arr >= threshold).mean() * 100.0

		men_color = '#648FFF'
		women_color = '#FFB000'
		if below_mask.any():
			plt.plot(x_grid[below_mask], y_grid[below_mask], color=men_color, lw=2)
		if above_mask.any():
			plt.plot(x_grid[above_mask], y_grid[above_mask], color=women_color, lw=2)
		if below_mask.any():
			plt.fill_between(
				x_grid[below_mask],
				y_grid[below_mask],
				alpha=0.5,
				color=men_color,
				label='Preference for Men\nAcross Settings'
			)
		if above_mask.any():
			plt.fill_between(
				x_grid[above_mask],
				y_grid[above_mask],
				alpha=0.3,
				color=women_color,
				label='Preference for Women\nAcross Settings'
			)
		plt.axvline(x=threshold, color='gray', linestyle='--', linewidth=2, alpha=0.9)
		plt.ylabel('Probability Density')
		ax = plt.gca()
		ax.set_xlim(-x_limit, x_limit)

		y_top = ax.get_ylim()[1]
		text_offset = 0.015 * x_limit
		plt.text(-text_offset, 0.04 * y_top, f"{below_pct:.0f}%", ha='right', va='center', fontsize=28)
		plt.text(text_offset, 0.04 * y_top, f"{above_pct:.0f}%", ha='left', va='center', fontsize=28)
	else:
		# Histogram fallback with same below/above threshold coloring
		threshold = 0.0
		below_vals = [v for v in vals if v < threshold]
		above_vals = [v for v in vals if v >= threshold]
		x_min, x_max = min(vals), max(vals)
		pad = max((x_max - x_min) * 0.25, 1e-3)
		x_limit = max(abs(x_min), abs(x_max)) + pad
		bins = np.linspace(-x_limit, x_limit, 20) if len(set(vals)) > 1 else 20
		if below_vals:
			plt.hist(below_vals, bins=bins, color='darkblue', alpha=0.6, edgecolor='black', label='Preference for Men\nAcross Settings')
		if above_vals:
			plt.hist(above_vals, bins=bins, color='purple', alpha=0.6, edgecolor='black', label='Preference for Women\nAcross Settings')
		plt.axvline(x=threshold, color='gray', linestyle='--', linewidth=2, alpha=0.9)
		plt.ylabel('Frequency')
		ax = plt.gca()
		ax.set_xlim(-x_limit, x_limit)

		total_count = len(vals)
		below_pct = 100.0 * len(below_vals) / total_count if total_count else 0.0
		above_pct = 100.0 * len(above_vals) / total_count if total_count else 0.0
		y_top = ax.get_ylim()[1]
		text_offset = 0.08 * x_limit
		plt.text(-text_offset, 0.25 * y_top, f"{below_pct:.0f}%", ha='right', va='center', fontsize=18)
		plt.text(text_offset, 0.25 * y_top, f"{above_pct:.0f}%", ha='left', va='center', fontsize=18)

	if split_by:
		print(split_by)
		if split_by.startswith("model: "):
			model_name = split_by.split(': ')[1].strip()
			plot_name = model_to_plotname.get(model_name, model_name)
			plt.title(f'{plot_name}')
		if split_by.startswith("job: "):
			job = split_by.split(': ')[1].strip()
			plt.title(f'{job}'.capitalize())
		if split_by.startswith("name: "):
			name = split_by.split(': ')[1].strip()
			plt.title(f'{name}'.capitalize())
	else: 
		plt.title("Wang et. al", pad=16)


	plt.xlabel(pretty_metric(metric), labelpad=12)
	plt.ylabel(plt.gca().get_ylabel(), labelpad=12)
	plt.xlim((-1,1))
	

	if vlines:
		for idx, (x_val, label, line_color) in enumerate(vlines):
			plt.axvline(
				x=x_val,
				color=line_color,
				linestyle=REF_LINE_STYLES[idx % len(REF_LINE_STYLES)],
				linewidth=2,
				alpha=0.9,
				label=label,
			)

	# Draw noise floor shaded region if provided (mean, std)
	if noise_floor is not None:
		try:
			nf_mean, nf_std = float(noise_floor[0]), float(noise_floor[1])
			nf_low = nf_mean - nf_std
			nf_high = nf_mean + nf_std
			plt.axvspan(nf_low, nf_high, color=REF_LINE_BROWN, alpha=0.18, label="Noise Floor")
		except Exception:
			pass

	if full_width:
		plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

	plt.grid(axis='y', alpha=0.3)
	plt.tight_layout()
	final_path = out_path
	if full_width:
		final_path = final_path.replace('.png', '_reported_settings.png')
	plt.savefig(repo_path(final_path))
	print(f"Created plot: {final_path}")
	plt.close()


def plot_split_density_cdf(
	split_dict: Dict[str, List[float]],
	metric: str,
	split_type: str,
	out_root: str,
	vlines: Optional[List[tuple]] = None,
	bw_threshold: Optional[float] = None,
) -> None:
	if not split_dict:
		return

	fig_den, ax_den = plt.subplots(figsize=(12, 7))
	fig_cdf, ax_cdf = plt.subplots(figsize=(12, 7))
	fig_hist, ax_hist = plt.subplots(figsize=(12, 7))

	keys = list(split_dict.keys())
	colors = plt.cm.tab20(np.linspace(0, 1, max(2, len(keys))))

	all_vals = []
	for k in keys:
		all_vals.extend(clean_values(split_dict[k]))
	bins = np.linspace(min(all_vals), max(all_vals), 30) if all_vals else 30

	plotted = False
	for i, key in enumerate(keys):
		vals = clean_values(split_dict[key])
		if not vals:
			continue

		ax_hist.hist(vals, bins=bins, alpha=0.45, edgecolor='black', color=colors[i], label=str(key))

		if len(vals) > 1 and np.std(vals) > 1e-9:
			kde = gaussian_kde(vals)
			if bw_threshold is not None:
				kde.set_bandwidth(bw_method=bw_threshold)
			x_min, x_max = min(vals), max(vals)
			pad = max((x_max - x_min) * 0.25, 1e-3)
			x_grid = np.linspace(x_min - pad, x_max + pad, 250)
			ax_den.plot(x_grid, kde(x_grid), lw=2, color=colors[i], label=str(key))

			sorted_vals = np.sort(vals)
			cdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
			ax_cdf.step(sorted_vals, cdf, where='post', lw=2, color=colors[i], label=str(key))

		plotted = True

	if not plotted:
		plt.close(fig_den)
		plt.close(fig_cdf)
		plt.close(fig_hist)
		return

	ax_hist.set_title(f"Wang: {pretty_metric(metric)} Histogram by {split_type}", pad=16)
	ax_hist.set_xlabel(pretty_metric(metric), labelpad=12)
	ax_hist.set_ylabel('Frequency', labelpad=12)
	if vlines:
		for idx, (x_val, label, _color) in enumerate(vlines):
			ax_hist.axvline(
				x=x_val,
				color=REF_LINE_BROWN,
				linestyle=REF_LINE_STYLES[idx % len(REF_LINE_STYLES)],
				linewidth=2,
				alpha=0.9,
				label=label,
			)
	ax_hist.grid(axis='y', alpha=0.3)
	ax_hist.legend(title=split_type, bbox_to_anchor=(1.02, 1), loc='upper left')
	fig_hist.tight_layout()
	hist_path = os.path.join(repo_path(out_root), f"{metric}_combined_histogram_by_{split_type}.png")
	fig_hist.savefig(repo_path(hist_path))
	print(f"Created plot: {hist_path}")

	ax_den.set_title(f"Wang: {pretty_metric(metric)} Density by {split_type}", pad=16)
	ax_den.set_xlabel(pretty_metric(metric), labelpad=12)
	ax_den.set_ylabel('Probability Density', labelpad=12)
	if vlines:
		for idx, (x_val, label, _color) in enumerate(vlines):
			ax_den.axvline(
				x=x_val,
				color=REF_LINE_BROWN,
				linestyle=REF_LINE_STYLES[idx % len(REF_LINE_STYLES)],
				linewidth=2,
				alpha=0.9,
				label=label,
			)
	ax_den.grid(axis='y', alpha=0.3)
	ax_den.legend(title=split_type, bbox_to_anchor=(1.02, 1), loc='upper left')
	fig_den.tight_layout()
	den_path = os.path.join(repo_path(out_root), f"{metric}_combined_density_by_{split_type}.png")
	fig_den.savefig(repo_path(den_path))
	print(f"Created plot: {den_path}")

	ax_cdf.set_title(f"Wang: {pretty_metric(metric)} CDF by {split_type}", pad=16)
	ax_cdf.set_xlabel(pretty_metric(metric), labelpad=12)
	ax_cdf.set_ylabel('Cumulative Probability', labelpad=12)
	ax_cdf.grid(axis='y', alpha=0.3)
	ax_cdf.legend(title=split_type, bbox_to_anchor=(1.02, 1), loc='upper left')
	fig_cdf.tight_layout()
	cdf_path = os.path.join(repo_path(out_root), f"{metric}_combined_cdf_by_{split_type}.png")
	fig_cdf.savefig(repo_path(cdf_path))
	print(f"Created plot: {cdf_path}")

	plt.close(fig_den)
	plt.close(fig_cdf)
	plt.close(fig_hist)


def plot_boxwhisker(split_dict: Dict[str, List[float]], metric: str, split_type: str, out_root: str) -> None:
	labels = [k for k, vals in split_dict.items() if len(clean_values(vals)) > 0]
	if not labels:
		return
	data = [clean_values(split_dict[k]) for k in labels]

	fig, ax = plt.subplots(figsize=(max(10, len(labels) * 0.65), 6))
	bp = ax.boxplot(data, patch_artist=True, widths=0.6)
	colors = plt.cm.tab20(np.linspace(0, 1, max(2, len(labels))))

	for i, patch in enumerate(bp['boxes']):
		patch.set_facecolor(colors[i])
		patch.set_alpha(0.7)
		patch.set_edgecolor('black')

	ax.set_xticks(np.arange(1, len(labels) + 1))
	ax.set_xticklabels([str(l).replace('-', '\n') for l in labels], rotation=45, ha='right')
	ax.set_xlabel(split_type, labelpad=12)
	ax.set_ylabel(pretty_metric(metric), labelpad=12)
	ax.set_title(f"Wang: {pretty_metric(metric)} by {split_type}", pad=16)
	ax.grid(axis='y', alpha=0.3)

	fig.tight_layout()
	box_path = os.path.join(repo_path(out_root), f"{metric}_boxwhisker_by_{split_type}.png")
	fig.savefig(repo_path(box_path), dpi=150)
	print(f"Created plot: {box_path}")
	plt.close(fig)


def plot_armstrong_zollo_violin(
	armstrong_vals: List[float],
	zollo_vals: List[float],
	metric: str,
	out_path: str,
) -> None:
	"""Plot Armstrong and Zollo overlaid histograms on one shared axes."""
	armstrong_vals = clean_values(armstrong_vals)
	zollo_vals = clean_values(zollo_vals)

	if not armstrong_vals and not zollo_vals:
		return

	fig, ax = plt.subplots(figsize=(9, 5.5))

	all_vals = armstrong_vals + zollo_vals
	x_min, x_max = min(all_vals), max(all_vals)
	pad = max((x_max - x_min) * 0.15, 1e-3)
	x_limit = max(abs(x_min), abs(x_max)) + pad
	bins = np.linspace(-x_limit, x_limit, 51)

	if armstrong_vals:
		ax.hist(
			armstrong_vals,
			bins=bins,
			color='#648FFF',
			alpha=0.55,
			edgecolor='black',
			label='Armstrong',
		)
	if zollo_vals:
		ax.hist(
			zollo_vals,
			bins=bins,
			color='#FFB000',
			alpha=0.45,
			edgecolor='black',
			label='Zollo',
		)

	ax.set_xlim(-x_limit, x_limit)
	ax.set_xlabel(pretty_metric(metric), labelpad=10)
	ax.set_ylabel('Frequency', labelpad=10)
	ax.set_title('Combinations with a Fixed Name Module', pad=12)
	ax.axvline(0.0, color='black', linestyle='--', linewidth=1.2, alpha=0.9)
	ax.grid(axis='y', alpha=0.3)
	ax.legend(loc='upper right')

	fig.tight_layout()
	az_path = out_path
	fig.savefig(repo_path(az_path), dpi=150)
	print(f"Created plot: {az_path}")
	plt.close(fig)


def main() -> None:
	cwd = str(REPO_ROOT)
	default_results_dir = os.path.join(repo_path(cwd), 'results')
	default_wang_results = os.path.join(repo_path(default_results_dir), 'wang_sampling_downsampling_analysis_original.csv')
	default_wang_plots_dir = os.path.join(repo_path(cwd), 'wang', 'plots')

	parser = argparse.ArgumentParser(description='Wang plotting script (parallel style to plot_yin_all.py).')
	parser.add_argument('--input_file', type=str, default=default_wang_results)
	parser.add_argument('--reference_input', help='Baseline aggregate CSV (defaults to input_file).')
	parser.add_argument('--output_dir', type=str, default='results/figs/wang')
	parser.add_argument('--split_by', type=str, default='all', choices=['all', 'model', 'job', 'name', 'occupation', 'job_bundle_occupation'])
	parser.add_argument('--metric', type=str, default='all', help='all or one of Wang metrics.')
	parser.add_argument('--bw_threshold', type=float, default=0.1, help='If provided, set bandwidth method for KDE (passed to gaussian_kde.set_bandwidth).')
	parser.add_argument('--armstrong_zollo', action='store_true', help='If set, create an overlaid histogram with fixed name_bundle armstrong and zollo.')
	parser.add_argument('--reproduced_settings', default=False, action='store_true', help='Add per-model reproduced score-difference settings from Wang analysis CSVs.')
	parser.add_argument('--reproduced_settings_dir', type=str, default=default_wang_plots_dir, help='Directory containing per-model statistical_results_*.csv files.')
	parser.add_argument('--noise_floor_mean', type=float, default=None, help='Mean of noise floor to plot (optional)')
	parser.add_argument('--noise_floor_std', type=float, default=None, help='Std dev of noise floor to plot (optional)')
	parser.add_argument('--full_width', action='store_true', help='Expand plot width so the x-axis uses nearly the entire page width.')
	
	args = parser.parse_args()

	ensure_dirs(args.output_dir)
	df = load_wang_data(args.input_file)

	if args.metric == 'all':
		metrics_to_plot = sorted(df['metric'].dropna().unique().tolist())
	else:
		metrics_to_plot = [args.metric]

	df = df[df['Random_State'] == 42.0] if 'Random_State' in df.columns else df
	df = df[df['name_bundle'] != 'seshadri'] if 'name_bundle' in df.columns else df

	random_state_column = next(
		(column for column in ('random_state', 'Random_State', 'Random State') if column in df.columns),
		None,
	)
	if random_state_column is not None:
		random_states = pd.to_numeric(df[random_state_column], errors='coerce')
		df = df[random_states == 42.0].copy()

	# Compute a reproducible sampled noise-floor early so downstream plotting
	# uses the same sampled mean/std when the user did not explicitly supply them.
	try:
		noise_floor_results = compute_default_framework_stats()
		nf_n, nf_mean, nf_std = noise_floor_results['wang']['n'], noise_floor_results['wang']['mean'], noise_floor_results['wang']['std']
		print(f"Noise floor (sampled n={nf_n}): mean={nf_mean}, std={nf_std}")
		args.noise_floor_mean = -1 * nf_mean
		args.noise_floor_std = nf_std
		
	except Exception as e:
		print(f"Warning: failed to compute sampled noise floor: {e}")
		args.noise_floor_mean = None
		args.noise_floor_std = None

	for metric in metrics_to_plot:
		vlines = None
		if metric in {'score_difference', 'score_diff', 'wang_score_diff'}:
			vlines = [
				(load_reproduced_reference('wang', metric, args.reference_input or args.input_file), REPRODUCED_SETTING_LABEL, REF_LINE_BROWN),
				(load_saved_wang_reference(args.reproduced_settings_dir), reported_setting_label('GPT-4o'), REF_LINE_BROWN),
			]
			if args.reproduced_settings:
				reported_lines = load_reproduced_setting_vlines(args.reproduced_settings_dir)
				vlines.extend(line for line in reported_lines if line[1] != reported_setting_label('GPT-4o'))

		metric_df = df[df['metric'] == metric].copy()
		if metric_df.empty:
			print(f"Skipping {metric}: no rows")
			continue

		if args.armstrong_zollo:
			az_dir = os.path.join(repo_path(args.output_dir), 'armstrong_zollo')
			ensure_dirs(az_dir)
			armstrong_vals = metric_df[metric_df['name_bundle'] == 'armstrong']['value'].tolist()
			zollo_vals = metric_df[metric_df['name_bundle'] == 'zollo']['value'].tolist()
			plot_armstrong_zollo_violin(
				armstrong_vals,
				zollo_vals,
				metric,
				os.path.join(repo_path(az_dir), f"{metric}_histogram_armstrong_zollo.png"),
			)
			continue

		if args.split_by == 'all':
			all_dir = os.path.join(repo_path(args.output_dir), 'all')
			ensure_dirs(all_dir)
			vals = clean_values(metric_df['value'].tolist())
			print(f"Found {len(vals)} values")
			if vals:
				plot_global_density(vals, metric, os.path.join(repo_path(all_dir), f"{metric}_global_density_all.png"), vlines=vlines, bw_threshold=args.bw_threshold, full_width=args.full_width)
				# Also save a version that highlights the noise floor
				nf_arg = (args.noise_floor_mean, args.noise_floor_std)
				plot_global_density(vals, metric, os.path.join(repo_path(all_dir), f"{metric}_global_density_all_with_noise_floor.png"), vlines=vlines, bw_threshold=args.bw_threshold, noise_floor=nf_arg, full_width=args.full_width)
			continue

		if args.split_by == 'model':
			key_col = 'model'
		elif args.split_by == 'job':
			key_col = 'job_bundle'
		elif args.split_by == 'name':
			key_col = 'name_bundle'
		elif args.split_by == 'occupation':
			key_col = 'occupation'
		elif args.split_by == 'job_bundle_occupation':
			# make one set of plots per job bundle, split by occupation
			for jb in metric_df['job_bundle'].dropna().unique():
				jb_df = metric_df[metric_df['job_bundle'] == jb]
				split_dict = {str(k): g['value'].tolist() for k, g in jb_df.groupby('occupation')}
				bundle_out = os.path.join(repo_path(args.output_dir), 'job_bundle_occupation', f"job_bundle_{safe_dirname(jb)}")
				ensure_dirs(bundle_out)
				plot_split_density_cdf(split_dict, metric, 'occupation', bundle_out, vlines=vlines, bw_threshold=args.bw_threshold)
				plot_boxwhisker(split_dict, metric, 'occupation', bundle_out)
			continue
		else:
			continue

		# Requested behavior: create split folder and one subfolder per unique key with per-group density plot.
		split_root = os.path.join(repo_path(args.output_dir), args.split_by)
		ensure_dirs(split_root)

		for key in metric_df[key_col].dropna().unique():
			group_df = metric_df[metric_df[key_col] == key]
			vals = clean_values(group_df['value'].tolist())
			if not vals:
				continue

			group_dir = os.path.join(repo_path(split_root), safe_dirname(key))
			ensure_dirs(group_dir)
			plot_global_density(
				vals,
				metric,
				os.path.join(repo_path(group_dir), f"{metric}_density_{args.split_by}_{safe_dirname(key)}.png"),
				vlines=vlines,
				split_by=f"{args.split_by}: {key}",
				bw_threshold=args.bw_threshold,
				full_width=args.full_width,
			)


if __name__ == '__main__':
	main()



'''
Plot global histograms: python plot_wang.py


python plot_wang.py --split_by all --bw_threshold 0.1 --metric score_difference;  

python plot_wang.py --split_by model  --metric score_difference; 
python plot_wang.py --split_by name  --metric score_difference; 
python plot_wang.py --split_by job  --metric score_difference; 

python plot_wang.py --split_by job_bundle_occupation --input_file results/wang_sampling_downsampling_analysis_original_detailed.csv ;



'''

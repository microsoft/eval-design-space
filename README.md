# Flexibility and Fragility in Generative AI Evaluation: Lessons from Audit Studies of Resume Screening

[![arXiv](https://img.shields.io/badge/arXiv-ARXIV_NUMBER)](https://arxiv.org/abs/ARXIV_NUMBER)

We propose a natural instantiation of Integrative Experimental Design for AI
evaluation, using prior studies as empirically grounded design choices for
large-scale combinatorial replication. We share the data and code to apply the
method to the case study of gender bias in hiring. We run **1,134 counterfactual
experiments** by combinatorially varying social group indicators (including
names), job-resume bundles, and models across three baseline task designs:
Armstrong, Wang, and Yin.

**Preprint:** [Flexibility and Fragility in Generative AI Evaluation: Lessons from Audit Studies of Resume Screening](https://arxiv.org/abs/ARXIV_NUMBER).
[Nicole Meister](https://nicolemeister.github.io/),
[Hannah Cha](https://www.hannahcha.com/),
[Mohammed Alsobay](https://malsobay.com/),
[Alexandra Chouldechova](https://scholar.google.com/citations?user=uoDW9hkAAAAJ&hl=en),
[P. Alex Dow](https://www.microsoft.com/en-us/research/people/dowalex/),
[Carlos Guestrin](https://guestrin.su.domains/),
[Jennifer Wortman Vaughan](https://www.jennwv.com/),
[Hanna Wallach](https://www.microsoft.com/en-us/research/people/wallach/),
[Tatsunori Hashimoto](https://thashim.github.io/), and
[Solon Barocas](http://solon.barocas.org/).

The arXiv links and citation currently use the placeholder `ARXIV_NUMBER`.

**Questions?** Please open an issue or contact us at nmeist@stanford.edu.

## Research corpora

The paper analyzes **40 studies** evaluating gender and/or racial bias in
generative AI resume screening: 39 examine gender bias and 28 examine racial
bias, with overlap between these groups.

- **Resume screening corpus (n = 40):** We manually annotated the studies'
  high-level conclusions to create Table 1.
  [Annotated findings](https://docs.google.com/spreadsheets/d/1fJWdlJuSVIeKcJXJAmBFFp__XzqSjvaf91KCeYqUusY/edit?gid=270490799#gid=270490799)
- **Reproducible subset (n = 9):** Papers with publicly available data sufficient
  to reconstruct their experiments. We characterized their designs using 46
  experimental dimensions grouped into 10 evaluation modules.
  [Annotations](https://docs.google.com/spreadsheets/d/1fJWdlJuSVIeKcJXJAmBFFp__XzqSjvaf91KCeYqUusY/edit?gid=0#gid=0)

The broader
[hiring-funnel annotations](https://docs.google.com/spreadsheets/d/1fJWdlJuSVIeKcJXJAmBFFp__XzqSjvaf91KCeYqUusY/edit?gid=248197715#gid=248197715)
are also available as a related resource.

## Combinatorial experiments

We combine values drawn from prior studies across four independently swappable
sets: **task design**, **social group indicator**, **job-resume**, and **model**.
Task design groups the interdependent task, prompt, runs, and outcome modules;
jobs and resumes are bundled to preserve plausible pairings. We use these
combinations to investigate variation and learnable structure in evaluation
outcomes.
[Module annotations](https://docs.google.com/spreadsheets/d/1hSQHJTcDGPQHMrS4Pxghk-icXkbje4Kdqywjizn5fjc/edit?gid=2056286674#gid=2056286674)

The canonical grid contains **378 configurations per task design**:
9 social group indicator bundles x 7 job-resume bundles x 6 models.
Each configuration produces the bias measure defined by its task design.
The nine studies supply social group indicators; seven supply job-resume
bundles. Gaebler and Lippens are excluded from the job-resume set because their
materials involve real people's resumes or work experience.

## Repository layout

| Location | Contents |
| --- | --- |
| `code\configs` | Task configurations and reported reference values |
| `code\modules` | Social group indicator and job-resume source data |
| `code\results` | Aggregate experiment data, separate noise-floor data, and analysis outputs |
| `code\utils` | Data processing, plotting, regression, and permutation-importance utilities |
| `code\wang` | Wang-specific generation and evaluation utilities |
| `code\together` | Together batch-processing utilities |
| `code\scripts` | Shell helpers for running experiments and analyses |

## Setup

Clone this repository, then create a virtual environment and install the
dependencies. The pins in `code\requirements.txt` target the Python 3.13
environment used for analysis; this repository does not require an editable
package installation.

```powershell
git clone https://github.com/microsoft/eval-design-space.git
Set-Location eval-design-space
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r code\requirements.txt
python -m pip check
```

On macOS/Linux, activate with `source .venv/bin/activate` and
replace backslashes in the example paths below with forward slashes.

**Before running any analysis or generation command below, change from the
repository root into `code`:**

```powershell
cd code
```

All commands below run from `eval-design-space\code`, not the repository root.
Their input and output paths are relative to that directory.

## Experiment data

The main inputs are:

| Task design | Aggregate CSV in `code\results` | Outcome |
| --- | --- | --- |
| Armstrong | `armstrong_sampling_downsampling_analysis_original.csv` | Regression coefficient for women |
| Wang | `wang_sampling_downsampling_analysis_original.csv` | Score difference |
| Yin | `yin_sampling_downsampling_analysis_original.csv` | Selection rate for women |

The nine social group indicator bundles are `armstrong`, `rozado`, `wen`, `wang`,
`karvonen`, `zollo`, `yin`, `gaeb`, and `lippens`. The seven job-resume bundles are
`armstrong`, `rozado`, `wen`, `wang`, `karvonen`, `zollo`, and `yin`.
The six models are GPT-4o-mini, GPT-5-nano, Llama 3.1 8B, Llama 3.3 70B,
Mistral Small 24B, and Mistral 7B. Scripts handle the stored model aliases.

The analysis grid uses women's outcomes; the Yin aggregate also retains
complementary male rows. Use the analysis loaders rather than treating every
aggregate row as a distinct grid configuration. Wang analyses reverse the stored
score-difference sign to use the protected-minus-privileged convention.

Noise-floor runs are stored separately as `armstrong_noise_floor.csv`,
`wang_noise_floor.csv`, and `yin_noise_floor.csv`. These files include the
reproduced baseline as well as reruns; that baseline also remains in the main
configuration grid. The noise-floor utility samples up to 25 candidate outcomes
with a fixed random seed.

## Reproduce plots and learn structure in design choices

Commands can overwrite existing outputs. Use a separate output directory or file
where the script supports `--output_dir` or `--output`.

### Task-design distributions

```powershell
python plot_armstrong.py --metric regression_coefficients --split_by all
python plot_yin_all.py --metric selection_rate --split_by all --reported_setting_models gpt-3.5-turbo gpt-4
python plot_wang.py --metric score_difference --split_by all --reproduced_settings --full_width
```

These write figures under `results\figs`. Legends distinguish the
**Reproduced Setting (GPT-4o-mini)** from model-specific **Reported Setting**
references. Wang's `--reproduced_settings` flag retains its legacy name but adds
the reported-model reference lines; `--full_width` displays the legend.
Its reference CSVs are under `wang\plots`.

### Cross-design summaries and noise floor

```powershell
python utils\plot_relative.py --no_wang_models
python utils\plot_male_preference_bar.py
python utils\swapped_module_results.py
python utils\noise_floor.py
```

`--no_wang_models` keeps the relative plot to the three task-design distributions
without additional per-model Wang rows; it does not exclude Wang social group
indicator bundles. The noise-floor command reads the three separate noise-floor
CSVs and writes `results\noise_floor_results.csv`.

### OLS regression

```powershell
python utils\meta_regression_analysis_intercept.py --exp_framework yin --metric selection_rate --model_spec main
python utils\meta_regression_analysis_intercept.py --exp_framework yin --metric selection_rate --model_spec interactions
python utils\meta_regression_analysis_intercept.py --exp_framework armstrong --metric regression_coefficients --model_spec main
python utils\meta_regression_analysis_intercept.py --exp_framework wang --metric score_difference --model_spec main
```

Regression outputs are saved under `results\meta\<framework>`. Coefficient plots
fit the full dataset, omit reference-category markers, and show 95% confidence
intervals. Observed-versus-predicted plots use held-out predictions from five
cross-validation folds; the dashed line is the identity line, not a fitted trend.
Use `--ablation all` for the supported ablation comparisons.

### Elastic Net and CatBoost

```powershell
python utils\meta_regression_predictors.py --exp_framework yin
python utils\meta_regression_predictors.py --exp_framework armstrong
python utils\meta_regression_predictors.py --exp_framework wang
```

The default comparison fits Elastic Net with pairwise interactions and CatBoost
with categorical inputs. Elastic Net uses standardized indicator features with
one reference category omitted per factor. Hyperparameters are selected by
three-fold inner cross-validation, minimizing RMSE, within each of five outer
training folds. Summaries are written to
`results\meta\<framework>\predictor_comparison.csv`.

Held-out predictive R-squared uses the **training-set mean** as its comparison
predictor by default. Scores are computed separately in each outer fold and
summarized by their mean and standard deviation. Pass `--r2-mode standard` to use
the conventional held-out-set mean instead. Full-data-fit R-squared is separate
from held-out predictive performance.

### Permutation importance

```powershell
python utils\feature_importance.py --exp_framework yin --metric selection_rate --model_spec main --ablation none --feature_importance
python utils\feature_importance.py --exp_framework yin --metric selection_rate --model_spec interactions --ablation none --feature_importance
```

The `--feature_importance` flag is required to generate permutation-importance
outputs. Each model is trained on four folds and evaluated on the held-out fold.
The utility shuffles each raw design dimension 30 times per fold and rebuilds
its indicator and interaction features without refitting. Reported importances
are the mean and standard deviation of the drop in held-out R-squared across
five folds x 30 shuffles. A synthetic random predictor provides a comparison
feature, so the reference R-squared can differ slightly from ordinary OLS.

The standalone script writes
`results\meta\<framework>\permutation_importance_ablation_none.csv` and
`permutation_importance_plot_ablation_none.png`. Its main and interaction runs
use the same filenames, so preserve the first outputs before running the second.
Alternatively, the OLS regression script's `--feature_importance` option produces
filenames containing the model specification.

## Generate new evaluations

Configure model access before running generation. The OpenAI backend reads
`OPENAI_API_KEY`; Together batch utilities read `TOGETHER_API_KEY`. Export
credentials in the environment rather than committing them. Local
OpenAI-compatible backends require an accessible endpoint and appropriate
configuration in `utils\model.py`; its existing machine-specific defaults are
not portable.

For example, after configuring OpenAI access:

```powershell
$env:OPENAI_API_KEY = "<your-api-key>"
python main.py --base_config_path configs\base_yin.yaml --name yin --job yin --model gpt-4o-mini
```

For Together batch generation, set `$env:TOGETHER_API_KEY` instead and use the
utilities in `together`. The OpenAI variable is `OPENAI_API_KEY`, not `OPENAI_KEY`.

Task configuration files control prompts, trial counts, and output locations.
Raw generation outputs are separate from the supplied aggregate CSVs; use the
analysis utilities to process new outputs. Consult each script's `--help` before
launching large or billable runs.

## Citation

If you use this code or data, please cite:

[Flexibility and Fragility in Generative AI Evaluation: Lessons from Audit Studies of Resume Screening](https://arxiv.org/abs/ARXIV_NUMBER).
Nicole Meister, Hannah Cha, Mohammed Alsobay, Alexandra Chouldechova, P. Alex Dow,
Carlos Guestrin, Jennifer Wortman Vaughan, Hanna Wallach, Tatsunori Hashimoto,
and Solon Barocas. 2026.

```bibtex
@misc{meister_2026,
  title={Flexibility and Fragility in Generative AI Evaluation: Lessons from Audit Studies of Resume Screening},
  author={Nicole Meister and Hannah Cha and Mohammed Alsobay and Alexandra Chouldechova and P. Alex Dow and Carlos Guestrin and Jennifer Wortman Vaughan and Hanna Wallach and Tatsunori Hashimoto and Solon Barocas},
  year={2026},
  eprint={ARXIV_NUMBER},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  url={https://arxiv.org/abs/ARXIV_NUMBER}
}
```

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit [Contributor License Agreements](https://cla.opensource.microsoft.com).

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Licenses

Code is licensed under the [Microsoft Research License Terms](https://github.com/microsoft/eval-design-space/blob/main/MICROSOFT_RESEARCH_LICENSE); all other materials are released under the [Community Data License Agreement – Permissive – Version 2.0](https://cdla.dev/permissive-2-0/).

## Privacy & Cookies

[Microsoft Privacy Statement](https://go.microsoft.com/fwlink/?LinkId=521839)

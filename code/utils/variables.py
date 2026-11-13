MODELS = {
    # "Mistral-7B-Instruct-v0.3": "mistralai/Mistral-7B-Instruct-v0.3",
    # "mistral-7b-v0.3": "mistralai/Mistral-7B-Instruct-v0.3",
    # # "mistral-7b-v0.2": "mistralai/Mistral-7B-Instruct-v0.2",
    # "meta-llama-3.1-8b-instruct-turbo": "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
    # # "mistral-small-24b": "mistralai/Mistral-Small-24B-Instruct-2501",
    # "meta-llama-3.3-70b-instruct-turbo": "meta-llama/Llama-3.3-70B-Instruct-Turbo",

    "mistral-7b-v0.3": 'tatsu/evaldims',
    "meta-llama-3.1-8b-instruct-turbo": 'tatsu/evaldims-llama',
    "meta-llama-3.3-70b-instruct-turbo": 'tatsu/evaldims-llama70b',
    # "mistral-small-24b": 'tatsu/evaldims-mistral24b',
}

MODELS_GPT = {
    "gpt-4o-mini": "gpt-4o-mini",
    # "gpt-4o-2024-11-20": "gpt-4o-2024-11-20",
    "gpt-5-nano-2025-08-07": "gpt-5-nano-2025-08-07",
}

# NAMES = ['armstrong']
# JOBS = ['armstrong']
model_to_plotname= {
	"meta-llama-3.1-8b-instruct-turbo": "Llama-3.1-8B-Instruct-Turbo",
	"mistral-small-24b": "Mistral-Small-24B",
	"meta-llama-3.3-70b-instruct-turbo": "Llama-3.3-70B-Instruct-Turbo",
	"gpt-5-nano-2025-08-07": "GPT-5 Nano",
	"gpt-4o-mini": "GPT-4o Mini",
    "gpt-4o-2024-11-20": "GPT-4o",
    "mistral-7b-v0.2": "Mistral-7B-v0.2",
	"mistral-7b-v0.3": "Mistral-7B-v0.3",
    "mistral-7b-v0.3": "Mistral-7B-Instruct-v0.3",
	
}


# NAMES = ['armstrong', 'rozado', 'wen', 'seshadri', 'karvonen', 'zollo', 'yin', 'gaeb', 'lippens']
# NAMES = ['armstrong', 'rozado', 'wen', 'wang', 'karvonen', 'zollo', 'yin', 'gaeb', 'lippens']
NAMES = ['lippens']
JOBS = ['armstrong', 'rozado', 'wen', 'wang', 'karvonen', 'zollo', 'yin']

armtrong_model_order = {
'mistral-7b-v0.2': '05-22-2024',
# 'mistral-7b-v0.3': '05-22-2024',  # Not supported in Together AI Batch API
'meta-llama-3.1-8b-instruct-turbo': '07-23-2024',
'gpt-4o-2024-11-20': '11-20-2024',
'meta-llama-3.3-70b-instruct-turbo': '12-06-2024',
'mistral-small-24b': '06-20-2025',
'gpt-5-nano-2025-08-07': '08-07-2025'}


yin_model_order = {'gpt-4o-mini': '07-18-2024',
'mistral-7b-v0.2': '05-22-2024',
# 'mistral-7b-v0.3': '05-22-2024',  # Not supported in Together AI Batch API
'meta-llama-3.1-8b-instruct-turbo': '07-23-2024',
'meta-llama-3.3-70b-instruct-turbo': '12-06-2024',
'mistral-small-24b': '06-20-2025',
'gpt-5-nano-2025-08-07': '08-07-2025'}

wang_model_order = {'gpt-4o-mini': '07-18-2024',
'mistral-7b-v0.2': '05-22-2024',
'meta-llama-3.1-8b-instruct-turbo': '07-23-2024',
'meta-llama-3.3-70b-instruct-turbo': '12-06-2024',
'mistral-small-24b': '06-20-2025',
'gpt-5-nano-2025-08-07': '08-07-2025'}


colors = ["purple", "darkblue", "green", "orange", "brown", "pink", "gray", "olive", "cyan", "magenta"]
linestyles = ['--', '-.', ':']
metric_to_plot = {'selection_rate': 'Selection Rate', 'score_difference': 'Score Difference', 'disparate_impact_ratio': 'Disparate Impact Ratio', 'regression_coefficients': 'Regression Coefficient'}
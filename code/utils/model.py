import os
import openai
import pandas as pd
import time
import datetime
import re
from openai import OpenAI

OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
VLLM_BASE_URL = 'http://172.24.67.98:8000/v1'
VLLM_BASE_URL_24B = 'http://172.24.67.98:8001/v1' 
VLLM_API_KEY = os.environ.get('VLLM_API_KEY', 'EMPTY')

MODEL_NAME_ALIASES = {
    'mistral-7b-v0.2': 'mistralai/Mistral-7B-Instruct-v0.2',
    'mistral-7b-instruct-v0.2': 'mistralai/Mistral-7B-Instruct-v0.2',
    'mistral-7b-instruct-v0.3': 'mistralai/Mistral-7B-Instruct-v0.3',
    'meta-llama-3.1-8b-instruct': 'meta-llama/Llama-3.1-8B-Instruct',
    'mistral-small-24b-instruct': 'mistralai/Mistral-Small-24B-Instruct-2501',
}

VLLM_MODEL_ALIASES = set(MODEL_NAME_ALIASES.keys())

class Model:
    def __init__(self, model_name, temperature=0.7, max_tokens=1024):
        self.original_model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.backend = self._infer_backend(model_name)
        self.model_name = self._resolve_model_name(model_name)
        self.client = self._build_client()

    def get_model_name(self, model_name):
        return self.model_name

    def _infer_backend(self, model_name):
        name = (model_name or '').strip().lower()
        if name.startswith('vllm:'):
            return 'vllm'
        if name in VLLM_MODEL_ALIASES:
            return 'vllm'
        return 'openai'

    def _resolve_model_name(self, model_name):
        name = (model_name or '').strip()
        lowered_name = name.lower()
        if lowered_name.startswith('vllm:'):
            name = name.split(':', 1)[1].strip()
            lowered_name = name.lower()
        return MODEL_NAME_ALIASES.get(lowered_name, name)

    def _get_vllm_base_url(self):
        raw_name = (self.original_model_name or '').strip().lower()
        if raw_name.startswith('vllm:'):
            raw_name = raw_name.split(':', 1)[1].strip().lower()

        if raw_name == 'mistral-small-24b-instruct':
            return VLLM_BASE_URL_24B

        return VLLM_BASE_URL

    def _build_client(self):
        if self.backend == 'vllm':
            if not VLLM_BASE_URL:
                raise ValueError(
                    'VLLM_BASE_URL must be set to use the vLLM backend '
                    f'for model {self.original_model_name}.'
                )

            base_url = self._get_vllm_base_url().rstrip('/')

            try:
                return OpenAI(api_key=VLLM_API_KEY, base_url=base_url)
            except Exception:
                alt_url = re.sub(r':8000$', ':8001', base_url)
                return OpenAI(api_key=VLLM_API_KEY, base_url=alt_url)

        return OpenAI(api_key=OPENAI_API_KEY)

    def _uses_max_completion_tokens(self):
        """Newer models (e.g., GPT-5 family) reject `max_tokens`."""
        name = (self.model_name or "").lower()
        return name.startswith("gpt-5")

    def _create_chat_completion(self, messages, req_temperature, req_max_tokens):
        base_kwargs = {
            "model": self.model_name,
            "messages": messages,
            "logprobs": False,
            "top_logprobs": None,
            "timeout": 60,
        }

        if req_temperature is not None:
            base_kwargs["temperature"] = req_temperature

        try_kwargs = dict(base_kwargs)
        if req_max_tokens is not None:
            if self.backend != 'vllm' and self._uses_max_completion_tokens():
                try_kwargs["extra_body"] = {"max_completion_tokens": req_max_tokens}
            else:
                try_kwargs["max_tokens"] = req_max_tokens

        return self.client.chat.completions.create(**try_kwargs)

    def model_dump(self, user_prompt, system_prompt="", temperature=None, max_tokens=None):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        req_temperature = self.temperature if temperature is None else temperature
        req_max_tokens = self.max_tokens if max_tokens is None else max_tokens
        try:
            response = self._create_chat_completion(messages, req_temperature, req_max_tokens)
        except Exception:
            try:
                response = self._create_chat_completion(messages, req_temperature, req_max_tokens)
            except Exception:
                return None

        return response.model_dump()

    def run_query(self, user_prompt, system_prompt="", temperature=None, max_tokens=None):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        req_temperature = self.temperature if temperature is None else temperature
        req_max_tokens = self.max_tokens if max_tokens is None else max_tokens
        try:
            response = self._create_chat_completion(messages, req_temperature, req_max_tokens)
            return response.choices[0].message.content
        except Exception as e:
            print(f"API attempt failed: {type(e).__name__}: {e}")
            try:
                response = self._create_chat_completion(messages, req_temperature, req_max_tokens)
                return response.choices[0].message.content
            except Exception as e2:
                print(f"Second API attempt also failed: {type(e2).__name__}: {e2}")
                return None
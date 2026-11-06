"""Client LLM qua API chuẩn OpenAI: vLLM, SGLang, llama.cpp server đều dùng được, đổi bằng `base_url` và `model`."""
from __future__ import annotations

import os
import time


class LLM:
    def __init__(self, base_url="http://localhost:8000/v1", model="Qwen/Qwen3-4B-Instruct-2507", temperature=0.0,
                 max_tokens=512, seed=0, timeout=60, api_key_env="LLM_API_KEY", extra_body=None, **_):
        from openai import OpenAI

        self.client = OpenAI(base_url=base_url, api_key=os.environ.get(api_key_env, "EMPTY"), timeout=timeout)
        self.model = model
        self.kw = dict(temperature=temperature, max_tokens=max_tokens, seed=seed)
        self.extra_body = extra_body or {}

    def chat(self, messages, **kw):
        t0 = time.perf_counter()
        resp = self.client.chat.completions.create(model=self.model, messages=messages,
                                                   extra_body=self.extra_body or None, **{**self.kw, **kw})
        usage = resp.usage
        meta = {"sec": time.perf_counter() - t0,
                "prompt_tokens": getattr(usage, "prompt_tokens", 0),
                "completion_tokens": getattr(usage, "completion_tokens", 0)}
        return resp.choices[0].message, meta

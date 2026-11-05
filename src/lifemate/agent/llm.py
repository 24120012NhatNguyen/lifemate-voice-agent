"""Client LLM qua API chuẩn OpenAI: vLLM, SGLang, llama.cpp server đều dùng được, đổi bằng `base_url` và `model`."""
from __future__ import annotations

import os
import time


class LLMTimeout(Exception):
    """Một lượt LLM chạy quá `timeout` riêng của nó."""


class LLM:
    def __init__(self, base_url="http://localhost:8000/v1", model="Qwen/Qwen3-4B-Instruct-2507", temperature=0.0,
                 max_tokens=512, seed=0, timeout=60, api_key_env="LLM_API_KEY", extra_body=None, **_):
        from openai import OpenAI

        self.client = OpenAI(base_url=base_url, api_key=os.environ.get(api_key_env, "EMPTY"), timeout=timeout)
        self.model = model
        self.kw = dict(temperature=temperature, max_tokens=max_tokens, seed=seed)
        self.extra_body = extra_body or {}

    def chat(self, messages, timeout=None, **kw):
        """timeout: giây cho riêng lượt này, không thử lại; quá giờ thì ném LLMTimeout."""
        from openai import APITimeoutError

        client = self.client.with_options(timeout=timeout, max_retries=0) if timeout else self.client
        t0 = time.perf_counter()
        try:
            resp = client.chat.completions.create(model=self.model, messages=messages,
                                                  extra_body=self.extra_body or None, **{**self.kw, **kw})
        except APITimeoutError as e:
            raise LLMTimeout(str(e)) from e
        usage = resp.usage
        meta = {"sec": time.perf_counter() - t0,
                "prompt_tokens": getattr(usage, "prompt_tokens", 0),
                "completion_tokens": getattr(usage, "completion_tokens", 0)}
        return resp.choices[0].message, meta

"""Model calls through LiteLLM, so any provider works: ollama/..., groq/..., gemini/...,
openrouter/..., nvidia_nim/..., openai/..., anthropic/... Keys come from the usual env vars."""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor

import litellm

litellm.suppress_debug_info = True
litellm.drop_params = True


def chat(model: str, prompt: str, max_tokens=1024, temperature=0.0, retries=4, system=None) -> str:
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
    err = None
    for attempt in range(retries):
        try:
            r = litellm.completion(model=model, messages=msgs, max_tokens=max_tokens, temperature=temperature, timeout=180)
            return r.choices[0].message.content or ""
        except Exception as e:  # rate limits and timeouts on free tiers: back off and retry
            err = e
            time.sleep(min(60, 3 * 2**attempt))
    return f"[error: {type(err).__name__}: {str(err)[:200]}]"


def chat_many(model: str, prompts, workers=4, **kw):
    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(lambda p: chat(model, p, **kw), prompts))


def embed(texts, model="ollama/nomic-embed-text"):
    r = litellm.embedding(model=model, input=list(texts))
    return [d["embedding"] for d in r.data]

from __future__ import annotations

import os
import re
import threading
from typing import Any, Dict, List, Tuple

ENGINE_LOCK = threading.Lock()
ENGINE_CACHE: Dict[str, Any] = {}

DEFAULT_SWIFT_MODEL = "meta-llama/Llama-3.3-70B-Instruct"
DEFAULT_SWIFT_CACHE_DIR = "/data/gpfs/projects/punim2168/hf_dir"
DEFAULT_SWIFT_SNAPSHOT = (
    "/data/gpfs/projects/punim2168/hf_dir/"
    "models--meta-llama--Llama-3.3-70B-Instruct/"
    "snapshots/6f6073b423013f6a7d4d9f39144961bfbfbc386b"
)


def _normalize_model(model: str) -> str:
    raw = str(model or "").strip()
    key = re.sub(r"[\s_]+", "-", raw.lower())
    aliases = {
        "gpt5.4": "gpt-5.4",
        "gpt-5-4": "gpt-5.4",
        "gpt-5.4": "gpt-5.4",
        "gpt5-mini": "gpt-5-mini",
        "gpt-5mini": "gpt-5-mini",
        "gpt-5-mini": "gpt-5-mini",
        "gemini-flash-3.1": "gemini-3-flash-preview",
        "gemini-3.1-flash": "gemini-3-flash-preview",
        "gemini-3-flash": "gemini-3-flash-preview",
        "gemini-flash-3": "gemini-3-flash-preview",
        "llama3.3": DEFAULT_SWIFT_MODEL,
        "llama3.3-70b": DEFAULT_SWIFT_MODEL,
        "llama3.3-70b-instruct": DEFAULT_SWIFT_MODEL,
        "llama-3.3": DEFAULT_SWIFT_MODEL,
        "llama-3.3-70b": DEFAULT_SWIFT_MODEL,
        "llama-3.3-70b-instruct": DEFAULT_SWIFT_MODEL,
        "meta-llama/llama-3.3-70b-instruct": DEFAULT_SWIFT_MODEL,
        "meta-llama/llama-3.3-70b": DEFAULT_SWIFT_MODEL,
    }
    return aliases.get(key, raw)


def _is_gemini_model(model: str) -> bool:
    return _normalize_model(model).lower().startswith("gemini-")


def _is_gpt5_model(model: str) -> bool:
    return _normalize_model(model).lower().startswith("gpt-5")


def _is_llama33_model(model: str) -> bool:
    resolved = _normalize_model(model)
    lowered = resolved.lower()
    compact = re.sub(r"[^a-z0-9]+", "", lowered)
    return resolved == DEFAULT_SWIFT_MODEL or compact.startswith("metallamallama33") or (
        "llama" in lowered and "3.3" in lowered and "70b" in lowered
    )


def _get_hf_token() -> str:
    return (
        os.getenv("HUGGINGFACE_TOKEN")
        or os.getenv("HF_TOKEN")
        or os.getenv("HUGGING_FACE_HUB_TOKEN")
        or ""
    ).strip()


def _get_swift_cache_dir() -> str:
    return os.getenv("SWIFT_HF_CACHE_DIR") or os.getenv("HF_HOME") or DEFAULT_SWIFT_CACHE_DIR


def _get_swift_max_batch_size() -> int:
    raw = (os.getenv("SWIFT_MAX_BATCH_SIZE") or os.getenv("LLAMA33_MAX_BATCH_SIZE") or "1").strip()
    try:
        value = int(raw)
    except ValueError:
        value = 1
    return max(1, value)


def _resolve_local_swift_model(model: str) -> str:
    resolved = _normalize_model(model)
    if _is_llama33_model(resolved):
        local_snapshot = os.getenv("SWIFT_LOCAL_LLAMA33_PATH") or DEFAULT_SWIFT_SNAPSHOT
        if os.path.isdir(local_snapshot):
            return local_snapshot
    return resolved


def _build_swift_engine(model: str) -> Any:
    from huggingface_hub import login
    from swift.llm import PtEngine

    resolved_model = _resolve_local_swift_model(model)
    hf_token = _get_hf_token()
    if hf_token:
        try:
            login(token=hf_token)
        except Exception:
            pass

    kwargs: Dict[str, Any] = {
        "max_batch_size": _get_swift_max_batch_size(),
        "use_hf": True,
        "cache_dir": _get_swift_cache_dir(),
    }
    kwargs = {k: v for k, v in kwargs.items() if v}
    if os.path.isdir(resolved_model):
        kwargs.pop("use_hf", None)
        kwargs.pop("cache_dir", None)
        kwargs.pop("hub_token", None)
        return PtEngine(resolved_model, model_type="llama3_1", **kwargs)

    try:
        return PtEngine(resolved_model, model_type="llama3_1", **kwargs)
    except TypeError:
        kwargs.pop("cache_dir", None)
        try:
            return PtEngine(resolved_model, **kwargs)
        except TypeError:
            return PtEngine(resolved_model, max_batch_size=_get_swift_max_batch_size(), use_hf=True)


def _get_swift_engine(model: str) -> Any:
    resolved_model = _normalize_model(model)
    with ENGINE_LOCK:
        engine = ENGINE_CACHE.get(resolved_model)
        if engine is None:
            engine = _build_swift_engine(resolved_model)
            ENGINE_CACHE[resolved_model] = engine
        return engine


def _clear_swift_engine(model: str | None = None) -> None:
    with ENGINE_LOCK:
        if model is None:
            ENGINE_CACHE.clear()
            return
        ENGINE_CACHE.pop(_normalize_model(model), None)


def _clear_cuda_memory() -> None:
    try:
        import gc

        gc.collect()
    except Exception:
        pass
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass


def _content_to_text(content: Any) -> str:
    if isinstance(content, list):
        return "".join(str(part.get("text", "")) if isinstance(part, dict) else str(part) for part in content)
    return str(content or "")


def _request_swift_texts(prompt: str, model: str, max_tokens: int, temperature: float, n: int = 1) -> List[str]:
    from swift.llm import InferRequest, RequestConfig

    engine = _get_swift_engine(model)
    request_count = max(1, int(n))
    infer_list = [InferRequest(messages=[{"role": "user", "content": prompt}]) for _ in range(request_count)]
    request_config = RequestConfig(max_tokens=int(max_tokens), temperature=float(temperature))
    resp_list = engine.infer(infer_list, request_config)
    outputs: List[str] = []
    for resp in resp_list:
        try:
            choice = resp.choices[0]
            content = choice.message.content
        except Exception:
            content = ""
        outputs.append(_content_to_text(content).strip())
    return outputs


def _request_swift_chat(prompt: str, model: str, max_tokens: int, temperature: float) -> str:
    texts = _request_swift_texts(prompt=prompt, model=model, max_tokens=max_tokens, temperature=temperature, n=1)
    return texts[0] if texts else ""


def _request_swift_candidates(
    prompt: str, model: str, max_tokens: int, temperature: float, sampling_n: int
) -> Tuple[List[Tuple[str, float, str]], str]:
    texts = _request_swift_texts(prompt=prompt, model=model, max_tokens=max_tokens, temperature=temperature, n=sampling_n)
    candidates: List[Tuple[str, float, str]] = []
    for rank, text in enumerate(texts):
        if text:
            candidates.append((text, -float(rank), "choice_order_no_logprobs"))
    return candidates, "choice_order_no_logprobs"

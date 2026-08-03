#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from table_localization.common.dataset_llm_utils import (  # noqa: E402
    DEFAULT_BASE_URL,
    DEFAULT_GEMINI_BASE_URL,
    _df_to_markdown,
    _ensure_dir,
    _extract_json_object,
    _get_api_key,
    _get_gemini_api_key,
    _is_gemini_model,
    _log,
    _load_dataset_split,
    _normalize_model,
    _normalize_dataset_item,
    _request_llm,
    _resolve_dataset_name,
    _wtq_item_to_df,
)
from table_localization.common.swift_infer_backend import (  # noqa: E402
    _is_llama33_model as _backend_is_llama33_model,
    _request_swift_candidates,
    _request_swift_chat,
)


METHOD_NAME = "opaware_table_localization"
DEFAULT_MODEL = "gpt5-mini"
DEFAULT_PROMPT_DIR = ROOT.parents[1] / "prompts"
OPENAI_MAX_CANDIDATES_PER_REQUEST = 8


def _as_text(value: Any) -> str:
    return str(value if value is not None else "")


def _safe_json_loads(text: str) -> Optional[Any]:
    try:
        return json.loads(text)
    except Exception:
        return None


def _normal_cell(value: Any) -> str:
    return re.sub(r"\s+", " ", _as_text(value).strip().lower())


def _table_linearization(table: pd.DataFrame, max_rows: int) -> str:
    shown = table.head(max_rows) if max_rows >= 0 else table
    lines = ["col : " + " | ".join(_as_text(col) for col in shown.columns)]
    for row_idx, (_, row) in enumerate(shown.iterrows(), start=1):
        lines.append(
            "row {} : ".format(row_idx)
            + " | ".join(_as_text(row[col]) for col in shown.columns)
        )
    return "\n".join(lines)


def _column_priority_json(table: pd.DataFrame, title: Optional[str], max_rows: int) -> str:
    shown = table.head(max_rows) if max_rows >= 0 else table
    list_table = [list(map(_as_text, table.columns))] + [
        [_as_text(cell) for cell in row] for row in shown.values.tolist()
    ]
    transposed = [
        [list_table[row_idx][col_idx] for row_idx in range(len(list_table))]
        for col_idx in range(len(list_table[0]))
    ]
    payload: Dict[str, Any] = {
        "columns": list(map(_as_text, table.columns)),
        "table_column_priority": transposed,
    }
    if title:
        payload = {"table_caption": title, **payload}
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _read_few_shot_prompt(prompt_path: Path, n_shots: int) -> str:
    text = prompt_path.read_text(encoding="utf-8")
    chunks: List[str] = []
    current: List[str] = []
    last_blank = False
    for line in text.splitlines(keepends=True):
        is_blank = line.strip() == ""
        if is_blank and last_blank:
            chunks.append("".join(current))
            current = []
        else:
            current.append(line)
        last_blank = is_blank
    if current:
        chunks.append("".join(current))
    chunks = [chunk for chunk in chunks if chunk.strip()]
    chunks = chunks[: max(0, n_shots)]
    if not chunks:
        return ""
    chunks[-1] = chunks[-1].strip()
    return "\n".join(chunks)


def _default_prompt_path(prompt_dir: Path, select_type: str) -> Path:
    if select_type == "col":
        return prompt_dir / "wtq" / "col.txt"
    if select_type == "row":
        return prompt_dir / "wtq" / "frow.txt"
    raise ValueError(f"Unsupported retrieval select_type: {select_type}")


def _build_retrieval_generate_prompt(
    question: str,
    table: pd.DataFrame,
    title: Optional[str],
    select_type: str,
    num_rows: int,
) -> str:
    if select_type == "col":
        table_json = _column_priority_json(table, title=title, max_rows=num_rows)
        return (
            "/*\n"
            "table = "
            + table_json
            + "\n*/\n"
            + "statement : "
            + question
            + ".\n"
        )

    if select_type == "row":
        table_text = _table_linearization(table, max_rows=num_rows)
        if title:
            table_text = "table caption : " + title + "\n" + table_text
        return "/*\n" + table_text + "\n*/\n" + "statement : " + question + "\nexplain :"

    raise ValueError(f"Unsupported retrieval select_type: {select_type}")


def _build_builtin_few_shot(select_type: str) -> str:
    if select_type == "col":
        return (
            "Use f_col() API to filter out useless columns in the table according to "
            "the statement and table. Return only one call like f_col([column_a, column_b]).\n"
            "Column names must be copied exactly from the given table columns."
        )
    if select_type == "row":
        return (
            "Use f_row() API to filter out useless rows in the table according to the "
            "statement and table. Return only one call like f_row([row 1, row 3]).\n"
            "Rows are numbered exactly as shown in the table. If all rows are needed, return f_row([*])."
        )
    raise ValueError(f"Unsupported retrieval select_type: {select_type}")


def _build_operation_aware_guidance(select_type: str) -> str:
    if select_type == "col":
        return (
            "First infer the latent table operation implied by the statement, such as locating an entity, "
            "comparing candidates, ranking, finding a neighboring row, or counting/aggregating. "
            "Then keep the columns that are jointly sufficient to carry out and verify that operation, "
            "not only the columns with the strongest surface overlap. Preserve identifier, comparison, "
            "ordering, counting, and answer-bearing columns whenever they are needed."
        )
    if select_type == "row":
        return (
            "First infer the latent table operation implied by the statement, such as locating an entity, "
            "comparing candidates, ranking, finding a neighboring row, or counting/aggregating. "
            "Then select an evidence row set that is sufficient to carry out and verify that operation, "
            "not only a guessed answer row. Preserve comparison domains, anchor rows, and neighboring rows "
            "whenever they are needed."
        )
    raise ValueError(f"Unsupported retrieval select_type: {select_type}")


def _build_prompt(
    question: str,
    table: pd.DataFrame,
    title: Optional[str],
    select_type: str,
    num_rows: int,
    prompt_dir: Path,
    n_shots: int,
    prompt_mode: str,
    require_original_prompts: bool,
    operation_guidance_mode: str,
) -> Tuple[str, str]:
    prompt_path = _default_prompt_path(prompt_dir, select_type)
    used_prompt = "builtin_zero_shot"
    few_shot = ""
    if prompt_mode == "original" or (prompt_mode == "auto" and prompt_path.exists()):
        if not prompt_path.exists():
            if require_original_prompts:
                raise FileNotFoundError(
                    f"Original prompt file is missing: {prompt_path}. "
                    "Restore the optional prompt templates, or use --prompt-mode builtin."
                )
        else:
            few_shot = _read_few_shot_prompt(prompt_path, n_shots=n_shots)
            used_prompt = str(prompt_path)
    if not few_shot:
        few_shot = _build_builtin_few_shot(select_type)
    if operation_guidance_mode == "light":
        few_shot = few_shot.rstrip() + "\n" + _build_operation_aware_guidance(select_type)
        used_prompt = f"{used_prompt}+operation_aware_light"

    generate_prompt = _build_retrieval_generate_prompt(
        question=question,
        table=table,
        title=title,
        select_type=select_type,
        num_rows=num_rows,
    )
    return few_shot + "\n\n" + generate_prompt, used_prompt


def _extract_message_content(choice: Dict[str, Any]) -> str:
    message = choice.get("message", {})
    content = message.get("content", "")
    if isinstance(content, list):
        return "".join(
            _as_text(part.get("text", "") if isinstance(part, dict) else part)
            for part in content
        )
    return _as_text(content)


def _choice_logprob_score(choice: Dict[str, Any]) -> Optional[float]:
    logprobs = choice.get("logprobs")
    if not isinstance(logprobs, dict):
        return None
    content = logprobs.get("content")
    if not isinstance(content, list):
        return None
    total = 0.0
    count = 0
    for token in content:
        if isinstance(token, dict) and isinstance(token.get("logprob"), (int, float)):
            total += float(token["logprob"])
            count += 1
    if count == 0:
        return None
    return total / count


def _request_openai_candidates(
    prompt: str,
    api_key: str,
    model: str,
    base_url: str,
    max_tokens: int,
    gpt5_completion_token_budget: int,
    temperature: float,
    top_p: float,
    sampling_n: int,
    stop: Optional[Sequence[str]],
    timeout: int,
    retries: int,
    request_logprobs: bool,
) -> Tuple[List[Tuple[str, float, str]], str]:
    if not api_key:
        raise RuntimeError("API_KEY or OPENAI_API_KEY is required for OpenAI models.")

    resolved_model = _normalize_model(model)
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    url = f"{base_url.rstrip('/')}/chat/completions"
    total_requested = max(1, sampling_n)
    all_candidates: List[Tuple[str, float, str]] = []
    score_sources: List[str] = []
    global_rank = 0

    # Modern chat APIs cap `n` per request. The default script uses n=8 so this
    # loop runs once; keeping the loop here only protects explicit larger runs.
    while len(all_candidates) < total_requested:
        batch_n = min(
            OPENAI_MAX_CANDIDATES_PER_REQUEST,
            total_requested - len(all_candidates),
        )
        payload: Dict[str, Any] = {
            "model": resolved_model,
            "messages": [{"role": "user", "content": prompt}],
            "n": batch_n,
            "stream": False,
        }
        if resolved_model.startswith("gpt-5"):
            payload["max_completion_tokens"] = max(max_tokens, gpt5_completion_token_budget)
        else:
            payload["temperature"] = temperature
            payload["top_p"] = top_p
            payload["max_tokens"] = max_tokens
        if stop and not resolved_model.startswith("gpt-5"):
            payload["stop"] = list(stop)
        if request_logprobs:
            payload["logprobs"] = True

        last_error: Optional[Exception] = None
        batch_candidates: List[Tuple[str, float, str]] = []
        for attempt in range(retries + 1):
            try:
                response = requests.post(url, headers=headers, json=payload, timeout=timeout)
                if response.ok:
                    data = response.json()
                    batch_score_source = "chat_logprobs"
                    for choice in data.get("choices", []):
                        text = _extract_message_content(choice).strip()
                        if not text:
                            global_rank += 1
                            continue
                        score = _choice_logprob_score(choice)
                        if score is None:
                            score = -float(global_rank)
                            batch_score_source = "choice_order_no_logprobs"
                        batch_candidates.append((text, float(score), batch_score_source))
                        score_sources.append(batch_score_source)
                        global_rank += 1
                    if batch_candidates:
                        break
                    finish_reason = data.get("choices", [{}])[0].get("finish_reason")
                    usage = data.get("usage")
                    raise RuntimeError(
                        "OpenAI returned no usable candidates. "
                        f"finish_reason={finish_reason} usage={usage} "
                        f"api_max_completion_tokens={payload.get('max_completion_tokens')}"
                    )
                last_error = RuntimeError(f"OpenAI request failed: {response.status_code} {response.text[:1000]}")
            except Exception as exc:
                last_error = exc
            if attempt < retries:
                time.sleep(2 * (attempt + 1))
        if not batch_candidates:
            if last_error is not None:
                raise last_error
            break
        all_candidates.extend(batch_candidates)

    if not all_candidates:
        return [], "none"
    if any(source == "chat_logprobs" for source in score_sources):
        return all_candidates[:total_requested], "chat_logprobs"
    return all_candidates[:total_requested], "choice_order_no_logprobs"


def _request_gemini_candidates(
    prompt: str,
    api_key: str,
    model: str,
    base_url: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    sampling_n: int,
    timeout: int,
    retries: int,
) -> Tuple[List[Tuple[str, float, str]], str]:
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY or GOOGLE_API_KEY is required for Gemini models.")

    resolved_model = _normalize_model(model)
    url = f"{base_url.rstrip('/')}/models/{resolved_model}:generateContent"
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": temperature,
            "topP": top_p,
            "candidateCount": max(1, sampling_n),
            "maxOutputTokens": max_tokens,
        },
    }
    headers = {"x-goog-api-key": api_key, "Content-Type": "application/json"}
    last_error: Optional[Exception] = None
    for attempt in range(retries + 1):
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=timeout)
            if response.ok:
                data = response.json()
                candidates: List[Tuple[str, float, str]] = []
                for rank, candidate in enumerate(data.get("candidates", [])):
                    parts = candidate.get("content", {}).get("parts", [])
                    text = "".join(_as_text(part.get("text", "")) for part in parts).strip()
                    if text:
                        candidates.append((text, -float(rank), "choice_order_no_logprobs"))
                if candidates:
                    return candidates, "choice_order_no_logprobs"
                raise RuntimeError(f"Gemini returned no usable candidates: {json.dumps(data)[:1000]}")
            last_error = RuntimeError(f"Gemini request failed: {response.status_code} {response.text[:1000]}")
        except Exception as exc:
            last_error = exc
        if attempt < retries:
            time.sleep(2 * (attempt + 1))
    if last_error is not None:
        raise last_error
    return [], "none"


def _request_candidates(
    prompt: str,
    openai_api_key: str,
    gemini_api_key: str,
    model: str,
    base_url: str,
    gemini_base_url: str,
    max_tokens: int,
    gpt5_completion_token_budget: int,
    temperature: float,
    top_p: float,
    sampling_n: int,
    stop: Optional[Sequence[str]],
    timeout: int,
    retries: int,
    request_logprobs: bool,
) -> Tuple[List[Tuple[str, float, str]], str]:
    if _backend_is_llama33_model(model):
        return _request_swift_candidates(
            prompt=prompt,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            sampling_n=sampling_n,
        )
    if _is_gemini_model(model):
        return _request_gemini_candidates(
            prompt=prompt,
            api_key=gemini_api_key,
            model=model,
            base_url=gemini_base_url,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            sampling_n=sampling_n,
            timeout=timeout,
            retries=retries,
        )
    return _request_openai_candidates(
        prompt=prompt,
        api_key=openai_api_key,
        model=model,
        base_url=base_url,
        max_tokens=max_tokens,
        gpt5_completion_token_budget=gpt5_completion_token_budget,
        temperature=temperature,
        top_p=top_p,
        sampling_n=sampling_n,
        stop=stop,
        timeout=timeout,
        retries=retries,
        request_logprobs=request_logprobs,
    )


def _request_llm_local(
    prompt: str,
    openai_api_key: str,
    gemini_api_key: str,
    model: str,
    base_url: str,
    gemini_base_url: str,
    max_tokens: int,
    temperature: float,
    timeout: int,
    gpt5_completion_token_budget: Optional[int] = None,
) -> str:
    if _backend_is_llama33_model(model):
        return _request_swift_chat(
            prompt=prompt,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
        )
    return _request_llm(
        prompt=prompt,
        openai_api_key=openai_api_key,
        gemini_api_key=gemini_api_key,
        model=model,
        base_url=base_url,
        gemini_base_url=gemini_base_url,
        max_tokens=max_tokens,
        temperature=temperature,
        timeout=timeout,
        gpt5_completion_token_budget=gpt5_completion_token_budget,
    )


def _parse_call_items(raw_output: str, function_name: str) -> List[str]:
    text = raw_output.strip()
    text = re.sub(r"^```(?:python)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)

    pattern = re.compile(rf"{re.escape(function_name)}\s*\(\s*\[(.*?)\]\s*\)", re.DOTALL)
    match = pattern.search(text)
    if not match:
        json_obj = _safe_json_loads(text)
        if isinstance(json_obj, list):
            return [_as_text(x).strip().strip("\"'") for x in json_obj if _as_text(x).strip()]
        if isinstance(json_obj, dict):
            key = "columns" if function_name == "f_col" else "rows"
            value = json_obj.get(key)
            if isinstance(value, list):
                return [_as_text(x).strip().strip("\"'") for x in value if _as_text(x).strip()]
        return []
    inside = match.group(1).strip()
    if not inside:
        return []

    json_like = "[" + inside + "]"
    parsed = _safe_json_loads(json_like)
    if isinstance(parsed, list):
        return [_as_text(x).strip().strip("\"'") for x in parsed if _as_text(x).strip()]

    items = []
    for item in re.split(r"\s*,\s*", inside):
        clean = item.strip().strip("\"'").strip()
        if clean:
            items.append(clean)
    return items


def _normalize_column_item(item: str, columns: Sequence[str]) -> Optional[str]:
    raw = item.strip().strip("\"'`")
    if not raw:
        return None
    exact = {str(col): str(col) for col in columns}
    if raw in exact:
        return exact[raw]
    lower_map = {_normal_cell(col): str(col) for col in columns}
    return lower_map.get(_normal_cell(raw))


def _normalize_row_item(item: str, num_rows: int) -> Optional[int | str]:
    raw = item.strip().strip("\"'`")
    if raw == "*":
        return "*"
    match = re.search(r"-?\d+", raw)
    if not match:
        return None
    row_number = int(match.group(0))
    # Row-selection prompts show data rows as 1-based ids, while our saved row ids are 0-based.
    row_id = row_number - 1
    if 0 <= row_id < num_rows:
        return row_id
    return None


def _ranked_union(
    parsed_candidates: Iterable[Tuple[List[Any], float]],
    top_groups: int,
) -> List[Any]:
    groups: Dict[Tuple[Any, ...], float] = {}
    for items, score in parsed_candidates:
        if not items:
            continue
        key = tuple(dict.fromkeys(items))
        groups[key] = groups.get(key, 0.0) + math.exp(score)
    ordered = sorted(groups.items(), key=lambda x: x[1], reverse=True)
    union: List[Any] = []
    for group, _ in ordered[:top_groups]:
        for item in group:
            if item not in union:
                union.append(item)
    return union


def _canonical_group_items(items: Sequence[Any]) -> List[Any]:
    if any(item == "*" for item in items):
        return ["*"]
    ordered: List[Any] = []
    for item in items:
        if item not in ordered:
            ordered.append(item)
    return ordered


def _aggregate_candidate_groups(
    parsed_candidates: Iterable[Tuple[List[Any], float]],
) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
    for items, score in parsed_candidates:
        canonical_items = _canonical_group_items(items)
        if not canonical_items:
            continue
        key = tuple(canonical_items)
        if key not in groups:
            groups[key] = {
                "items": canonical_items,
                "mass": 0.0,
                "votes": 0,
                "best_score": score,
            }
        groups[key]["mass"] += math.exp(score)
        groups[key]["votes"] += 1
        groups[key]["best_score"] = max(float(groups[key]["best_score"]), float(score))
    ordered = sorted(
        groups.values(),
        key=lambda x: (float(x["mass"]), int(x["votes"]), float(x["best_score"])),
        reverse=True,
    )
    return ordered


def _group_disagreement(groups: Sequence[Dict[str, Any]]) -> float:
    if not groups:
        return 1.0
    total_mass = sum(float(group["mass"]) for group in groups)
    if total_mass <= 0:
        return 1.0
    top_share = float(groups[0]["mass"]) / total_mass
    return max(0.0, min(1.0, 1.0 - top_share))


def _adaptive_top_groups(
    groups: Sequence[Dict[str, Any]],
    max_groups: int,
    adaptive_mode: str,
) -> int:
    if not groups or max_groups <= 0:
        return 0
    capped = min(max_groups, len(groups))
    if adaptive_mode == "none":
        return capped
    disagreement = _group_disagreement(groups)
    dynamic = 1 + int(round(disagreement * max(0, capped - 1)))
    return max(1, min(capped, dynamic))


def _union_prefix_from_groups(
    groups: Sequence[Dict[str, Any]],
    top_groups: int,
) -> List[Any]:
    if top_groups <= 0:
        return []
    union: List[Any] = []
    for group in groups[:top_groups]:
        items = list(group.get("items", []))
        if "*" in items:
            return ["*"]
        for item in items:
            if item not in union:
                union.append(item)
    return union


def _item_coverage(
    selected_items: Sequence[Any],
    group_items: Sequence[Any],
) -> bool:
    selected_set = set(selected_items)
    group_set = set(group_items)
    if "*" in selected_set:
        return True
    if "*" in group_set:
        return "*" in selected_set
    return group_set.issubset(selected_set)


def _coverage_mass(
    selected_items: Sequence[Any],
    groups: Sequence[Dict[str, Any]],
) -> float:
    total_mass = sum(float(group["mass"]) for group in groups)
    if total_mass <= 0:
        return 0.0
    covered_mass = sum(
        float(group["mass"])
        for group in groups
        if _item_coverage(selected_items, list(group.get("items", [])))
    )
    return covered_mass / total_mass


def _effective_item_count(items: Sequence[Any], full_count: int) -> int:
    if any(item == "*" for item in items):
        return full_count
    return len(items)


def _prefix_candidates(
    groups: Sequence[Dict[str, Any]],
    max_groups: int,
    adaptive_mode: str,
) -> List[Tuple[int, List[Any]]]:
    effective_max = _adaptive_top_groups(groups, max_groups=max_groups, adaptive_mode=adaptive_mode)
    prefixes: List[Tuple[int, List[Any]]] = []
    seen = set()
    for k in range(1, effective_max + 1):
        items = _union_prefix_from_groups(groups, top_groups=k)
        signature = tuple(items)
        if items and signature not in seen:
            prefixes.append((k, items))
            seen.add(signature)
    return prefixes


def _joint_candidate_score(
    row_items: Sequence[Any],
    col_items: Sequence[Any],
    row_groups: Sequence[Dict[str, Any]],
    col_groups: Sequence[Dict[str, Any]],
    num_rows: int,
    num_cols: int,
) -> Dict[str, float]:
    row_coverage = _coverage_mass(row_items, row_groups)
    col_coverage = _coverage_mass(col_items, col_groups)
    support = math.sqrt(max(row_coverage, 1e-8) * max(col_coverage, 1e-8))
    full_cells = max(1, num_rows * num_cols)
    cell_count = max(1, _effective_item_count(row_items, num_rows) * _effective_item_count(col_items, num_cols))
    cell_ratio = cell_count / float(full_cells)
    disagreement = (_group_disagreement(row_groups) + _group_disagreement(col_groups)) / 2.0
    # Keep a slight compactness bias, but make it softer so the union stage
    # does not over-trim evidence that may be needed downstream.
    compactness_alpha = max(0.10, 0.24 - 0.12 * disagreement)
    score = support / (cell_ratio ** compactness_alpha)
    return {
        "score": score,
        "row_coverage": row_coverage,
        "col_coverage": col_coverage,
        "support": support,
        "cell_count": float(cell_count),
        "cell_ratio": cell_ratio,
        "disagreement": disagreement,
        "compactness_alpha": compactness_alpha,
    }


def _select_joint_candidate(
    table: pd.DataFrame,
    row_groups: Sequence[Dict[str, Any]],
    col_groups: Sequence[Dict[str, Any]],
    max_row_groups: int,
    max_col_groups: int,
    adaptive_mode: str,
) -> Optional[Dict[str, Any]]:
    row_prefixes = _prefix_candidates(row_groups, max_groups=max_row_groups, adaptive_mode=adaptive_mode)
    col_prefixes = _prefix_candidates(col_groups, max_groups=max_col_groups, adaptive_mode=adaptive_mode)
    if not row_prefixes or not col_prefixes:
        return None

    best: Optional[Dict[str, Any]] = None
    for row_k, row_items in row_prefixes:
        for col_k, col_items in col_prefixes:
            metrics = _joint_candidate_score(
                row_items=row_items,
                col_items=col_items,
                row_groups=row_groups,
                col_groups=col_groups,
                num_rows=len(table),
                num_cols=len(table.columns),
            )
            candidate = {
                "rows": row_items,
                "columns": col_items,
                "row_top_groups": row_k,
                "col_top_groups": col_k,
                **metrics,
            }
            if best is None:
                best = candidate
                continue
            better = (
                candidate["score"],
                candidate["support"],
                candidate["row_coverage"] + candidate["col_coverage"],
                -candidate["cell_count"],
            ) > (
                best["score"],
                best["support"],
                best["row_coverage"] + best["col_coverage"],
                -best["cell_count"],
            )
            if better:
                best = candidate
    return best


def _summarize_groups(groups: Sequence[Dict[str, Any]], top_k: int) -> str:
    if top_k <= 0 or not groups:
        return "none"
    lines: List[str] = []
    for i, group in enumerate(groups[:top_k], start=1):
        items = group.get("items", [])
        lines.append(
            f"{i}. items={json.dumps(items, ensure_ascii=False)} "
            f"mass={float(group.get('mass', 0.0)):.3f} "
            f"votes={int(group.get('votes', 0))}"
        )
    return "\n".join(lines)


def _build_answerability_prompt(
    question: str,
    table: pd.DataFrame,
    selected_rows: Sequence[Any],
    selected_cols: Sequence[Any],
    row_groups: Sequence[Dict[str, Any]],
    col_groups: Sequence[Dict[str, Any]],
    top_groups: int,
) -> str:
    rows = list(selected_rows)
    cols = [str(x) for x in selected_cols]
    current_rows_view = table.loc[rows, :] if rows and rows != ["*"] else table.head(min(len(table), 12))
    current_subtable = table.loc[rows, cols] if rows and cols and rows != ["*"] else pd.DataFrame()
    return f"""You are checking whether a selected evidence table is sufficient to answer a WikiTableQuestions question.

Return JSON only:
{{
  "sufficient": bool,
  "rows": list[int] | ["*"],
  "columns": list[str]
}}

Rules:
- Keep all current rows and columns unless you need a minimal superset.
- If insufficient, expand only using the candidate groups shown below.
- Prefer the smallest superset that still looks sufficient.
- Rows are zero-based ids.
- Column names must be copied exactly from the table.

Question:
{question}

Current selected rows with all columns:
{_df_to_markdown(current_rows_view) if not current_rows_view.empty else "EMPTY"}

Current selected sub-table:
{_df_to_markdown(current_subtable) if not current_subtable.empty else "EMPTY"}

Top row candidate groups:
{_summarize_groups(row_groups, top_groups)}

Top column candidate groups:
{_summarize_groups(col_groups, top_groups)}

Return only valid JSON."""


def _parse_answerability_output(raw_text: str, table: pd.DataFrame, fallback_rows: Sequence[Any], fallback_cols: Sequence[Any]) -> Dict[str, Any]:
    try:
        obj = _extract_json_object(raw_text)
    except Exception:
        obj = {}
    rows = obj.get("rows", fallback_rows)
    cols = obj.get("columns", fallback_cols)
    return {
        "sufficient": bool(obj.get("sufficient", False)),
        "rows": [row for row in (_normalize_row_item(str(x), len(table)) for x in _as_list(rows)) if row is not None],
        "columns": [
            col
            for col in (_normalize_column_item(str(x), list(table.columns)) for x in _as_list(cols))
            if col is not None
        ],
    }


def _result_for_json(result: Dict[str, Any]) -> Dict[str, Any]:
    sub_table = result.get("sub_table")
    if isinstance(sub_table, pd.DataFrame):
        sub_table_records = sub_table.reset_index().rename(columns={"index": "row_id"}).to_dict(orient="records")
        sub_table_markdown = _df_to_markdown(sub_table)
        sub_table_shape = list(sub_table.shape)
    else:
        sub_table_records = []
        sub_table_markdown = ""
        sub_table_shape = [0, 0]
    return {
        "rows": result.get("rows", []),
        "columns": result.get("columns", []),
        "status": result.get("status"),
        "reason": result.get("reason"),
        "sub_table_shape": sub_table_shape,
        "sub_table_records": sub_table_records,
        "sub_table_markdown": sub_table_markdown,
        "source": result.get("source"),
    }


def _load_existing_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    records: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Bad JSONL at {path}:{line_no}: {exc}") from exc
    return records


def _full_table_result(table: pd.DataFrame, reason: str) -> Dict[str, Any]:
    rows = list(range(len(table)))
    columns = [str(col) for col in table.columns]
    return {
        "rows": rows,
        "columns": columns,
        "sub_table": table,
        "status": "accept",
        "reason": reason,
        "source": "full_table_fallback",
    }


def _build_subtable_result(
    table: pd.DataFrame,
    rows: Sequence[int | str],
    columns: Sequence[str],
    reason: Optional[str],
) -> Dict[str, Any]:
    if "*" in rows:
        row_ids = list(range(len(table)))
    else:
        row_ids = []
        for row in rows:
            if isinstance(row, int) and 0 <= row < len(table) and row not in row_ids:
                row_ids.append(row)

    col_ids: List[str] = []
    for col in columns:
        if col in table.columns and col not in col_ids:
            col_ids.append(col)

    sub_table = table.loc[row_ids, col_ids] if row_ids and col_ids else pd.DataFrame()
    status = "accept" if not sub_table.empty else "reject"
    if not row_ids:
        reason = (reason + "; " if reason else "") + "no grounded rows"
    if not col_ids:
        reason = (reason + "; " if reason else "") + "no grounded columns"
    return {
        "rows": row_ids,
        "columns": col_ids,
        "sub_table": sub_table,
        "status": status,
        "reason": reason,
        "source": "opaware_col_row_union",
    }


def _run_one(
    question: str,
    title: Optional[str],
    table: pd.DataFrame,
    openai_api_key: str,
    gemini_api_key: str,
    model: str,
    base_url: str,
    gemini_base_url: str,
    prompt_dir: Path,
    prompt_mode: str,
    require_original_prompts: bool,
    col_num_rows: int,
    row_num_rows: int,
    n_shots: int,
    max_tokens: int,
    gpt5_completion_token_budget: int,
    temperature: float,
    top_p: float,
    sampling_n: int,
    union_col: int,
    union_row: int,
    stop: Optional[Sequence[str]],
    timeout: int,
    retries: int,
    request_logprobs: bool,
    operation_guidance_mode: str,
    aggregation_mode: str,
    adaptive_union_mode: str,
    enable_answerability_check: bool,
    enable_answerability_expand: bool,
    answerability_top_groups: int,
    answerability_max_new_tokens: int,
    answerability_completion_budget: int,
) -> Dict[str, Any]:
    col_prompt, col_prompt_source = _build_prompt(
        question=question,
        table=table,
        title=title,
        select_type="col",
        num_rows=col_num_rows,
        prompt_dir=prompt_dir,
        n_shots=n_shots,
        prompt_mode=prompt_mode,
        require_original_prompts=require_original_prompts,
        operation_guidance_mode=operation_guidance_mode,
    )
    row_prompt, row_prompt_source = _build_prompt(
        question=question,
        table=table,
        title=title,
        select_type="row",
        num_rows=row_num_rows,
        prompt_dir=prompt_dir,
        n_shots=n_shots,
        prompt_mode=prompt_mode,
        require_original_prompts=require_original_prompts,
        operation_guidance_mode=operation_guidance_mode,
    )

    col_candidates, col_score_source = _request_candidates(
        prompt=col_prompt,
        openai_api_key=openai_api_key,
        gemini_api_key=gemini_api_key,
        model=model,
        base_url=base_url,
        gemini_base_url=gemini_base_url,
        max_tokens=max_tokens,
        gpt5_completion_token_budget=gpt5_completion_token_budget,
        temperature=temperature,
        top_p=top_p,
        sampling_n=sampling_n,
        stop=stop,
        timeout=timeout,
        retries=retries,
        request_logprobs=request_logprobs,
    )
    row_candidates, row_score_source = _request_candidates(
        prompt=row_prompt,
        openai_api_key=openai_api_key,
        gemini_api_key=gemini_api_key,
        model=model,
        base_url=base_url,
        gemini_base_url=gemini_base_url,
        max_tokens=max_tokens,
        gpt5_completion_token_budget=gpt5_completion_token_budget,
        temperature=temperature,
        top_p=top_p,
        sampling_n=sampling_n,
        stop=stop,
        timeout=timeout,
        retries=retries,
        request_logprobs=request_logprobs,
    )

    parsed_cols: List[Tuple[List[str], float]] = []
    for text, score, _ in col_candidates:
        items = _parse_call_items(text, "f_col")
        cols = [
            col
            for col in (_normalize_column_item(item, list(table.columns)) for item in items)
            if col is not None
        ]
        parsed_cols.append((cols, score))

    parsed_rows: List[Tuple[List[int | str], float]] = []
    for text, score, _ in row_candidates:
        items = _parse_call_items(text, "f_row")
        rows = [
            row
            for row in (_normalize_row_item(item, len(table)) for item in items)
            if row is not None
        ]
        parsed_rows.append((rows, score))

    aggregated_col_groups = _aggregate_candidate_groups(parsed_cols)
    aggregated_row_groups = _aggregate_candidate_groups(parsed_rows)

    selected_cols: List[Any] = []
    selected_rows: List[Any] = []
    reconciliation: Dict[str, Any] = {
        "aggregation_mode": aggregation_mode,
        "adaptive_union_mode": adaptive_union_mode,
        "row_group_disagreement": _group_disagreement(aggregated_row_groups),
        "col_group_disagreement": _group_disagreement(aggregated_col_groups),
        "row_groups": aggregated_row_groups,
        "col_groups": aggregated_col_groups,
    }

    if aggregation_mode == "joint_reconcile":
        selected = _select_joint_candidate(
            table=table,
            row_groups=aggregated_row_groups,
            col_groups=aggregated_col_groups,
            max_row_groups=union_row,
            max_col_groups=union_col,
            adaptive_mode=adaptive_union_mode,
        )
        if selected is not None:
            selected_rows = list(selected["rows"])
            selected_cols = list(selected["columns"])
            reconciliation["selected_candidate"] = selected
    else:
        effective_col_groups = _adaptive_top_groups(
            aggregated_col_groups,
            max_groups=union_col,
            adaptive_mode=adaptive_union_mode,
        )
        effective_row_groups = _adaptive_top_groups(
            aggregated_row_groups,
            max_groups=union_row,
            adaptive_mode=adaptive_union_mode,
        )
        selected_cols = _union_prefix_from_groups(aggregated_col_groups, top_groups=effective_col_groups)
        selected_rows = _union_prefix_from_groups(aggregated_row_groups, top_groups=effective_row_groups)
        reconciliation["selected_candidate"] = {
            "rows": selected_rows,
            "columns": selected_cols,
            "row_top_groups": effective_row_groups,
            "col_top_groups": effective_col_groups,
        }

    reason_parts = []
    if not selected_cols:
        selected_cols = [str(col) for col in table.columns]
        reason_parts.append("column parse failed; kept all columns")
    if not selected_rows:
        selected_rows = ["*"]
        reason_parts.append("row parse failed; kept all rows")

    silver_result = _build_subtable_result(
        table=table,
        rows=selected_rows,
        columns=selected_cols,
        reason="; ".join(reason_parts) if reason_parts else None,
    )
    if silver_result.get("status") == "accept":
        silver_result["source"] = (
            "opaware_joint_reconcile"
            if aggregation_mode == "joint_reconcile"
            else "opaware_col_row_union"
        )

    answerability: Optional[Dict[str, Any]] = None
    raw_answerability = ""
    if enable_answerability_check and silver_result.get("status") == "accept" and silver_result.get("source") != "full_table_fallback":
        try:
            raw_answerability = _request_llm_local(
                prompt=_build_answerability_prompt(
                    question=question,
                    table=table,
                    selected_rows=silver_result.get("rows", []),
                    selected_cols=silver_result.get("columns", []),
                    row_groups=aggregated_row_groups,
                    col_groups=aggregated_col_groups,
                    top_groups=max(1, answerability_top_groups),
                ),
                openai_api_key=openai_api_key,
                gemini_api_key=gemini_api_key,
                model=model,
                base_url=base_url,
                gemini_base_url=gemini_base_url,
                max_tokens=answerability_max_new_tokens,
                temperature=0.0,
                timeout=timeout,
                gpt5_completion_token_budget=answerability_completion_budget,
            )
            answerability = _parse_answerability_output(
                raw_answerability,
                table=table,
                fallback_rows=silver_result.get("rows", []),
                fallback_cols=silver_result.get("columns", []),
            )
            if enable_answerability_expand and not answerability.get("sufficient"):
                merged_rows: List[Any] = []
                for row in list(silver_result.get("rows", [])) + list(answerability.get("rows", [])):
                    if row not in merged_rows:
                        merged_rows.append(row)
                merged_cols: List[Any] = []
                for col in list(silver_result.get("columns", [])) + list(answerability.get("columns", [])):
                    if col not in merged_cols:
                        merged_cols.append(col)
                expanded = _build_subtable_result(
                    table=table,
                    rows=merged_rows,
                    columns=merged_cols,
                    reason="answerability_expanded",
                )
                if expanded.get("status") == "accept":
                    silver_result = expanded
                    silver_result["source"] = "opaware_answerability_expanded"
        except Exception as exc:
            answerability = {
                "sufficient": False,
                "rows": list(silver_result.get("rows", [])),
                "columns": list(silver_result.get("columns", [])),
                "error": str(exc),
            }

    return {
        "silver_result": silver_result,
        "col_prompt_source": col_prompt_source,
        "row_prompt_source": row_prompt_source,
        "col_prompt": col_prompt,
        "row_prompt": row_prompt,
        "col_candidates": [
            {"text": text, "score": score, "score_source": score_source}
            for text, score, score_source in col_candidates
        ],
        "row_candidates": [
            {"text": text, "score": score, "score_source": score_source}
            for text, score, score_source in row_candidates
        ],
        "parsed_col_candidates": [
            {"items": items, "score": score} for items, score in parsed_cols
        ],
        "parsed_row_candidates": [
            {"items": items, "score": score} for items, score in parsed_rows
        ],
        "aggregated_col_groups": aggregated_col_groups,
        "aggregated_row_groups": aggregated_row_groups,
        "reconciliation": reconciliation,
        "col_score_source": col_score_source,
        "row_score_source": row_score_source,
        "answerability": answerability,
        "raw_answerability": raw_answerability,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Operation-Aware Table Localization: retrieve evidence sub-tables with operation-guided row/column selection."
    )
    parser.add_argument("--dataset", default="wikitq")
    parser.add_argument("--split", default="test")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--gemini-base-url", default=DEFAULT_GEMINI_BASE_URL)
    parser.add_argument("--api-key", default="")
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--start-idx", type=int, default=0)
    parser.add_argument("--max-items", type=int, default=150)
    parser.add_argument("--save-every", type=int, default=20)
    parser.add_argument("--append-jsonl", action="store_true")
    parser.add_argument("--quiet", action="store_true")

    parser.add_argument(
        "--prompt-dir",
        type=Path,
        default=DEFAULT_PROMPT_DIR,
        help="Optional directory that contains original templates/wtq/col.txt and templates/wtq/frow.txt.",
    )
    parser.add_argument(
        "--prompt-mode",
        choices=["auto", "original", "builtin"],
        default="auto",
        help="auto uses optional original prompts when available; builtin uses a compact local prompt.",
    )
    parser.add_argument(
        "--require-original-prompts",
        action="store_true",
        help="Fail if optional original prompt files are missing.",
    )

    parser.add_argument("--col-num-rows", type=int, default=3)
    parser.add_argument("--row-num-rows", type=int, default=100)
    parser.add_argument("--n-shots", type=int, default=14)
    parser.add_argument("--max-new-tokens", type=int, default=150)
    parser.add_argument(
        "--gpt5-completion-token-budget",
        type=int,
        default=2048,
        help="API-level max_completion_tokens for GPT-5 models; GPT-5 budgets include hidden reasoning tokens.",
    )
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument(
        "--sampling-n",
        type=int,
        default=8,
        help="Number of candidates per col/row prompt. OpenAI chat APIs currently allow at most 8 per request.",
    )
    parser.add_argument("--union-col", type=int, default=4)
    parser.add_argument("--union-row", type=int, default=2)
    parser.add_argument(
        "--aggregation-mode",
        choices=["joint_reconcile", "independent_union"],
        default="joint_reconcile",
        help=(
            "How to combine sampled row/column candidates. "
            "joint_reconcile searches over row/column prefix combinations and chooses the most "
            "support-dense evidence sub-table."
        ),
    )
    parser.add_argument(
        "--adaptive-union-mode",
        choices=["disagreement", "none"],
        default="disagreement",
        help=(
            "Whether the number of retained row/column groups should adapt to candidate disagreement. "
            "disagreement keeps wider evidence when the samples are less consistent."
        ),
    )
    parser.add_argument(
        "--operation-guidance-mode",
        choices=["none", "light"],
        default="light",
        help=(
            "Append a lightweight operation-aware instruction to the retrieval prompts. "
            "This keeps the f_row/f_col retrieval interface unchanged while encouraging "
            "the model to retrieve evidence sets instead of isolated answer guesses."
        ),
    )
    parser.add_argument("--stop", default="\n\n")
    parser.add_argument(
        "--request-logprobs",
        action="store_true",
        help="Ask chat API for logprobs when supported; otherwise candidates are ranked by returned choice order.",
    )
    parser.add_argument(
        "--no-full-table-fallback",
        action="store_true",
        help="Disable final full-table fallback. Default keeps every saved sub-table non-empty.",
    )
    parser.add_argument(
        "--enable-answerability-check",
        action="store_true",
        help="Run a lightweight answerability check over the current union using only the saved candidate groups.",
    )
    parser.add_argument(
        "--enable-answerability-expand",
        action="store_true",
        help="If answerability says the current union is insufficient, expand it with the minimal suggested rows/columns.",
    )
    parser.add_argument(
        "--answerability-top-groups",
        type=int,
        default=4,
        help="How many top row/column candidate groups to expose to the answerability prompt.",
    )
    parser.add_argument(
        "--answerability-max-new-tokens",
        type=int,
        default=96,
        help="Completion budget for the lightweight answerability prompt.",
    )
    parser.add_argument(
        "--answerability-completion-budget",
        type=int,
        default=256,
        help="GPT-5 completion-token budget for the lightweight answerability prompt.",
    )
    parser.add_argument(
        "--save-path",
        default="results/ours/subtables/gpt5mini_opaware_wikitq.pkl",
    )
    parser.add_argument(
        "--jsonl-path",
        default="results/ours/subtables/gpt5mini_opaware_wikitq.jsonl",
    )
    args = parser.parse_args()

    if args.prompt_mode == "original":
        args.require_original_prompts = True

    openai_api_key = (args.api_key or _get_api_key()).strip()
    gemini_api_key = (args.api_key or _get_gemini_api_key()).strip()
    provider = "gemini" if _is_gemini_model(args.model) else ("llama33" if _backend_is_llama33_model(args.model) else "openai")
    if provider == "openai" and not openai_api_key:
        raise SystemExit(
            "API_KEY or OPENAI_API_KEY is required for OpenAI models. "
            "Set one of them or pass --api-key YOUR_KEY before running."
        )
    if provider == "gemini" and not gemini_api_key:
        raise SystemExit(
            "GEMINI_API_KEY or GOOGLE_API_KEY is required for Gemini models. "
            "Set one of them or pass --api-key YOUR_KEY before running."
        )

    dataset_name = _resolve_dataset_name(args.dataset)
    resolved_model = _normalize_model(args.model)
    stop = None if args.stop.lower() in {"", "none", "null"} else [args.stop.encode("utf-8").decode("unicode_escape")]
    api_temperature = "default" if resolved_model.startswith("gpt-5") else args.temperature
    api_top_p = "default" if resolved_model.startswith("gpt-5") else args.top_p
    api_stop = None if resolved_model.startswith("gpt-5") else stop
    full_table_fallback_enabled = not args.no_full_table_fallback

    _log(f"[Init] method={METHOD_NAME} dataset={dataset_name} split={args.split}", args.quiet)
    _log(
        f"[Init] provider={provider} model={resolved_model} prompt_mode={args.prompt_mode} prompt_dir={args.prompt_dir}",
        args.quiet,
    )
    _log(
        "[Init] Operation-Aware Table Localization params "
        f"temperature={args.temperature} top_p={args.top_p} sampling_n={args.sampling_n} "
        f"max_tokens={args.max_new_tokens} col_num_rows={args.col_num_rows} row_num_rows={args.row_num_rows} "
        f"n_shots={args.n_shots} union_col={args.union_col} union_row={args.union_row} "
        f"aggregation_mode={args.aggregation_mode} adaptive_union_mode={args.adaptive_union_mode} "
        f"operation_guidance_mode={args.operation_guidance_mode} stop={stop} "
        f"api_temperature={api_temperature} api_top_p={api_top_p} api_stop={api_stop} "
        f"full_table_fallback={full_table_fallback_enabled}",
        args.quiet,
    )

    dataset = _load_dataset_split(args.dataset, args.split)
    end_idx = len(dataset)
    if args.max_items > 0:
        end_idx = min(end_idx, args.start_idx + args.max_items)
    total_to_run = max(0, end_idx - args.start_idx)
    _log(f"[Init] loaded {len(dataset)} examples; running {total_to_run} examples from idx={args.start_idx}", args.quiet)

    _ensure_dir(args.save_path)
    _ensure_dir(args.jsonl_path)

    existing_results = _load_existing_jsonl(Path(args.jsonl_path)) if args.append_jsonl else []
    results: List[Dict[str, Any]] = list(existing_results)
    done_indices = {int(record.get("idx")) for record in existing_results if "idx" in record}
    jsonl_mode = "a" if args.append_jsonl else "w"
    if existing_results:
        _log(
            f"[Init] append_jsonl=true loaded_existing={len(existing_results)} from {args.jsonl_path}",
            args.quiet,
        )
    with open(args.jsonl_path, jsonl_mode, encoding="utf-8") as jsonl_file:
        for i in range(args.start_idx, end_idx):
            if i in done_indices:
                continue
            item = _normalize_dataset_item(dict(dataset[i]), args.dataset, idx=i)
            question = str(item["question"])
            gold_answers = [str(x) for x in item.get("answers", [])]
            table = _wtq_item_to_df(item)
            title = str(item.get("table", {}).get("name") or item.get("id") or f"wikitq_{i}")
            local_idx = len(results) + 1
            _log(
                f"[Item {local_idx}/{total_to_run}] idx={i} rows={table.shape[0]} cols={table.shape[1]} question={question[:120]}",
                args.quiet,
            )

            try:
                run_result = _run_one(
                    question=question,
                    title=title,
                    table=table,
                    openai_api_key=openai_api_key,
                    gemini_api_key=gemini_api_key,
                    model=args.model,
                    base_url=args.base_url,
                    gemini_base_url=args.gemini_base_url,
                    prompt_dir=args.prompt_dir,
                    prompt_mode=args.prompt_mode,
                    require_original_prompts=args.require_original_prompts,
                    col_num_rows=args.col_num_rows,
                    row_num_rows=args.row_num_rows,
                    n_shots=args.n_shots,
                    max_tokens=args.max_new_tokens,
                    gpt5_completion_token_budget=args.gpt5_completion_token_budget,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    sampling_n=args.sampling_n,
                    union_col=args.union_col,
                    union_row=args.union_row,
                    stop=stop,
                    timeout=args.timeout,
                    retries=args.retries,
                    request_logprobs=args.request_logprobs,
                    operation_guidance_mode=args.operation_guidance_mode,
                    aggregation_mode=args.aggregation_mode,
                    adaptive_union_mode=args.adaptive_union_mode,
                    enable_answerability_check=args.enable_answerability_check,
                    enable_answerability_expand=args.enable_answerability_expand,
                    answerability_top_groups=args.answerability_top_groups,
                    answerability_max_new_tokens=args.answerability_max_new_tokens,
                    answerability_completion_budget=args.answerability_completion_budget,
                )
            except Exception as exc:
                run_result = {
                    "silver_result": {
                        "rows": [],
                        "columns": [],
                        "sub_table": pd.DataFrame(),
                        "status": "reject",
                        "reason": f"generation failed: {exc}",
                        "source": "generation_failed",
                    },
                    "col_prompt_source": "",
                    "row_prompt_source": "",
                    "col_prompt": "",
                    "row_prompt": "",
                    "col_candidates": [],
                    "row_candidates": [],
                    "parsed_col_candidates": [],
                    "parsed_row_candidates": [],
                    "aggregated_col_groups": [],
                    "aggregated_row_groups": [],
                    "reconciliation": {
                        "aggregation_mode": args.aggregation_mode,
                        "adaptive_union_mode": args.adaptive_union_mode,
                    },
                    "col_score_source": "none",
                    "row_score_source": "none",
                    "answerability": None,
                    "raw_answerability": "",
                }

            if full_table_fallback_enabled and run_result["silver_result"].get("status") != "accept":
                run_result["silver_result"] = _full_table_result(
                    table,
                    reason=f"fallback_full_table_after_failure: {run_result['silver_result'].get('reason')}",
                )

            silver = run_result["silver_result"]
            route_status = str(silver.get("source") or "")
            record = {
                "idx": i,
                "question": question,
                "gold_answers": gold_answers,
                "route_status": route_status,
                "method": METHOD_NAME,
                "model": resolved_model,
                "provider": provider,
                "prompt_mode": args.prompt_mode,
                "col_prompt_source": run_result["col_prompt_source"],
                "row_prompt_source": run_result["row_prompt_source"],
                "col_prompt": run_result["col_prompt"],
                "row_prompt": run_result["row_prompt"],
                "col_candidates": run_result["col_candidates"],
                "row_candidates": run_result["row_candidates"],
                "parsed_col_candidates": run_result["parsed_col_candidates"],
                "parsed_row_candidates": run_result["parsed_row_candidates"],
                "aggregated_col_groups": run_result["aggregated_col_groups"],
                "aggregated_row_groups": run_result["aggregated_row_groups"],
                "reconciliation": run_result["reconciliation"],
                "col_score_source": run_result["col_score_source"],
                "row_score_source": run_result["row_score_source"],
                "answerability": run_result.get("answerability"),
                "raw_answerability": run_result.get("raw_answerability", ""),
                "silver_result": silver,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "sampling_n": args.sampling_n,
                "max_new_tokens": args.max_new_tokens,
                "gpt5_completion_token_budget": args.gpt5_completion_token_budget,
                "col_num_rows": args.col_num_rows,
                "row_num_rows": args.row_num_rows,
                "n_shots": args.n_shots,
                "union_col": args.union_col,
                "union_row": args.union_row,
                "aggregation_mode": args.aggregation_mode,
                "adaptive_union_mode": args.adaptive_union_mode,
                "operation_guidance_mode": args.operation_guidance_mode,
                "stop": stop,
                "full_table_fallback_enabled": full_table_fallback_enabled,
                "uses_gold_answer_for_generation": False,
                "generation_inputs": ["question", "table_caption", "table_columns", "table_rows"],
            }
            results.append(record)
            done_indices.add(i)

            json_record = dict(record)
            json_record["silver_result"] = _result_for_json(silver)
            jsonl_file.write(json.dumps(json_record, ensure_ascii=False) + "\n")
            jsonl_file.flush()

            _log(
                f"[Item {local_idx}/{total_to_run}] status={silver.get('status')} route={route_status} "
                f"rows={len(silver.get('rows') or [])} cols={len(silver.get('columns') or [])}",
                args.quiet,
            )
            if silver.get("reason"):
                _log(f"[Item {local_idx}/{total_to_run}] reason={silver.get('reason')}", args.quiet)

            if len(results) % args.save_every == 0:
                with open(args.save_path, "wb") as f:
                    pickle.dump(results, f)
                accepted = sum(1 for x in results if x["silver_result"].get("status") == "accept")
                full = sum(1 for x in results if x["silver_result"].get("source") == "full_table_fallback")
                _log(f"[Save] processed={len(results)} accepted={accepted} full_table={full}", args.quiet)

    with open(args.save_path, "wb") as f:
        pickle.dump(results, f)
    accepted = sum(1 for x in results if x["silver_result"].get("status") == "accept")
    full = sum(1 for x in results if x["silver_result"].get("source") == "full_table_fallback")
    _log(
        f"[Done] processed={len(results)} accepted={accepted} full_table={full} jsonl_path={args.jsonl_path}",
        args.quiet,
    )


if __name__ == "__main__":
    main()

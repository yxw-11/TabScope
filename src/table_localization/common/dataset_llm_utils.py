#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import os
import pickle
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from table_localization.common.swift_infer_backend import (
    _is_llama33_model as _backend_is_llama33_model,
    _request_swift_chat,
)
from table_localization.retrieval.anchor_sub import AnchorSub


DEFAULT_MODEL = "gpt5-mini"
DEFAULT_BASE_URL = "https://api.openai.com/v1"
# DEFAULT_BASE_URL = "https://api.aaai.vip/v1"  # Using AAAVIP proxy for better performance in some regions
DEFAULT_GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
VALID_OPERATORS = {"==", "contains", ">", ">=", "<", "<=", "!="}
VALID_ROW_SCOPES = {"matched_rows", "candidate_domain", "all_rows", "entity_rows"}


def _get_api_key() -> str:
    raw = os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY") or ""
    return raw.strip()


def _get_gemini_api_key() -> str:
    raw = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or ""
    return raw.strip()


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
        "llama3.3": "meta-llama/Llama-3.3-70B-Instruct",
        "llama3.3-70b": "meta-llama/Llama-3.3-70B-Instruct",
        "llama3.3-70b-instruct": "meta-llama/Llama-3.3-70B-Instruct",
        "llama-3.3": "meta-llama/Llama-3.3-70B-Instruct",
        "llama-3.3-70b": "meta-llama/Llama-3.3-70B-Instruct",
        "llama-3.3-70b-instruct": "meta-llama/Llama-3.3-70B-Instruct",
        "meta-llama/llama-3.3-70b-instruct": "meta-llama/Llama-3.3-70B-Instruct",
        "meta-llama/llama-3.3-70b": "meta-llama/Llama-3.3-70B-Instruct",
    }
    return aliases.get(key, raw)


def _is_llama33_model(model: str) -> bool:
    return _backend_is_llama33_model(model)


def _is_gemini_model(model: str) -> bool:
    return _normalize_model(model).lower().startswith("gemini-")


def _is_gpt5_model(model: str) -> bool:
    return _normalize_model(model).lower().startswith("gpt-5")


def _ensure_dir(path: str) -> None:
    Path(path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)


def _log(message: str, quiet: bool = False) -> None:
    if not quiet:
        print(message, flush=True)


def _resolve_dataset_name(dataset_name: str) -> str:
    name = dataset_name.strip()
    if name.lower() in {"wikitq", "wikitablequestions", "wtq"}:
        return "Stanford/wikitablequestions"
    return name


def _is_slqa_dataset(dataset_name: str) -> bool:
    name = str(dataset_name or "").strip().lower()
    return name == "slqa" or name.endswith("/slqa") or name.endswith("slqa.jsonl")


def _find_local_dataset_file(dataset_name: str) -> Path:
    raw = Path(str(dataset_name).strip()).expanduser()
    if raw.exists():
        return raw.resolve()
    candidates = [
        PROJECT_ROOT / "data" / f"{str(dataset_name).strip()}.jsonl",
        PROJECT_ROOT / "data" / Path(str(dataset_name).strip()).name,
        ROOT / "datasets" / f"{str(dataset_name).strip()}.jsonl",
        ROOT / "datasets" / Path(str(dataset_name).strip()).name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"Could not resolve local dataset file for {dataset_name!r}.")


def _load_local_jsonl_dataset(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                record = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Failed to parse JSONL at {path}:{line_no}: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"Expected object at {path}:{line_no}, got {type(record).__name__}.")
            records.append(record)
    return records


def _load_dataset_split(dataset_name: str, split: str) -> Sequence[Dict[str, Any]]:
    resolved = _resolve_dataset_name(dataset_name)
    if _is_slqa_dataset(dataset_name) or _is_slqa_dataset(resolved):
        return _load_local_jsonl_dataset(_find_local_dataset_file(dataset_name))
    from datasets import load_dataset

    return load_dataset(resolved)[split]


def _slqa_item_to_wtq(item: Dict[str, Any], idx: Optional[int] = None) -> Dict[str, Any]:
    table_text = item.get("table_text") or []
    header = [str(x) for x in table_text[0]] if table_text else []
    rows = [[str(x) for x in row] for row in table_text[1:]] if len(table_text) > 1 else []
    answers = item.get("answer", [])
    if answers is None:
        answers = []
    if not isinstance(answers, list):
        answers = [answers]
    normalized_id = item.get("ids") or item.get("id") or (f"slqa_{idx}" if idx is not None else "slqa")
    return {
        **item,
        "id": str(normalized_id),
        "question": str(item.get("statement", item.get("question", ""))),
        "answers": [str(x) for x in answers],
        "table": {
            "header": header,
            "rows": rows,
            "name": str(normalized_id),
        },
    }


def _normalize_dataset_item(item: Dict[str, Any], dataset_name: str, idx: Optional[int] = None) -> Dict[str, Any]:
    if _is_slqa_dataset(dataset_name):
        return _slqa_item_to_wtq(item, idx=idx)
    return item


def _wtq_item_to_df(item: Dict[str, Any]) -> pd.DataFrame:
    table = item.get("table", {})
    if table:
        header = [str(x) for x in table.get("header", [])]
        rows = [[str(x) for x in row] for row in table.get("rows", [])]
        if header:
            return pd.DataFrame(rows, columns=header)
        return pd.DataFrame(rows)
    table_text = item.get("table_text") or []
    if table_text:
        header = [str(x) for x in table_text[0]]
        rows = [[str(x) for x in row] for row in table_text[1:]]
        if header:
            return pd.DataFrame(rows, columns=header)
        return pd.DataFrame(rows)
    return pd.DataFrame()


def _df_to_markdown(table: pd.DataFrame) -> str:
    rows = []
    headers = ["row_id"] + [str(col) for col in table.columns]
    rows.append("| " + " | ".join(headers) + " |")
    rows.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row_id, row in table.iterrows():
        cells = [str(row_id)] + [str(row[col]) for col in table.columns]
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join(rows)


def _build_anchor_prompt(question: str, table: pd.DataFrame, gold_answers: List[str], include_gold: bool) -> str:
    gold_block = json.dumps(gold_answers, ensure_ascii=False) if include_gold else "HIDDEN"
    return f"""You are generating evidence anchors for WikiTableQuestions.

Task:
Given a question and the full table, output an anchor_spec JSON object that can be grounded into a silver sub-table.
The local code will ground your JSON into real table rows and columns. Do not output SQL, DSL, code, markdown, or explanations.

Anchor spec schema:
{{
  "columns": list[str],
  "row_anchors": [
    {{"column": str | null, "value": any, "operator": "==" | "contains" | ">" | ">=" | "<" | "<=" | "!="}}
  ],
  "answer_values": list[any],
  "row_scope": "matched_rows" | "candidate_domain" | "all_rows" | "entity_rows",
  "entity_values": list[any],
  "entity_column": str | null,
  "requires_answer_cell": bool
}}

Guidelines:
- Choose columns that are necessary to answer the question or verify the evidence.
- Use row_anchors for filter conditions, comparisons, dates, entities, or other constraints in the question.
- Use "matched_rows" when anchors identify exactly the evidence rows.
- Use "candidate_domain" when the question asks for an argmax/argmin, comparison, ranking, count, or domain filtered by anchors.
- Use "all_rows" only when the whole table is the candidate domain.
- Use "entity_rows" when the question is about specific named entities.
- Put likely final answer cell values in answer_values. If gold answers are visible, use them only as answer/evidence cell anchors.
- Set requires_answer_cell to true when the sub-table should contain the answer cell; otherwise false.
- Column names and values must be copied from the table when possible.

Question:
{question}

Gold answers:
{gold_block}

Full table:
{_df_to_markdown(table)}

Return only valid JSON for the anchor_spec."""


def _request_openai_chat(
    prompt: str,
    api_key: str,
    model: str,
    base_url: str,
    max_tokens: int,
    temperature: float,
    timeout: int,
    gpt5_completion_token_budget: Optional[int] = None,
    retries: int = 2,
) -> str:
    url = f"{base_url.rstrip('/')}/chat/completions"
    resolved_model = _normalize_model(model)
    completion_budget = max_tokens
    if _is_gpt5_model(model) and gpt5_completion_token_budget is not None:
        completion_budget = int(gpt5_completion_token_budget)
    payload = {
        "model": resolved_model,
        "messages": [
            {"role": "system", "content": "You output valid JSON only."},
            {"role": "user", "content": prompt},
        ],
        "max_completion_tokens": completion_budget,
    }
    if not _is_gpt5_model(model):
        payload["temperature"] = temperature
        payload["top_p"] = 1
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    last_error: Optional[Exception] = None
    for attempt in range(retries + 1):
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=timeout)
            if response.ok:
                data = response.json()
                choices = data.get("choices") or []
                if not choices:
                    raise RuntimeError(f"OpenAI returned no choices. usage={data.get('usage')}")
                choice = choices[0]
                message = choice.get("message") or {}
                content = message.get("content")
                if isinstance(content, list):
                    content = "".join(
                        str(part.get("text", "")) if isinstance(part, dict) else str(part)
                        for part in content
                    )
                content = str(content or "").strip()
                if content:
                    return content
                raise RuntimeError(
                    "OpenAI returned empty message content. "
                    f"finish_reason={choice.get('finish_reason')} "
                    f"usage={data.get('usage')} "
                    f"api_max_completion_tokens={completion_budget}"
                )
            last_error = RuntimeError(
                f"OpenAI request failed: {response.status_code} {response.text[:1000]}"
            )
        except Exception as exc:
            last_error = exc

        if attempt < retries:
            time.sleep(2 * (attempt + 1))

    if last_error is not None:
        raise last_error
    return ""


def _request_gemini_generate_content(
    prompt: str,
    api_key: str,
    model: str,
    base_url: str,
    max_tokens: int,
    temperature: float,
    timeout: int,
    retries: int = 2,
) -> str:
    resolved_model = _normalize_model(model)
    url = f"{base_url.rstrip('/')}/models/{resolved_model}:generateContent"
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": prompt}],
            }
        ],
        "generationConfig": {
            "temperature": temperature,
            "topP": 1,
            "maxOutputTokens": max_tokens,
            "responseMimeType": "application/json",
        },
        "systemInstruction": {
            "parts": [{"text": "You output valid JSON only."}],
        },
    }
    headers = {
        "x-goog-api-key": api_key,
        "Content-Type": "application/json",
    }

    last_error: Optional[Exception] = None
    for attempt in range(retries + 1):
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=timeout)
            if response.ok:
                data = response.json()
                parts = data["candidates"][0]["content"].get("parts", [])
                return "".join(str(part.get("text", "")) for part in parts)
            last_error = RuntimeError(
                f"Gemini request failed: {response.status_code} {response.text[:1000]}"
            )
        except Exception as exc:
            last_error = exc

        if attempt < retries:
            time.sleep(2 * (attempt + 1))

    if last_error is not None:
        raise last_error
    return ""


def _request_llm(
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
    if _is_llama33_model(model):
        return _request_swift_chat(
            prompt=prompt,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
        )
    if _is_gemini_model(model):
        if not gemini_api_key:
            raise RuntimeError("GEMINI_API_KEY or GOOGLE_API_KEY is required for Gemini models.")
        return _request_gemini_generate_content(
            prompt=prompt,
            api_key=gemini_api_key,
            model=model,
            base_url=gemini_base_url,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout=timeout,
        )

    if not openai_api_key:
        raise RuntimeError("API_KEY or OPENAI_API_KEY is required for OpenAI models.")
    return _request_openai_chat(
        prompt=prompt,
        api_key=openai_api_key,
        model=model,
        base_url=base_url,
        max_tokens=max_tokens,
        temperature=temperature,
        timeout=timeout,
        gpt5_completion_token_budget=gpt5_completion_token_budget,
    )


def _extract_json_object(text: str) -> Dict[str, Any]:
    cleaned = str(text or "").strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        obj = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            raise
        obj = json.loads(match.group(0))
    if not isinstance(obj, dict):
        raise ValueError("anchor_spec must be a JSON object")
    return obj


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, (tuple, set)):
        return list(value)
    return [value]


def _sanitize_anchor_spec(spec: Dict[str, Any]) -> Dict[str, Any]:
    row_anchors = []
    for anchor in _as_list(spec.get("row_anchors", [])):
        if not isinstance(anchor, dict):
            continue
        operator = str(anchor.get("operator", "==") or "==")
        if operator not in VALID_OPERATORS:
            operator = "=="
        row_anchors.append(
            {
                "column": anchor.get("column", None),
                "value": anchor.get("value", None),
                "operator": operator,
            }
        )

    row_scope = str(spec.get("row_scope", "matched_rows") or "matched_rows")
    if row_scope not in VALID_ROW_SCOPES:
        row_scope = "matched_rows"

    return {
        "columns": [str(x) for x in _as_list(spec.get("columns", [])) if x is not None],
        "row_anchors": row_anchors,
        "answer_values": _as_list(spec.get("answer_values", [])),
        "row_scope": row_scope,
        "entity_values": _as_list(spec.get("entity_values", [])),
        "entity_column": spec.get("entity_column", None),
        "requires_answer_cell": bool(spec.get("requires_answer_cell", True)),
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
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate WikiTQ sub-tables with GPT anchor specs and AnchorSub grounding.")
    parser.add_argument("--dataset", type=str, default="wikitq")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--base-url", type=str, default=DEFAULT_BASE_URL)
    parser.add_argument("--gemini-base-url", type=str, default=DEFAULT_GEMINI_BASE_URL)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--max-new-tokens", type=int, default=700)
    parser.add_argument(
        "--gpt5-completion-token-budget",
        type=int,
        default=4096,
        help="Completion-token budget for GPT-5 models; includes hidden reasoning tokens.",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--start-idx", type=int, default=0)
    parser.add_argument("--max-items", type=int, default=-1)
    parser.add_argument("--save-every", type=int, default=20)
    parser.add_argument("--save-path", type=str, default="results/subtables/gpt_anchor_sub_wikitq.pkl")
    parser.add_argument("--jsonl-path", type=str, default="results/subtables/gpt_anchor_sub_wikitq.jsonl")
    parser.add_argument("--api-key", type=str, default="")
    parser.add_argument("--hide-gold-from-llm", action="store_true")
    parser.add_argument("--append-jsonl", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    openai_api_key = (args.api_key or _get_api_key()).strip()
    gemini_api_key = (args.api_key or _get_gemini_api_key()).strip()

    dataset_name = _resolve_dataset_name(args.dataset)
    resolved_model = _normalize_model(args.model)
    provider = "swift" if _is_llama33_model(args.model) else ("gemini" if _is_gemini_model(args.model) else "openai")
    gpt5_budget = args.gpt5_completion_token_budget if _is_gpt5_model(args.model) else None
    _log(f"[Init] dataset={dataset_name} split={args.split}", args.quiet)
    _log(f"[Init] provider={provider} model={resolved_model}", args.quiet)
    if gpt5_budget is not None:
        _log(f"[Init] gpt5_completion_token_budget={gpt5_budget}", args.quiet)
    _log("[Init] loading dataset...", args.quiet)
    dataset = _load_dataset_split(args.dataset, args.split)
    end_idx = len(dataset)
    if args.max_items > 0:
        end_idx = min(end_idx, args.start_idx + args.max_items)
    total_to_run = max(0, end_idx - args.start_idx)
    _log(f"[Init] loaded {len(dataset)} examples; running {total_to_run} examples from idx={args.start_idx}", args.quiet)

    _ensure_dir(args.save_path)
    _ensure_dir(args.jsonl_path)
    _log(f"[Init] save_path={args.save_path}", args.quiet)
    _log(f"[Init] jsonl_path={args.jsonl_path}", args.quiet)

    anchor_sub = AnchorSub()
    results: List[Dict[str, Any]] = []
    jsonl_mode = "a" if args.append_jsonl else "w"

    with open(args.jsonl_path, jsonl_mode, encoding="utf-8") as jsonl_file:
        for i in range(args.start_idx, end_idx):
            item = _normalize_dataset_item(dict(dataset[i]), args.dataset, idx=i)
            question = str(item["question"])
            gold_answers = [str(x) for x in item.get("answers", [])]
            table = _wtq_item_to_df(item)
            local_idx = len(results) + 1
            _log(
                f"[Item {local_idx}/{total_to_run}] idx={i} rows={table.shape[0]} cols={table.shape[1]} question={question[:120]}",
                args.quiet,
            )

            prompt = _build_anchor_prompt(
                question=question,
                table=table,
                gold_answers=gold_answers,
                include_gold=not args.hide_gold_from_llm,
            )
            _log(f"[Item {local_idx}/{total_to_run}] requesting {provider}...", args.quiet)
            raw_output = _request_llm(
                prompt=prompt,
                openai_api_key=openai_api_key,
                gemini_api_key=gemini_api_key,
                model=args.model,
                base_url=args.base_url,
                gemini_base_url=args.gemini_base_url,
                max_tokens=args.max_new_tokens,
                temperature=args.temperature,
                timeout=args.timeout,
                gpt5_completion_token_budget=gpt5_budget,
            )

            try:
                anchor_spec = _sanitize_anchor_spec(_extract_json_object(raw_output))
                silver_result = anchor_sub.generate(
                    question=question,
                    table=table,
                    gold_answer=gold_answers,
                    anchor_spec=anchor_spec,
                )
                parse_error = None
            except Exception as exc:
                anchor_spec = {}
                silver_result = {
                    "rows": [],
                    "columns": [],
                    "sub_table": pd.DataFrame(),
                    "status": "reject",
                    "reason": f"anchor generation failed: {exc}",
                }
                parse_error = str(exc)
                _log(f"[Item {local_idx}/{total_to_run}] parse/grounding error: {parse_error}", args.quiet)

            record = {
                "idx": i,
                "question": question,
                "gold_answers": gold_answers,
                "raw_output": raw_output,
                "anchor_spec": anchor_spec,
                "silver_result": silver_result,
                "parse_error": parse_error,
            }
            results.append(record)

            json_record = dict(record)
            json_record["silver_result"] = _result_for_json(silver_result)
            jsonl_file.write(json.dumps(json_record, ensure_ascii=False) + "\n")
            jsonl_file.flush()
            status = silver_result.get("status")
            rows = len(silver_result.get("rows", []) or [])
            columns = len(silver_result.get("columns", []) or [])
            reason = silver_result.get("reason")
            if reason:
                _log(f"[Item {local_idx}/{total_to_run}] status={status} rows={rows} cols={columns} reason={reason}", args.quiet)
            else:
                _log(f"[Item {local_idx}/{total_to_run}] status={status} rows={rows} cols={columns}", args.quiet)

            if len(results) % args.save_every == 0:
                with open(args.save_path, "wb") as f:
                    pickle.dump(results, f)
                accepted = sum(1 for x in results if x["silver_result"].get("status") == "accept")
                _log(f"[Save] processed={len(results)} accepted={accepted}", args.quiet)

    with open(args.save_path, "wb") as f:
        pickle.dump(results, f)
    accepted = sum(1 for x in results if x["silver_result"].get("status") == "accept")
    _log(f"[Done] processed={len(results)} accepted={accepted} save_path={args.save_path}", args.quiet)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import pickle
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from table_localization.common.dataset_llm_utils import (  # noqa: E402
    DEFAULT_BASE_URL,
    _ensure_dir,
    _get_api_key,
    _is_gpt5_model,
    _load_dataset_split,
    _normalize_dataset_item,
    _normalize_model,
)
from table_localization.evaluate.evaluator import evaluate_prediction  # noqa: E402
from table_localization.common.swift_infer_backend import (  # noqa: E402
    _clear_cuda_memory,
    _clear_swift_engine,
    _is_llama33_model as _backend_is_llama33_model,
    _request_swift_chat,
)


COT_PROMPT = """You are an advanced AI capable of analyzing and understanding information within tables. Read the table below.

{table}

Based on the given table, answer the following question:

{question}

Let's think step by step, and then answer the question.
"""

STRICT_RULES = """Rules (strict):
OUTPUT FORMAT: exactly two sections:
   <think>
   ...
   </think>
   <answer>...</answer>

Here is an example:

Title: opponent_info

Question: who was the opponent in the first game of the season?

Table Content:

| row_id | Date           | Opponent         |
|--------|----------------|------------------|
| 1      | 15 August 1987 | Derby County     |
| 2      | 18 August 1987 | Coventry City    |
| 3      | 22 August 1987 | West Ham United  |
| 4      | 29 August 1987 | Chelsea          |
| 5      | 31 August 1987 | Arsenal          |


<think>
The table lists games in chronological order. The first game is the first row, with date 15 August 1987.
The opponent in that row is Derby County.
Therefore, the opponent in the first game of the season was Derby County.
</think>
<answer>Derby County</answer>
"""


def _log(message: str, quiet: bool = False) -> None:
    if not quiet:
        print(message, flush=True)


def _load_jsonl(path: Path) -> Dict[int, Dict[str, Any]]:
    out: Dict[int, Dict[str, Any]] = {}
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                print(
                    f"[Warn] Bad JSONL at {path}:{line_no}: {exc}. "
                    f"Skipping this line and continuing with later valid records.",
                    flush=True,
                )
                continue
            if "idx" not in obj:
                raise ValueError(f"Missing idx at {path}:{line_no}")
            out[int(obj["idx"])] = obj
    return out


def _extract_answer(text: str) -> str:
    text = str(text or "")
    match = re.search(r"<answer>(.*?)</answer>", text, flags=re.IGNORECASE | re.DOTALL)
    if match:
        return _clean_answer(match.group(1))
    match = re.search(r"Answer\s*:\s*(.*)", text, flags=re.IGNORECASE)
    if match:
        return _clean_answer(match.group(1))
    return ""


def _clean_answer(answer: str) -> str:
    answer = str(answer or "").strip()
    answer = re.sub(r"^\*+", "", answer)
    answer = re.sub(r"\*+$", "", answer)
    answer = re.sub(r"\s+", " ", answer)
    return answer.strip()


def _needs_full_table_fallback(text: str) -> bool:
    lowered = str(text or "").lower()
    signals = [
        "not enough information",
        "insufficient information",
        "cannot answer from the table",
        "cannot be answered from the table",
        "cannot be determined",
        "not possible to determine",
        "need more information",
        "missing information",
        "not enough data",
        "answer cannot be determined",
        "table does not contain",
        "not present in the table",
        "not available in the table",
        "does not provide",
        "no information",
    ]
    return any(signal in lowered for signal in signals)


def _is_oom_error(exc: Exception) -> bool:
    text = str(exc or "").lower()
    return "out of memory" in text or "cuda oom" in text or "cuda error: out of memory" in text


def _is_empty_length_error(exc: Exception) -> bool:
    text = str(exc or "")
    lowered = text.lower()
    if "openai returned empty message" not in lowered:
        return False
    return "finish_reason': 'length'" in text or 'finish_reason": "length"' in text


def _markdown_escape(value: Any) -> str:
    text = "" if value is None else str(value)
    text = text.replace("\n", " ").replace("\r", " ")
    text = text.replace("|", "\\|")
    return text.strip()


def _records_to_markdown(records: Sequence[Dict[str, Any]], columns: Sequence[str]) -> str:
    headers = ["row_id"] + [str(col) for col in columns]
    lines = ["| " + " | ".join(_markdown_escape(x) for x in headers) + " |"]
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for rec in records:
        cells = [_markdown_escape(rec.get("row_id", ""))]
        cells.extend(_markdown_escape(rec.get(col, "")) for col in columns)
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _full_table_markdown(item: Dict[str, Any]) -> str:
    table = item.get("table", {}) or {}
    headers = [str(x) for x in table.get("header", [])]
    rows = table.get("rows", []) or []
    records: List[Dict[str, Any]] = []
    for i, row in enumerate(rows):
        rec = {"row_id": i}
        for col, value in zip(headers, row):
            rec[col] = value
        records.append(rec)
    return _records_to_markdown(records, headers)


def _ground_rows_columns_from_item(item: Dict[str, Any], rows: Sequence[Any], columns: Sequence[str]) -> List[Dict[str, Any]]:
    table = item.get("table", {}) or {}
    headers = [str(x) for x in table.get("header", [])]
    raw_rows = table.get("rows", []) or []
    if not headers or not raw_rows:
        return []

    valid_cols: List[str] = [str(col) for col in columns if str(col) in headers]
    if not valid_cols:
        return []

    col_to_idx = {name: idx for idx, name in enumerate(headers)}
    grounded: List[Dict[str, Any]] = []
    for row_value in rows:
        try:
            row_id = int(row_value)
        except (TypeError, ValueError):
            continue
        if not (0 <= row_id < len(raw_rows)):
            continue
        src_row = raw_rows[row_id]
        rec: Dict[str, Any] = {"row_id": row_id}
        for col in valid_cols:
            col_idx = col_to_idx[col]
            rec[col] = src_row[col_idx] if col_idx < len(src_row) else ""
        grounded.append(rec)
    return grounded


def _subtable_markdown(
    record: Dict[str, Any],
    item: Dict[str, Any],
    fallback_full_table: bool,
) -> Tuple[str, str, List[int], List[str]]:
    silver = (record or {}).get("silver_result") or {}
    columns = [str(x) for x in (silver.get("columns") or [])]
    rows = list(silver.get("rows") or [])
    records = silver.get("sub_table_records") or []
    if records and columns:
        return _records_to_markdown(records, columns), "subtable_records", rows, columns
    if rows and columns:
        grounded_records = _ground_rows_columns_from_item(item, rows, columns)
        if grounded_records:
            return _records_to_markdown(grounded_records, columns), "grounded_rows_columns", rows, columns
    markdown = str(silver.get("sub_table_markdown") or "").strip()
    if markdown:
        return markdown, "subtable_markdown", rows, columns
    if fallback_full_table:
        return _full_table_markdown(item), "full_table_fallback", rows, columns
    return "| row_id | EMPTY |\n| --- | --- |", "empty_subtable", rows, columns


def _build_prompt(question: str, table_md: str, include_rules: bool) -> str:
    if include_rules:
        return (
            STRICT_RULES.rstrip()
            + "\n\nNow solve the actual target question below. Do not answer the example.\n\n"
            + COT_PROMPT.format(table=table_md, question=question).rstrip()
        )
    return COT_PROMPT.format(table=table_md, question=question)


def _request_openai_chat(
    prompt: str,
    api_key: str,
    model: str,
    base_url: str,
    max_tokens: int,
    temperature: float,
    timeout: int,
    gpt5_completion_token_budget: Optional[int],
    retries: int = 2,
) -> str:
    if _backend_is_llama33_model(model):
        return _request_swift_chat(prompt=prompt, model=model, max_tokens=max_tokens, temperature=temperature)
    resolved_model = _normalize_model(model)
    payload: Dict[str, Any] = {
        "model": resolved_model,
        "messages": [
            {"role": "system", "content": "You are a precise TableQA assistant. Follow the required XML-like output format exactly."},
            {"role": "user", "content": prompt},
        ],
        "max_completion_tokens": int(gpt5_completion_token_budget or max_tokens),
    }
    if not _is_gpt5_model(model):
        payload["temperature"] = temperature
        payload["top_p"] = 1
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    url = f"{base_url.rstrip('/')}/chat/completions"
    last_error: Optional[Exception] = None
    for attempt in range(retries + 1):
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=timeout)
            if response.ok:
                data = response.json()
                choices = data.get("choices") or []
                if not choices:
                    raise RuntimeError(f"OpenAI returned no choices: usage={data.get('usage')}")
                message = choices[0].get("message") or {}
                content = message.get("content")
                if isinstance(content, list):
                    content = "".join(str(part.get("text", "")) if isinstance(part, dict) else str(part) for part in content)
                content = str(content or "").strip()
                if content:
                    return content
                raise RuntimeError(f"OpenAI returned empty message. choice={choices[0]} usage={data.get('usage')}")
            last_error = RuntimeError(f"OpenAI request failed: {response.status_code} {response.text[:1000]}")
        except Exception as exc:
            last_error = exc
        if attempt < retries:
            time.sleep(2 * (attempt + 1))
    if last_error:
        raise last_error
    return ""


def _request_with_oom_guard(
    prompt: str,
    api_key: str,
    model: str,
    base_url: str,
    max_tokens: int,
    temperature: float,
    timeout: int,
    gpt5_completion_token_budget: Optional[int],
    quiet: bool,
    idx: int,
    stage: str,
) -> str:
    token_schedule: List[int] = []
    for candidate in [max_tokens, min(max_tokens, 384), min(max_tokens, 256), min(max_tokens, 192), 128]:
        candidate = max(64, int(candidate))
        if candidate not in token_schedule:
            token_schedule.append(candidate)

    last_error: Optional[Exception] = None
    for attempt_no, current_tokens in enumerate(token_schedule, start=1):
        try:
            if attempt_no > 1:
                _log(
                    f"[OOM Retry] idx={idx} stage={stage} retry={attempt_no}/{len(token_schedule)} max_tokens={current_tokens}",
                    quiet,
                )
            text = _request_openai_chat(
                prompt=prompt,
                api_key=api_key,
                model=model,
                base_url=base_url,
                max_tokens=current_tokens,
                temperature=temperature,
                timeout=timeout,
                gpt5_completion_token_budget=gpt5_completion_token_budget,
            )
            _clear_cuda_memory()
            return text
        except Exception as exc:
            last_error = exc
            if _backend_is_llama33_model(model) and _is_oom_error(exc):
                _log(
                    f"[OOM Guard] idx={idx} stage={stage} max_tokens={current_tokens} error={exc}",
                    quiet,
                )
                _clear_cuda_memory()
                time.sleep(2)
                continue
            raise

    if last_error is not None:
        raise last_error
    return ""


def _load_existing_results(path: Path) -> Dict[int, Dict[str, Any]]:
    if not path.exists():
        return {}
    out: Dict[int, Dict[str, Any]] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            out[int(obj["idx"])] = obj
    return out


def _accuracy(records: Sequence[Dict[str, Any]]) -> Tuple[int, int, float, int]:
    counted = [x for x in records if not x.get("skip_for_acc")]
    skipped = len(records) - len(counted)
    n = len(counted)
    correct = sum(1 for x in counted if x.get("correct"))
    return correct, n, correct / n if n else 0.0, skipped


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one shared COT QA pipeline on generated WikiTQ/SLQA sub-tables.")
    parser.add_argument("--dataset", type=str, default="wikitq")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--subtable-jsonl", required=True)
    parser.add_argument("--method-name", default="subtable_method")
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--output-pkl", default="")
    parser.add_argument("--model", type=str, default="llama3.3")
    parser.add_argument("--base-url", type=str, default=DEFAULT_BASE_URL)
    parser.add_argument("--api-key", type=str, default="")
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--gpt5-completion-token-budget", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--start-idx", type=int, default=0)
    parser.add_argument("--max-items", type=int, default=-1)
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--acc-log-path", type=str, default="", help="Optional checkpoint accuracy log JSONL. Defaults to a sibling *_acc.jsonl file.")
    parser.add_argument("--include-rules", action="store_true", help="Append the strict rules/example block after the user template.")
    parser.add_argument("--fallback-full-table", action="store_true", help="Use full table if a sub-table record lacks markdown/records.")
    parser.add_argument("--fallback-on-missing-info", action="store_true", help="If the first sub-table answer suggests missing information, retry once with the full table.")
    parser.add_argument("--skip-on-oom", action="store_true", help="If all guarded retries OOM, record the item but exclude it from accuracy.")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    api_key = (args.api_key or _get_api_key()).strip()
    if not api_key and not _backend_is_llama33_model(args.model):
        raise RuntimeError("API_KEY or OPENAI_API_KEY is required.")

    dataset = _load_dataset_split(args.dataset, args.split)
    predictions = _load_jsonl(Path(args.subtable_jsonl))
    existing = _load_existing_results(Path(args.output_jsonl)) if args.resume else {}
    done = set(existing)

    indices = sorted(idx for idx in predictions if idx >= args.start_idx and idx < len(dataset))
    if args.max_items > 0:
        indices = indices[: args.max_items]
    total = len(indices)

    _ensure_dir(args.output_jsonl)
    if args.output_pkl:
        _ensure_dir(args.output_pkl)
    acc_log_path = Path(args.acc_log_path) if args.acc_log_path else Path(args.output_jsonl).with_name(Path(args.output_jsonl).stem + "_acc.jsonl")
    _ensure_dir(str(acc_log_path))
    mode = "a" if args.resume and Path(args.output_jsonl).exists() else "w"
    acc_mode = "a" if args.resume and acc_log_path.exists() else "w"
    all_records = [existing[idx] for idx in sorted(existing)] if existing else []

    _log(f"[Init] method={args.method_name} model={_normalize_model(args.model)}", args.quiet)
    _log(f"[Init] subtable_jsonl={args.subtable_jsonl}", args.quiet)
    _log(f"[Init] output_jsonl={args.output_jsonl}", args.quiet)
    _log(f"[Init] acc_log_path={acc_log_path}", args.quiet)
    _log(f"[Init] total_indices={total} already_done={len(done)}", args.quiet)

    with open(args.output_jsonl, mode, encoding="utf-8") as out, open(acc_log_path, acc_mode, encoding="utf-8") as acc_out:
        for pos, idx in enumerate(indices, start=1):
            if idx in done:
                continue
            item = _normalize_dataset_item(dict(dataset[idx]), args.dataset, idx=idx)
            question = str(item["question"])
            gold_answers = [str(x) for x in item.get("answers", [])]
            sub_record = predictions[idx]
            table_md, table_source, rows, columns = _subtable_markdown(sub_record, item, args.fallback_full_table)
            prompt = _build_prompt(question=question, table_md=table_md, include_rules=args.include_rules)

            _log(f"[Item {pos}/{total}] idx={idx} method={args.method_name} rows={len(rows)} cols={len(columns)} source={table_source}", args.quiet)
            raw = ""
            answer = ""
            error = None
            fallback_used = False
            skip_for_acc = False
            skip_reason = ""
            try:
                raw = _request_with_oom_guard(
                    prompt=prompt,
                    api_key=api_key,
                    model=args.model,
                    base_url=args.base_url,
                    max_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    timeout=args.timeout,
                    gpt5_completion_token_budget=args.gpt5_completion_token_budget if _is_gpt5_model(args.model) else None,
                    quiet=args.quiet,
                    idx=idx,
                    stage="subtable",
                )
                answer = _extract_answer(raw)
                if args.fallback_on_missing_info and _needs_full_table_fallback(raw):
                    fallback_used = True
                    full_table_md = _full_table_markdown(item)
                    full_prompt = _build_prompt(question=question, table_md=full_table_md, include_rules=args.include_rules)
                    _log(f"[Fallback] idx={idx} retrying with full table due to missing-info signal", args.quiet)
                    raw = _request_with_oom_guard(
                        prompt=full_prompt,
                        api_key=api_key,
                        model=args.model,
                        base_url=args.base_url,
                        max_tokens=args.max_new_tokens,
                        temperature=args.temperature,
                        timeout=args.timeout,
                        gpt5_completion_token_budget=args.gpt5_completion_token_budget if _is_gpt5_model(args.model) else None,
                        quiet=args.quiet,
                        idx=idx,
                        stage="full_table_fallback",
                    )
                    answer = _extract_answer(raw)
            except Exception as exc:
                error = str(exc)
                if args.skip_on_oom and _backend_is_llama33_model(args.model) and _is_oom_error(exc):
                    skip_for_acc = True
                    skip_reason = "oom"
                    raw = f"OOM_SKIPPED: {exc}"
                    answer = ""
                    _log(f"[Skip OOM] idx={idx} excluded from accuracy", args.quiet)
                elif _is_empty_length_error(exc):
                    skip_for_acc = True
                    skip_reason = "empty_length"
                    raw = f"EMPTY_RESPONSE_SKIPPED: {exc}"
                    answer = ""
                    _log(f"[Skip Empty] idx={idx} excluded from accuracy due to empty length-limited response", args.quiet)
                else:
                    _log(f"[Fatal] idx={idx} error={exc}", args.quiet)
                    _log(traceback.format_exc(), args.quiet)
                    raise
            finally:
                if _backend_is_llama33_model(args.model):
                    _clear_cuda_memory()

            gold = ",".join(gold_answers)
            correct = False if skip_for_acc else evaluate_prediction(gold, answer)
            result = {
                "idx": idx,
                "method": args.method_name,
                "model": _normalize_model(args.model),
                "question": question,
                "gold_answers": gold_answers,
                "prediction": answer,
                "raw_response": raw,
                "correct": correct,
                "error": error,
                "skip_for_acc": skip_for_acc,
                "skip_reason": skip_reason,
                "fallback_used": fallback_used,
                "table_source": "full_table_fallback_missing_info" if fallback_used else table_source,
                "subtable_shape": [len(rows), len(columns)],
                "rows": rows,
                "columns": columns,
            }
            out.write(json.dumps(result, ensure_ascii=False) + "\n")
            out.flush()
            all_records.append(result)

            correct_n, counted_n, acc, skipped_n = _accuracy(all_records)
            _log(
                f"[Item {pos}/{total}] idx={idx} pred={answer} gold={gold_answers} correct={correct} fallback={fallback_used} skip={skip_for_acc}",
                args.quiet,
            )
            if pos % args.save_every == 0 or pos == total:
                if args.output_pkl:
                    with open(args.output_pkl, "wb") as f:
                        pickle.dump(all_records, f)
                acc_rec = {
                    "seen": len(all_records),
                    "counted": counted_n,
                    "correct": correct_n,
                    "skipped": skipped_n,
                    "acc": acc,
                    "last_idx": idx,
                }
                acc_out.write(json.dumps(acc_rec, ensure_ascii=False) + "\n")
                acc_out.flush()
                _log(
                    f"[ACC] correct={correct_n}/{counted_n} acc={acc:.6f} skipped={skipped_n}",
                    args.quiet,
                )

    if args.output_pkl:
        with open(args.output_pkl, "wb") as f:
            pickle.dump(all_records, f)


if __name__ == "__main__":
    main()

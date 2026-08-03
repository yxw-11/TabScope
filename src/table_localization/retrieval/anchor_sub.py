from __future__ import annotations

import math
import re
from collections.abc import Sequence
from numbers import Number
from typing import Any, List, Optional, Set, Tuple

import pandas as pd


_NUMERIC_RE = re.compile(r"^[+-]?(?:\d+(?:\.\d+)?|\.\d+)$")
_GROUPED_NUMERIC_RE = re.compile(r"^[+-]?\d{1,3}(?:,\d{3})+(?:\.\d+)?$")


def _is_missing(value: Any) -> bool:
    try:
        return bool(pd.isna(value))
    except Exception:
        return value is None


def _collapse_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _strip_grouping_commas(text: str) -> str:
    def _replace(match: re.Match[str]) -> str:
        return match.group(0).replace(",", "")

    return _GROUPED_NUMERIC_RE.sub(_replace, text)


def _normalize_text(text: str) -> str:
    text = str(text).strip().lower()
    text = text.replace("\u00a0", " ")
    text = _strip_grouping_commas(text)
    text = re.sub(r"\s+", " ", text)
    cleaned = []
    for ch in text:
        if ch.isalnum() or ch in {".", "%", "-", " "}:
            cleaned.append(ch)
    text = "".join(cleaned)
    text = _collapse_spaces(text)
    if text in {"true", "yes"}:
        return "yes"
    if text in {"false", "no"}:
        return "no"
    return text


def _coerce_number(value: Any) -> Optional[float]:
    if _is_missing(value) or isinstance(value, bool):
        return None
    if isinstance(value, Number):
        amount = float(value)
        if math.isnan(amount) or math.isinf(amount):
            return None
        return amount
    text = _normalize_text(value)
    if not text or "%" in text:
        return None
    if _NUMERIC_RE.fullmatch(text):
        try:
            amount = float(text)
        except Exception:
            return None
        if math.isnan(amount) or math.isinf(amount):
            return None
        return amount
    return None


def normalize_value(value: Any) -> Any:
    if _is_missing(value):
        return ""
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, Number):
        amount = float(value)
        if math.isnan(amount) or math.isinf(amount):
            return ""
        if abs(amount - round(amount)) <= 1e-6:
            return int(round(amount))
        return amount
    if isinstance(value, str):
        text = _normalize_text(value)
        if not text:
            return ""
        if text in {"yes", "no"}:
            return text
        numeric = _coerce_number(text)
        if numeric is not None:
            if abs(numeric - round(numeric)) <= 1e-6:
                return int(round(numeric))
            return numeric
        return text
    if isinstance(value, (list, tuple, set)):
        return [normalize_value(x) for x in value]
    return normalize_value(str(value))


def _canonical_text(value: Any) -> str:
    norm = normalize_value(value)
    if isinstance(norm, list):
        parts = [_canonical_text(x) for x in norm]
        return _collapse_spaces(", ".join(part for part in parts if part))
    if isinstance(norm, (int, float)) and not isinstance(norm, bool):
        if isinstance(norm, int) or abs(float(norm) - round(float(norm))) <= 1e-6:
            return str(int(round(float(norm))))
        return ("%f" % float(norm)).rstrip("0").rstrip(".")
    return str(norm)


def _tokenize(text: Any) -> Set[str]:
    return {token for token in _canonical_text(text).split() if token}


def _expand_atomic_values(value: Any) -> List[Any]:
    if _is_missing(value):
        return []
    if isinstance(value, (list, tuple, set)):
        out: List[Any] = []
        for item in value:
            out.extend(_expand_atomic_values(item))
        return out
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        normalized = _normalize_text(stripped)
        if "," in stripped and not _GROUPED_NUMERIC_RE.fullmatch(normalized):
            return [part.strip() for part in stripped.split(",") if part.strip()]
        return [stripped]
    return [value]


def _ordered_union(*groups: Sequence[Any]) -> List[Any]:
    seen: Set[Any] = set()
    out: List[Any] = []
    for group in groups:
        for item in group:
            if item not in seen:
                out.append(item)
                seen.add(item)
    return out


def _compare_cells(cell_value: Any, target_value: Any, operator: str) -> bool:
    cell_text = _canonical_text(cell_value)
    target_text = _canonical_text(target_value)
    cell_num = _coerce_number(cell_value)
    target_num = _coerce_number(target_value)

    if operator == "contains":
        return bool(target_text) and target_text in cell_text

    if operator == "==":
        if cell_num is not None and target_num is not None:
            return abs(cell_num - target_num) <= 1e-6
        return cell_text == target_text

    if operator == "!=":
        return not _compare_cells(cell_value, target_value, "==")

    if cell_num is not None and target_num is not None:
        if operator == ">":
            return cell_num > target_num + 1e-6
        if operator == ">=":
            return cell_num > target_num - 1e-6 or abs(cell_num - target_num) <= 1e-6
        if operator == "<":
            return cell_num < target_num - 1e-6
        if operator == "<=":
            return cell_num < target_num + 1e-6 or abs(cell_num - target_num) <= 1e-6

    if operator == ">":
        return cell_text > target_text
    if operator == ">=":
        return cell_text >= target_text
    if operator == "<":
        return cell_text < target_text
    if operator == "<=":
        return cell_text <= target_text
    return False


def _match_column_name(requested: Any, actual_columns: Sequence[Any]) -> Optional[Any]:
    if requested is None:
        return None
    request_text = _canonical_text(requested)
    if not request_text:
        return None

    pairs = [(col, _canonical_text(col)) for col in actual_columns]

    for col, col_text in pairs:
        if col_text == request_text:
            return col

    for col, col_text in pairs:
        if request_text in col_text or col_text in request_text:
            return col

    request_tokens = _tokenize(request_text)
    best_col = None
    best_score = 0.0
    for col, col_text in pairs:
        col_tokens = _tokenize(col_text)
        if not request_tokens or not col_tokens:
            continue
        overlap = len(request_tokens & col_tokens)
        if overlap <= 0:
            continue
        score = overlap / float(max(len(request_tokens), len(col_tokens)))
        if score > best_score:
            best_score = score
            best_col = col
    return best_col


def _ground_columns(requested_columns: Sequence[Any], actual_columns: Sequence[Any]) -> List[Any]:
    grounded: List[Any] = []
    seen: Set[Any] = set()
    for requested in requested_columns:
        matched = _match_column_name(requested, actual_columns)
        if matched is not None and matched not in seen:
            grounded.append(matched)
            seen.add(matched)
    return grounded


def _search_rows_for_value(
    table: pd.DataFrame,
    value: Any,
    operator: str,
    preferred_column: Any = None,
) -> Tuple[List[Any], List[Any]]:
    if preferred_column is not None and preferred_column in table.columns:
        columns_to_search = [preferred_column]
    else:
        columns_to_search = list(table.columns)

    matched_rows: List[Any] = []
    matched_columns: List[Any] = []
    seen_rows: Set[Any] = set()
    seen_columns: Set[Any] = set()

    for column in columns_to_search:
        column_matched = False
        for row_idx, cell_value in table[column].items():
            if _compare_cells(cell_value, value, operator):
                column_matched = True
                if row_idx not in seen_rows:
                    matched_rows.append(row_idx)
                    seen_rows.add(row_idx)
        if column_matched and column not in seen_columns:
            matched_columns.append(column)
            seen_columns.add(column)

    if matched_rows:
        return matched_rows, matched_columns

    if preferred_column is not None and preferred_column in table.columns and len(columns_to_search) == 1:
        return _search_rows_for_value(table, value, operator, preferred_column=None)

    return [], []


def _search_rows_for_any_values(
    table: pd.DataFrame,
    values: Sequence[Any],
    preferred_column: Any = None,
) -> Tuple[List[Any], List[Any]]:
    matched_rows: List[Any] = []
    matched_columns: List[Any] = []
    seen_rows: Set[Any] = set()
    seen_columns: Set[Any] = set()

    for value in values:
        value_rows, value_columns = _search_rows_for_value(table, value, "==", preferred_column=preferred_column)
        for row_idx in value_rows:
            if row_idx not in seen_rows:
                matched_rows.append(row_idx)
                seen_rows.add(row_idx)
        for column in value_columns:
            if column not in seen_columns:
                matched_columns.append(column)
                seen_columns.add(column)

    return matched_rows, matched_columns


def _intersect_row_groups(groups: Sequence[Sequence[Any]]) -> List[Any]:
    if not groups:
        return []
    intersection = set(groups[0])
    for group in groups[1:]:
        intersection &= set(group)
    return [row for row in groups[0] if row in intersection]


class AnchorSub:
    def generate(self, question, table, gold_answer, anchor_spec) -> dict:
        del question

        if not isinstance(table, pd.DataFrame):
            raise TypeError("table must be a pandas.DataFrame")

        anchor_spec = anchor_spec or {}
        requested_columns = anchor_spec.get("columns", []) or []
        row_anchors = anchor_spec.get("row_anchors", []) or []
        answer_values = anchor_spec.get("answer_values", []) or []
        row_scope = str(anchor_spec.get("row_scope", "matched_rows") or "matched_rows")
        entity_values = anchor_spec.get("entity_values", []) or []
        entity_column = anchor_spec.get("entity_column", None)
        requires_answer_cell = bool(anchor_spec.get("requires_answer_cell", False))

        grounded_columns = _ground_columns(requested_columns, list(table.columns))

        row_anchor_groups: List[List[Any]] = []
        row_anchor_columns: List[Any] = []
        row_anchor_all_grounded = bool(row_anchors)
        for anchor in row_anchors:
            operator = str(anchor.get("operator", "==") or "==")
            column = anchor.get("column", None)
            grounded_column = _match_column_name(column, list(table.columns)) if column is not None else None
            if grounded_column is not None and grounded_column not in row_anchor_columns:
                row_anchor_columns.append(grounded_column)
            rows, matched_columns = _search_rows_for_value(
                table,
                anchor.get("value", None),
                operator,
                preferred_column=grounded_column,
            )
            if not rows:
                row_anchor_all_grounded = False
            else:
                row_anchor_groups.append(rows)
            for matched_column in matched_columns:
                if matched_column not in row_anchor_columns:
                    row_anchor_columns.append(matched_column)

        if row_anchors and row_anchor_all_grounded and row_anchor_groups:
            row_anchor_rows = _intersect_row_groups(row_anchor_groups)
        else:
            row_anchor_rows = []

        answer_candidates = _expand_atomic_values(answer_values) + _expand_atomic_values(gold_answer)
        answer_rows, answer_columns = _search_rows_for_any_values(table, answer_candidates)

        grounded_entity_column = _match_column_name(entity_column, list(table.columns)) if entity_column is not None else None
        entity_rows, entity_columns = _search_rows_for_any_values(
            table,
            _expand_atomic_values(entity_values),
            preferred_column=grounded_entity_column,
        )

        if row_scope == "matched_rows":
            rows = row_anchor_rows if (row_anchors and row_anchor_all_grounded) else answer_rows
        elif row_scope == "candidate_domain":
            base_rows = row_anchor_rows if (row_anchors and row_anchor_all_grounded) else list(table.index)
            rows = _ordered_union(base_rows, answer_rows)
        elif row_scope == "all_rows":
            rows = list(table.index)
        elif row_scope == "entity_rows":
            rows = _ordered_union(entity_rows, answer_rows)
        else:
            rows = row_anchor_rows if (row_anchors and row_anchor_all_grounded) else answer_rows

        columns = _ordered_union(
            grounded_columns,
            row_anchor_columns,
            answer_columns,
            entity_columns,
            [grounded_entity_column] if grounded_entity_column is not None else [],
        )
        columns = [col for col in columns if col in table.columns]
        rows = [row for row in rows if row in table.index]

        sub_table = table.loc[rows, columns]

        status = "accept"
        reason = None
        if sub_table.empty:
            status = "reject"
            reason = "empty sub-table"
        elif not columns:
            status = "reject"
            reason = "no grounded columns"
        elif not rows:
            status = "reject"
            reason = "no grounded rows"
        elif requires_answer_cell and not answer_rows:
            status = "reject"
            reason = "answer rows not found"

        return {
            "rows": rows,
            "columns": columns,
            "sub_table": sub_table,
            "status": status,
            "reason": reason,
        }


def evaluate_tabsd_subtable(tabsd_sub_table, silver_result, full_table_shape=None):
    if not isinstance(tabsd_sub_table, pd.DataFrame):
        raise TypeError("tabsd_sub_table must be a pandas.DataFrame")

    silver_result = silver_result or {}
    silver_rows = set(silver_result.get("rows", []) or [])
    silver_columns = set(silver_result.get("columns", []) or [])
    tabsd_rows = set(tabsd_sub_table.index.tolist())
    tabsd_columns = set(tabsd_sub_table.columns.tolist())

    row_intersection = silver_rows & tabsd_rows
    column_intersection = silver_columns & tabsd_columns
    cell_intersection_size = len(row_intersection) * len(column_intersection)

    silver_row_size = len(silver_rows)
    silver_column_size = len(silver_columns)
    silver_cell_size = silver_row_size * silver_column_size
    tabsd_row_size = len(tabsd_rows)
    tabsd_column_size = len(tabsd_columns)
    tabsd_cell_size = tabsd_row_size * tabsd_column_size

    def _ratio(numerator: float, denominator: float) -> float:
        if denominator <= 0:
            return 0.0
        return float(numerator) / float(denominator)

    row_recall = _ratio(len(row_intersection), silver_row_size)
    column_recall = _ratio(len(column_intersection), silver_column_size)
    cell_recall = _ratio(cell_intersection_size, silver_cell_size)

    row_precision = _ratio(len(row_intersection), tabsd_row_size)
    column_precision = _ratio(len(column_intersection), tabsd_column_size)
    cell_precision = _ratio(cell_intersection_size, tabsd_cell_size)

    compression_ratio = None
    if full_table_shape is not None:
        try:
            full_rows, full_columns = full_table_shape
            full_table_cells = int(full_rows) * int(full_columns)
            if full_table_cells > 0:
                compression_ratio = 1.0 - (float(tabsd_cell_size) / float(full_table_cells))
        except Exception:
            compression_ratio = None

    return {
        "row_recall": row_recall,
        "column_recall": column_recall,
        "cell_recall": cell_recall,
        "row_precision": row_precision,
        "column_precision": column_precision,
        "cell_precision": cell_precision,
        "compression_ratio": compression_ratio,
    }

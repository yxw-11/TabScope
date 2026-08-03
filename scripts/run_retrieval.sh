#!/usr/bin/env bash
set -euo pipefail

DATASET="${DATASET:-wikitq}"
MODEL="${MODEL:-gpt5-mini}"
SPLIT="${SPLIT:-test}"
MAX_ITEMS="${MAX_ITEMS:--1}"
START_IDX="${START_IDX:-0}"

case "${MODEL}" in
  gpt5-mini|gpt-5-mini) MODEL_TAG="gpt5mini" ;;
  llama3.3|llama-3.3|meta-llama/Llama-3.3-70B-Instruct) MODEL_TAG="llama33" ;;
  *) MODEL_TAG="$(printf '%s' "${MODEL}" | tr '/: .' '____')" ;;
esac

OUT_PREFIX="results/ours/subtables/${MODEL_TAG}_opaware_${DATASET}"

PYTHONPATH="${PYTHONPATH:-}:src" python -m table_localization.retrieval.operation_aware_retrieval   --dataset "${DATASET}"   --split "${SPLIT}"   --model "${MODEL}"   --start-idx "${START_IDX}"   --max-items "${MAX_ITEMS}"   --prompt-mode "${PROMPT_MODE:-builtin}"   --operation-guidance-mode light   --aggregation-mode joint_reconcile   --adaptive-union-mode disagreement   --save-path "${OUT_PREFIX}.pkl"   --jsonl-path "${OUT_PREFIX}.jsonl"   "$@"

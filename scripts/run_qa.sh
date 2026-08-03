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

SUBTABLE_JSONL="${SUBTABLE_JSONL:-results/ours/subtables/${MODEL_TAG}_opaware_${DATASET}.jsonl}"
OUT_PREFIX="results/ours/qa/${MODEL_TAG}_opaware_${DATASET}_qa"
ACC_PATH="results/ours/accuracy/${MODEL_TAG}_opaware_${DATASET}_acc.jsonl"

PYTHONPATH="${PYTHONPATH:-}:src" python -m table_localization.qa.evaluate_retrieved_qa   --dataset "${DATASET}"   --split "${SPLIT}"   --model "${MODEL}"   --subtable-jsonl "${SUBTABLE_JSONL}"   --method-name "OperationAwareTableLocalization-${MODEL_TAG}-${DATASET}"   --output-jsonl "${OUT_PREFIX}.jsonl"   --output-pkl "${OUT_PREFIX}.pkl"   --acc-log-path "${ACC_PATH}"   --start-idx "${START_IDX}"   --max-items "${MAX_ITEMS}"   --fallback-on-missing-info   --skip-on-oom   "$@"

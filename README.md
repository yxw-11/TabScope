# TabScope: Operation-Aware Table Localization

## File Content

- `src/table_localization/retrieval/operation_aware_retrieval.py`: Operation-Aware Table Localization retrieval. It samples `f_col` and `f_row`, adds operation guidance, reconciles row/column candidates, and writes silver sub-table records.
- `src/table_localization/qa/evaluate_retrieved_qa.py`: QA evaluation after retrieval. It runs COT QA on the retrieved sub-table, saves predictions, and appends checkpoint accuracy logs.
- `src/table_localization/common/swift_infer_backend.py`: Llama 3.3 inference through the existing `swift_infer` conda runtime.
- `data/silver_subtables/`: JSONL silver sub-table datasets produced by our method.
- `results/ours/`: generated sub-tables, QA outputs, and accuracy logs kept under stable names.

## Environment

Use the existing conda environment:

```bash
conda activate swift_infer
pip install -r requirements-minimal.txt
```

The minimal dependency file intentionally excludes heavyweight model-runtime packages. For `llama3.3`, the `swift_infer` environment must already provide `swift`, `torch`, CUDA, and the Llama 3.3 weights/cache. The backend reads these environment variables when needed:

```bash
export SWIFT_LOCAL_LLAMA33_PATH=/path/to/Llama-3.3-70B-Instruct/snapshot
export SWIFT_HF_CACHE_DIR=/path/to/hf/cache
```

For GPT-5 mini:

```bash
export OPENAI_API_KEY=...
```

## Run Retrieval

GPT-5 mini on WikiTQ:

```bash
MODEL=gpt5-mini DATASET=wikitq ./scripts/run_retrieval.sh
```

Llama 3.3 on SLQA:

```bash
MODEL=llama3.3 DATASET=slqa ./scripts/run_retrieval.sh
```

Useful controls:

```bash
START_IDX=0 MAX_ITEMS=100 MODEL=gpt5-mini DATASET=wikitq ./scripts/run_retrieval.sh
PROMPT_MODE=original MODEL=llama3.3 DATASET=wikitq ./scripts/run_retrieval.sh --sampling-n 4
```

`PROMPT_MODE=builtin` is the default so the repo is self-contained. Use `PROMPT_MODE=original` only if optional original prompt templates are restored.

## Run QA After Retrieval

```bash
MODEL=gpt5-mini DATASET=wikitq ./scripts/run_qa.sh
MODEL=llama3.3 DATASET=slqa ./scripts/run_qa.sh
```

By default, QA reads:

```text
results/ours/subtables/{model_tag}_opaware_{dataset}.jsonl
```

and writes:

```text
results/ours/qa/{model_tag}_opaware_{dataset}_qa.jsonl
results/ours/qa/{model_tag}_opaware_{dataset}_qa.pkl
results/ours/accuracy/{model_tag}_opaware_{dataset}_acc.jsonl
```

To evaluate an alternate retrieved file:

```bash
SUBTABLE_JSONL=/path/to/subtables.jsonl MODEL=gpt5-mini DATASET=wikitq ./scripts/run_qa.sh
```

## Retained Results

| Dataset | Model | Silver sub-table JSONL | QA JSONL | Accuracy log | Final logged accuracy |
|---|---|---|---|---|---|
| WikiTQ | GPT-5 mini | `results/ours/subtables/gpt5mini_opaware_wikitq.jsonl` | `results/ours/qa/gpt5mini_opaware_wikitq_qa.jsonl` | `results/ours/accuracy/gpt5mini_opaware_wikitq_acc.jsonl` | 1556/1895 = 0.8211 |
| SLQA | GPT-5 mini | `results/ours/subtables/gpt5mini_opaware_slqa.jsonl` | `results/ours/qa/gpt5mini_opaware_slqa_qa.jsonl` | `results/ours/accuracy/gpt5mini_opaware_slqa_acc.jsonl` | 756/1099 = 0.6879 |

These are the retained experiment outputs found in the original directory. Some WikiTQ/llama logs are partial runs; use the line counts in the files as the exact covered subset.

## Dataset Link

You can download by `https://figshare.com/s/41a54cfac3a4aa13b98c`

## Layout

```text
.
├── data/
│   ├── slqa.jsonl
│   └── silver_subtables/
├── results/
│   └── ours/
│       ├── accuracy/
│       ├── qa/
│       └── subtables/
├── scripts/
│   ├── run_qa.sh
│   └── run_retrieval.sh
└── src/table_localization/
    ├── common/
    ├── evaluate/
    ├── qa/
    └── retrieval/
```

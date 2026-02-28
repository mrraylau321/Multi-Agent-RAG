# Code Subdirectory Runbook

This folder contains the executable code, datasets, and evaluation scripts for the multi-agent RAG pipeline.

For the full project overview and architecture, use the repository root `README.md`.

## Quick Run (from `Code/`)

```bash
python src/gui.py
```

## Batch Evaluation (from `Code/`)

```bash
python evaluation/build_predictions_full_pipeline_parallel.py
python evaluation/reorder_and_normalize_predictions.py \
  --input_file predictions_full_pipeline.jsonl \
  --output_file predictions_full_pipeline_reordered.jsonl \
  --gold_file ./data/validation.jsonl
python evaluation/official_eval_hotpotqa.py --gold ./data/validation.jsonl --pred predictions_full_pipeline_reordered.jsonl
python evaluation/official_eval_retrieval.py --gold ./data/validation.jsonl --pred predictions_full_pipeline_reordered.jsonl
```
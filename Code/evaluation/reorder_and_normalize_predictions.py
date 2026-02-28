import json
from pathlib import Path
from typing import Dict, List, Any
from charset_normalizer import from_path


def load_id_order(path: Path) -> List[str]:
    """Load question IDs from a JSONL file in order."""
    ids: List[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            ids.append(obj["id"])
    return ids


def load_predictions(path: Path) -> Dict[str, Any]:
    """Load predictions JSONL into a dict keyed by id."""
    probe = from_path(str(path)).best()
    enc = probe.encoding or "utf-8"
    # print(f"Detected encoding: {enc} (confidence {probe.chaos:.3f})")
    preds: Dict[str, Any] = {}
    with path.open("r", encoding=enc) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            preds[obj["id"]] = obj
    return preds


def normalize_answer(answer: Any) -> Any:
    """
    Normalize answers:
    - If exactly "Yes" -> "yes"
    - If exactly "No"  -> "no"
    Otherwise unchanged.
    """
    if isinstance(answer, str):
        if answer == "Yes":
            return "yes"
        if answer == "No":
            return "no"
    return answer


def reorder_and_fix(
    ordered_ids: List[str],
    predictions_by_id: Dict[str, Any],
    out_path: Path,
    label: str,
) -> None:
    """
    Reorder predictions to match ordered_ids and normalize Yes/No answers.

    Only writes rows for which a prediction exists. Missing IDs are skipped.
    """
    written = 0
    missing = 0
    with out_path.open("w", encoding="utf-8") as out_f:
        for qid in ordered_ids:
            pred = predictions_by_id.get(qid)
            # Skip if there is no prediction for this id.
            if pred is None:
                missing += 1
                continue

            if "answer" in pred:
                pred["answer"] = normalize_answer(pred["answer"])

            json.dump(pred, out_f, ensure_ascii=False)
            out_f.write("\n")
            written += 1

    print(f"[{label}] Wrote {written} predictions to {out_path} (skipped {missing} missing)")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    data_dir = root / "data"
    eval_dir = root / "evaluation"

    # Test set: predictions_full_pipeline_testset.jsonl vs data/test.jsonl
    try:
        test_ids = load_id_order(data_dir / "test.jsonl")
        test_preds = load_predictions(eval_dir / "predictions_full_pipeline_testset.jsonl")
        reorder_and_fix(
            ordered_ids=test_ids,
            predictions_by_id=test_preds,
            out_path=eval_dir / "predictions_full_pipeline_testset_reordered.jsonl",
            label="test",
        )
    except Exception as e:
        print(f'Failed to reorder testset file: {e}')

    # Validation set: predictions_full_pipeline.jsonl vs data/validation.jsonl
    try:
        val_ids = load_id_order(data_dir / "validation.jsonl")
        val_preds = load_predictions(eval_dir / "predictions_full_pipeline.jsonl")
        reorder_and_fix(
            ordered_ids=val_ids,
            predictions_by_id=val_preds,
            out_path=eval_dir / "predictions_full_pipeline_reordered.jsonl",
            label="validation",
        )
    except Exception as e:
        print(f'Failed to reorder validation file: {e}')


if __name__ == "__main__":
    main()
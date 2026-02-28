import os
import sys
import json
import re
import codecs
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock, Thread
from queue import Queue, Empty
from typing import List, Dict, Tuple, Generator, Optional, Set

from tqdm import tqdm
from dotenv import load_dotenv

# ---------------- Environment and paths ----------------
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["ACCELERATE_NUM_PROCESSES"] = "1"
os.environ["ACCELERATE_MIXED_PRECISION"] = "no"
os.environ["ACCELERATE_DYNAMO_BACKEND"] = "no"

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
load_dotenv(dotenv_path=os.path.join(PROJECT_ROOT, '.env'))

# ---------------- Project imports ----------------
from src.retrieval import load_documents, HybridRetriever
from src.llm_services import OpenRouterGenerator, RAGPrompt
from src.multi_agent_flow import Orchestrator


# ---------------- Utilities ----------------

ID_SALVAGE_REGEX = re.compile(r'"id"\s*:\s*"([^"]+)"')
WHITESPACE_RE = re.compile(r"\s+")


def _normalize_final_answer(ans: str) -> str:
    s = (ans or "").strip().strip('"').strip("'").replace("_", " ")
    s = s.replace('*', '')
    for key in ('Answer:', 'answer:', 'ANSWER:'):
        if key in s:
            try:
                s = s.split(key, 1)[1].strip()
            except Exception:
                pass
    s = WHITESPACE_RE.sub(" ", s).strip()
    return s


def _create_ranked_scores(doc_ids: List[str], rrf_k: int = 60) -> List[List[object]]:
    """Assigns a reciprocal rank fusion (RRF) score to a list of documents."""
    return [[doc_id, 1.0 / (rrf_k + i + 1)] for i, doc_id in enumerate(doc_ids)]


def _normalize_id(val) -> Optional[str]:
    if val is None:
        return None
    s = str(val).strip()
    return s if s else None


def load_completed_ids(output_path: str, badlines_path: Optional[str] = None) -> Set[str]:
    """
    Robustly scan predictions file to collect completed IDs.
    - Tolerates BOM, encoding glitches, truncated lines.
    - Salvages "id" using regex if JSON parsing fails.
    - Logs problematic lines optionally.
    """
    completed: Set[str] = set()
    total_lines = 0
    parsed_ok = 0
    with_id = 0
    salvaged = 0
    bad_lines: List[Tuple[str, str]] = []

    if not os.path.exists(output_path):
        print(f"No existing predictions at {output_path}")
        return completed

    with codecs.open(output_path, 'r', encoding='utf-8-sig', errors='replace') as f:
        for raw_line in f:
            total_lines += 1
            line = raw_line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                parsed_ok += 1
                if 'id' in obj and obj['id'] is not None:
                    id_val = _normalize_id(obj['id'])
                    if id_val:
                        completed.add(id_val)
                        with_id += 1
                else:
                    m = ID_SALVAGE_REGEX.search(line)
                    if m:
                        id_val = _normalize_id(m.group(1))
                        if id_val:
                            completed.add(id_val)
                            salvaged += 1
                    else:
                        bad_lines.append(("no_id_in_json", line[:500]))
            except Exception:
                m = ID_SALVAGE_REGEX.search(line)
                if m:
                    id_val = _normalize_id(m.group(1))
                    if id_val:
                        completed.add(id_val)
                        salvaged += 1
                else:
                    bad_lines.append(("json_parse_error", line[:500]))

    print(
        f"Scan predictions: total_lines={total_lines}, parsed_ok={parsed_ok}, "
        f"with_id={with_id}, salvaged={salvaged}, unique_ids={len(completed)}"
    )

    if badlines_path and bad_lines:
        try:
            os.makedirs(os.path.dirname(badlines_path), exist_ok=True)
            with open(badlines_path, 'w', encoding='utf-8') as bf:
                for reason, snippet in bad_lines[:2000]:
                    bf.write(f"{reason}\t{snippet}\n")
            print(f"Wrote {len(bad_lines)} problematic lines to {badlines_path}")
        except Exception:
            traceback.print_exc()

    return completed


def load_test_examples(test_path: str) -> List[Dict]:
    """Load and normalize test examples; skip rows lacking id or text."""
    examples: List[Dict] = []
    bad_json = 0
    missing_id = 0
    missing_text = 0
    with codecs.open(test_path, 'r', encoding='utf-8-sig', errors='replace') as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                bad_json += 1
                continue
            qid = _normalize_id(obj.get('id'))
            if not qid:
                missing_id += 1
                continue
            text = obj.get('text', '')
            if not isinstance(text, str) or not text.strip():
                missing_text += 1
                continue
            obj['id'] = qid
            obj['text'] = text.strip()
            examples.append(obj)
    print(
        f"Loaded test: total={len(examples)}, bad_json={bad_json}, "
        f"missing_id={missing_id}, missing_text={missing_text}"
    )
    return examples


# ---------------- Writer thread (single writer for atomic lines) ----------------
class LineWriter:
    def __init__(self, output_path: str):
        self.output_path = output_path
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        self.q: Queue = Queue(maxsize=10000)
        self.stop_token = object()
        self.thread = Thread(target=self._run, daemon=True)
        self.started = False

    def start(self):
        if not self.started:
            self.thread.start()
            self.started = True

    def write(self, obj: Dict):
        self.q.put(obj)

    def close(self):
        # Signal and join
        self.q.put(self.stop_token)
        self.thread.join()

    def _run(self):
        with open(self.output_path, 'a', encoding='utf-8') as out_f:
            while True:
                item = self.q.get()
                if item is self.stop_token:
                    out_f.flush()
                    os.fsync(out_f.fileno())
                    break
                try:
                    line = json.dumps(item, ensure_ascii=False)
                except Exception as e:
                    # Fallback: log minimal info
                    line = json.dumps(
                        {"_writer_error": str(e), "_raw": str(item)}, ensure_ascii=False
                    )
                out_f.write(line + "\n")
                # flush + fsync to minimize risk of truncated lines on crash
                out_f.flush()
                os.fsync(out_f.fileno())


# ---------------- Main pipeline ----------------
def build_predictions_full_pipeline_parallel(
    output_path: str,
    log_path: Optional[str] = None,
    num_samples: Optional[int] = None,
    initial_top_k: int = 20,
    agent_rounds: int = 5,
    rerank_top_k: int = 10,
    num_workers: int = 10,
) -> None:
    """
    Parallel full pipeline with robust resume and single-writer thread.
    """
    print("--- Building predictions_full_pipeline (parallel, robust) ---")

    # Load data
    documents = load_documents()
    test_path = os.path.join(PROJECT_ROOT, 'data', 'test.jsonl')

    # Initialize shared components
    retriever = HybridRetriever(documents, dense_model_name='mxbai-embed-large:335m')
    llm = OpenRouterGenerator()
    prompts = RAGPrompt()

    # Robust resume: collect already completed IDs
    badlines = os.path.join(PROJECT_ROOT, 'evaluation', 'predictions_full_pipeline.badlines.txt')
    completed_ids = load_completed_ids(output_path, badlines_path=badlines)
    if completed_ids:
        print(f"Resuming: found {len(completed_ids)} completed predictions in existing file.")

    # Load and normalize full test set
    all_examples = load_test_examples(test_path)
    # Filter to remaining examples
    remaining_examples = [ex for ex in all_examples if ex['id'] not in completed_ids]
    if num_samples is not None:
        remaining_examples = remaining_examples[:num_samples]

    # Show set stats for sanity-check
    val_ids = set(ex['id'] for ex in all_examples)
    print(
        f"Total examples: {len(all_examples)} | Already done: {len(completed_ids)} | Remaining: {len(remaining_examples)} | "
        f"val_only={len(val_ids - completed_ids)} | pred_only={len(completed_ids - val_ids)}"
    )

    if not remaining_examples:
        print("No remaining examples to process.")
        return

    # Ensure directories exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if log_path:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

    # Optional log writer (separate file; writing is lightweight here)
    log_lock = Lock()

    # Single writer thread for predictions
    writer = LineWriter(output_path)
    writer.start()

    def process_example(example: Dict) -> None:
        qid = example['id']
        query = example['text']

        # Perform initial retrieval
        retrieved_ids = retriever.retrieve(query, top_k=initial_top_k)

        orchestrator = Orchestrator(retriever, llm, prompts, max_rounds=agent_rounds)
        orchestrator_gen = orchestrator.run(
            question=query,
            initial_top_k=initial_top_k,
            rerank_top_k=rerank_top_k,
            initial_retrieved_docs_ids=retrieved_ids, # Pass initial retrieval results
            use_reranker_agent=True # Enable Orchestrator's internal reranker
        )

        predicted_answer, final_doc_ids = "No answer found", []
        log_lines: List[str] = []
        try:
            while True:
                message = next(orchestrator_gen)
                # message could be str or structured; cast to str for logging
                log_lines.append(str(message))
        except StopIteration as e:
            try:
                predicted_answer, final_doc_ids = e.value
            except Exception:
                # In case e.value is unexpected
                predicted_answer, final_doc_ids = "No answer found", []

        predicted_answer = (predicted_answer or "").strip()
        retrieved_with_scores = _create_ranked_scores(final_doc_ids)
        normalized_answer = _normalize_final_answer(predicted_answer)

        out_line = {
            "id": _normalize_id(qid),
            "text": query,
            "answer": normalized_answer.strip(),
            "retrieved_docs": retrieved_with_scores,
        }

        # Enqueue to single writer
        writer.write(out_line)

        # Write logs (synchronous but small; protected)
        if log_path:
            with log_lock:
                with open(log_path, 'a', encoding='utf-8') as log_f:
                    log_f.write(f"### ID: {qid}\n")
                    for line in log_lines:
                        log_f.write(str(line) + "\n")
                    log_f.write("\n")

    # Dispatch work to thread pool
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(process_example, ex) for ex in remaining_examples]
        for _ in tqdm(as_completed(futures), total=len(futures), desc="Parallel full-pipeline predictions"):
            pass

    # Close writer to flush and fsync the last data
    writer.close()

    print(f"Wrote predictions to: {output_path}")
    if log_path:
        print(f"Wrote logs to: {log_path}")

if __name__ == "__main__":
    output = os.path.join(PROJECT_ROOT, 'evaluation', 'predictions_full_pipeline.jsonl')
    logs = os.path.join(PROJECT_ROOT, 'evaluation', 'predictions_full_pipeline.log')

    build_predictions_full_pipeline_parallel(
        output_path=output,
        log_path=logs,
        num_samples=None,
        initial_top_k=20,
        agent_rounds=5,
        rerank_top_k=10,
        num_workers=24,
    )
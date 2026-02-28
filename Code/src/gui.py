import os
import sys
import gradio as gr
import json
import random
import html as html_lib
import re
from urllib.parse import quote

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

from src.gui_retrieval import load_documents, GuiHybridRetriever, BM25Retriever, StaticEmbeddingRetriever, GuiOllamaDenseRetriever, ColBERTRetriever
from src.llm_services import OllamaGenerator, RAGPrompt
from src.multi_agent_flow import Orchestrator
from src.colbert_reranker import ColBERTReranker

# Global application setup
print("Starting application setup...")
documents = load_documents()
doc_id_map = {doc['id']: doc for doc in documents}
llm = OllamaGenerator()
prompts = RAGPrompt()
TRAIN_PATH = os.path.join(PROJECT_ROOT, 'data', 'train.jsonl')
VAL_PATH = os.path.join(PROJECT_ROOT, 'data', 'validation.jsonl')
TEST_PATH = os.path.join(PROJECT_ROOT, 'data', 'test.jsonl')

session_state = {"last_context_docs": []}

def _load_jsonl_safe(path):
    if not os.path.exists(path): return []
    items = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try: items.append(json.loads(line))
                except: pass
    return items

train_data = _load_jsonl_safe(TRAIN_PATH)
val_data = _load_jsonl_safe(VAL_PATH)
test_data = _load_jsonl_safe(TEST_PATH)

def _doc_link(doc_id: str) -> str:
    doc = doc_id_map.get(doc_id)
    if not doc: return doc_id
    title = html_lib.escape(doc_id)
    body = html_lib.escape(doc.get('text', ''))
    html_page = f"<html><head><meta charset='utf-8'></head><body><h3>{title}</h3><pre style='white-space:pre-wrap'>{body}</pre></body></html>"
    data_url = "data:text/html;charset=utf-8," + quote(html_page)
    return f"[{doc_id}]({data_url})"

def _links_markdown(docs_with_scores):
    if not docs_with_scores: return ""
    links = []
    if isinstance(docs_with_scores, list) and docs_with_scores and isinstance(docs_with_scores[0], (list, tuple)) and len(docs_with_scores[0]) == 2:
        for doc_id, score in docs_with_scores:
            link = _doc_link(doc_id)
            links.append(f"{link} (Score: {score:.3f})")
    else:
        for doc_id in docs_with_scores:
            links.append(_doc_link(doc_id))
    return "\n".join([f"- {lnk}" for lnk in links])

def _normalize_final_answer(ans: str) -> str:
    s = (ans or "").strip().strip('"').strip("'").replace("_", " ")
    s = s.replace('*','')
    for key in ('Answer:', 'answer:', 'ANSWER:'):
        if key in s:
            try: s = s.split(key, 1)[1].strip()
            except Exception: pass
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _create_ranked_scores(doc_ids: list[str], rrf_k: int = 60) -> list[list[object]]:
    return [[doc_id, 1.0 / (rrf_k + i + 1)] for i, doc_id in enumerate(doc_ids)]

print("Application setup complete.")

def _find_sample_by_text(text):
    if not text: return None
    for s in train_data + val_data + test_data:
        if s.get('text', '') == text:
            return s
    return None

def on_query_changed(query):
    sample = _find_sample_by_text(query)
    if not sample: return "", "", gr.update(choices=[], value=None)
    gt = _format_gt(sample)
    s_ids = sample.get('supporting_ids', [])
    s_ids_for_markdown = [[s_id, 1.0] for s_id in s_ids if not isinstance(s_id, (list, tuple))]
    links = _links_markdown(s_ids_for_markdown)
    choices = [s_id[0] if isinstance(s_id, (list, tuple)) else s_id for s_id in s_ids]
    first_choice = choices[0] if choices else None
    return gt, links, gr.update(choices=choices, value=first_choice)

def run_pipeline(query, strategy, dense_model, use_colbert, use_agentic):
    trace_lines = []
    
    gt_text, gt_links, gt_choices_update = on_query_changed(query)
    yield {
        ground_truth_output: gt_text,
        gt_doc_links_output: gt_links,
        gt_doc_select: gt_choices_update,
        intermediate_steps_output: "Initializing pipeline...",
        reasoning_output: "",
        live_extracted_answer_output: "Processing...",
        final_answer_output: "",
        model_doc_links_output: "",
        model_doc_select: gr.update(choices=[], value=None)
    }

    def context_update_callback(docs):
        session_state["last_context_docs"] = docs

    trace_lines.append(f"Strategy: {strategy}, Dense Model: {dense_model}, Colbert: {use_colbert}, Agentic: {use_agentic}")
    yield {intermediate_steps_output: "\n".join(trace_lines)}

    retriever = None
    if strategy == "Hybrid":
        retriever = GuiHybridRetriever(documents, dense_model_name=dense_model)
    elif strategy == "BM25 Only":
        retriever = BM25Retriever(documents)
    elif strategy == "ColBERT":
        retriever = ColBERTRetriever(documents)
    else: # Static (GloVe)
        retriever = StaticEmbeddingRetriever(documents)
        
    retrieved_ids = retriever.retrieve(query, top_k=20)
    trace_lines.append(f"Initial retrieval ({strategy}): Found {len(retrieved_ids)} documents.")
    yield {intermediate_steps_output: "\n".join(trace_lines)}

    final_doc_ids_for_pipeline = retrieved_ids[:10]
    if use_colbert:
        trace_lines.append("Applying ColBERT reranking...")
        yield {intermediate_steps_output: "\n".join(trace_lines)}
        reranker = ColBERTReranker()
        candidate_docs = [doc_id_map[doc_id] for doc_id in retrieved_ids if doc_id in doc_id_map]
        candidate_tuples = [(doc['id'], doc['text']) for doc in candidate_docs]
        reranked_ids = reranker.rerank(query, candidate_tuples, top_k=10)
        final_doc_ids_for_pipeline = reranked_ids
        trace_lines.append(f"ColBERT reranked to top {len(reranked_ids)} documents.")
        yield {intermediate_steps_output: "\n".join(trace_lines)}
    
    orchestrator = Orchestrator(retriever, llm, prompts, max_rounds=5 if use_agentic else 1)
    
    orchestrator_gen = orchestrator.run(
        question=query, 
        context_callback=context_update_callback,
        initial_retrieved_docs_ids=final_doc_ids_for_pipeline,
        use_reranker_agent=not use_colbert
    )
    
    final_doc_ids = []
    predicted_answer = "No answer found"
    try:
        while True:
            progress_message = next(orchestrator_gen)
            update_dict = {}
            if isinstance(progress_message, str):
                if progress_message.startswith("REASONING::"):
                    update_dict[reasoning_output] = progress_message.replace("REASONING::", "").strip()
                elif progress_message.startswith("EXTRACTED::"):
                    update_dict[live_extracted_answer_output] = progress_message.replace("EXTRACTED::", "").strip()
                else:
                    trace_lines.append(progress_message.replace("TRACE::", "").strip())
                    update_dict[intermediate_steps_output] = "\n".join(trace_lines)
            yield update_dict
    except StopIteration as e:
        predicted_answer, final_doc_ids = e.value

    normalized_answer = _normalize_final_answer(predicted_answer)
    final_docs_with_scores = _create_ranked_scores(final_doc_ids)
    doc_links_md = _links_markdown(final_docs_with_scores)
    
    yield {
        live_extracted_answer_output: normalized_answer,
        final_answer_output: normalized_answer,
        model_doc_links_output: doc_links_md,
        model_doc_select: gr.update(choices=final_doc_ids, value=(final_doc_ids[0] if final_doc_ids else None))
    }

def get_last_context():
    docs = session_state.get("last_context_docs", [])
    if not docs: return "No context available."
    return "\n\n".join([f"--- DOC ID: {doc['id']} ---\n{doc['text']}" for doc in docs])

def _format_gt(sample):
    if not sample: return ""
    ans = sample.get('answer', '')
    sids = sample.get('supporting_ids', [])
    sids_flat = [item[0] if isinstance(item, (list, tuple)) else item for item in sids]
    return f"{ans}\n\nSupporting IDs: {', '.join(sids_flat) if sids_flat else ''}"

def pick_random(data):
    if not data: return "", "", "", gr.update(choices=[], value=None)
    s = random.choice(data)
    gt = _format_gt(s)
    s_ids_raw = s.get('supporting_ids', [])
    s_ids_for_markdown = [[s_id, 1.0] for s_id in s_ids_raw if not isinstance(s_id, (list, tuple))]
    links = _links_markdown(s_ids_for_markdown)
    choices = [s_id[0] if isinstance(s_id, (list, tuple)) else s_id for s_id in s_ids_raw]
    first_choice = choices[0] if choices else None
    return s.get('text', ''), gt, links, gr.update(choices=choices, value=first_choice)

def show_document(doc_id: str):
    if not doc_id: return ""
    doc = doc_id_map.get(doc_id)
    return doc.get('text', f"Document not found: {doc_id}") if doc else f"Document not found: {doc_id}"

with gr.Blocks() as demo:
    gr.Markdown("# Multi-Agent RAG System")
    
    with gr.Row():
        query_input = gr.Textbox(label="Enter your query", lines=5, scale=4)
        submit_button = gr.Button("Run Pipeline", scale=1)
    
    with gr.Row():
        btn_train = gr.Button("Random Question (Train)")
        btn_val = gr.Button("Random Question (Validation)")
        btn_test = gr.Button("Random Question (Test)")

    with gr.Row():
        retrieval_strategy = gr.Radio(["Hybrid", "BM25 Only", "Static (GloVe)", "ColBERT"], label="Retrieval Strategy", value="Hybrid")
        dense_model_select = gr.Radio(
            ["mxbai-embed-large:335m", "qwen3-embedding:latest", "bge-m3:latest"],
            label="Dense Model (for Hybrid)",
            value="mxbai-embed-large:335m"
        )
        with gr.Column():
            use_colbert_reranker = gr.Checkbox(label="Use ColBERT Reranker", value=False)
            agentic_workflow_toggle = gr.Checkbox(label="Enable Full Agentic Pipeline", value=True)

    def update_dense_model_visibility(strategy):
        return gr.update(visible=strategy == "Hybrid")

    retrieval_strategy.change(fn=update_dense_model_visibility, inputs=retrieval_strategy, outputs=dense_model_select)
        
    with gr.Row():
        with gr.Column(scale=1):
            live_extracted_answer_output = gr.Textbox(label="Live Extracted Answer", interactive=False)
            final_answer_output = gr.Textbox(label="Final Answer", interactive=False)
        reasoning_output = gr.Textbox(label="Reasoning", lines=8, interactive=False, scale=2)
    
    ground_truth_output = gr.Textbox(label="Ground Truth", lines=4, interactive=False)
    
    with gr.Row():
        intermediate_steps_output = gr.Textbox(label="Reasoning Pipeline Trace", lines=15, interactive=False)
        context_viewer = gr.Textbox(label="Full Agent Context", lines=15, interactive=False)

    with gr.Row():
        btn_show_context = gr.Button("Show Current Context")

    with gr.Row():
        gt_doc_links_output = gr.Markdown(value="**Ground Truth Docs**")
        model_doc_links_output = gr.Markdown(value="**Retrieved Docs**")

    with gr.Row():
        gt_doc_select = gr.Dropdown(choices=[], label="Ground Truth Doc ID")
        btn_show_gt = gr.Button("Show GT Doc")
        model_doc_select = gr.Dropdown(choices=[], label="Model Doc ID")
        btn_show_model = gr.Button("Show Model Doc")

    doc_viewer = gr.Textbox(label="Document Viewer", lines=20, interactive=False)

    submit_button.click(
        fn=run_pipeline, 
        inputs=[query_input, retrieval_strategy, dense_model_select, use_colbert_reranker, agentic_workflow_toggle],
        outputs=[
            intermediate_steps_output, reasoning_output, live_extracted_answer_output,
            final_answer_output, model_doc_links_output, model_doc_select,
            ground_truth_output, gt_doc_links_output, gt_doc_select
        ]
    )
    query_input.change(fn=on_query_changed, inputs=query_input, outputs=[ground_truth_output, gt_doc_links_output, gt_doc_select])
    btn_train.click(lambda: pick_random(train_data), outputs=[query_input, ground_truth_output, gt_doc_links_output, gt_doc_select])
    btn_val.click(lambda: pick_random(val_data), outputs=[query_input, ground_truth_output, gt_doc_links_output, gt_doc_select])
    btn_test.click(lambda: pick_random(test_data), outputs=[query_input, ground_truth_output, gt_doc_links_output, gt_doc_select])
    btn_show_gt.click(fn=show_document, inputs=gt_doc_select, outputs=doc_viewer)
    btn_show_model.click(fn=show_document, inputs=model_doc_select, outputs=doc_viewer)
    btn_show_context.click(fn=get_last_context, outputs=context_viewer)

if __name__ == "__main__":
    server_name = os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0")
    server_port = int(os.environ.get("GRADIO_SERVER_PORT", "7860"))
    share = os.environ.get("GRADIO_SHARE", "false").lower() in ("1", "true", "yes")
    demo.queue().launch(server_name=server_name, server_port=server_port, share=share)
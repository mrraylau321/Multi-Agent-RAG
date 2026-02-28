import re
import json
from src.retrieval import HybridRetriever
from src.llm_services import OpenRouterGenerator, RAGPrompt

class ReasoningAnswerer:
    def __init__(self, llm: OpenRouterGenerator, prompts: RAGPrompt):
        self.llm = llm
        self.prompts = prompts

    def answer(self, question: str, context_docs: list[dict], useful_information: str = "N/A") -> str:
        context = "\n\n---\n\n".join([f"ID: {d['id']}\n{d['text']}" for d in context_docs])
        prompt = self.prompts.reasoning_answerer_template.format(question=question, context=context, useful_information=useful_information)
        return self.llm.generate(prompt).strip()

class AnswerExtractor:
    def __init__(self, llm: OpenRouterGenerator, prompts: RAGPrompt):
        self.llm = llm
        self.prompts = prompts

    def extract(self, question: str, reasoned_answer: str) -> str:
        prompt = self.prompts.answer_extractor_template.format(question=question, reasoned_answer=reasoned_answer)
        return self.llm.generate(prompt).strip()

class AnswerReviewer:
    def __init__(self, llm: OpenRouterGenerator, prompts: RAGPrompt):
        self.llm = llm
        self.prompts = prompts

    def review(self, question: str, answer: str, context_docs: list[dict]) -> tuple[str, str | None]:
        context = "\n\n---\n\n".join([f"ID: {d['id']}\n{d['text']}" for d in context_docs])
        prompt = self.prompts.self_check_template.format(question=question, answer=answer, context=context)
        evaluation_text = self.llm.generate(prompt)
        
        t = (evaluation_text or '').strip()
        if re.search(r"^\s*good\s*$", t, re.IGNORECASE):
            return "GOOD", None
        
        m = re.search(r"obvious\s*answer\s*:\s*(.+?)\s*\|\s*evidence\s*:\s*(.+)", t, re.IGNORECASE)
        if m:
            ans = m.group(1).strip() if m.group(1) else ""
            evidence = m.group(2).strip() if m.group(2) else ""
            return "OBVIOUS", f"Answer: {ans} | Evidence: \"{evidence}\""

        m = re.search(r"bad\s*:\s*(.+)", t, re.IGNORECASE)
        if m:
            reason = m.group(1).strip()
            return "BAD", reason
            
        return "UNKNOWN", t

class KeySearchAgent:
    def __init__(self, llm: OpenRouterGenerator, prompts: RAGPrompt):
        self.llm = llm
        self.prompts = prompts

    def generate_keys(self, question: str, context_docs: list[dict], useful_information: str = "N/A", num_keys: int = 3) -> list[str]:
        context = "\n\n---\n\n".join([f"ID: {d['id']}\n{d['text'][:500]}..." for d in context_docs])
        prompt = self.prompts.key_generation_template.format(num_keys=num_keys, question=question, context=context, useful_information=useful_information)
        raw_keys = self.llm.generate(prompt)
        keys = [k.strip() for k in raw_keys.split("\n") if k.strip() and not k.startswith("- ")]
        return keys[:num_keys]

class AnswerSimplifier:
    def __init__(self, llm: OpenRouterGenerator, prompts: RAGPrompt):
        self.llm = llm
        self.prompts = prompts

    def simplify(self, question: str, answer: str) -> str:
        prompt = self.prompts.answer_simplifier_template.format(question=question, answer=answer)
        return self.llm.generate(prompt).strip()

class SpecialGuessAgent:
    def __init__(self, llm: OpenRouterGenerator, prompts: RAGPrompt):
        self.llm = llm
        self.prompts = prompts

    def guess(self, question: str, context_docs: list[dict]) -> str:
        context = "\n\n---\n\n".join([f"ID: {d['id']}\n{d['text']}" for d in context_docs])
        prompt = self.prompts.special_guess_template.format(question=question, context=context)
        return self.llm.generate(prompt).strip()

class SynthesizerAgent:
    def __init__(self, llm: OpenRouterGenerator, prompts: RAGPrompt):
        self.llm = llm
        self.prompts = prompts

    def synthesize(self, question: str, candidates: list[dict], useful_information: str = "N/A") -> str:
        candidate_answers = "\n".join([f"- {c['answer']}" for c in candidates])
        prompt = self.prompts.synthesizer_template.format(question=question, candidate_answers=candidate_answers, useful_information=useful_information)
        return self.llm.generate(prompt).strip()

class FinalGuesserAgent:
    def __init__(self, llm: OpenRouterGenerator, prompts: RAGPrompt):
        self.llm = llm
        self.prompts = prompts

    def guess(self, question: str, context_docs: list[dict], useful_information: str = "N/A") -> str:
        context = "\n\n---\n\n".join([f"ID: {d['id']}\n{d['text']}" for d in context_docs])
        prompt = self.prompts.final_guesser_template.format(question=question, context=context, useful_information=useful_information)
        return self.llm.generate(prompt).strip()

class InformationSummarizerAgent:
    def __init__(self, llm: OpenRouterGenerator, prompts: RAGPrompt):
        self.llm = llm
        self.prompts = prompts

    def summarize(self, question: str, context_docs: list[dict]) -> str:
        context = "\n\n".join([f"ID: {d['id']}\n{d['text']}" for d in context_docs])
        prompt = self.prompts.summarizer_template.format(question=question, context=context)
        summary = self.llm.generate(prompt).strip()
        if "no useful information found" in summary.lower():
            return ""
        return summary

class RerankerAgent:
    def __init__(self, llm: OpenRouterGenerator, prompts: RAGPrompt):
        self.llm = llm
        self.prompts = prompts

    def rerank(self, question: str, context_docs: list[dict], useful_information: str) -> list[str]:
        context = "\n\n".join([f"ID: {d['id']}\n{d['text']}" for d in context_docs])
        prompt = self.prompts.reranker_template.format(question=question, context=context, useful_information=useful_information)
        response = self.llm.generate(prompt).strip()
        return [doc_id.strip() for doc_id in response.split(',') if doc_id.strip()]

class AnswerGenerationAgent:
    def __init__(self, llm: OpenRouterGenerator, prompts: RAGPrompt):
        self.llm = llm
        self.prompts = prompts

    def generate(self, question: str, context_docs: list[dict]) -> str:
        context = "\n\n---\n\n".join([f"ID: {d['id']}\n{d['text']}" for d in context_docs])
        prompt = self.prompts.final_answer_generator_template.format(question=question, context=context)
        return self.llm.generate(prompt).strip()

class Orchestrator:
    def __init__(self, retriever: HybridRetriever, llm: OpenRouterGenerator, prompts: RAGPrompt, max_rounds: int = 5):
        self.retriever = retriever
        self.reasoner = ReasoningAnswerer(llm, prompts)
        self.extractor = AnswerExtractor(llm, prompts)
        self.reviewer = AnswerReviewer(llm, prompts)
        self.key_searcher = KeySearchAgent(llm, prompts)
        self.simplifier = AnswerSimplifier(llm, prompts)
        self.synthesizer = SynthesizerAgent(llm, prompts)
        self.guesser = FinalGuesserAgent(llm, prompts)
        self.summarizer = InformationSummarizerAgent(llm, prompts)
        self.reranker = RerankerAgent(llm, prompts)
        self.generator = AnswerGenerationAgent(llm, prompts)
        self.max_rounds = max_rounds

    def run(self, question: str, initial_top_k: int = 20, rerank_top_k: int = 10, context_callback=None, use_reranker_agent: bool = True, initial_retrieved_docs_ids: list[str] = None):
        if initial_retrieved_docs_ids:
            yield f"TRACE: Round 1: Using {len(initial_retrieved_docs_ids)} pre-retrieved documents..."
            retrieved_ids = initial_retrieved_docs_ids
        else:
            yield f"TRACE: Round 1: Initial retrieval for question: '{question}'..."
            retrieved_ids = self.retriever.retrieve(question, top_k=initial_top_k)
        
        all_retrieved_ids = set(retrieved_ids)
        context_docs = self.retriever.get_docs_by_ids(retrieved_ids)
        if context_callback: context_callback(context_docs)
        yield f"TRACE: Found {len(retrieved_ids)} initial documents."

        candidate_answers = []
        useful_information = "N/A"

        for i in range(self.max_rounds):
            round_num = i + 1
            yield f"TRACE: --- Round {round_num} ---"
            
            if use_reranker_agent:
                yield "TRACE: Reranking retrieved documents..."
                llm_reranked_ids = self.reranker.rerank(question, context_docs, useful_information)
            else:
                yield "TRACE: Skipping RerankerAgent as per request."
                llm_reranked_ids = retrieved_ids


            # --- Padding Logic to ensure 10 documents ---
            # Start with the LLM's preferred docs, removing duplicates while preserving order
            final_reranked_ids = list(dict.fromkeys(llm_reranked_ids))
            reranked_set = set(final_reranked_ids)

            # If after reranking and deduplication, we have fewer than rerank_top_k documents,
            # pad with documents from the initial retrieval order, maintaining their original order.
            current_reranked_set = set(final_reranked_ids)
            for doc_id in retrieved_ids:
                if len(final_reranked_ids) >= rerank_top_k:
                    break
                if doc_id not in current_reranked_set:
                    final_reranked_ids.append(doc_id)
                    current_reranked_set.add(doc_id)
            
            # If after all padding, we still have fewer than rerank_top_k documents (e.g., if initial_top_k was small),
            # ensure we take up to rerank_top_k from whatever we have.
            if len(final_reranked_ids) < rerank_top_k:
                final_reranked_ids = (final_reranked_ids + retrieved_ids)[:rerank_top_k]
            else:
                final_reranked_ids = final_reranked_ids[:rerank_top_k]
            # --- End Padding Logic ---

            reranked_ids = final_reranked_ids # Use the padded list from now on
            yield f"TRACE: Top {len(reranked_ids)} reranked IDs (padded): {reranked_ids}"
            
            reranked_docs = self.retriever.get_docs_by_ids(reranked_ids)

            yield f"TRACE: Generating reasoned answer from top {len(reranked_docs)} docs (Useful Info: {useful_information != 'N/A'})..."
            reasoned_answer = self.reasoner.answer(question, reranked_docs, useful_information)
            yield f"REASONING::{reasoned_answer}"

            yield "TRACE: Extracting final answer..."
            extracted_answer = self.extractor.extract(question, reasoned_answer)
            yield f"EXTRACTED::{extracted_answer}"
            
            candidate_answers.append({"answer": extracted_answer, "doc_ids": reranked_ids})

            concrete_answers = [c['answer'] for c in candidate_answers if "information cannot be extracted" not in c['answer'].lower()]
            if len(concrete_answers) > 1 and len(set(concrete_answers)) == 1:
                yield f"TRACE: Conclusion: All concrete answers are identical ('{concrete_answers[0]}'). Ending search rounds early."
                break

            yield "TRACE: Reviewing extracted answer..."
            verdict, payload = self.reviewer.review(question, extracted_answer, reranked_docs)
            yield f"TRACE: Review verdict: {verdict} - Details: {payload or 'N/A'}"
            
            yield "TRACE: Summarizing useful information from this round..."
            new_summary = self.summarizer.summarize(question, reranked_docs)
            if new_summary:
                useful_information = f"{useful_information}\n{new_summary}".strip() if useful_information != "N/A" else new_summary
                yield f"TRACE: Updated useful information: {useful_information}"

            if verdict in ("GOOD", "OBVIOUS"):
                yield "TRACE: Conclusion: Answer is satisfactory. Ending search rounds."
                break
            
            if i == self.max_rounds - 1:
                yield "TRACE: Conclusion: Max rounds reached. Proceeding to final synthesis."
                break

            if verdict == "BAD":
                yield "TRACE: Verdict is BAD. Generating new search keys..."
                new_keys = self.key_searcher.generate_keys(question, context_docs, useful_information)
                yield f"TRACE: Generated new keys: {new_keys}"
                
                if not new_keys:
                    yield "TRACE: No new keys generated. Ending search."
                    break

                yield f"TRACE: Retrieving {initial_top_k} new documents..."
                retrieved_ids = self.retriever.retrieve(" ".join(new_keys), top_k=initial_top_k)
                all_retrieved_ids.update(retrieved_ids)
                context_docs = self.retriever.get_docs_by_ids(retrieved_ids)
                if context_callback: context_callback(context_docs)
                yield f"TRACE: Context now contains {len(retrieved_ids)} documents."
            else:
                yield "TRACE: Review verdict is UNKNOWN. Ending workflow."
                break
        
        yield "TRACE: --- Finalization Stage ---"
        
        concrete_answers = [c for c in candidate_answers if "information cannot be extracted" not in c['answer'].lower()]
        all_doc_ids = {doc_id for c in candidate_answers for doc_id in c['doc_ids']}
        
        if not concrete_answers:
            yield "TRACE: All rounds failed. Engaging Final Answer Generation Agent..."
            full_context_docs = self.retriever.get_docs_by_ids(list(all_doc_ids))
            generated_reasoning = self.generator.generate(question, full_context_docs)
            yield f"REASONING::{generated_reasoning}"
            yield "TRACE: Extracting from generated answer..."
            final_answer = self.extractor.extract(question, generated_reasoning)
            final_doc_ids = list(all_doc_ids)
        else:
            if len(set(c['answer'] for c in concrete_answers)) == 1:
                yield "TRACE: All concrete answers are identical. Skipping synthesis."
                final_answer = concrete_answers[-1]['answer']
                final_doc_ids = concrete_answers[-1]['doc_ids']
            else:
                yield "TRACE: Synthesizing the best answer from concrete candidates..."
                synthesized_answer = self.synthesizer.synthesize(question, concrete_answers, useful_information)
                yield f"EXTRACTED::{synthesized_answer}"
                final_answer = synthesized_answer
                
                final_doc_ids = concrete_answers[-1]['doc_ids']
                found_match = False
                for candidate in reversed(concrete_answers):
                    if candidate['answer'] == synthesized_answer:
                        final_doc_ids = candidate['doc_ids']
                        found_match = True
                        break
                if found_match:
                    yield "TRACE: Matched synthesized answer to a candidate. Using its specific doc list."
                else:
                    yield "TRACE: Synthesized answer is novel. Using doc list from last concrete answer as fallback."

        if "information cannot be extracted" in final_answer.lower():
            yield "TRACE: No definitive answer found. Engaging Final Guesser Agent..."
            full_context_docs = self.retriever.get_docs_by_ids(list(all_doc_ids))
            json_response_str = self.guesser.guess(question, full_context_docs, useful_information)
            yield f"TRACE: Guesser Agent Raw Output: {json_response_str}"

            try:
                cleaned_json_str = re.sub(r'```json\s*|\s*```', '', json_response_str, flags=re.DOTALL).strip()
                guesser_output = json.loads(cleaned_json_str)
                final_answer = guesser_output.get("answer", "Error: Missing 'answer' key in JSON.")
                final_doc_ids = guesser_output.get("supporting_doc_ids", list(all_doc_ids))
                yield "TRACE: Successfully parsed Guesser JSON."
            except json.JSONDecodeError:
                yield "TRACE: Fallback: Guesser agent did not return valid JSON. Using last concrete answer."
                final_answer = concrete_answers[-1]['answer'] if concrete_answers else "Information cannot be extracted from the context."
                final_doc_ids = concrete_answers[-1]['doc_ids'] if concrete_answers else list(all_doc_ids)

        if len(final_doc_ids) < rerank_top_k:
            yield f"TRACE: Final list has {len(final_doc_ids)} docs, need {rerank_top_k}. Performing intelligent padding..."
            needed = rerank_top_k - len(final_doc_ids)
            
            padding_candidates_ids = list(all_retrieved_ids - set(final_doc_ids))
            
            if padding_candidates_ids:
                yield f"TRACE: Reranking {len(padding_candidates_ids)} remaining documents to find the best {needed} to add..."
                padding_candidate_docs = self.retriever.get_docs_by_ids(padding_candidates_ids)
                
                # Use the reranker to find the best documents from the remaining pool
                reranked_padding_ids = self.reranker.rerank(question, padding_candidate_docs, useful_information)
                
                # Take the top `needed` docs and append them
                docs_to_add = reranked_padding_ids[:needed]
                final_doc_ids.extend(docs_to_add)
                yield f"TRACE: Padded with {len(docs_to_add)} documents: {docs_to_add}"
            else:
                yield "TRACE: No remaining documents to perform intelligent padding with."

        yield "TRACE: Simplifying final answer..."
        simplified_answer = self.simplifier.simplify(question, final_answer)
        yield f"EXTRACTED::{simplified_answer}"
        
        return simplified_answer, final_doc_ids[:rerank_top_k]

import os
import re
import json
import ollama
import openai
import requests

class RAGPrompt:
    """A centralized class to hold all prompt templates."""
    def __init__(self):
        self.self_check_template = """
        
You are a meticulous quality control assistant. Your task is to rigorously evaluate a generated answer based on a user's question and the retrieved context. You must check if ALL parts of the question are addressed by the context.

Here is the user's question:
{question}

Here is the retrieved context that the answer was based on:
{context}

Here is the generated answer:
{answer}

Respond on a SINGLE LINE with EXACTLY ONE of the following (no extra text):
- GOOD
- OBVIOUS ANSWER: <answer> | EVIDENCE: <brief quote from context that makes it obvious>
- BAD: <brief reason>

Rules:
- Before declaring an answer GOOD or OBVIOUS, you MUST verify that every single constraint and entity in the question is supported by the context.
- If the answer is an admission of failure (e.g., "cannot determine", "information not available", "context is insufficient"), you MUST output "BAD: The context is insufficient to answer the question.". This is critical to allow the system to search for more information.
- If the answer is directly and clearly supported by a specific fact in the context that addresses all parts of the question, output "OBVIOUS ANSWER: ... | EVIDENCE: ...".
- If the answer is a correct "yes" or "no" and is generally supported, output "GOOD".
- If the answer is wrong, incomplete, or relies on assumptions not in the context, output "BAD: <reason>". For example, if the question is about Person A and Person B, but the context only mentions Person A, the answer is BAD.
        """

        self.key_generation_template = """
You are an expert query decomposition assistant for multi-hop question answering.
Your task is to identify what information is MISSING from the provided evidence and propose up to {num_keys} short, distinct search keys to find that missing information.

Analyze the user's question, the summary of useful information found so far, and the evidence retrieved in the current round. Based on your analysis, generate search keys for the next hop of information needed to fully answer the question.

Rules:
- First, think about what is missing.
- Then, generate search keys that are concrete, specific, and target the missing entities or relationships.
- Output exactly one search key per line.
- No numbering, no bullet points, no extra text.

Question: {question}

Useful Information Found So Far:
---
{useful_information}
---

Evidence Retrieved This Round:
---
{context}
---

Based on the above, what information is still needed? What are the next search keys to find it?
Search Keys:
"""

        self.reasoning_answerer_template = """
You are a reasoning assistant. Your task is to answer the user's question based ONLY on the provided context and the summary of useful information.
Provide a step-by-step explanation of how you arrived at the answer, citing specific facts.
Conclude your explanation with the final, concise answer.

Rules:
- In the reasoning process, you could cite the specific facts from the context that support your answer.
- In the answering process, DO NOT answer with POSSIBLY MULTIPLE ANSWERS, the answer must be a single answer.
For example, "Who is the richest person in the world?", the answer MUST BE ONLY "Elon Musk", not "Jeff Bezos or Elon Musk", pick the most likely answer.
- State the possibility in the explanation, but the final answer must be a single answer.
- In the explanation, you could cite the specific facts from the context that support your answer.


Useful Information Found So Far:
---
{useful_information}
---

Context For This Round:
---
{context}
---

Question: {question}
Reasoned Answer:
"""

        self.answer_extractor_template = """
You are an answer extraction assistant.
Your task is to extract the final, concise answer from the reasoned answer provided below.

Rules:
- The final answer should be as short as possible while being accurate.
- For yes/no questions, the answer must be ONLY "yes" or "no".
- For questions asking style look like a yes/no question, but is not a yes/no question, the answer must be the answer to the question.
For example, "Was Tom or Alan more successful?" is not a yes/no question, the answer MUST BE ONLY "Tom" or "Alan", not "yes" or "no".
- For questions about numbers, it must be a single number.
For example, "What is the population of Tokyo?" is not a number question, the answer MUST BE ONLY "37400000", not "37400000 people".
- For questions about dates, it must be a single date, while date format can be reference to the context.
- For questions about names, it must be a single name.
For example, "What is the name of the president of the United States?" is not a name question, the answer MUST BE ONLY "Joe Biden", not "Joe Biden".
- For "Which" questions, it must be a single entity.
For example, "Which country is the largest in the world?" is not a which question, the answer MUST BE ONLY "Russia", not "Russia is the largest country in the world".
- For "Who" questions, it must be a single entity.
For example, "Who is the president of the United States?" is not a who question, the answer MUST BE ONLY "Joe Biden", not "Joe Biden is the president of the United States".
- For "What" questions, it must be a single entity.
For example, "What is the capital of France?" is not a what question, the answer MUST BE ONLY "Paris", not "Paris is the capital of France".
- For "How" questions, it must be a single number.
- If the reasoned answer indicates that the information is not available or the context is insufficient, your final answer must be EXACTLY "Information cannot be extracted from the context."
- Simplify the answer to the most direct and concise form.
For example, "Who killed John F. Kennedy?", the answer MUST BE ONLY "Lee Harvey Oswald", not "Lee Harvey Oswald killed John F. Kennedy".
Another example, "What is the capital of France?", the answer MUST BE ONLY "Paris", not "Paris is the capital of France".


Original Question: {question}

Reasoned Answer:
---
{reasoned_answer}
---

Final Answer:
"""
        # Adding the missing templates back
        self.answer_guess_template = """
You are an answer extraction assistant.
Given the original question and the evidence, extract a single best answer.
You must output a SINGLE JSON object with the following structure and no extra text:
{{
  "answer": "short direct answer string, no explanations"
}}

Rules:
- Answer strictly from the evidence; do not guess or add explanations.
- For yes/no questions, it must be ONLY "yes" or "no".
- For questions asking style look like a yes/no question, but is not a yes/no question, the answer must be the answer to the question.
For example, "Was Tom or Alan more successful?" is not a yes/no question, the answer MUST BE ONLY "Tom" or "Alan", not "yes" or "no".
- For questions about numbers, it must be a single number.
- For questions about dates, it must be a single date.
- For questions about names, it must be a single name.
- For questions about organizations, it must be a single organization.
- For questions about events, it must be a single event.
- For questions about places, it must be a single place.
- For questions about things, it must be a single thing.
- You must attempt to answer the question based on the provided evidence, even if the information is incomplete. Synthesize the most plausible answer from the context.

Question: {question}

Evidence:
---
{context}
---

JSON:
        """

        self.comparison_answer_template = """
You are a reasoning assistant for answering comparison questions.
Your task is to extract the relevant facts for each entity, compare them, and determine the final answer.
You MUST output a single JSON object with your reasoning and the final answer.

Example:
Question: Who has more F1 wins, Lewis Hamilton or Michael Schumacher?
Context: Lewis Hamilton has 103 wins. Michael Schumacher has 91 wins.
JSON:
{{
  "reasoning": "Based on the context, Lewis Hamilton has 103 wins and Michael Schumacher has 91 wins. Since 103 > 91, Lewis Hamilton has more wins.",
  "final_answer": "Lewis Hamilton"
}}

Question: {question}

Evidence:
---
{context}
---

JSON:
"""

        self.answer_simplifier_template = """
You are an expert at simplifying answers.
Your task is to extract the absolute shortest possible answer from the provided text, based on the original question.

Original Question: {question}

Answer to Simplify:
---
{answer}
---

Simplified Answer:
"""

        self.synthesizer_template = """
You are a synthesizer assistant. Your task is to analyze a list of candidate answers and determine the single best one, using the summary of useful information as a guide.

Rules:
- Review all candidate answers provided below.
- If one or more candidates provide a concrete answer, select the most accurate and comprehensive one based on the useful information.
- If ALL candidates suggest that the information is not available, you MUST respond with EXACTLY "Information cannot be extracted from the context."

Original Question: {question}

Useful Information Found So Far:
---
{useful_information}
---

Candidate Answers:
---
{candidate_answers}
---

Based on the candidates and the useful information, what is the single best answer?
Best Answer:
"""

        self.final_guesser_template = """
You are a pragmatic Q&A assistant. Your task is to provide the best possible answer to the user's question, even with limited information. You MUST format your response as a single JSON object.

Rules:
- You must make a best-effort guess to answer the question based on the provided context and the useful information summary.
- You must identify the top 10 most relevant document IDs from the context that support your guess.
- Your entire output must be a single JSON object with no extra text before or after.

The JSON object must have the following structure:
{{
  "answer": "Your best-effort, guessed answer to the question.",
  "supporting_doc_ids": ["doc_id_1", "doc_id_2", "doc_id_3", "doc_id_4", "doc_id_5", "doc_id_6", "doc_id_7", "doc_id_8", "doc_id_9", "doc_id_10"]
}}

Original Question: {question}

Useful Information Found So Far:
---
{useful_information}
---

Full Context (with Document IDs):
---
{context}
---

JSON Output:
"""

        self.summarizer_template = """
You are a fact summarization assistant. Your task is to read the provided context and extract a list of key facts relevant to the user's question, citing the source for each fact.

Rules:
- For each distinct piece of information, cite its Document ID.
- Be concise and to the point.
- If no relevant facts are found, output "No useful information found."

Example:
Original Question: Who earns more, Tom or Alan?
Context:
ID: doc-1234
Tom's salary is 5000 USD per month.

ID: doc-2345
Alan makes 8000 CAD a year.
---
Summary of Useful Information:
- Tom's salary is 5000 USD per month (from doc-1234).
- Alan's salary is 8000 CAD a year (from doc-2345).

Original Question: {question}

Context:
---
{context}
---

Summary of Useful Information:
"""

        self.final_answer_generator_template = """
You are a reasoning assistant.
All previous attempts to answer the question have failed. This is the last attempt.
Your task is to synthesize a final answer based ONLY on the entire context provided below.
Provide a step-by-step explanation of how you arrive at the answer, then conclude with the final, concise answer.

If the information is truly not in the context, your final answer must be EXACTLY "Information cannot be extracted from the context."

Context:
---
{context}
---

Question: {question}
Reasoned Answer:
"""

        self.reranker_template = """
You are a document reranking assistant. Your task is to analyze a list of retrieved documents and create a ranked list of EXACTLY 10 document IDs.

Rules:
- Your primary goal is to rank the documents based on their relevance to the user's question and the facts gathered so far.
- Prioritize documents that directly address the question or fill in missing information.
- Give high priority to documents already cited as sources.
- Your output MUST be a single line containing a comma-separated list of EXACTLY 10 document IDs.
- If you believe fewer than 10 of the provided documents are relevant, you MUST fill the list with the next-best available documents until the list contains 10 IDs. Do not return a list with fewer than 10 IDs.
- Do NOT include any explanation or extra text.

Example Output:
doc-1234,doc-5678,doc-1122,doc-3344,doc-9988,doc-7766,doc-5544,doc-1212,doc-3434,doc-8877

Original Question: {question}

Useful Information Found So Far:
---
{useful_information}
---

Retrieved Documents:
---
{context}
---

Top 10 Document IDs (comma-separated):
"""

        self.special_guess_template = """
You are a pragmatic Q&A assistant. Your task is to answer the question based on the limited context.
Output ONLY the most likely answer based on the text. Do not add any extra words, explanations, or apologies.
Do not say "Based on the context" or mention that the information is limited. Just state the answer.
"""

class OllamaGenerator:
    """Generator that uses a local Ollama model."""
    def __init__(self, model_name="qwen2.5:7b-instruct"):
        self.model_name = model_name
        ollama_host = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
        self.client = ollama.Client(host=ollama_host)
        print(f"OllamaGenerator initialized with model: {self.model_name}")

    def generate(self, prompt, stop=None):
        options = {'num_predict': 1000, 'temperature': 0.0}
        if stop:
            options['stop'] = stop
        response = self.client.generate(model=self.model_name, prompt=prompt, options=options)
        return response['response']

class OpenRouterGenerator:
    """Generator that uses the OpenRouter API."""
    def __init__(self, model_name="qwen/qwen-2.5-7b-instruct"):
        self.model_name = model_name
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key: raise ValueError("OPENROUTER_API_KEY not set.")
        self.client = openai.OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key, timeout=30.0)
        print(f"OpenRouterGenerator initialized with model: {self.model_name}")

    def generate(self, prompt, stop=None):
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=1000,
            stop=stop
        )
        return response.choices[0].message.content.strip()

class HFGenerator:
    """Generator that uses the Hugging Face text generation inference router with an OpenAI-compatible client."""

    def __init__(self, model_name: str = "Qwen/Qwen2.5-7B-Instruct"):
        self.model_name = model_name
        api_key = os.environ.get("HUGGINGFACE_API_KEY")
        if not api_key:
            raise ValueError("HUGGINGFACE_API_KEY not set.")
        # Hugging Face OpenAI-compatible router endpoint
        self.client = openai.OpenAI(
            base_url="https://router.huggingface.co/v1",
            api_key=api_key,
            timeout=60.0,
        )

    def generate(self, prompt: str, stop=None) -> str:
        """
        Call the HF router using the OpenAI chat.completions interface.
        Mirrors the interface of the other Generator classes: returns a plain string.
        """
        kwargs = dict(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=1000,
        )
        if stop:
            kwargs["stop"] = stop

        completion = self.client.chat.completions.create(**kwargs)
        # Align with other generators by returning only the message content as a string
        return (completion.choices[0].message.content or "").strip()


class POEGenerator:
    """Generator that calls the Poe API via an OpenAI-compatible client."""

    def __init__(self, model_name: str = "qwen-2.5-7b-t"):  # Poe's Qwen 2.5 7B Turbo model, but without instruction tuning (POE did not have an instruct variant at time of writing)
        self.model_name = model_name
        api_key = os.environ.get("POE_API_KEY")
        if not api_key:
            raise ValueError("POE_API_KEY not set.")
        self.client = openai.OpenAI(
            base_url="https://api.poe.com/v1",
            api_key=api_key,
            timeout=60.0,
        )

    def generate(self, prompt: str, stop=None) -> str:
        """
        Use Poe's OpenAI-compatible /chat/completions endpoint.
        Returns just the assistant message content as a string.
        """
        kwargs = dict(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=1000,
        )
        if stop:
            kwargs["stop"] = stop

        chat = self.client.chat.completions.create(**kwargs)
        return (chat.choices[0].message.content or "").strip()
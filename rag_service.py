"""Module 8 — Integration Task: RAG Service."""

import json
import os
import re
from typing import List

import weaviate
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from index_helpers import bm25_search, dense_search, hybrid_search  # noqa: F401


CLASS_NAME = "Post"
WEAVIATE_URL = os.getenv("WEAVIATE_URL", "http://localhost:8080")
GENERATOR_MODEL = "google/flan-t5-base"
EMBEDDER_MODEL = "all-MiniLM-L6-v2"

ABSTAIN_PHRASES = [
    "i don't know",
    "i do not know",
    "not in the context",
    "the context does not",
    "cannot be answered",
    "no information",
]

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "but", "by",
    "can", "could", "did", "do", "does", "for", "from", "had", "has",
    "have", "he", "her", "him", "his", "i", "if", "in", "into", "is",
    "it", "its", "just", "may", "me", "might", "my", "no", "nor", "not",
    "of", "on", "or", "our", "out", "over", "she", "so", "some", "such",
    "than", "that", "the", "their", "them", "then", "there", "these",
    "they", "this", "those", "to", "too", "under", "until", "up", "was",
    "we", "were", "what", "when", "where", "which", "while", "who",
    "will", "with", "would", "you", "your",
}


_tokenizer = AutoTokenizer.from_pretrained(GENERATOR_MODEL)
_model = AutoModelForSeq2SeqLM.from_pretrained(GENERATOR_MODEL)
_embedder = SentenceTransformer(EMBEDDER_MODEL)

_client: weaviate.Client | None = None


def _get_client() -> weaviate.Client:
    global _client
    if _client is None:
        _client = weaviate.Client(WEAVIATE_URL)
    return _client


_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def _tokenize_for_groundedness(text: str) -> set[str]:
    tokens = _TOKEN_RE.findall(text.lower())
    return {token for token in tokens if token and token not in STOPWORDS}


def _whole_word_match(keyword: str, answer: str) -> bool:
    pattern = r"\b" + re.escape(keyword) + r"\b"
    return re.search(pattern, answer, flags=re.IGNORECASE) is not None


def retrieve(query: str, k: int = 5) -> List[dict]:
    client = _get_client()

    doc_ids = hybrid_search(
        client,
        query,
        k,
        _embedder,
        alpha=0.5,
    )

    results = []

    for doc_id in doc_ids:
        response = (
            client.query
            .get(CLASS_NAME, ["doc_id", "title", "answer_text"])
            .with_where({
                "path": ["doc_id"],
                "operator": "Equal",
                "valueText": doc_id,
            })
            .with_limit(1)
            .do()
        )

        posts = response.get("data", {}).get("Get", {}).get(CLASS_NAME) or []

        if posts:
            post = posts[0]
            results.append({
                "doc_id": post.get("doc_id", ""),
                "title": post.get("title", ""),
                "answer_text": post.get("answer_text", ""),
            })

    return results



def build_prompt(query: str, contexts: List[dict]) -> str:
    context_lines = []

    for i, context in enumerate(contexts, start=1):
        title = context.get("title", "")
        answer_text = context.get("answer_text", "")
        truncated_answer = " ".join(answer_text.split()[:80])
        context_lines.append(f"[{i}] {title}: {truncated_answer}")

    context_block = "\n".join(context_lines)

    return (
        "Answer the question using only the context. If the context does not contain the answer, say \"I don't know.\"\n\n"
        "Context:\n"
        f"{context_block}\n\n"
        f"Question: {query}\n"
        "Answer:"
    )


def generate(prompt: str) -> str:
    inputs = _tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )

    outputs = _model.generate(
        **inputs,
        max_new_tokens=128,
        num_beams=1,
    )

    return _tokenizer.decode(outputs[0], skip_special_tokens=True)


def rag_pipeline(query: str, k: int = 5) -> dict:
    contexts = retrieve(query, k)
    prompt = build_prompt(query, contexts)
    answer = generate(prompt)

    return {
        "query": query,
        "answer": answer,
        "contexts": contexts,
        "prompt": prompt,
    }


def groundedness_score(answer: str, contexts: List[dict]) -> float:
    answer_tokens = [
        token for token in _TOKEN_RE.findall(answer.lower())
        if token and token not in STOPWORDS
    ]

    if not answer_tokens:
        return 0.0

    context_text = " ".join(
        context.get("answer_text", "") for context in contexts
    )

    context_tokens = set(
        token for token in _TOKEN_RE.findall(context_text.lower())
        if token and token not in STOPWORDS
    )

    overlap_count = sum(
        1 for token in answer_tokens if token in context_tokens
    )

    return overlap_count / len(answer_tokens)


def evaluate_rag(eval_path: str) -> dict:
    with open(eval_path, "r", encoding="utf-8") as file:
        rows = [json.loads(line) for line in file if line.strip()]

    answerable_recalls = []
    answerable_groundedness = []
    borderline_abstentions = []
    borderline_groundedness = []
    per_question = []

    for row_index, row in enumerate(rows):
        question = row["question"]
        difficulty = row["difficulty"]
        expected_keywords = row.get("expected_answer_keywords", [])

        result = rag_pipeline(question)
        answer = result["answer"]
        contexts = result["contexts"]
        groundedness = groundedness_score(answer, contexts)

        record = {
            "row_index": row_index,
            "difficulty": difficulty,
            "question": question,
            "answer": answer,
            "groundedness": groundedness,
        }

        if difficulty in {"single_fact", "single_doc_synthesis"}:
            matched_keywords = [
                keyword
                for keyword in expected_keywords
                if _whole_word_match(keyword, answer)
            ]

            recall = (
                len(matched_keywords) / len(expected_keywords)
                if expected_keywords else 0.0
            )

            answerable_recalls.append(recall)
            answerable_groundedness.append(groundedness)
            record["matched_keywords"] = matched_keywords

        elif difficulty == "borderline":
            answer_lower = answer.lower()

            phrase_abstained = any(
                phrase in answer_lower for phrase in ABSTAIN_PHRASES
            )

            short_low_groundedness = (
                len(answer.strip()) <= 20 and groundedness <= 0.2
            )

            abstained = phrase_abstained or short_low_groundedness

            borderline_abstentions.append(1.0 if abstained else 0.0)
            borderline_groundedness.append(groundedness)
            record["abstained"] = abstained

        per_question.append(record)

    return {
        "answer_keyword_recall_main": (
            sum(answerable_recalls) / len(answerable_recalls)
            if answerable_recalls else 0.0
        ),
        "borderline_abstain_rate": (
            sum(borderline_abstentions) / len(borderline_abstentions)
            if borderline_abstentions else 0.0
        ),
        "mean_groundedness_main": (
            sum(answerable_groundedness) / len(answerable_groundedness)
            if answerable_groundedness else 0.0
        ),
        "mean_groundedness_borderline": (
            sum(borderline_groundedness) / len(borderline_groundedness)
            if borderline_groundedness else 0.0
        ),
        "per_question": per_question,
    }
if __name__ == "__main__":
    results = evaluate_rag("data/rag_eval.jsonl")

    print("Keyword Recall:", results["answer_keyword_recall_main"])
    print("Abstain Rate:", results["borderline_abstain_rate"])
    print("Groundedness Main:", results["mean_groundedness_main"])
    print("Groundedness Borderline:", results["mean_groundedness_borderline"])

    for row in results["per_question"]:
        print("=" * 50)
        print("Question:", row["question"])
        print("Answer:", row["answer"])
        print("Groundedness:", row["groundedness"])
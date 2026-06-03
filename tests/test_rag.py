"""Learner-written tests for the RAG pipeline."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rag_service import build_prompt, groundedness_score, retrieve


def test_retrieve_returns_k_or_fewer():
    results = retrieve("git rebase", k=5)

    assert isinstance(results, list)
    assert len(results) <= 5

    for item in results:
        assert isinstance(item, dict)
        assert "doc_id" in item
        assert "title" in item
        assert "answer_text" in item


def test_build_prompt_includes_context():
    contexts = [
        {
            "doc_id": "1",
            "title": "Git Rebase",
            "answer_text": "Rebase rewrites commit history.",
        }
    ]

    prompt = build_prompt("What is rebase?", contexts)

    assert "Answer the question using only the context." in prompt
    assert "[1] Git Rebase: Rebase rewrites commit history." in prompt
    assert "Question: What is rebase?" in prompt
    assert prompt.endswith("Answer:")


def test_groundedness_zero_for_unrelated_answer():
    contexts = [
        {"answer_text": "the quick brown fox jumps over the lazy dog"}
    ]

    score = groundedness_score("xyzzy plugh quux", contexts)

    assert score <= 0.1
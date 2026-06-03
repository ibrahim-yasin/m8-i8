# Module 8 Integration — Retrieval-Augmented Service

## Overview

This project implements a complete Retrieval-Augmented Generation (RAG) service using Weaviate as the retrieval layer and Google FLAN-T5 Base as the generator. The pipeline retrieves the top-k relevant documents from a technical Q&A corpus using hybrid retrieval (BM25 + dense retrieval), injects the retrieved context into a structured prompt, generates an answer using FLAN-T5 Base, and evaluates the system using groundedness, keyword recall, and abstention metrics on a 30-question evaluation set.

## Setup

1. Start Weaviate locally:

```bash
docker run -d --name weaviate-int \
  -p 8080:8080 \
  -e DEFAULT_VECTORIZER_MODULE=none \
  -e ENABLE_MODULES= \
  -e PERSISTENCE_DATA_PATH=/var/lib/weaviate \
  semitechnologies/weaviate:1.24.10
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Ingest the corpus:

```bash
python ingest.py
```

4. Run tests:

```bash
pytest tests/ -v
```

## How to Run

Example usage:

```python
from rag_service import rag_pipeline

result = rag_pipeline(
    "What HTML attribute did HTML4 use to set the content-type via a meta tag?"
)

print(result)
```

Response format:

```python
{
    "query": "...",
    "answer": "...",
    "contexts": [...],
    "prompt": "..."
}
```

The pipeline follows four stages:

1. Retrieve top-k documents from Weaviate.
2. Build a context-injected prompt.
3. Generate an answer using FLAN-T5 Base.
4. Return the answer, prompt, and retrieved contexts.

## Evaluation Output

Canonical evaluation results:

```text
answer_keyword_recall_main: 0.0
borderline_abstain_rate: 0.8
mean_groundedness_main: 0.6133
mean_groundedness_borderline: 0.7333
```

Example evaluation rows:

```text
Question:
What HTML attribute did HTML4 use to set the content-type via a meta tag?

Answer:
I don't know.

Groundedness:
1.0
```

```text
Question:
Which Mach exception name on iOS signals a memory access violation often surfaced during PhoneGap testing?

Answer:
pattern

Groundedness:
1.0
```

The low answer keyword recall is expected under the canonical methodology because FLAN-T5 Base frequently abstains even when the correct answer exists in the retrieved context.

## Known Limitations

1. The baseline answer_keyword_recall_main is intentionally low under the canonical prompt configuration. FLAN-T5 Base frequently abstains on answerable questions, resulting in recall values typically between 0.00 and 0.10.

2. FLAN-T5 Base has a 512-token input limit. Retrieved contexts must be truncated, which can remove useful information and reduce answer quality.

3. The groundedness_score implementation relies on token overlap and does not properly recognize paraphrases or semantically equivalent wording.

4. The STOPWORDS list is English-only and may not generalize well to multilingual corpora.

5. Hybrid retrieval quality depends on the underlying embeddings and corpus coverage. If relevant information is not retrieved, the generator cannot produce a correct answer.

6. Greedy decoding (num_beams=1) improves reproducibility but may reduce answer quality compared to more advanced decoding strategies.

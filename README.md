# RAG Translation Server

A **Retrieval-Augmented Generation (RAG)** backend for translation prompts, suitable for few-shot prompt techniques, built with Python and FastAPI.  

Given a source sentence and a language pair, the server retrieves the most similar translation examples from its vector database and constructs a structured prompt that can be fed directly into any LLM translator.

---

## Features

| Feature | Details |
|---|---|
| **Store translation pairs** | Store translation pairs (source - target) in a thread-safe in-memory vector database |
| **Vector similarity search** | TF-IDF (char n-gram) + FAISS cosine similarity |
| **RAG-based prompts** | Retrieve RAG-based prompts for few-shot translation using similarity search. |
| **Language-aware retrieval** | Separate index per `(src, tgt)` language pair |
| **Stammering detection** | Multi-signal algorithm (n-gram repetition, char-level, length anomaly) |
| **Docker support** | Single-container |

---

## Architecture

```
rag-translation-server/
├── app/
│   ├── main.py          # FastAPI routes
│   ├── models.py        # Pydantic request/response models
│   ├── database.py      # Vector DB management (TF-IDF + FAISS)
│   └── stammering.py    # Stammering detection algorithm
├── data/
│   ├── translation_pairs.jsonl      # Sample pairs for /pairs
│   ├── translation_requests.jsonl   # Sample queries for /prompt
│   └── stammering_tests.jsonl       # Test cases for /stammering
├── requirements.txt
├── Dockerfile
```

### Similarity Search

Each `(source_language, target_language)` pair gets its own FAISS index.  
Source sentences are vectorised with a **character n-gram TF-IDF** (n=2–4, sublinear TF), which is language-agnostic and handles morphological variation well.  
Vectors are L2-normalised so inner product equals **cosine similarity**.  
The top-K (max 4) most similar pairs are returned.

### Stammering Detection

Five complementary signals are computed and summed into a score:

1. **Repeated token n-grams** (n = 1–6) — weighted by n-gram size and excess count  
2. **Repeated character windows** (10–40 chars) — catches sub-word / morphological repetitions  
3. **Unique-token ratio** — low ratio ⟹ heavy duplication  
4. **Consecutive duplicate tokens / bigrams** — near-certain stammer  
5. **Length anomaly** — translation >3.5× longer than source  

A total score ≥ 1.0 is flagged as stammering.

---

## Quick Start

### Docker

```bash
# Docker (build + start):
docker compose up --build -d

# the following times
docker compose up -d

# stop
docker compose down
```

---

## API Reference

### `POST /pairs` — Add a translation pair

**Query params:** `source_language`, `target_language`, `sentence`, `translation`


```json
{
  "source_language": "en",
  "target_language": "it",
  "sentence": "How are you?",
  "translation": "Come stai?"
}
```

**Response:** `{"status": "ok"}`

---

### `GET /prompt` — Get a RAG translation prompt

**Query params:** `source_language`, `target_language`, `sentence`

```
GET /prompt?source_language=en&target_language=it&sentence=Good+morning
```

**Response:**
```json
{
  "prompt": "You are a professional translator...\nExample 1 (similarity: 0.87):\n  English: Good morning everyone\n  Italian: Buongiorno a tutti\n..."
}
```

**Notes**:

Returns top-K (up to 4) most similar translation examples from the database.
Prompt includes human-readable examples and the target sentence placeholder.

---

### `GET /stammering` — Detect stammering

**Query params:** `sentence`, `translation`

```
GET /stammering?sentence=Hello+world&translation=Ciao+ciao+ciao+ciao+mondo+mondo
```

**Response:** `{"has_stammer": true}`

---

### `GET /health` — Health check

```json
{"status": 42}
```
**Notes**:

status is the total number of translation pairs stored in the vector database.
Useful for monitoring container health in production.

---

## Using the Client Script

The provided `client.py` can populate the database, request prompts, and run stammering tests interactively.

```bash
python client.py
```

The script expects these files (paths configurable in `FILES` dict):
- `translation_pairs.jsonl`
- `translation_requests.jsonl`
- `stammering_tests.jsonl`

Sample files are provided under `data/`.

---

## Interactive Docs

With the server running, visit:
- **Swagger UI:** http://localhost:8000/docs
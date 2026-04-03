from __future__ import annotations

import logging
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import numpy as np

import faiss
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)

@dataclass
class _LanguageShard:
    """Stores all translation pairs for one (src, tgt) language pair."""

    records: list[dict] = field(default_factory=list)
    # Fitted vectoriser (re-fit whenever new records are added)
    vectorizer: Any = None
    # Dense matrix of TF-IDF vectors  shape (n, d) – numpy float32
    matrix: np.ndarray | None = None
    # FAISS flat index (cosine via Inner Product on L2-normalised vecs)
    index: Any = None

    def rebuild_index(self) -> None:
        """Re-vectorise all source sentences and rebuild the FAISS / numpy index."""
        
        if not self.records:
            return

        sentences = [r["sentence"] for r in self.records]

        if self.vectorizer is None:
            self.vectorizer = TfidfVectorizer(analyzer="char_wb",
                                              ngram_range=(2, 4),
                                              sublinear_tf=True,
                                              min_df=1)
            
        self.vectorizer.fit(sentences)
        raw = self.vectorizer.transform(sentences).toarray().astype(np.float32)

        # L2-normalise so inner product == cosine similarity
        norms = np.linalg.norm(raw, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.matrix = raw / norms

        d = self.matrix.shape[1]
        self.index = faiss.IndexFlatIP(d)
        self.index.add(self.matrix)

    def query(self, query_sentence: str, top_k: int = 4) -> list[dict]:
        """Return up to top_k records ordered by descending cosine similarity."""

        if not self.records:
            return []
        
        if self.vectorizer is None:
            self.rebuild_index()

        # Vectorise the query
        qvec = self.vectorizer.transform([query_sentence]).toarray().astype(np.float32)

        norm = np.linalg.norm(qvec)
        if norm > 0:
            qvec = qvec / norm

        n = len(self.records)
        k = min(top_k, n)

        if self.index is not None:
            scores, indices = self.index.search(qvec, k)  # type: ignore[arg-type]
            scores = scores[0]
            indices = indices[0]
        else:
            # numpy fallback
            sims = (self.matrix @ qvec.T).flatten()
            indices = np.argsort(sims)[::-1][:k]
            scores = sims[indices]

        results = []
        for idx, score in zip(indices, scores):
            if idx < 0:
                continue
            rec = dict(self.records[idx])
            rec["score"] = float(score)
            results.append(rec)

        return results


#  VectorDatabase
class VectorDatabase:
    """Thread-safe in-memory vector database for translation pairs."""

    def __init__(self) -> None:
        self._shards: dict[tuple[str, str], _LanguageShard] = defaultdict(_LanguageShard)
        self._lock = threading.Lock()

    # public API
    def add_pair(self, pair) -> None:
        key = (pair.source_language.lower(), pair.target_language.lower())
        with self._lock:
            shard = self._shards[key]
            shard.records.append({
                "source_language": pair.source_language,
                "target_language": pair.target_language,
                "sentence": pair.sentence,
                "translation": pair.translation,
            })
            if len(shard.records) % 50 == 0:    
                shard.rebuild_index()

    def search(self,
               source_language: str,
               target_language: str,
               query: str,
               top_k: int = 4) -> list[dict]:
        
        key = (source_language.lower(), target_language.lower())
        with self._lock:
            shard = self._shards.get(key)
            if shard is None or not shard.records:
                return []
            return shard.query(query, top_k)

    def count(self) -> int:
        with self._lock:
            return sum(len(s.records) for s in self._shards.values())
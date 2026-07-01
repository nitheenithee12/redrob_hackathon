"""
semantic.py
===========
The "beyond keyword" recall/semantic layer.

Two paths are provided:

1. DEFAULT — TF-IDF cosine (scikit-learn). Zero downloads, zero network, fits
   the 5-min CPU budget on 100K docs comfortably, and is 100% reproducible on
   any machine. We fit a word 1-2gram TF-IDF over the *candidate corpus itself*
   so the vocabulary is data-driven, then score each candidate against a dense
   JD query vector. This captures phrase-level relevance (e.g. "embedding-based
   retrieval" scoring against a JD query that mentions "embeddings", "retrieval",
   "vector search") beyond a single exact keyword, while remaining an auditable,
   deterministic transform.

2. OPTIONAL — local sentence-embedding re-rank (`--embeddings`). If a
   sentence-transformers model is already cached on disk (no network at ranking
   time), we can compute dense cosine similarity for a truer semantic signal.
   This is disabled by default so the guaranteed-reproducible path never depends
   on model weights being present. See README for how to enable it.

Only the semantic *tilt* (8% of the score, see scoring.py) comes from here; role
logic stays in charge. That is intentional: pure embedding similarity is exactly
what ranks the JD's keyword-stuffer and honeypot traps highly.
"""

from __future__ import annotations

from typing import List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


class TfidfSemantic:
    def __init__(self, jd_query: str):
        self.jd_query = jd_query
        self.vectorizer: TfidfVectorizer | None = None
        self._jd_vec = None

    def fit(self, corpus: List[str]) -> None:
        self.vectorizer = TfidfVectorizer(
            max_features=40000,
            ngram_range=(1, 2),
            min_df=3,
            sublinear_tf=True,
            stop_words="english",
        )
        self.vectorizer.fit(corpus)
        self._jd_vec = self.vectorizer.transform([self.jd_query])

    def similarity(self, texts: List[str]) -> np.ndarray:
        """Cosine similarity of each text to the JD query, in [0, 1]."""
        assert self.vectorizer is not None and self._jd_vec is not None
        X = self.vectorizer.transform(texts)               # rows are L2-normalised
        sims = (X @ self._jd_vec.T).toarray().ravel()      # cosine (both normalised)
        # spread the distribution a little so it is a useful tilt, not a flat 0.05
        return np.clip(sims * 3.0, 0.0, 1.0)


def try_embedding_semantic(jd_query: str, texts: List[str], model_dir: str):
    """Optional dense-embedding path. Only used when explicitly requested AND a
    local model is available. Never downloads at ranking time.

    Returns an array of cosine similarities in [0,1], or None if unavailable.
    """
    try:
        from sentence_transformers import SentenceTransformer  # noqa: WPS433
    except Exception:
        return None
    try:
        model = SentenceTransformer(model_dir)  # a local path; no network
        q = model.encode([jd_query], normalize_embeddings=True)
        emb = model.encode(texts, normalize_embeddings=True, batch_size=256)
        sims = (emb @ q.T).ravel()
        return np.clip((sims + 1.0) / 2.0, 0.0, 1.0)
    except Exception:
        return None

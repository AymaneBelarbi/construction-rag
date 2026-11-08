"""INDEX, step 1: turn text into vectors.

An embedder is any function that takes a list of texts and returns one list of
floats per text. Keeping it a plain function means tests can pass a fake and
real providers (see llm.py) can be swapped in without touching other files.

The default embedder here is a "hashing" embedder: no model, no network. It
counts words into a fixed number of buckets. It is weaker than a neural
embedding on meaning, but it is deterministic, free, and good enough to
exercise the whole pipeline. Hybrid search (retrieve.py) adds keyword matching
on top, which covers exact IDs.
"""
import hashlib
import math
import re
from collections.abc import Callable, Mapping

Embedder = Callable[[list[str]], list[list[float]]]

DIMENSIONS = 256
# Saved with the index. Bump the "v3" whenever tokenize() or _bucket() changes, so old
# cached vectors are not reused with the new logic.
HASHING_ID = f"hashing-v3-{DIMENSIONS}"
# A word is letters and digits in any language (Müller, café); hyphens join the parts of an ID or date.
_TOKEN = re.compile(r"[^\W_]+(?:-[^\W_]+)*")


# Words that say nothing about which record is meant. Keyword search drops them from the QUESTION in every
# store (PostgreSQL's ranking has no notion of rare words, and BM25 would score them low anyway).
STOPWORDS = frozenset(
    "a an the is are was were be been am do does did have has had of in on at to for from by with and or "
    "which what who whom whose when where how why that this these those it its as any all some me my we "
    "show list give tell please".split()
)


def _is_lone_letter(token: str) -> bool:
    return len(token) == 1 and token.isalpha()


def tokenize(text: str) -> list[str]:
    """Lowercase words. A hyphenated ID like 'rfi-003' is kept whole AND split into 'rfi' and '003', so an
    exact-ID search and a loose search both work. A single letter ("contractor's" leaves an 's') is not a word."""
    tokens: list[str] = []
    for match in _TOKEN.findall(text.lower()):
        tokens.append(match)
        if "-" in match:
            tokens.extend(part for part in match.split("-") if not _is_lone_letter(part))
    return [token for token in tokens if not _is_lone_letter(token)]


def tokenize_query(text: str) -> list[str]:
    """Words of a question for KEYWORD search. A hyphenated ID stays one exact token (rfi-003) and is not
    split, so the query RFI-003 cannot match a record that only mentions RFI-011 and CE-003. Common words
    (STOPWORDS) are dropped."""
    return [token for token in _TOKEN.findall(text.lower())
            if not _is_lone_letter(token) and token not in STOPWORDS]


# Vector matches weaker than this are ignored (and, when nothing else matches, the answer is a refusal).
# The right value depends on the embedding model: unrelated texts score near 0 with the offline hashing
# embedder and noticeably higher with real models. The OpenAI figure was calibrated on 2026-11-07 against the
# ingested sample project (`python -m evals.retrieval_eval --store postgres --real-embeddings --verbose`): twelve
# held-out questions unrelated to construction scored at most 0.26, legitimate questions at least 0.36. 0.3 sits
# between them. Questions near the topic that the records cannot answer ("is the project over budget", 0.34)
# score above it; no cut-off separates those from weak real questions, so they are reported by the eval, not gated.
_DEFAULT_MIN_VECTOR_SCORE = 0.2
_MIN_VECTOR_SCORE_BY_MODEL = {
    HASHING_ID: 0.32,  # 256 buckets are noisy: a three-word nonsense query reaches 0.30 by hash collisions
    "text-embedding-3-small": 0.3,
}


def min_vector_score_for(embedder_id: str, env: Mapping[str, str]) -> float:
    """The similarity cut-off for this embedding model. RAG_MIN_VECTOR_SCORE overrides it (a number from 0 to 1)."""
    override = env.get("RAG_MIN_VECTOR_SCORE", "").strip()
    if override:
        try:
            value = float(override)
        except ValueError:
            raise ValueError(f"RAG_MIN_VECTOR_SCORE must be a number from 0 to 1, got {override!r}") from None
        if not 0.0 <= value <= 1.0:  # also rejects nan
            raise ValueError(f"RAG_MIN_VECTOR_SCORE must be a number from 0 to 1, got {override!r}")
        return value
    model = embedder_id.rsplit(":", 1)[-1]  # "api:<base url>:<model>" -> "<model>"
    return _MIN_VECTOR_SCORE_BY_MODEL.get(embedder_id, _MIN_VECTOR_SCORE_BY_MODEL.get(model, _DEFAULT_MIN_VECTOR_SCORE))


def _bucket(token: str) -> int:
    # hashlib, not the built-in hash(): Python randomises hash() on every run,
    # which would make saved vectors useless after a restart.
    digest = hashlib.md5(token.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") % DIMENSIONS


def hashing_embed(texts: list[str]) -> list[list[float]]:
    vectors: list[list[float]] = []
    for text in texts:
        vector = [0.0] * DIMENSIONS
        for token in tokenize(text):
            if token not in STOPWORDS:  # "for" and "the" must not make unrelated texts look alike
                vector[_bucket(token)] += 1.0
        length = math.sqrt(sum(x * x for x in vector))
        if length > 0:
            vector = [x / length for x in vector]  # unit length, so dot product = cosine similarity
        vectors.append(vector)
    return vectors

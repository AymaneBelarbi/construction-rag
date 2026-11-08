"""BM25 keyword scoring, used by the in-memory store.

(The PostgreSQL store uses the database's own full-text ranking instead.)"""
import math
from collections import Counter


class Bm25:
    """BM25 scores how well each document matches the query words. Rare words
    count more than common ones, and a word repeated in a short document counts
    more than the same word in a long one."""

    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.term_counts = [Counter(doc) for doc in docs]
        self.lengths = [len(doc) for doc in docs]
        self.avg_length = (sum(self.lengths) / len(docs)) if docs else 0.0
        self.doc_freq: Counter[str] = Counter()
        for counts in self.term_counts:
            self.doc_freq.update(counts.keys())  # in how many documents does each word appear?

    def scores(self, query_tokens: list[str]) -> list[float]:
        total_docs = len(self.term_counts)
        result: list[float] = []
        for counts, length in zip(self.term_counts, self.lengths, strict=True):
            score = 0.0
            for token in set(query_tokens):
                tf = counts.get(token, 0)
                if tf == 0:
                    continue
                df = self.doc_freq[token]
                idf = math.log(1 + (total_docs - df + 0.5) / (df + 0.5))
                norm = tf + self.k1 * (1 - self.b + self.b * length / self.avg_length)
                score += idf * tf * (self.k1 + 1) / norm
            result.append(score)
        return result

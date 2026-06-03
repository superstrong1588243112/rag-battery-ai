import math
import os
import pickle
from collections import defaultdict

import faiss
import jieba
import numpy as np
from langchain.docstore.document import Document
from sklearn.feature_extraction.text import TfidfVectorizer

from config import Config


class HybridVectorDB:
    def __init__(self, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.documents = []
        self.doc_lengths = []
        self.avg_doc_length = 0
        self.inverted_index = defaultdict(list)
        self.doc_count = 0
        self.vectorizer = None
        self.index = None
        self.embedding_dim = 0

    @staticmethod
    def has_index(path):
        return (
            os.path.exists(os.path.join(path, "index.faiss"))
            and os.path.exists(os.path.join(path, "meta.pkl"))
        )

    def _tokenize(self, text):
        import re

        text = re.sub(r"[^\w\s\u4e00-\u9fff]", " ", text)
        words = jieba.lcut(text)
        return [word.lower() for word in words if len(word.strip()) > 1]

    def _build_bm25(self, documents):
        self.documents = documents
        self.doc_count = len(documents)
        self.doc_lengths = []
        self.inverted_index = defaultdict(list)

        for doc_id, doc in enumerate(documents):
            content = doc.page_content if isinstance(doc, Document) else str(doc)
            tokens = self._tokenize(content)
            self.doc_lengths.append(len(tokens))

            counts = defaultdict(int)
            for token in tokens:
                counts[token] += 1
            for token, count in counts.items():
                self.inverted_index[token].append((doc_id, count))

        self.avg_doc_length = (
            sum(self.doc_lengths) / len(self.doc_lengths) if self.doc_lengths else 0
        )

    def _idf(self, term):
        doc_freq = len(self.inverted_index.get(term, []))
        if doc_freq == 0 or self.doc_count == 0:
            return 0
        return math.log((self.doc_count - doc_freq + 0.5) / (doc_freq + 0.5) + 1)

    def _bm25_score_one(self, query_tokens, doc_id):
        if self.avg_doc_length == 0:
            return 0
        score = 0.0
        doc_length = max(1, self.doc_lengths[doc_id])
        for token in query_tokens:
            postings = self.inverted_index.get(token)
            if not postings:
                continue
            idf = self._idf(token)
            for idx, freq in postings:
                if idx != doc_id:
                    continue
                numerator = freq * (self.k1 + 1)
                denominator = freq + self.k1 * (1 - self.b + self.b * doc_length / self.avg_doc_length)
                score += idf * numerator / denominator
                break
        return score

    def _search_bm25(self, query, top_k):
        tokens = self._tokenize(query)
        if not tokens:
            return []
        scores = []
        for doc_id in range(self.doc_count):
            score = self._bm25_score_one(tokens, doc_id)
            if score > 0:
                scores.append((doc_id, score))
        scores.sort(key=lambda item: item[1], reverse=True)
        return scores[:top_k]

    def _build_vectors(self, documents):
        texts = [doc.page_content for doc in documents]
        self.vectorizer = TfidfVectorizer(
            tokenizer=self._tokenize,
            token_pattern=None,
            max_features=Config.TFIDF_MAX_FEATURES,
            sublinear_tf=True,
            norm="l2",
        )
        matrix = self.vectorizer.fit_transform(texts)
        dense = matrix.astype(np.float32).toarray()
        if dense.size == 0:
            dense = np.zeros((len(texts), 1), dtype=np.float32)
        faiss.normalize_L2(dense)
        self.embedding_dim = dense.shape[1]
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        self.index.add(dense)

    def create_index(self, documents):
        print(f"[索引] 创建混合索引，片段数: {len(documents)}")
        self._build_bm25(documents)
        self._build_vectors(documents)
        print(f"[索引] 完成，向量维度: {self.embedding_dim}")

    def search(self, query, top_k=5):
        if self.doc_count == 0:
            return []

        candidate_count = max(top_k * 4, 20)
        combined = {}

        vector_scores = []
        if self.index is not None and self.vectorizer is not None:
            query_vec = self.vectorizer.transform([query]).astype(np.float32).toarray()
            if query_vec.shape[1] == self.embedding_dim:
                faiss.normalize_L2(query_vec)
                scores, ids = self.index.search(query_vec, min(candidate_count, self.doc_count))
                vector_scores = [
                    (int(doc_id), float(score))
                    for doc_id, score in zip(ids[0], scores[0])
                    if int(doc_id) >= 0
                ]

        bm25_scores = self._search_bm25(query, candidate_count)
        max_bm25 = max((score for _, score in bm25_scores), default=1.0)

        for rank, (doc_id, score) in enumerate(vector_scores, start=1):
            combined.setdefault(doc_id, {
                "doc_id": doc_id,
                "vector_score": 0.0,
                "bm25_score": 0.0,
                "bm25_raw_score": 0.0,
                "vector_rank": None,
                "bm25_rank": None,
            })
            combined[doc_id]["vector_score"] = max(0.0, score)
            combined[doc_id]["vector_rank"] = rank

        for rank, (doc_id, raw_score) in enumerate(bm25_scores, start=1):
            combined.setdefault(doc_id, {
                "doc_id": doc_id,
                "vector_score": 0.0,
                "bm25_score": 0.0,
                "bm25_raw_score": 0.0,
                "vector_rank": None,
                "bm25_rank": None,
            })
            combined[doc_id]["bm25_raw_score"] = raw_score
            combined[doc_id]["bm25_score"] = raw_score / max_bm25 if max_bm25 > 0 else 0.0
            combined[doc_id]["bm25_rank"] = rank

        results = []
        for item in combined.values():
            combined_score = (
                Config.VECTOR_WEIGHT * item["vector_score"]
                + Config.BM25_WEIGHT * item["bm25_score"]
            )
            doc = self.documents[item["doc_id"]]
            results.append({
                "doc_id": item["doc_id"],
                "document": doc,
                "distance": 1 - combined_score,
                "similarity": combined_score,
                "vector_similarity": item["vector_score"],
                "bm25_score": item["bm25_score"],
                "bm25_raw_score": item["bm25_raw_score"],
                "combined_score": combined_score,
                "vector_rank": item["vector_rank"],
                "bm25_rank": item["bm25_rank"],
            })

        results.sort(key=lambda item: item["combined_score"], reverse=True)
        for rank, item in enumerate(results[:top_k], start=1):
            item["rank"] = rank
            item["final_rank"] = rank
        return results[:top_k]

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        if self.index is None:
            raise RuntimeError("Cannot save an empty vector index")
        faiss.write_index(self.index, os.path.join(path, "index.faiss"))
        data = {
            "documents": self.documents,
            "doc_lengths": self.doc_lengths,
            "avg_doc_length": self.avg_doc_length,
            "inverted_index": dict(self.inverted_index),
            "doc_count": self.doc_count,
            "k1": self.k1,
            "b": self.b,
            "vectorizer": self.vectorizer,
            "embedding_dim": self.embedding_dim,
        }
        with open(os.path.join(path, "meta.pkl"), "wb") as file:
            pickle.dump(data, file)
        print(f"[保存] 混合索引已保存到 {path}")

    def load(self, path):
        with open(os.path.join(path, "meta.pkl"), "rb") as file:
            data = pickle.load(file)
        self.documents = data["documents"]
        self.doc_lengths = data["doc_lengths"]
        self.avg_doc_length = data["avg_doc_length"]
        self.inverted_index = defaultdict(list, data["inverted_index"])
        self.doc_count = data["doc_count"]
        self.k1 = data["k1"]
        self.b = data["b"]
        self.vectorizer = data["vectorizer"]
        self.embedding_dim = data["embedding_dim"]
        self.index = faiss.read_index(os.path.join(path, "index.faiss"))
        print(f"[加载] 混合索引已加载，片段数: {self.doc_count}")

    def is_empty(self):
        return self.doc_count == 0


VectorDB = HybridVectorDB

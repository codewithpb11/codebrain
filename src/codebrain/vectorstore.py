"""Vector store with ChromaDB primary and in-memory fallback."""

import os
from typing import List, Optional

from .chunker import Chunk
from .config import settings
from .embedder import Embedder


class VectorStore:
    """Vector store wrapper with ChromaDB primary and in-memory fallback."""

    def __init__(self, persist_dir: str = None, collection_name: str = None):
        self.persist_dir = persist_dir or settings.chroma_persist_dir
        self.collection_name = collection_name or settings.collection_name
        self._client = None
        self._collection = None
        self._memory_store = None  # Fallback in-memory store
        self._use_chroma = True

    def _init_chroma(self):
        """Try to initialize ChromaDB."""
        try:
            import chromadb
            from chromadb.config import Settings as ChromaSettings

            os.makedirs(self.persist_dir, exist_ok=True)
            self._client = chromadb.PersistentClient(
                path=self.persist_dir,
                settings=ChromaSettings(anonymized_telemetry=False),
            )
            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            self._use_chroma = True
        except ImportError:
            self._use_chroma = False
            self._memory_store = InMemoryStore()

    def _ensure_init(self):
        if self._collection is None and self._memory_store is None:
            self._init_chroma()

    def add_chunks(self, chunks: List[Chunk], embedder: Embedder):
        """Add chunks to the vector store with embeddings."""
        self._ensure_init()

        texts = [chunk.content for chunk in chunks]
        embeddings = embedder.embed(texts)

        if self._use_chroma and self._collection:
            metadatas = []
            ids = []
            for i, chunk in enumerate(chunks):
                metadatas.append({
                    "source_path": chunk.source_path,
                    "language": chunk.language,
                    "chunk_type": chunk.chunk_type,
                    "start_line": chunk.start_line,
                    "end_line": chunk.end_line,
                    "name": chunk.metadata.get("name", ""),
                })
                ids.append(f"{chunk.source_path}:{chunk.start_line}:{i}")

            batch_size = 64
            for i in range(0, len(texts), batch_size):
                self._collection.add(
                    ids=ids[i:i + batch_size],
                    embeddings=embeddings[i:i + batch_size],
                    documents=texts[i:i + batch_size],
                    metadatas=metadatas[i:i + batch_size],
                )
        elif self._memory_store:
            self._memory_store.add_chunks(chunks, embeddings)

    def search(self, query: str, embedder: Embedder, n_results: int = 5) -> List[dict]:
        """Search for relevant chunks given a query."""
        self._ensure_init()

        query_embedding = embedder.embed([query])[0]

        if self._use_chroma and self._collection:
            results = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                include=["documents", "metadatas", "distances"],
            )

            formatted = []
            for i in range(len(results["ids"][0])):
                formatted.append({
                    "id": results["ids"][0][i],
                    "content": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "distance": results["distances"][0][i],
                })
            return formatted
        elif self._memory_store:
            return self._memory_store.search(query_embedding, n_results)
        return []

    def clear(self):
        """Clear all data from the store."""
        self._ensure_init()
        if self._use_chroma and self._collection:
            try:
                self._client.delete_collection(self.collection_name)
            except Exception:
                pass
            self._collection = None
        elif self._memory_store:
            self._memory_store.clear()

    def count(self) -> int:
        """Return the number of chunks in the store."""
        self._ensure_init()
        try:
            if self._use_chroma and self._collection:
                return self._collection.count()
            elif self._memory_store:
                return self._memory_store.count()
        except Exception:
            pass
        return 0

    def list_source_paths(self, limit: int = 100) -> List[str]:
        """Return unique indexed file paths."""
        self._ensure_init()
        paths = set()
        try:
            if self._use_chroma and self._collection:
                count = self._collection.count()
                if count == 0:
                    return []
                result = self._collection.get(limit=min(count, 5000), include=["metadatas"])
                for meta in result.get("metadatas", []):
                    if meta and meta.get("source_path"):
                        paths.add(meta["source_path"])
            elif self._memory_store:
                for chunk in self._memory_store.chunks:
                    path = chunk["metadata"].get("source_path")
                    if path:
                        paths.add(path)
        except Exception:
            pass
        return sorted(paths)[:limit]


class InMemoryStore:
    """Simple in-memory vector store using cosine similarity."""

    def __init__(self):
        self.chunks = []
        self.embeddings = []

    def add_chunks(self, chunks: List[Chunk], embeddings: List[List[float]]):
        for chunk, embedding in zip(chunks, embeddings):
            self.chunks.append({
                "content": chunk.content,
                "metadata": {
                    "source_path": chunk.source_path,
                    "language": chunk.language,
                    "chunk_type": chunk.chunk_type,
                    "start_line": chunk.start_line,
                    "end_line": chunk.end_line,
                    "name": chunk.metadata.get("name", ""),
                },
            })
            self.embeddings.append(embedding)

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def search(self, query_embedding: List[float], n_results: int = 5) -> List[dict]:
        scores = []
        for i, emb in enumerate(self.embeddings):
            sim = self._cosine_similarity(query_embedding, emb)
            scores.append((sim, i))

        scores.sort(reverse=True)
        results = []
        for sim, idx in scores[:n_results]:
            chunk = self.chunks[idx]
            results.append({
                "id": f"{chunk['metadata']['source_path']}:{chunk['metadata']['start_line']}",
                "content": chunk["content"],
                "metadata": chunk["metadata"],
                "distance": 1.0 - sim,  # Convert similarity to distance
            })
        return results

    def clear(self):
        self.chunks = []
        self.embeddings = []

    def count(self) -> int:
        return len(self.chunks)

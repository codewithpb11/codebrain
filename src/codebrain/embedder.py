"""Embedding model providers with graceful fallback."""

import re
from abc import ABC, abstractmethod
from typing import List

from .config import settings


class Embedder(ABC):
    """Abstract base class for embedding providers."""

    @abstractmethod
    def embed(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of texts into vectors."""
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the embedding dimension."""
        pass


class LocalEmbedder(Embedder):
    """Local embedding using sentence-transformers."""

    def __init__(self, model_name: str = None):
        self.model_name = model_name or settings.embedding_model
        self._model = None
        self._dimension = None

    def _load_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError:
                raise ImportError(
                    "sentence-transformers is not installed. "
                    "Install it with: pip install 'codebrain[local]' "
                    "or switch to OpenAI embeddings by setting EMBEDDING_PROVIDER=openai"
                )
            self._model = SentenceTransformer(self.model_name)
            test_vec = self._model.encode(["test"])
            self._dimension = len(test_vec[0])
        return self._model

    def embed(self, texts: List[str]) -> List[List[float]]:
        model = self._load_model()
        embeddings = model.encode(texts, show_progress_bar=len(texts) > 10)
        return embeddings.tolist()

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            self._load_model()
        return self._dimension


class OpenAIEmbedder(Embedder):
    """OpenAI embedding provider."""

    def __init__(self, model_name: str = None, api_key: str = None):
        self.model_name = model_name or settings.openai_embedding_model
        self.api_key = api_key or settings.openai_api_key
        self._client = None
        self._dimension = 1536

    def _get_client(self):
        if self._client is None:
            try:
                import openai
            except ImportError:
                raise ImportError(
                    "openai is not installed. "
                    "Install it with: pip install 'codebrain[openai]'"
                )
            self._client = openai.OpenAI(api_key=self.api_key)
        return self._client

    def embed(self, texts: List[str]) -> List[List[float]]:
        client = self._get_client()
        batch_size = 100
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            response = client.embeddings.create(
                model=self.model_name,
                input=batch,
            )
            all_embeddings.extend([item.embedding for item in response.data])
        return all_embeddings

    @property
    def dimension(self) -> int:
        return self._dimension


class SimpleEmbedder(Embedder):
    """
    A simple bag-of-words embedder that requires no external dependencies.
    Uses term frequency vectors with a fixed vocabulary.
    This is a fallback for when no other embedder is available.
    """

    def __init__(self, vocab_size: int = 5000):
        self.vocab_size = vocab_size
        self._vocab = {}
        self._dimension = vocab_size

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization."""
        text = text.lower()
        # Extract words and identifiers
        tokens = re.findall(r'[a-z]+|[0-9]+|[^\w\s]', text)
        return tokens

    def _build_vocab(self, texts: List[str]):
        """Build vocabulary from texts."""
        from collections import Counter
        all_tokens = []
        for text in texts:
            all_tokens.extend(self._tokenize(text))

        counter = Counter(all_tokens)
        most_common = counter.most_common(self.vocab_size - 1)
        self._vocab = {token: idx for idx, (token, _) in enumerate(most_common)}

    def embed(self, texts: List[str]) -> List[List[float]]:
        if not self._vocab:
            self._build_vocab(texts)

        embeddings = []
        for text in texts:
            tokens = self._tokenize(text)
            vec = [0.0] * self.vocab_size
            for token in tokens:
                if token in self._vocab:
                    vec[self._vocab[token]] += 1.0

            # Normalize
            norm = sum(x * x for x in vec) ** 0.5
            if norm > 0:
                vec = [x / norm for x in vec]

            embeddings.append(vec)
        return embeddings

    @property
    def dimension(self) -> int:
        return self._dimension


def get_embedder() -> Embedder:
    """Factory function to get the configured embedder."""
    if settings.embedding_provider == "openai":
        try:
            import openai  # noqa: F401
            return OpenAIEmbedder()
        except ImportError:
            pass
    elif settings.embedding_provider == "local":
        try:
            import sentence_transformers  # noqa: F401
            return LocalEmbedder()
        except ImportError:
            pass

    # Ultimate fallback
    return SimpleEmbedder()
    """Factory function to get the configured embedder."""
    if settings.embedding_provider == "openai":
        try:
            return OpenAIEmbedder()
        except ImportError:
            pass
    elif settings.embedding_provider == "local":
        try:
            return LocalEmbedder()
        except ImportError:
            pass

    # Ultimate fallback
    return SimpleEmbedder()

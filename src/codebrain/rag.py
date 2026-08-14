"""RAG pipeline orchestration."""

from datetime import datetime, timezone
from typing import AsyncGenerator, List, Optional

from .chunker import chunk_document
from .config import settings
from .embedder import get_embedder
from .llm import LLMResponse, get_llm_provider
from .parser import CodeDocument, documents_from_contents, parse_codebase
from .vectorstore import VectorStore


SYSTEM_PROMPT = """You are CodeBrain, an expert codebase assistant. You help developers understand code, find bugs, explain architecture, and suggest improvements.

When answering:
- Reference specific files and line numbers from the context
- Explain your reasoning clearly
- If you're unsure, say so rather than guessing
- Provide code examples when helpful
- Keep responses concise but thorough

The user is asking about their codebase. Use the provided context to answer accurately."""


class RAGPipeline:
    """End-to-end RAG pipeline for codebase Q&A."""

    def __init__(self):
        self.embedder = get_embedder()
        self.vector_store = VectorStore()
        self.llm = get_llm_provider()
        self._index_info = self._empty_index_info()

    def _empty_index_info(self) -> dict:
        return {
            "status": "empty",
            "source_name": None,
            "source_type": None,
            "files": 0,
            "chunks": 0,
            "languages": [],
            "file_paths": [],
            "indexed_at": None,
        }

    def _update_index_info(
        self,
        documents: List[CodeDocument],
        clear: bool,
        source_name: Optional[str] = None,
        source_type: Optional[str] = None,
    ) -> None:
        new_paths = sorted({d.relative_path for d in documents})
        new_languages = sorted({d.language for d in documents})

        if clear:
            self._index_info = self._empty_index_info()

        if not clear and self._index_info["file_paths"]:
            new_paths = sorted(set(self._index_info["file_paths"]) | set(new_paths))
            new_languages = sorted(set(self._index_info["languages"]) | set(new_languages))
            if source_name and self._index_info["source_name"]:
                source_name = f"{self._index_info['source_name']} + {source_name}"

        self._index_info.update({
            "status": "ready",
            "source_name": source_name or self._index_info.get("source_name") or "Unknown",
            "source_type": source_type or self._index_info.get("source_type"),
            "files": len(new_paths),
            "chunks": self.vector_store.count(),
            "languages": new_languages,
            "file_paths": new_paths[:200],
            "indexed_at": datetime.now(timezone.utc).isoformat(),
        })

    def _index_documents(
        self,
        documents: List[CodeDocument],
        clear: bool = True,
        source_name: Optional[str] = None,
        source_type: Optional[str] = None,
    ) -> dict:
        """Chunk and embed a list of documents into the vector store."""
        if clear:
            self.vector_store.clear()
            self._index_info = self._empty_index_info()

        self._index_info["status"] = "indexing"

        if not documents:
            self._index_info["status"] = "error"
            return {"status": "error", "message": "No code files found", "chunks": 0}

        all_chunks = []
        for doc in documents:
            chunks = chunk_document(doc, settings.chunk_size, settings.chunk_overlap)
            all_chunks.extend(chunks)

        self.vector_store.add_chunks(all_chunks, self.embedder)
        self._update_index_info(documents, clear=clear, source_name=source_name, source_type=source_type)

        return {
            "status": "success",
            "files": self._index_info["files"],
            "chunks": self._index_info["chunks"],
            "languages": self._index_info["languages"],
            "source_name": self._index_info["source_name"],
            "source_type": self._index_info["source_type"],
            "file_paths": self._index_info["file_paths"],
            "indexed_at": self._index_info["indexed_at"],
        }

    def index_codebase(self, root_path: str, progress_callback=None, clear: bool = True) -> dict:
        """Index a codebase directory into the vector store."""
        import os
        documents = parse_codebase(root_path, progress_callback=progress_callback)
        source_name = os.path.basename(os.path.abspath(root_path)) or root_path
        return self._index_documents(
            documents,
            clear=clear,
            source_name=source_name,
            source_type="path",
        )

    def index_uploaded_files(
        self,
        files: List[dict],
        merge: bool = False,
        source_name: Optional[str] = None,
    ) -> dict:
        """Index files uploaded from the browser folder picker."""
        entries = [(f["path"], f["content"]) for f in files]
        documents = documents_from_contents(entries)
        if not source_name and files:
            first_path = files[0]["path"].replace("\\", "/")
            source_name = first_path.split("/")[0] if "/" in first_path else first_path
        return self._index_documents(
            documents,
            clear=not merge,
            source_name=source_name,
            source_type="folder",
        )

    def index_snippets(self, snippets: List[dict], merge: bool = True) -> dict:
        """Index manually pasted code snippets."""
        entries = []
        for snippet in snippets:
            path = snippet.get("path") or "pasted/snippet.txt"
            content = snippet.get("content", "")
            language = snippet.get("language")
            doc = documents_from_contents([(path, content)], language=language)
            entries.extend(doc)
        return self._index_documents(
            entries,
            clear=not merge,
            source_name="Manual snippets",
            source_type="snippet",
        )

    def _build_context(self, results: List[dict]) -> str:
        """Build context string from retrieval results."""
        context_parts = []
        for i, result in enumerate(results):
            meta = result["metadata"]
            source = meta["source_path"]
            start = meta["start_line"]
            end = meta["end_line"]
            chunk_type = meta.get("chunk_type", "code")
            name = meta.get("name", "")

            header = f"### [{i+1}] {source}"
            if name:
                header += f" - {name}"
            header += f" (lines {start}-{end}, {chunk_type})"

            context_parts.append(f"{header}\n```{meta.get('language', 'text')}\n{result['content']}\n```\n")

        return "\n".join(context_parts)

    def _build_user_message(self, question: str, context: str, attached_code: Optional[str] = None) -> str:
        """Build the user message with optional attached code and retrieved context."""
        parts = []
        if attached_code and attached_code.strip():
            parts.append(
                "The user attached this code directly (not from the index):\n"
                f"```\n{attached_code.strip()}\n```\n"
            )
        if context:
            parts.append(f"Here is the relevant code context from the codebase:\n\n{context}")
        parts.append(f"Question: {question}")
        return "\n\n".join(parts)

    async def ask(
        self,
        question: str,
        n_results: int = 5,
        temperature: float = 0.3,
        attached_code: Optional[str] = None,
    ) -> LLMResponse:
        """Ask a question about the codebase."""
        # Retrieve relevant chunks
        results = self.vector_store.search(question, self.embedder, n_results)

        if not results and not (attached_code and attached_code.strip()):
            return LLMResponse(
                content="I don't have any codebase indexed yet. Please index a codebase first, paste code in the sidebar, or attach code to your question."
            )

        context = self._build_context(results) if results else ""

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": self._build_user_message(question, context, attached_code),
            },
        ]

        response = await self.llm.chat(messages, temperature=temperature)

        # Add source references to the response
        sources = [r["metadata"]["source_path"] for r in results]
        unique_sources = list(dict.fromkeys(sources))  # preserve order, remove dups
        if unique_sources:
            source_list = "\n".join(f"- {s}" for s in unique_sources[:5])
            response.content += f"\n\n**Sources:**\n{source_list}"

        return response

    async def stream_ask(
        self,
        question: str,
        n_results: int = 5,
        temperature: float = 0.3,
        attached_code: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        """Stream a response to a codebase question."""
        results = self.vector_store.search(question, self.embedder, n_results)

        if not results and not (attached_code and attached_code.strip()):
            yield "I don't have any codebase indexed yet. Please index a codebase first, paste code in the sidebar, or attach code to your question."
            return

        context = self._build_context(results) if results else ""

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": self._build_user_message(question, context, attached_code),
            },
        ]

        async for token in self.llm.stream_chat(messages, temperature=temperature):
            yield token

    def get_stats(self) -> dict:
        """Get indexing statistics."""
        count = self.vector_store.count()
        file_paths = self.vector_store.list_source_paths()

        if count > 0 and self._index_info["status"] == "empty":
            self._index_info.update({
                "status": "ready",
                "files": len(file_paths),
                "chunks": count,
                "file_paths": file_paths[:200],
                "source_name": self._index_info.get("source_name") or "Previously indexed",
                "source_type": self._index_info.get("source_type") or "unknown",
            })

        return {
            "indexed_chunks": count,
            "embedding_provider": settings.embedding_provider,
            "llm_provider": settings.llm_provider,
            "embedding_model": settings.embedding_model,
            "llm_model": getattr(settings, f"{settings.llm_provider}_model", "unknown"),
            "index_status": self._index_info["status"],
            "source_name": self._index_info["source_name"],
            "source_type": self._index_info["source_type"],
            "indexed_files": self._index_info["files"] or len(file_paths),
            "languages": self._index_info["languages"],
            "file_paths": file_paths[:50],
            "indexed_at": self._index_info["indexed_at"],
        }

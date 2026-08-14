"""FastAPI server for CodeBrain."""

import os
import threading
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from .rag import RAGPipeline

# Global pipeline instance
_pipeline: Optional[RAGPipeline] = None


def get_pipeline() -> RAGPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline()
    return _pipeline


class IndexRequest(BaseModel):
    path: str
    merge: bool = False


class FileUpload(BaseModel):
    path: str
    content: str


class UploadIndexRequest(BaseModel):
    files: list[FileUpload]
    merge: bool = False
    source_name: Optional[str] = None


class SnippetItem(BaseModel):
    content: str
    path: str = "pasted/snippet.py"
    language: Optional[str] = None


class SnippetsIndexRequest(BaseModel):
    snippets: list[SnippetItem]
    merge: bool = True


class AskRequest(BaseModel):
    question: str
    n_results: int = 5
    temperature: float = 0.3
    stream: bool = False
    attached_code: Optional[str] = None


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage]
    n_results: int = 5
    temperature: float = 0.3
    stream: bool = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    def preload_models():
        try:
            print("Loading AI models in the background...")
            get_pipeline()
            print("AI models ready.")
        except Exception as exc:
            print(f"Warning: model preload failed: {exc}")

    threading.Thread(target=preload_models, daemon=True).start()
    yield


app = FastAPI(
    title="CodeBrain",
    description="RAG-powered codebase chatbot",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Serve the frontend."""
    frontend_path = os.path.join(os.path.dirname(__file__), "..", "..", "frontend", "index.html")
    if os.path.exists(frontend_path):
        return FileResponse(frontend_path)
    return {"message": "CodeBrain API is running. Visit /docs for API documentation."}


@app.get("/api/stats")
async def stats():
    """Get indexing statistics."""
    pipeline = get_pipeline()
    return pipeline.get_stats()


@app.post("/api/index")
async def index_codebase(request: IndexRequest):
    """Index a codebase directory."""
    pipeline = get_pipeline()

    if not os.path.exists(request.path):
        raise HTTPException(status_code=404, detail="Path not found")

    if not os.path.isdir(request.path):
        raise HTTPException(status_code=400, detail="Path must be a directory")

    result = pipeline.index_codebase(request.path, clear=not request.merge)
    return result


@app.post("/api/index/upload")
async def index_upload(request: UploadIndexRequest):
    """Index files uploaded from the browser folder picker."""
    pipeline = get_pipeline()

    if not request.files:
        raise HTTPException(status_code=400, detail="No files provided")

    files = [{"path": f.path, "content": f.content} for f in request.files]
    return pipeline.index_uploaded_files(
        files,
        merge=request.merge,
        source_name=request.source_name,
    )


@app.post("/api/index/snippets")
async def index_snippets(request: SnippetsIndexRequest):
    """Index manually pasted code snippets."""
    pipeline = get_pipeline()

    if not request.snippets:
        raise HTTPException(status_code=400, detail="No snippets provided")

    snippets = [
        {"content": s.content, "path": s.path, "language": s.language}
        for s in request.snippets
    ]
    return pipeline.index_snippets(snippets, merge=request.merge)


@app.post("/api/ask")
async def ask(request: AskRequest):
    """Ask a question about the codebase."""
    pipeline = get_pipeline()

    if request.stream:
        async def generate() -> AsyncGenerator[str, None]:
            async for token in pipeline.stream_ask(
                request.question,
                n_results=request.n_results,
                temperature=request.temperature,
                attached_code=request.attached_code,
            ):
                yield token

        return StreamingResponse(generate(), media_type="text/plain")

    response = await pipeline.ask(
        request.question,
        n_results=request.n_results,
        temperature=request.temperature,
        attached_code=request.attached_code,
    )
    return {"answer": response.content, "usage": response.usage}


@app.post("/api/chat")
async def chat(request: ChatRequest):
    """Chat with the codebase using message history."""
    pipeline = get_pipeline()

    # Convert messages to dict format
    messages = [{"role": m.role, "content": m.content} for m in request.messages]

    # Use the last user message as the question for retrieval
    last_user_msg = None
    for m in reversed(messages):
        if m["role"] == "user":
            last_user_msg = m["content"]
            break

    if not last_user_msg:
        raise HTTPException(status_code=400, detail="No user message found")

    # Retrieve context
    results = pipeline.vector_store.search(last_user_msg, pipeline.embedder, request.n_results)
    context = pipeline._build_context(results) if results else ""

    # Build messages with context
    system_msg = {
        "role": "system",
        "content": f"You are CodeBrain, an expert codebase assistant. Use the provided code context to answer accurately.\n\nCode Context:\n{context}",
    }

    full_messages = [system_msg] + messages

    if request.stream:
        async def generate() -> AsyncGenerator[str, None]:
            async for token in pipeline.llm.stream_chat(full_messages, temperature=request.temperature):
                yield token

        return StreamingResponse(generate(), media_type="text/plain")

    response = await pipeline.llm.chat(full_messages, temperature=request.temperature)
    return {"answer": response.content, "usage": response.usage}


@app.get("/api/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok"}

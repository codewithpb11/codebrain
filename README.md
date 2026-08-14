# CodeBrain - RAG Codebase Chatbot

A production-ready RAG (Retrieval-Augmented Generation) pipeline for chatting with your codebase. Uses **free local AI** by default (Ollama), with optional cloud providers.

![Version](https://img.shields.io/badge/version-0.1.0-blue)
![Python](https://img.shields.io/badge/python-3.9+-blue)
![License](https://img.shields.io/badge/license-MIT-green)

## Features

- **100% Free by Default**: Uses Ollama (local AI) and sentence-transformers (local embeddings) — no API keys needed
- **Smart Code Chunking**: Parses code by function, class, and file boundaries with context preservation
- **Multiple LLM Providers**: Ollama (free/local), OpenAI, or Anthropic
- **Vector Storage**: ChromaDB for fast semantic retrieval with in-memory fallback
- **Web UI**: Clean, responsive chat interface (dark mode by default)
- **API**: FastAPI backend with async streaming support

## Quick Start (Free)

### 1. Install Ollama

Download and install from [ollama.com](https://ollama.com). It runs entirely on your machine.

Then pull a model:
```bash
ollama pull llama3.2
```

### 2. Install CodeBrain

```bash
# Clone the repository
git clone https://github.com/yourusername/codebrain.git
cd codebrain

# Install dependencies
pip install -r requirements.txt
pip install sentence-transformers
```

### 3. Index Your Codebase

```bash
# Index the current directory
python src/codebrain/cli.py index .

# Or index a specific project
python src/codebrain/cli.py index /path/to/your/project
```

### 4. Start the Chat Server

```bash
python src/codebrain/cli.py serve
```

Open `http://localhost:8000` in your browser.

### 5. Ask Questions

```bash
# CLI mode
python src/codebrain/cli.py ask "How does authentication work?"

# Or use the web UI
```

## Using Cloud Providers (Optional)

If you prefer OpenAI or Anthropic instead of Ollama:

1. Create a `.env` file:
```env
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-your-key
```

2. Restart the server

## Configuration

Create a `.env` file (or set environment variables):

```env
# Default: Ollama (free, local)
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2

# Embeddings (local by default, no API key needed)
EMBEDDING_PROVIDER=local
EMBEDDING_MODEL=all-MiniLM-L6-v2

# Vector Store
CHROMA_PERSIST_DIR=./chroma_db
COLLECTION_NAME=codebase
```

## Architecture

```
codebrain/
├── src/codebrain/
│   ├── parser.py        # Code file parsing & language detection
│   ├── chunker.py       # Code-aware document chunking
│   ├── embedder.py      # Embedding models (local, OpenAI, fallback)
│   ├── vectorstore.py   # ChromaDB + in-memory fallback
│   ├── llm.py           # LLM providers (Ollama, OpenAI, Anthropic)
│   ├── rag.py           # RAG pipeline orchestration
│   ├── server.py        # FastAPI app
│   └── cli.py           # Command-line interface
├── frontend/
│   └── index.html       # Web UI
├── pyproject.toml
└── README.md
```

## How It Works

1. **Parse**: Recursively scans your codebase, skipping build artifacts and dependencies
2. **Chunk**: Intelligently splits code by functions, classes, and sections
3. **Embed**: Converts chunks into semantic vectors using local embeddings
4. **Store**: Saves vectors in ChromaDB for fast similarity search
5. **Retrieve**: Finds the most relevant code snippets for your question
6. **Generate**: Sends retrieved context + question to your LLM for an informed answer

## Supported Languages

Python, JavaScript, TypeScript, Go, Rust, Java, C/C++, C#, Ruby, PHP, Swift, Kotlin, Scala, and more.

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web UI |
| `/api/health` | GET | Health check |
| `/api/stats` | GET | Indexing statistics |
| `/api/index` | POST | Index a codebase directory |
| `/api/ask` | POST | Ask a question |
| `/api/chat` | POST | Chat with message history |

## Development

```bash
# Run tests
pytest

# Start dev server with auto-reload
uvicorn codebrain.server:app --reload
```

## License

MIT

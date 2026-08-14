"""CLI interface for CodeBrain."""

import argparse
import asyncio
import os
import sys

from tqdm import tqdm

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from codebrain.rag import RAGPipeline


def print_progress(current, total, item):
    """Simple progress printer."""
    pass  # tqdm handles this


def index_command(args):
    """Index a codebase."""
    pipeline = RAGPipeline()

    print(f"Indexing: {args.path}")
    print("This may take a while depending on codebase size...")

    result = pipeline.index_codebase(args.path)

    if result["status"] == "success":
        print(f"\nDone! Indexed {result['files']} files into {result['chunks']} chunks.")
        print(f"Languages found: {', '.join(result['languages'])}")
    else:
        print(f"\nError: {result['message']}")
        sys.exit(1)


async def ask_command(args):
    """Ask a question."""
    pipeline = RAGPipeline()

    if args.stream:
        print("Answer: ", end="", flush=True)
        async for token in pipeline.stream_ask(args.question, n_results=args.results):
            print(token, end="", flush=True)
        print()
    else:
        response = await pipeline.ask(args.question, n_results=args.results)
        print(f"\nAnswer:\n{response.content}")


def stats_command(args):
    """Show stats."""
    pipeline = RAGPipeline()
    stats = pipeline.get_stats()
    print(f"Indexed chunks: {stats['indexed_chunks']}")
    print(f"Embedding: {stats['embedding_provider']} ({stats['embedding_model']})")
    print(f"LLM: {stats['llm_provider']} ({stats['llm_model']})")


def serve_command(args):
    """Start the web server."""
    import threading
    import time
    import urllib.error
    import urllib.request
    import webbrowser

    import uvicorn
    from codebrain.server import app

    bind_host = args.host or "127.0.0.1"
    port = args.port or 8000
    url = f"http://127.0.0.1:{port}"

    print(f"Starting CodeBrain at {url}")
    print("Press Ctrl+C to stop the server.")

    if not args.no_browser:
        def open_when_ready():
            for _ in range(60):
                try:
                    with urllib.request.urlopen(f"{url}/api/health", timeout=1) as response:
                        if response.status == 200:
                            webbrowser.open(url)
                            print(f"Opened {url} in your browser.")
                            return
                except (urllib.error.URLError, TimeoutError, OSError):
                    time.sleep(0.5)
            print(f"Server is running. Open {url} in your browser.")

        threading.Thread(target=open_when_ready, daemon=True).start()

    uvicorn.run(app, host=bind_host, port=port, log_level="info")


def main():
    parser = argparse.ArgumentParser(description="CodeBrain - RAG Codebase Chatbot")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Index command
    index_parser = subparsers.add_parser("index", help="Index a codebase")
    index_parser.add_argument("path", nargs="?", default=".", help="Path to codebase directory (default: current directory)")

    # Ask command
    ask_parser = subparsers.add_parser("ask", help="Ask a question")
    ask_parser.add_argument("question", help="Your question")
    ask_parser.add_argument("--results", "-n", type=int, default=5, help="Number of chunks to retrieve")
    ask_parser.add_argument("--stream", "-s", action="store_true", help="Stream the response")

    # Stats command
    stats_parser = subparsers.add_parser("stats", help="Show indexing stats")

    # Serve command
    serve_parser = subparsers.add_parser("serve", help="Start the web server")
    serve_parser.add_argument("--host", default="127.0.0.1", help="Host to bind to (default: 127.0.0.1)")
    serve_parser.add_argument("--port", "-p", type=int, default=8000, help="Port to bind to")
    serve_parser.add_argument("--no-browser", action="store_true", help="Do not open the browser automatically")

    args = parser.parse_args()

    if args.command == "index":
        index_command(args)
    elif args.command == "ask":
        asyncio.run(ask_command(args))
    elif args.command == "stats":
        stats_command(args)
    elif args.command == "serve":
        serve_command(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

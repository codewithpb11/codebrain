"""Code file parsing utilities."""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass
class CodeDocument:
    """Represents a parsed code file."""
    path: str
    content: str
    language: str
    relative_path: str


# File extension to language mapping
EXTENSION_MAP = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".cpp": "cpp",
    ".c": "c",
    ".h": "c",
    ".hpp": "cpp",
    ".cs": "csharp",
    ".rb": "ruby",
    ".php": "php",
    ".swift": "swift",
    ".kt": "kotlin",
    ".scala": "scala",
    ".r": "r",
    ".m": "objective-c",
    ".mm": "objective-c",
    ".sh": "bash",
    ".bash": "bash",
    ".zsh": "bash",
    ".ps1": "powershell",
    ".pl": "perl",
    ".lua": "lua",
    ".elm": "elm",
    ".hs": "haskell",
    ".ex": "elixir",
    ".exs": "elixir",
    ".clj": "clojure",
    ".fs": "fsharp",
    ".fsx": "fsharp",
    ".dart": "dart",
    ".jl": "julia",
    ".nim": "nim",
    ".cr": "crystal",
    ".v": "v",
    ".zig": "zig",
    ".odin": "odin",
    ".coffee": "coffeescript",
    ".vue": "vue",
    ".svelte": "svelte",
    ".sql": "sql",
    ".md": "markdown",
    ".mdx": "markdown",
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".toml": "toml",
    ".ini": "ini",
    ".cfg": "ini",
    ".conf": "ini",
    ".Dockerfile": "dockerfile",
    ".dockerfile": "dockerfile",
    ".tf": "terraform",
    ".hcl": "hcl",
    ".graphql": "graphql",
    ".gql": "graphql",
    ".proto": "protobuf",
    ".html": "html",
    ".css": "css",
    ".scss": "scss",
    ".sass": "scss",
    ".less": "less",
    ".xml": "xml",
    ".svg": "xml",
}

# Directories to skip
SKIP_DIRS = {
    ".git", ".svn", ".hg", "node_modules", "__pycache__", ".venv",
    "venv", "env", ".env", "dist", "build", "target", "vendor",
    ".idea", ".vscode", ".vs", "bin", "obj", "out", "coverage",
    ".pytest_cache", ".mypy_cache", ".tox", ".eggs", "*.egg-info",
    ".next", ".nuxt", "public", "static", "assets", "uploads",
    "tmp", "temp", "log", "logs", "backup", "backups",
    "chroma_db", ".chroma", "embeddings",
}

# Files to skip
SKIP_FILES = {
    ".DS_Store", "Thumbs.db", ".gitignore", ".gitattributes",
    ".editorconfig", ".prettierrc", ".eslintrc", ".babelrc",
    "package-lock.json", "yarn.lock", "pnpm-lock.yaml",
    "Cargo.lock", "poetry.lock", "Pipfile.lock",
    ".eslintcache", ".stylelintcache", ".parcel-cache",
}

# Binary-ish extensions to skip
SKIP_EXTENSIONS = {
    ".exe", ".dll", ".so", ".dylib", ".bin",
    ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".ico", ".svg",
    ".mp3", ".mp4", ".avi", ".mov", ".wmv", ".flv",
    ".zip", ".tar", ".gz", ".bz2", ".7z", ".rar",
    ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
    ".woff", ".woff2", ".ttf", ".otf", ".eot",
    ".wasm", ".class", ".o", ".a", ".lib",
}


def get_language(file_path: str) -> Optional[str]:
    """Detect programming language from file path."""
    path = Path(file_path)
    ext = path.suffix.lower()
    name = path.name.lower()

    # Check for extensionless files like Dockerfile
    if name in ("dockerfile", "makefile", "rakefile", "gemfile", "vagrantfile"):
        return name.replace("file", "")
    if name == "dockerfile":
        return "dockerfile"
    if name in ("makefile", "gnumakefile"):
        return "makefile"

    return EXTENSION_MAP.get(ext)


def should_skip(path: str, root: str) -> bool:
    """Determine if a file or directory should be skipped."""
    rel_path = os.path.relpath(path, root)
    parts = Path(rel_path).parts

    # Skip hidden directories/files (except .github, .vscode configs, etc.)
    for part in parts[:-1] if os.path.isfile(path) else parts:
        if part.startswith(".") and part not in {".github", ".vscode", ".ci"}:
            return True
        if part in SKIP_DIRS:
            return True

    if os.path.isfile(path):
        name = os.path.basename(path)
        if name in SKIP_FILES:
            return True
        ext = Path(path).suffix.lower()
        if ext in SKIP_EXTENSIONS:
            return True
        # Skip minified files
        if ".min." in name:
            return True

    return False


def parse_codebase(root_path: str, progress_callback=None) -> List[CodeDocument]:
    """Parse all code files in a directory into documents."""
    documents = []
    root = os.path.abspath(root_path)

    all_files = []
    for dirpath, dirnames, filenames in os.walk(root):
        # Filter out skip directories in-place to prevent os.walk from descending
        dirnames[:] = [d for d in dirnames if not should_skip(os.path.join(dirpath, d), root)]

        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            if should_skip(file_path, root):
                continue
            all_files.append(file_path)

    total = len(all_files)
    for i, file_path in enumerate(all_files):
        try:
            language = get_language(file_path)
            if language is None:
                # Try to read as text anyway for unknown extensions
                pass

            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()

            # Skip empty files
            if not content.strip():
                continue

            # Skip files that are clearly binary (high ratio of null bytes)
            if "\x00" in content[:1024]:
                continue

            rel_path = os.path.relpath(file_path, root)
            documents.append(CodeDocument(
                path=file_path,
                content=content,
                language=language or "text",
                relative_path=rel_path,
            ))

            if progress_callback:
                progress_callback(i + 1, total, rel_path)

        except Exception as e:
            # Log and continue
            if progress_callback:
                progress_callback(i + 1, total, f"ERROR: {rel_path} - {e}")

    return documents


def should_skip_relative_path(relative_path: str) -> bool:
    """Determine if a relative file path should be skipped (for uploads)."""
    parts = Path(relative_path.replace("\\", "/")).parts
    if not parts:
        return True

    for part in parts[:-1]:
        if part.startswith(".") and part not in {".github", ".vscode", ".ci"}:
            return True
        if part in SKIP_DIRS:
            return True

    name = parts[-1]
    if name in SKIP_FILES:
        return True
    ext = Path(name).suffix.lower()
    if ext in SKIP_EXTENSIONS:
        return True
    if ".min." in name:
        return True
    return False


def document_from_content(
    relative_path: str,
    content: str,
    language: Optional[str] = None,
) -> Optional[CodeDocument]:
    """Create a CodeDocument from in-memory content (uploads / pasted snippets)."""
    relative_path = relative_path.replace("\\", "/").lstrip("/")
    if should_skip_relative_path(relative_path):
        return None
    if not content or not content.strip():
        return None
    if "\x00" in content[:1024]:
        return None

    detected = language or get_language(relative_path)
    return CodeDocument(
        path=relative_path,
        content=content,
        language=detected or "text",
        relative_path=relative_path,
    )


def documents_from_contents(
    entries: List[tuple[str, str]],
    language: Optional[str] = None,
) -> List[CodeDocument]:
    """Build CodeDocument list from (relative_path, content) pairs."""
    documents = []
    for relative_path, content in entries:
        doc = document_from_content(relative_path, content, language=language)
        if doc:
            documents.append(doc)
    return documents

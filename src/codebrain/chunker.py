"""Smart code-aware chunking strategies."""

import re
from dataclasses import dataclass
from typing import List

from .parser import CodeDocument


@dataclass
class Chunk:
    """A chunk of code with metadata."""
    content: str
    source_path: str
    language: str
    chunk_type: str  # "function", "class", "file", "section"
    start_line: int
    end_line: int
    metadata: dict


def count_lines(text: str) -> int:
    return text.count("\n") + 1


def chunk_by_size(document: CodeDocument, chunk_size: int = 1000, overlap: int = 200) -> List[Chunk]:
    """Simple size-based chunking with overlap."""
    content = document.content
    chunks = []
    start = 0
    lines = content.split("\n")
    total_lines = len(lines)
    current_line = 1

    while start < len(content):
        end = start + chunk_size
        chunk_text = content[start:end]

        # Try to break at newline
        if end < len(content):
            next_newline = content.find("\n", end)
            if next_newline != -1 and next_newline - end < 100:
                end = next_newline + 1
                chunk_text = content[start:end]

        # Calculate line numbers
        chunk_lines = chunk_text.count("\n")
        end_line = min(current_line + chunk_lines, total_lines)

        chunks.append(Chunk(
            content=chunk_text,
            source_path=document.relative_path,
            language=document.language,
            chunk_type="section",
            start_line=current_line,
            end_line=end_line,
            metadata={"strategy": "size"},
        ))

        current_line += chunk_lines
        start = end - overlap
        if start <= 0 or start >= len(content):
            break

    return chunks


PYTHON_DEF_PATTERN = re.compile(r"^(async\s+)?(def|class)\s+(\w+)", re.MULTILINE)
JS_TS_DEF_PATTERN = re.compile(r"^(export\s+)?(async\s+)?(function|class|const|let|var)\s+(\w+)", re.MULTILINE)
GO_DEF_PATTERN = re.compile(r"^func\s+(\([^)]+\)\s+)?(\w+)", re.MULTILINE)
RUST_DEF_PATTERN = re.compile(r"^(pub\s+)?(fn|struct|enum|impl|trait)\s+(\w+)", re.MULTILINE)


def find_definitions(content: str, language: str) -> List[tuple]:
    """Find top-level definitions in code."""
    definitions = []

    if language == "python":
        for match in PYTHON_DEF_PATTERN.finditer(content):
            definitions.append((match.start(), match.group(0), match.group(3), match.group(2)))
    elif language in ("javascript", "typescript"):
        for match in JS_TS_DEF_PATTERN.finditer(content):
            definitions.append((match.start(), match.group(0), match.group(4), match.group(3)))
    elif language == "go":
        for match in GO_DEF_PATTERN.finditer(content):
            definitions.append((match.start(), match.group(0), match.group(2), "func"))
    elif language == "rust":
        for match in RUST_DEF_PATTERN.finditer(content):
            definitions.append((match.start(), match.group(0), match.group(3), match.group(2)))

    return sorted(definitions, key=lambda x: x[0])


def chunk_by_definitions(document: CodeDocument, chunk_size: int = 1000) -> List[Chunk]:
    """Chunk by function/class definitions with smart boundaries."""
    content = document.content
    definitions = find_definitions(content, document.language)

    if not definitions:
        return chunk_by_size(document, chunk_size)

    chunks = []
    lines = content.split("\n")

    for i, (start_pos, signature, name, def_type) in enumerate(definitions):
        # Determine end position (start of next definition or end of file)
        if i + 1 < len(definitions):
            end_pos = definitions[i + 1][0]
        else:
            end_pos = len(content)

        chunk_text = content[start_pos:end_pos].strip()
        if not chunk_text:
            continue

        # Calculate line numbers
        start_line = content[:start_pos].count("\n") + 1
        end_line = content[:end_pos].count("\n") + 1

        # If chunk is too large, fall back to size-based chunking for this segment
        if len(chunk_text) > chunk_size * 2:
            sub_doc = CodeDocument(
                path=document.path,
                content=chunk_text,
                language=document.language,
                relative_path=document.relative_path,
            )
            sub_chunks = chunk_by_size(sub_doc, chunk_size, overlap=200)
            for sc in sub_chunks:
                sc.start_line += start_line - 1
                sc.end_line += start_line - 1
            chunks.extend(sub_chunks)
        else:
            chunks.append(Chunk(
                content=chunk_text,
                source_path=document.relative_path,
                language=document.language,
                chunk_type=def_type,
                start_line=start_line,
                end_line=end_line,
                metadata={"name": name, "strategy": "definition"},
            ))

    return chunks


def chunk_document(document: CodeDocument, chunk_size: int = 1000, overlap: int = 200) -> List[Chunk]:
    """Intelligently chunk a document based on its language."""
    # For supported languages, try definition-based chunking
    if document.language in ("python", "javascript", "typescript", "go", "rust", "java", "csharp"):
        chunks = chunk_by_definitions(document, chunk_size)
        if chunks:
            return chunks

    # Fallback to size-based chunking
    return chunk_by_size(document, chunk_size, overlap)

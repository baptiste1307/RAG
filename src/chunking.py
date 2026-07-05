import ast
import json
from pathlib import Path
from typing import TypedDict
from tqdm import tqdm
from .config import global_state as state
from .models import MinimalSource, TextChunk


class MarkdownSection(TypedDict):
    """Markdown section with original character boundaries."""

    text: str
    first_character_index: int
    last_character_index: int


def _read_file(path: Path) -> str:
    """Return the content of the given file as a string."""
    try:
        with open(path, "r", encoding="utf-8") as file:
            return file.read()
    except (OSError, ValueError):
        return ""


def _validate_chunk_settings(max_chunk_size: int) -> None:
    """Validate shared chunking settings."""
    if max_chunk_size <= 0 or max_chunk_size > 2000:
        raise ValueError("max_chunk_size must be positive (max 2000)")
    if state.overlap < 0 or state.overlap >= max_chunk_size:
        raise ValueError("state.overlap must be >= 0 and < max_chunk_size")


def _chunk_text_range(
    file_path: Path,
    text: str,
    first_character_index: int,
    max_chunk_size: int,
) -> list[TextChunk]:
    """Chunk text while preserving indexes from the original file."""
    chunks: list[TextChunk] = []
    step = max_chunk_size - state.overlap

    for offset in range(0, len(text), step):
        chunk_text = text[offset: offset + max_chunk_size]
        start = first_character_index + offset
        end = first_character_index + offset + len(chunk_text)

        chunks.append(
            TextChunk(
                text=chunk_text,
                source=MinimalSource(
                    file_path=str(file_path),
                    first_character_index=start,
                    last_character_index=end,
                ),
            )
        )

    return chunks


def _chunk_text_file(file_path: Path, max_chunk_size: int) -> list[TextChunk]:
    """Chunk a file with fixed-size text chunks."""
    content = _read_file(file_path)
    return _chunk_text_range(file_path, content, 0, max_chunk_size)


def _get_md_sections(content: str) -> list[MarkdownSection]:
    """Split Markdown content into heading-based sections."""
    if not content:
        return []

    sections: list[MarkdownSection] = []
    section_start = 0
    position = 0
    seen_heading = False

    for line in content.splitlines(keepends=True):
        is_heading = line.lstrip().startswith("#")

        if is_heading and seen_heading:
            sections.append(
                {
                    "text": content[section_start:position],
                    "first_character_index": section_start,
                    "last_character_index": position,
                }
            )
            section_start = position

        if is_heading:
            seen_heading = True

        position += len(line)

    sections.append(
        {
            "text": content[section_start:],
            "first_character_index": section_start,
            "last_character_index": len(content),
        }
    )

    return [section for section in sections if section["text"]]


def _chunk_md(file_path: Path, max_chunk_size: int) -> list[TextChunk]:
    """Chunk Markdown by sections, with fixed-size fallback per section."""
    content = _read_file(file_path)
    chunks: list[TextChunk] = []

    for section in _get_md_sections(content):
        chunks.extend(
            _chunk_text_range(
                file_path,
                section["text"],
                section["first_character_index"],
                max_chunk_size,
            )
        )

    return chunks


def _line_start_indexes(content: str) -> list[int]:
    """Return the character index where each source line starts."""
    starts = [0]
    for index, char in enumerate(content):
        if char == "\n":
            starts.append(index + 1)
    return starts


def _node_range(content: str, node: ast.AST) -> tuple[int, int] | None:
    """Convert a Python AST node line range to character indexes."""
    lineno = getattr(node, "lineno", None)
    end_lineno = getattr(node, "end_lineno", None)
    if lineno is None or end_lineno is None:
        return None

    line_starts = _line_start_indexes(content)
    start = line_starts[lineno - 1]
    if end_lineno < len(line_starts):
        end = line_starts[end_lineno]
    else:
        end = len(content)

    return start, end


def _chunk_py(file_path: Path, max_chunk_size: int) -> list[TextChunk]:
    """Chunk Python files by top-level classes and functions."""
    content = _read_file(file_path)
    if not content:
        return []

    try:
        tree = ast.parse(content)
    except SyntaxError:
        return _chunk_text_file(file_path, max_chunk_size)

    chunks: list[TextChunk] = []
    for node in tree.body:
        if not isinstance(
            node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            continue

        node_range = _node_range(content, node)
        if node_range is None:
            continue

        start, end = node_range
        chunks.extend(
            _chunk_text_range(
                file_path,
                content[start:end],
                start,
                max_chunk_size,
            )
        )

    if not chunks:
        return _chunk_text_file(file_path, max_chunk_size)

    return chunks


def _chunk_file(file_path: Path, max_chunk_size: int) -> list[TextChunk]:
    """Chunk the given file according to its type."""
    if file_path.suffix == ".py":
        return _chunk_py(file_path, max_chunk_size)
    if file_path.suffix == ".md":
        return _chunk_md(file_path, max_chunk_size)
    return _chunk_text_file(file_path, max_chunk_size)


def _collect_files() -> list[Path]:
    """Collect all Python and Markdown files recursively."""
    try:
        return [
            file_path
            for file_path in state.repo_path.rglob("*")
            if file_path.is_file() and file_path.suffix in {".py", ".md"}
        ]
    except (OSError, ValueError) as e:
        raise RuntimeError(f"Failed to read {state.repo_path}: {e}") from e


def _chunk_repo(max_chunk_size: int) -> list[TextChunk]:
    """Chunk each supported file and return all chunks."""
    chunks: list[TextChunk] = []
    files = _collect_files()

    for file_path in tqdm(files, desc="Chunking repository"):
        chunks.extend(_chunk_file(file_path, max_chunk_size))

    return chunks


def create_and_save_chunks(max_chunk_size: int = 2000) -> list[TextChunk]:
    """Create repository chunks and save them as JSON."""
    _validate_chunk_settings(max_chunk_size)
    state.save_chunks_path.parent.mkdir(parents=True, exist_ok=True)

    chunks = _chunk_repo(max_chunk_size)
    data = [chunk.model_dump() for chunk in chunks]

    with open(state.save_chunks_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    return chunks

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import yaml
from tqdm import tqdm

from common.io_utils import write_jsonl
from common.logging_utils import get_logger
from common.schemas import DocumentRecord, build_metadata, normalize_doc_type
from common.text_utils import normalize_newlines, normalize_whitespace, pack_paragraphs, stable_id

logger = get_logger(__name__)

_HEADING_RE = re.compile(r"^(?P<hashes>#{1,6})\s+(?P<title>.+?)\s*#*\s*$")
_FRONT_MATTER_RE = re.compile(r"\A---\n(?P<body>.*?)\n---\n?", re.DOTALL)
_SUPPORTED_SUFFIXES = {".md", ".markdown"}


@dataclass(frozen=True)
class MarkdownSection:
    """Represent a Markdown heading section with normalized content."""

    title: str
    level: int
    path: str
    content: str


def _strip_front_matter(markdown: str) -> tuple[str, dict[str, str]]:
    """Remove simple YAML front matter and return scalar string metadata."""
    match = _FRONT_MATTER_RE.match(markdown)
    if not match:
        return markdown, {}

    raw_body = match.group("body")
    metadata: dict[str, str] = {}
    try:
        loaded = yaml.safe_load(raw_body) or {}
        if not isinstance(loaded, dict):
            loaded = {}
    except yaml.YAMLError:
        loaded = {}

    for key, value in loaded.items():
        if isinstance(value, str) and value.strip():
            metadata[str(key)] = value.strip()

    return markdown[match.end():], metadata


def _clean_heading_title(title: str) -> str:
    """Normalize a Markdown heading title."""
    return normalize_whitespace(title.strip().strip("#").strip()) or "Untitled Section"


def _extract_title(path: Path, front_matter: dict[str, str], sections: list[MarkdownSection]) -> str:
    """Pick the best title for a Markdown document."""
    title = front_matter.get("title")
    if title:
        return title

    top_heading = next((section.title for section in sections if section.level == 1), None)
    if top_heading:
        return top_heading

    return path.stem.replace("-", " ").replace("_", " ").strip() or path.stem


def _split_sections(
    markdown: str,
    default_title: str,
    split_level: int = 1,
) -> list[MarkdownSection]:
    """Split Markdown into heading-scoped sections while ignoring fenced code headings."""
    lines = normalize_newlines(markdown).split("\n")
    sections: list[MarkdownSection] = []
    heading_stack: dict[int, str] = {}
    current_title = default_title
    current_level = 1
    current_path = default_title
    current_lines: list[str] = []
    in_fence = False

    def flush() -> None:
        """Append the buffered section content and reset the section line buffer."""
        nonlocal current_lines
        content = "\n".join(current_lines).strip()
        if content:
            sections.append(
                MarkdownSection(
                    title=current_title,
                    level=current_level,
                    path=current_path,
                    content=content,
                )
            )
        current_lines = []

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```") or stripped.startswith("~~~"):
            in_fence = not in_fence
            current_lines.append(line)
            continue

        match = _HEADING_RE.match(stripped) if not in_fence else None
        if match and len(match.group("hashes")) > split_level:
            match = None

        if not match:
            current_lines.append(line)
            continue

        flush()
        current_level = len(match.group("hashes"))
        current_title = _clean_heading_title(match.group("title"))
        heading_stack[current_level] = current_title
        for stacked_level in sorted([level for level in heading_stack if level > current_level]):
            heading_stack.pop(stacked_level, None)

        path_parts = [
            heading_stack[level]
            for level in sorted(heading_stack)
            if level <= current_level and heading_stack.get(level)
        ]
        current_path = " > ".join(path_parts) if path_parts else current_title

    flush()

    if sections:
        return sections

    body = "\n".join(lines).strip()
    if not body:
        return []
    return [
        MarkdownSection(
            title=default_title,
            level=1,
            path=default_title,
            content=body,
        )
    ]


def _markdown_paragraphs(section: MarkdownSection) -> list[str]:
    """Return paragraph blocks from a section without flattening fenced code."""
    paragraphs: list[str] = []
    current: list[str] = []
    in_fence = False

    def flush() -> None:
        """Append the buffered paragraph text and reset the paragraph line buffer."""
        nonlocal current
        text = "\n".join(current).strip()
        if text:
            paragraphs.append(text)
        current = []

    for line in normalize_newlines(section.content).split("\n"):
        stripped = line.strip()
        if stripped.startswith("```") or stripped.startswith("~~~"):
            in_fence = not in_fence
            current.append(line)
            continue

        if not in_fence and not stripped:
            flush()
            continue

        current.append(line)

    flush()
    return paragraphs


def _build_markdown_records(
    markdown_path: Path,
    title: str,
    section: MarkdownSection,
    target_chars: int,
    min_chars: int,
    split_level: int,
    product: str | None,
    doc_version: str | None,
    doc_type: str | None,
    created_at: datetime | None,
    updated_at: datetime | None,
    ingested_at: datetime,
) -> list[DocumentRecord]:
    """Build prechunked Markdown records from one heading section."""
    paragraphs = _markdown_paragraphs(section)
    if not paragraphs:
        return []

    chunks = pack_paragraphs(paragraphs, target_chars, min_chars)
    records: list[DocumentRecord] = []
    for block_index, chunk_text in enumerate(chunks):
        text = chunk_text.strip()
        if not text:
            continue

        source_ref = f"markdown::{markdown_path}#section={section.path}:block={block_index}"
        doc_id = stable_id(
            "markdown-section",
            str(markdown_path.resolve()),
            section.path,
            str(block_index),
            text[:120],
        )
        markdown = f"{'#' * min(section.level, 6)} {section.title}\n\n{text}\n"
        metadata = build_metadata(
            product=product,
            doc_version=doc_version,
            doc_type=doc_type,
            section_path=section.path,
            created_at=created_at,
            updated_at=updated_at,
            ingested_at=ingested_at,
        )
        paragraph_count = len(
            _markdown_paragraphs(
                MarkdownSection(section.title, section.level, section.path, text)
            )
        )
        split_mode = "chapter_whole" if target_chars <= 0 and split_level == 1 else "section_paragraph"
        metadata.update(
            {
                "file_name": markdown_path.name,
                "markdown_title": title,
                "section_title": section.title,
                "section_level": section.level,
                "active_heading": section.title,
                "split_mode": split_mode,
                "chunk_strategy": "paragraph_bound",
                "markdown_split_level": split_level,
                "block_index": block_index,
                "paragraph_count": paragraph_count,
            }
        )

        records.append(
            DocumentRecord(
                doc_id=doc_id,
                source_type="markdown",
                source_ref=source_ref,
                source_path=str(markdown_path),
                title=title,
                text=text,
                markdown=markdown,
                created_at=created_at,
                metadata=metadata,
            )
        )

    return records


def extract_markdown_records(
    markdown_path: Path,
    target_chars: int = 0,
    min_chars: int = 0,
    split_level: int = 1,
    product: str | None = None,
    doc_version: str | None = None,
    doc_type: str | None = None,
) -> list[DocumentRecord]:
    """Extract a Markdown file into structure-aware paragraph-block records."""
    if split_level < 1 or split_level > 6:
        raise ValueError("split_level must be between 1 and 6")

    ingested_at = datetime.now(timezone.utc)
    stat = markdown_path.stat()
    updated_at = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)
    created_at = updated_at
    raw_markdown = normalize_newlines(markdown_path.read_text(encoding="utf-8"))
    body_markdown, front_matter = _strip_front_matter(raw_markdown)
    initial_title = front_matter.get("title") or markdown_path.stem
    sections = _split_sections(body_markdown, initial_title, split_level=split_level)
    title = _extract_title(markdown_path, front_matter, sections)

    records: list[DocumentRecord] = []
    for section in sections:
        records.extend(
            _build_markdown_records(
                markdown_path=markdown_path,
                title=title,
                section=section,
                target_chars=target_chars,
                min_chars=min_chars,
                split_level=split_level,
                product=product,
                doc_version=doc_version,
                doc_type=doc_type,
                created_at=created_at,
                updated_at=updated_at,
                ingested_at=ingested_at,
            )
        )

    for record in records:
        record.metadata["markdown_metadata"] = front_matter
    return records


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for ingest markdown."""
    parser = argparse.ArgumentParser(
        description="Extract Markdown files into structure-aware paragraph block records."
    )
    parser.add_argument("--input-dir", default="data/raw/markdown")
    parser.add_argument("--output", default="data/staging/documents/markdown_documents.jsonl")
    parser.add_argument(
        "--target-chars",
        type=int,
        default=0,
        help=(
            "Target text size for each paragraph block record. "
            "Use 0 to keep each split-level section whole."
        ),
    )
    parser.add_argument(
        "--min-chars",
        type=int,
        default=0,
        help="Minimum text size for each paragraph block record.",
    )
    parser.add_argument(
        "--split-level",
        type=int,
        default=1,
        help="Deepest Markdown heading level that starts a new record (1 keeps H1 chapters whole).",
    )
    parser.add_argument("--product", default="", help="Product label for indexed metadata.")
    parser.add_argument("--doc-version", default="", help="Document version label.")
    parser.add_argument(
        "--doc-type",
        default="",
        help="Document type metadata (guide, api, release-note).",
    )
    return parser.parse_args()


def main() -> None:
    """Run the ingest markdown entrypoint."""
    args = parse_args()
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        logger.warning("Markdown input directory does not exist, writing empty output: %s", input_dir)
        write_jsonl(args.output, [])
        return

    product = args.product.strip() or None
    doc_version = args.doc_version.strip() or None
    doc_type = normalize_doc_type(args.doc_type)

    markdown_paths = sorted(
        path
        for path in input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in _SUPPORTED_SUFFIXES
    )
    if not markdown_paths:
        logger.warning("No Markdown files found in %s", input_dir)
        write_jsonl(args.output, [])
        return

    all_rows: list[dict] = []
    for markdown_path in tqdm(markdown_paths, desc="Extracting Markdown"):
        try:
            records = extract_markdown_records(
                markdown_path=markdown_path,
                target_chars=args.target_chars,
                min_chars=args.min_chars,
                split_level=args.split_level,
                product=product,
                doc_version=doc_version,
                doc_type=doc_type,
            )
            all_rows.extend(record.model_dump(mode="json") for record in records)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to parse Markdown %s: %s", markdown_path, exc)

    count = write_jsonl(args.output, all_rows)
    logger.info("Wrote %s Markdown structured documents to %s", count, args.output)


if __name__ == "__main__":
    main()

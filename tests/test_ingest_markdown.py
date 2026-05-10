from __future__ import annotations

from pathlib import Path

from ingestion.ingest_markdown import extract_markdown_records
from ingestion.normalize_docs import _chunk_docs


def test_extract_markdown_records_uses_heading_sections(tmp_path: Path) -> None:
    """Verify Markdown ingestion emits records with heading paths and fenced code intact."""
    markdown_path = tmp_path / "guide.md"
    markdown_path.write_text(
        """---
title: Decision Guide
---
# Overview

Intro paragraph for the guide.

## Configuration

Use this setting carefully.

```python
# Not A Heading
value = 1
```
""",
        encoding="utf-8",
    )

    records = extract_markdown_records(
        markdown_path,
        target_chars=80,
        min_chars=0,
        split_level=6,
        product="Decisioning",
        doc_type="guide",
    )

    assert records
    assert {record.source_type for record in records} == {"markdown"}
    assert records[0].title == "Decision Guide"
    assert records[0].metadata["markdown_title"] == "Decision Guide"
    assert records[0].metadata["section_path"] == "Overview"
    assert any(
        record.metadata["section_path"] == "Overview > Configuration"
        for record in records
    )
    assert any("# Not A Heading" in record.text for record in records)


def test_extract_markdown_records_keeps_h1_chapter_whole_by_default(tmp_path: Path) -> None:
    """Verify default Markdown splitting preserves each H1 chapter as one record."""
    markdown_path = tmp_path / "guide.md"
    markdown_path.write_text(
        """# Chapter One

First paragraph.

## Details

Second paragraph stays in the same chapter record.

# Chapter Two

Third paragraph.
""",
        encoding="utf-8",
    )

    records = extract_markdown_records(markdown_path)

    assert len(records) == 2
    assert records[0].metadata["section_path"] == "Chapter One"
    assert "## Details" in records[0].text
    assert "Second paragraph stays in the same chapter record." in records[0].text
    assert records[1].metadata["section_path"] == "Chapter Two"


def test_markdown_prechunked_records_are_preserved(tmp_path: Path) -> None:
    """Verify prechunked Markdown records are not split again by normalization."""
    markdown_path = tmp_path / "guide.md"
    markdown_path.write_text(
        "# Guide\n\nFirst paragraph.\n\nSecond paragraph.\n",
        encoding="utf-8",
    )

    records = extract_markdown_records(markdown_path, target_chars=20, min_chars=0)
    chunks = _chunk_docs(records, chunk_size=20, chunk_overlap=5, min_chunk_chars=0)

    assert len(chunks) == len(records)
    assert [chunk.text for chunk in chunks] == [record.text for record in records]
    assert {chunk.source_type for chunk in chunks} == {"markdown"}

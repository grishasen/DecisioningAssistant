#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-}:src"

python3 -m ingestion.ingest_pdfs \
  --input-dir data/raw/pdf \
  --output data/staging/documents/pdf_documents.jsonl

python3 -m ingestion.ingest_markdown \
  --input-dir data/raw/markdown \
  --output data/staging/documents/markdown_documents.jsonl

python3 -m ingestion.ingest_webex \
  --input-dir data/raw/webex \
  --output data/staging/documents/webex_documents.jsonl

python3 -m ingestion.normalize_docs --config configs/sources.yaml

"""Ingest company policies into ChromaDB.

Usage (from backend/):
    python -m scripts.ingest                       # rebuild from every file in data/policy/
    python -m scripts.ingest path/to/policy.pdf    # add / refresh one file (PDF, DOCX, TXT or MD)
"""

import sys
from pathlib import Path

from app.services.ingestion import IngestionError, ingest_all, ingest_policy


def main() -> int:
    try:
        result = ingest_policy(Path(sys.argv[1])) if len(sys.argv) > 1 else ingest_all()
    except IngestionError as exc:
        print(f"Ingestion failed: {exc}", file=sys.stderr)
        return 1

    print(f"Ingested {result['chunks']} chunks from {result['source']}:")
    for s in result["sections"]:
        print(f"  {s['label'] or s['section_title']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

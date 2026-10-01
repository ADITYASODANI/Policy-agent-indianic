"""Ingest the company policy into ChromaDB.

Usage (from backend/):
    python -m scripts.ingest                       # uses POLICY_PATH from settings
    python -m scripts.ingest path/to/policy.pdf    # PDF, DOCX or TXT
"""

import sys
from pathlib import Path

from app.services.ingestion import IngestionError, ingest_policy


def main() -> int:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    try:
        result = ingest_policy(path)
    except IngestionError as exc:
        print(f"Ingestion failed: {exc}", file=sys.stderr)
        return 1

    print(f"Ingested {result['chunks']} chunks from {result['source']}:")
    for s in result["sections"]:
        print(f"  Section {s['section_number']:>2} — {s['section_title']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

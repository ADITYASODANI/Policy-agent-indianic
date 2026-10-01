"""Policy ingestion: load PDF/DOCX/TXT/MD -> split by section -> embed -> ChromaDB.

Several policy documents live side by side in the vector store. Each chunk carries its
`source` file, so a document can be added or refreshed without touching the others.
"""

import logging
import re
from dataclasses import dataclass
from pathlib import Path

from docx import Document as DocxDocument
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from app.config import get_settings
from app.services.vectorstore import get_vectorstore, reset_vectorstore

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}

# Markdown exports (e.g. from an HR portal): "#### Maternity Leave".
_MD_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
# Navigation / boilerplate blocks that carry no policy content.
_BOILERPLATE_HEADINGS = {"index", "table of contents", "contents", "revision history"}
_NAV_NOISE = re.compile(r"keyboard\\?_arrow\\?_\w+|^\W*go back\W*$", re.IGNORECASE)

# Top-level headings look like "4. EMPLOYEE HEALTH, SAFETY & WORKPLACE SECURITY".
_CAPS_HEADING = re.compile(r"^\s*(\d{1,2})\.\s+([A-Z][A-Z0-9 &,/()'’\-–—]{2,120})\s*$")
# Fallback for documents whose headings are not upper-case.
_ANY_HEADING = re.compile(r"^\s*(\d{1,2})\.\s+([A-Z][^.\n]{2,100})\s*$")

_SMALL_WORDS = {"a", "an", "and", "at", "by", "for", "from", "in", "of", "on", "or", "the", "to"}

OVERVIEW_SECTION = "0"
OVERVIEW_TITLE = "Policy Information"


class IngestionError(Exception):
    pass


@dataclass
class Section:
    number: str
    title: str
    text: str


def load_text(path: Path) -> str:
    """Extract raw text from a PDF, DOCX or TXT file."""
    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise IngestionError(f"Unsupported file type '{ext}'. Use PDF, DOCX or TXT.")
    if not path.is_file():
        raise IngestionError(f"Policy file not found: {path}")

    try:
        if ext == ".pdf":
            reader = PdfReader(str(path))
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
        elif ext == ".docx":
            doc = DocxDocument(str(path))
            parts = [p.text for p in doc.paragraphs]
            for table in doc.tables:
                for row in table.rows:
                    parts.append(" | ".join(cell.text for cell in row.cells))
            text = "\n".join(parts)
        else:
            text = path.read_text(encoding="utf-8")
    except Exception as exc:  # pypdf/python-docx raise a variety of errors
        raise IngestionError(f"Could not read policy file: {exc}") from exc

    if not text.strip():
        raise IngestionError("No text could be extracted from the policy file.")
    return text


def clean_text(text: str) -> str:
    lines = [line.rstrip() for line in text.replace("\r\n", "\n").split("\n")]
    text = "\n".join(lines)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def title_case(heading: str) -> str:
    """'REMOTE WORK & WORK-FROM-HOME (WFH) RESPONSIBILITIES' -> 'Remote Work & Work-From-Home (WFH) Responsibilities'."""
    words = []
    for i, word in enumerate(heading.split()):
        if word.startswith("(") and word.endswith(")"):
            words.append(word.upper())  # acronyms such as (POSH), (WFH)
        elif i > 0 and word.lower() in _SMALL_WORDS:
            words.append(word.lower())
        else:
            words.append("-".join(part.capitalize() for part in word.split("-")))
    return " ".join(words)


def extract_metadata(text: str, policy_name: str | None = None) -> dict[str, str]:
    """Policy name / version / effective date. The configured defaults describe the primary
    Code of Conduct only; any other document that names itself gets blanks instead."""
    settings = get_settings()
    primary = policy_name is None
    meta = {
        "policy_name": policy_name or settings.policy_name,
        "version": settings.policy_version if primary else "",
        "effective_date": settings.policy_effective_date if primary else "",
    }
    if m := re.search(r"Effective Date\s*:\s*(.+)", text, re.IGNORECASE):
        meta["effective_date"] = m.group(1).strip()
    if m := re.search(r"\bVersion\s*(\d+(?:\.\d+)?)", text, re.IGNORECASE):
        meta["version"] = f"Version {m.group(1)}"
    return meta


def split_sections(text: str) -> list[Section]:
    lines = text.split("\n")
    pattern = _CAPS_HEADING
    if not any(_CAPS_HEADING.match(line) for line in lines):
        pattern = _ANY_HEADING

    sections: list[Section] = []
    current = Section(OVERVIEW_SECTION, OVERVIEW_TITLE, "")
    buffer: list[str] = []

    def flush() -> None:
        body = "\n".join(buffer).strip()
        if body or current.number != OVERVIEW_SECTION:
            sections.append(Section(current.number, current.title, body))

    for line in lines:
        match = pattern.match(line)
        if match:
            flush()
            title = match.group(2).strip()
            if title.isupper():
                title = title_case(title)
            current = Section(match.group(1), title, "")
            buffer = []
        else:
            buffer.append(line)
    flush()

    if len([s for s in sections if s.number != OVERVIEW_SECTION]) == 0:
        raise IngestionError("No numbered policy sections were detected in the document.")
    return sections


def is_markdown(text: str) -> bool:
    return any(_MD_HEADING.match(line) for line in text.split("\n"))


def _plain(text: str) -> str:
    """Drop markdown emphasis/escapes and turn links into readable text."""
    text = re.sub(r"\[([^\]]+)\]\(mailto:[^)]*\)", r"\1", text)
    text = re.sub(
        r"\[([^\]]+)\]\((https?://[^)]*)\)",
        lambda m: m.group(1) if m.group(1) == m.group(2) else f"{m.group(1)} ({m.group(2)})",
        text,
    )
    return text.replace("\\_", "_").replace("**", "").strip()


def policy_code(name: str) -> str:
    """Short id prefix for a policy's sections: 'Maternity & Paternity Leave Policy' -> 'MPL'."""
    words = [w for w in re.findall(r"[A-Za-z0-9]+", name) if w.lower() != "policy"]
    return "".join(w[0].upper() for w in words) or "DOC"


def split_markdown_sections(text: str) -> tuple[str | None, list[Section]]:
    """Split a markdown policy into (title, sections).

    The first heading is the document title when it sits above the section headings.
    Sections are the shallowest remaining heading level; deeper headings stay inside their
    section as plain lines. Index / revision-history blocks are dropped. Section numbers
    are the plain position (1, 2, 3 ...) within the document.
    """
    blocks: list[list] = []  # [level, heading, body lines]
    for line in text.split("\n"):
        if match := _MD_HEADING.match(line):
            blocks.append([len(match.group(1)), _plain(match.group(2)), []])
        elif blocks and not _NAV_NOISE.search(line):
            blocks[-1][2].append(line)
    blocks = [b for b in blocks if b[1].lower() not in _BOILERPLATE_HEADINGS]
    if len(blocks) < 2:
        raise IngestionError("No headed policy sections were detected in the document.")

    section_level = min(b[0] for b in blocks[1:])
    title = blocks.pop(0)[1] if blocks[0][0] < section_level else None

    sections: list[Section] = []
    buffer: list[str] = []
    current: Section | None = None

    def flush() -> None:
        if current is not None:
            current.text = "\n".join(buffer).strip()

    for level, heading, body in blocks:
        if level <= section_level:
            flush()
            current = Section(str(len(sections) + 1), heading, "")
            sections.append(current)
            buffer = []
        elif current is None:
            continue
        else:
            buffer.append(heading)  # sub-heading, kept as a plain line
        buffer.extend(re.sub(r"^(\s*)[*+]\s+", r"\1- ", _plain(line)) for line in body)
    flush()

    sections = [s for s in sections if s.text]
    if not sections:
        raise IngestionError("No policy content was found under the headings.")
    return title, sections


@dataclass
class ParsedPolicy:
    source: str
    meta: dict[str, str]
    sections: list[Section]
    numbered: bool  # True: "Section 4" style numbering; False: identified by policy name + title


def parse_policy(path: Path, source_name: str | None = None) -> ParsedPolicy:
    source = source_name or path.name
    text = clean_text(load_text(path))
    if is_markdown(text):
        title, sections = split_markdown_sections(text)
        name = title or Path(source).stem.replace("_", " ").strip()
        code = policy_code(name)
        sections = [Section(f"{code}.{s.number}", s.title, s.text) for s in sections]
        return ParsedPolicy(source, extract_metadata(text, policy_name=name), sections, numbered=False)
    return ParsedPolicy(source, extract_metadata(text), split_sections(text), numbered=True)


def build_chunks(parsed: ParsedPolicy) -> list[Document]:
    settings = get_settings()
    sections, base_meta, source = parsed.sections, parsed.meta, parsed.source
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    docs: list[Document] = []
    for section in sections:
        if not section.text:
            continue
        if section.number == OVERVIEW_SECTION:
            label = heading = section.title
        elif parsed.numbered:
            label = heading = f"Section {section.number} — {section.title}"
        else:
            # Not numbered in the document, so the policy name identifies it; the id lets the
            # LLM cite the section.
            label = f"{base_meta['policy_name']} — {section.title}"
            heading = f"{label} (Section ID: {section.number})"
        order = int(section.number.rsplit(".", 1)[-1])
        for i, piece in enumerate(splitter.split_text(section.text)):
            # The heading is prepended so every chunk carries its section context
            # for both embedding similarity and the LLM prompt.
            docs.append(
                Document(
                    page_content=f"{heading}\n{piece}",
                    metadata={
                        **base_meta,
                        "section_number": section.number,
                        "section_title": section.title,
                        "section_label": label,
                        "section_order": order,
                        "chunk_index": i,
                        "source": source,
                    },
                )
            )
    return docs


def _summary(parsed: ParsedPolicy, chunks: list[Document]) -> dict:
    labels = {c.metadata["section_number"]: c.metadata["section_label"] for c in chunks}
    return {
        "source": parsed.source,
        "chunks": len(chunks),
        "sections": [
            {"section_number": s.number, "section_title": s.title, "label": labels.get(s.number)}
            for s in parsed.sections
            if s.number != OVERVIEW_SECTION
        ],
    }


def _store_chunks(parsed: ParsedPolicy) -> list[Document]:
    chunks = build_chunks(parsed)
    ids = [f"{parsed.source}:s{d.metadata['section_number']}-c{d.metadata['chunk_index']}" for d in chunks]
    get_vectorstore().add_documents(chunks, ids=ids)
    logger.info("Ingested %d chunks from %s", len(chunks), parsed.source)
    return chunks


def ingest_policy(path: Path, source_name: str | None = None) -> dict:
    """Add one policy document, replacing an earlier version of the same file.
    Other documents in the vector store are left untouched."""
    parsed = parse_policy(Path(path), source_name)
    get_vectorstore().delete(where={"source": parsed.source})
    return _summary(parsed, _store_chunks(parsed))


def ingest_all(directory: Path | None = None) -> dict:
    """Rebuild the vector store from every supported file in the policy directory."""
    directory = Path(directory or get_settings().policy_dir)
    files = sorted(p for p in directory.glob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS)
    if not files:
        raise IngestionError(f"No policy files (PDF, DOCX, TXT, MD) found in {directory}")

    parsed_docs = []
    for file in files:  # parse everything first so a bad file cannot leave the store half-wiped
        try:
            parsed_docs.append(parse_policy(file))
        except IngestionError as exc:
            raise IngestionError(f"{file.name}: {exc}") from exc

    reset_vectorstore()
    summaries = [_summary(p, _store_chunks(p)) for p in parsed_docs]
    return {
        "source": ", ".join(s["source"] for s in summaries),
        "chunks": sum(s["chunks"] for s in summaries),
        "sections": [sec for s in summaries for sec in s["sections"]],
    }

"""Retrieval-augmented answering: rewrite -> similarity search -> Groq -> validated answer."""

import json
import logging
from dataclasses import dataclass, field

from groq import RateLimitError
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

from app.config import get_settings
from app.services.ingestion import OVERVIEW_SECTION
from app.services.prompts import ANSWER_PROMPT, NOT_FOUND_MESSAGE, REWRITE_PROMPT
from app.services.vectorstore import chunk_count, get_vectorstore

logger = logging.getLogger(__name__)


class PolicyNotIngested(Exception):
    pass


class LLMUnavailable(Exception):
    pass


class LLMRateLimited(LLMUnavailable):
    pass


@dataclass
class RagAnswer:
    answer: str
    found_in_policy: bool
    policy_references: list[dict] = field(default_factory=list)
    recommended_action: str | None = None


def _llm(model: str, json_mode: bool = False) -> ChatGroq:
    settings = get_settings()
    if not settings.groq_api_key:
        raise LLMUnavailable("GROQ_API_KEY is not configured.")
    llm = ChatGroq(
        model=model,
        api_key=settings.groq_api_key,
        temperature=settings.llm_temperature,
        max_retries=4,  # the Groq client backs off using the retry-after header
        timeout=60,
    )
    if json_mode:
        return llm.bind(response_format={"type": "json_object"})
    return llm


def rewrite_query(question: str, recent_turns: list[dict]) -> str:
    """Normalise spelling / Hinglish / follow-ups into an English search query."""
    history = "\n".join(f"Employee: {t['question']}\nAssistant: {t['answer']}" for t in recent_turns[-3:])
    try:
        chain = REWRITE_PROMPT | _llm(get_settings().groq_rewrite_model) | StrOutputParser()
        rewritten = chain.invoke({"history": history or "(none)", "question": question}).strip()
        return rewritten.strip('"').splitlines()[0][:300] if rewritten else question
    except Exception as exc:
        # Retrieval still works on the raw question, so do not fail the request.
        logger.warning("Query rewrite failed, using original question: %s", exc)
        return question


def _chunk_key(doc: Document) -> str:
    return f"{doc.metadata.get('section_number')}-{doc.metadata.get('chunk_index')}"


def retrieve(primary: str, original: str, k: int, extra: int = 2) -> list[Document]:
    """Top-k chunks for the normalised query, plus up to `extra` new chunks for the raw text.

    The raw question is only a fallback: Hinglish or misspelt text embeds poorly, so
    letting it compete on score would push relevant chunks out of the top k.
    """
    store = get_vectorstore()
    docs = store.similarity_search(primary, k=k)
    if original.strip() and original.strip().lower() != primary.strip().lower():
        seen = {_chunk_key(d) for d in docs}
        added = 0
        for doc in store.similarity_search(original, k=k):
            if added >= extra:
                break
            if _chunk_key(doc) not in seen:
                docs.append(doc)
                seen.add(_chunk_key(doc))
                added += 1
    return docs


def _format_context(docs: list[Document]) -> str:
    return "\n\n---\n\n".join(doc.page_content for doc in docs)


def _parse_json(raw: str) -> dict:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.strip("`").removeprefix("json").strip()
    start, end = raw.find("{"), raw.rfind("}")
    return json.loads(raw[start : end + 1])


def answer_question(question: str, recent_turns: list[dict] | None = None) -> RagAnswer:
    settings = get_settings()
    if chunk_count() == 0:
        raise PolicyNotIngested("The policy has not been ingested yet.")

    search_query = rewrite_query(question, recent_turns or [])
    docs = retrieve(search_query, question, k=settings.retrieval_k)
    logger.info("Query %r -> %r, sections %s", question, search_query,
                [d.metadata["section_number"] for d in docs])

    meta = docs[0].metadata if docs else {}
    chain = ANSWER_PROMPT | _llm(settings.groq_model, json_mode=True) | StrOutputParser()
    try:
        raw = chain.invoke(
            {
                "policy_name": meta.get("policy_name", settings.policy_name),
                "policy_version": meta.get("version", settings.policy_version),
                "effective_date": meta.get("effective_date", settings.policy_effective_date),
                "whistleblower_url": settings.whistleblower_url,
                "context": _format_context(docs),
                "question": f"{question}\n(Interpreted as: {search_query})"
                if search_query != question
                else question,
            }
        )
    except RateLimitError as exc:
        logger.warning("Groq rate limit hit: %s", exc)
        raise LLMRateLimited(
            "The assistant is receiving too many questions right now. Please try again in a few seconds."
        ) from exc
    except Exception as exc:
        logger.exception("Groq call failed")
        raise LLMUnavailable("The language model is currently unavailable.") from exc

    try:
        data = _parse_json(raw)
    except (ValueError, json.JSONDecodeError):
        logger.error("Unparseable LLM output: %s", raw[:500])
        return RagAnswer(answer=NOT_FOUND_MESSAGE, found_in_policy=False)

    return _validate(data, docs)


def _validate(data: dict, docs: list[Document]) -> RagAnswer:
    """Keep the model honest: references must come from retrieved chunks."""
    kind = str(data.get("kind", "not_found")).lower()
    answer = str(data.get("answer") or "").strip()

    if kind == "smalltalk" and answer:
        return RagAnswer(answer=answer, found_in_policy=False)
    if kind != "policy" or not answer:
        return RagAnswer(answer=NOT_FOUND_MESSAGE, found_in_policy=False)

    titles = {d.metadata["section_number"]: d.metadata["section_title"] for d in docs}
    cited = [str(s).strip().removeprefix("Section").strip() for s in data.get("section_numbers") or []]
    valid = [s for s in dict.fromkeys(cited) if s in titles]
    if not valid and docs:
        valid = [docs[0].metadata["section_number"]]  # every answer must carry a reference

    references = [{"section_number": s, "section_title": titles[s]} for s in valid]
    references.sort(key=lambda r: int(r["section_number"]) if r["section_number"].isdigit() else 0)

    action = data.get("recommended_action")
    action = str(action).strip() if action and str(action).strip().lower() not in {"null", "none", "n/a"} else None

    url = get_settings().whistleblower_url.rstrip("/")
    if "13" in valid and url not in answer and (not action or url not in action):
        note = f"Report through the official anonymous reporting platform: {get_settings().whistleblower_url}"
        action = f"{action} {note}" if action else note

    return RagAnswer(
        answer=answer,
        found_in_policy=True,
        policy_references=[r for r in references if r["section_number"] != OVERVIEW_SECTION] or references,
        recommended_action=action,
    )

"""Retrieval-augmented answering: master prompt (understand + write search query) -> similarity search -> Groq -> validated answer."""

import json
import logging
import time
from dataclasses import dataclass, field

from groq import RateLimitError
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

from app.config import get_settings
from app.services.ingestion import OVERVIEW_SECTION
from app.services.prompts import (
    ANSWER_PROMPT,
    MASTER_PROMPT,
    NOT_FOUND_MESSAGES,
    OVERVIEW_STYLE,
    SPECIFIC_STYLE,
    WHISTLEBLOWER_NOTES,
)
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
    language: str = "English"


def _llm(model: str, json_mode: bool = False, long_output: bool = False) -> ChatGroq:
    settings = get_settings()
    if not settings.groq_api_key:
        raise LLMUnavailable("GROQ_API_KEY is not configured.")
    extra = {}
    if long_output:
        # Whole-policy answers (esp. Devanagari) need far more output than the default budget,
        # and reasoning tokens count against it.
        extra = {"max_tokens": settings.overview_max_tokens, "reasoning_effort": "low"}
    llm = ChatGroq(
        model=model,
        api_key=settings.groq_api_key,
        temperature=settings.llm_temperature,
        max_retries=4,  # the Groq client backs off using the retry-after header
        timeout=90 if long_output else 60,
        **extra,
    )
    if json_mode:
        return llm.bind(response_format={"type": "json_object"})
    return llm


@dataclass
class QueryAnalysis:
    intent: str
    search_query: str
    language: str = "English"
    scope: str = "specific"  # "overview" = the employee wants a whole policy explained
    overview_policy: str | None = None  # which policy an overview is about; None = all


def list_policies() -> list[dict]:
    """The distinct policies in the vector store, by name."""
    got = get_vectorstore().get(include=["metadatas"])
    seen: dict[str, dict] = {}
    for meta in got["metadatas"]:
        seen.setdefault(meta["policy_name"], meta)
    return [seen[name] for name in sorted(seen)]


def _describe_policy(meta: dict) -> str:
    details = ", ".join(x for x in (meta.get("version"), f"effective {meta['effective_date']}" if meta.get("effective_date") else "") if x)
    return f"{meta['policy_name']} ({details})" if details else meta["policy_name"]


SUPPORTED_LANGUAGES = {"english": "English", "hindi": "Hindi"}


def _normalise_language(value: object, default: str = "English") -> str:
    return SUPPORTED_LANGUAGES.get(str(value or "").strip().lower(), default)


def understand_query(question: str, recent_turns: list[dict], policy_names: list[str]) -> QueryAnalysis:
    """First LLM pass (master prompt): work out what the employee wants, write a policy
    search query and decide the answer language (kept across the session)."""
    settings = get_settings()
    session_language = _normalise_language(recent_turns[-1].get("language")) if recent_turns else "English"
    history = "\n".join(f"Employee: {t['question']}\nAssistant: {t['answer']}" for t in recent_turns[-3:])
    logger.info("[Layer 1] user prompt: %r (session language: %s)", question, session_language)
    started = time.perf_counter()
    raw = ""
    try:
        chain = MASTER_PROMPT | _llm(settings.groq_rewrite_model, json_mode=True) | StrOutputParser()
        raw = chain.invoke(
            {
                "policy_names": "; ".join(policy_names),
                "history": history or "(none)",
                "current_language": session_language,
                "question": question,
            }
        )
        data = _parse_json(raw)
        query_text = str(data.get("search_query") or "").strip()
        if query_text:
            analysis = QueryAnalysis(
                intent=str(data.get("intent") or "").strip()[:300],
                search_query=query_text.splitlines()[0][:300],
                language=_normalise_language(data.get("answer_language"), session_language),
                scope="overview" if str(data.get("scope", "")).strip().lower() == "overview" else "specific",
                overview_policy=next(
                    (n for n in policy_names if n.lower() == str(data.get("overview_policy") or "").strip().lower()),
                    None,
                ),
            )
            logger.info("[Layer 1] response: %s | took %.2fs", raw.strip().replace("\n", " "), time.perf_counter() - started)
            return analysis
    except Exception as exc:
        # Retrieval still works on the raw question, so do not fail the request.
        logger.warning("[Layer 1] failed after %.2fs, using original question: %s | raw=%r",
                       time.perf_counter() - started, exc, raw[:300])
    return QueryAnalysis(intent="", search_query=question, language=session_language)


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


def retrieve_all(limit: int, policy_name: str | None = None) -> list[Document]:
    """Every chunk of one policy (or of all policies), in document order, for whole-policy questions."""
    got = get_vectorstore().get(include=["documents", "metadatas"])
    docs = [Document(page_content=c, metadata=m) for c, m in zip(got["documents"], got["metadatas"])]
    if policy_name:
        docs = [d for d in docs if d.metadata.get("policy_name") == policy_name]

    def order(doc: Document) -> tuple[str, int, int]:
        meta = doc.metadata
        return (meta.get("policy_name", ""), int(meta.get("section_order") or 0), int(meta.get("chunk_index") or 0))

    return sorted(docs, key=order)[:limit]


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

    total_started = time.perf_counter()
    policies = list_policies()
    analysis = understand_query(question, recent_turns or [], [p["policy_name"] for p in policies])
    search_query = analysis.search_query
    language = analysis.language

    retrieval_started = time.perf_counter()
    overview = analysis.scope == "overview"
    if overview:
        docs = retrieve_all(settings.overview_max_chunks, analysis.overview_policy)
    else:
        docs = retrieve(search_query, question, k=settings.retrieval_k)
    logger.info("[Retrieval] scope=%s%s search %r -> %d chunks, sections %s | took %.2fs", analysis.scope,
                f" ({analysis.overview_policy or 'all policies'})" if overview else "", search_query, len(docs),
                sorted({d.metadata["section_number"] for d in docs}), time.perf_counter() - retrieval_started)

    chain = ANSWER_PROMPT | _llm(settings.groq_model, json_mode=True, long_output=overview) | StrOutputParser()
    answer_started = time.perf_counter()
    try:
        raw = chain.invoke(
            {
                "policies": "; ".join(_describe_policy(p) for p in policies),
                "whistleblower_url": settings.whistleblower_url,
                "answer_language": language,
                "answer_style": OVERVIEW_STYLE if overview else SPECIFIC_STYLE,
                "context": _format_context(docs),
                "question": f"{question}\n(Interpreted as: {analysis.intent})"
                if analysis.intent
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

    logger.info("[Layer 2] answer (%s): %s | took %.2fs", language, raw.strip().replace("\n", " "),
                time.perf_counter() - answer_started)

    try:
        data = _parse_json(raw)
    except (ValueError, json.JSONDecodeError):
        logger.error("Unparseable LLM output: %s", raw[:500])
        return RagAnswer(answer=NOT_FOUND_MESSAGES[language], found_in_policy=False, language=language)

    result = _validate(data, docs, language)
    logger.info("[Total] request took %.2fs", time.perf_counter() - total_started)
    return result


def _validate(data: dict, docs: list[Document], language: str = "English") -> RagAnswer:
    """Keep the model honest: references must come from retrieved chunks."""
    kind = str(data.get("kind", "not_found")).lower()
    answer = str(data.get("answer") or "").strip()

    if kind == "smalltalk" and answer:
        return RagAnswer(answer=answer, found_in_policy=False, language=language)
    if kind != "policy" or not answer:
        return RagAnswer(answer=NOT_FOUND_MESSAGES[language], found_in_policy=False, language=language)

    titles = {d.metadata["section_number"]: d.metadata["section_title"] for d in docs}
    labels = {d.metadata["section_number"]: d.metadata.get("section_label") for d in docs}
    cited = [str(s).strip().removeprefix("Section").strip() for s in data.get("section_numbers") or []]
    valid = [s for s in dict.fromkeys(cited) if s in titles]
    if not valid and docs:
        valid = [docs[0].metadata["section_number"]]  # every answer must carry a reference

    references = [{"section_number": s, "section_title": titles[s], "label": labels[s]} for s in valid]
    references.sort(key=lambda r: (0, int(r["section_number"])) if r["section_number"].isdigit() else (1, 0))

    action = data.get("recommended_action")
    action = str(action).strip() if action and str(action).strip().lower() not in {"null", "none", "n/a"} else None

    url = get_settings().whistleblower_url.rstrip("/")
    if "13" in valid and url not in answer and (not action or url not in action):
        note = WHISTLEBLOWER_NOTES[language].format(url=get_settings().whistleblower_url)
        action = f"{action} {note}" if action else note

    return RagAnswer(
        answer=answer,
        found_in_policy=True,
        policy_references=[r for r in references if r["section_number"] != OVERVIEW_SECTION] or references,
        recommended_action=action,
        language=language,
    )

"""MongoDB chat history. Stores only Q/A content, references and an anonymous session id."""

import logging
from datetime import datetime

from pymongo import DESCENDING, MongoClient
from pymongo.collection import Collection
from pymongo.errors import PyMongoError

from app.config import get_settings

logger = logging.getLogger(__name__)

_client: MongoClient | None = None


class HistoryUnavailable(Exception):
    pass


def _collection() -> Collection:
    global _client
    settings = get_settings()
    if _client is None:
        _client = MongoClient(settings.mongodb_uri, serverSelectionTimeoutMS=2500, tz_aware=True)
    coll = _client[settings.mongodb_db][settings.mongodb_collection]
    return coll


def ensure_indexes() -> None:
    try:
        _collection().create_index([("session_id", 1), ("timestamp", DESCENDING)])
    except PyMongoError as exc:
        logger.warning("MongoDB not reachable, chat history disabled for now: %s", exc)


def save_turn(
    *,
    question: str,
    answer: str,
    policy_references: list[dict],
    recommended_action: str | None,
    found_in_policy: bool,
    session_id: str | None,
    timestamp: datetime,
) -> None:
    sections = [f"Section {r['section_number']}" for r in policy_references if r["section_number"] != "0"]
    doc = {
        "question": question,
        "answer": answer,
        "policy_section": ", ".join(sections) or None,
        "policy_references": policy_references,
        "recommended_action": recommended_action,
        "found_in_policy": found_in_policy,
        "session_id": session_id,
        "timestamp": timestamp,
    }
    try:
        _collection().insert_one(doc)
    except PyMongoError as exc:
        raise HistoryUnavailable(str(exc)) from exc


def get_history(session_id: str | None, limit: int = 50) -> list[dict]:
    query = {"session_id": session_id} if session_id else {}
    try:
        cursor = (
            _collection()
            .find(query, {"_id": 0, "session_id": 0})
            .sort("timestamp", DESCENDING)
            .limit(limit)
        )
        return list(reversed(list(cursor)))
    except PyMongoError as exc:
        raise HistoryUnavailable(str(exc)) from exc


def delete_session(session_id: str) -> int:
    try:
        return _collection().delete_many({"session_id": session_id}).deleted_count
    except PyMongoError as exc:
        raise HistoryUnavailable(str(exc)) from exc

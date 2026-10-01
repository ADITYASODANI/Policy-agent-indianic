from datetime import datetime

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)
    # Anonymous, client-generated id used only to group a conversation.
    session_id: str | None = Field(default=None, max_length=64)


class PolicyReference(BaseModel):
    section_number: str
    section_title: str


class ChatResponse(BaseModel):
    answer: str
    policy_references: list[PolicyReference] = []
    recommended_action: str | None = None
    found_in_policy: bool
    session_id: str | None = None
    timestamp: datetime


class HistoryItem(BaseModel):
    question: str
    answer: str
    policy_section: str | None = None
    policy_references: list[PolicyReference] = []
    recommended_action: str | None = None
    found_in_policy: bool = True
    timestamp: datetime


class HistoryResponse(BaseModel):
    items: list[HistoryItem]


class IngestResponse(BaseModel):
    message: str
    source: str
    chunks: int
    sections: list[PolicyReference]

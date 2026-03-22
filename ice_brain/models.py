"""
Pydantic models for ice_brain – OpenAI-compatible request/response schemas
plus internal router result type.
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Auth models
# ---------------------------------------------------------------------------

class LoginRequest(BaseModel):
    username: str
    password: str = ""   # empty string is valid for first-login check


class LoginResponse(BaseModel):
    user_id: str
    username: str
    role: str
    first_login: bool
    token: Optional[str] = None  # None when first_login=True (must set password first)


class SetPasswordRequest(BaseModel):
    user_id: str
    new_password: str


# ---------------------------------------------------------------------------
# OpenAI-compatible request models
# ---------------------------------------------------------------------------

class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "main"
    messages: List[ChatMessage]
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=512, ge=1, le=32768)
    stream: bool = False
    # Optional user-id extension (not part of standard OpenAI spec)
    user: Optional[str] = "default"
    # Session token issued by /auth/login
    session_token: Optional[str] = None
    # Optional IANA timezone name sent by the client (e.g. "Europe/Berlin").
    # Used to show the correct local time in the system prompt.
    timezone: Optional[str] = None


# ---------------------------------------------------------------------------
# OpenAI-compatible response models
# ---------------------------------------------------------------------------

class ChatCompletionChoice(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: str = "stop"


class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str = "main"
    choices: List[ChatCompletionChoice]
    usage: UsageInfo = Field(default_factory=UsageInfo)
    # Extra debug field – not part of standard OpenAI spec
    router_intent: Optional[str] = None


# ---------------------------------------------------------------------------
# Router result
# ---------------------------------------------------------------------------

class RouterResult(BaseModel):
    intent: str = "general"
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    entities: Dict[str, Any] = Field(default_factory=dict)

"""HTTP 잡 API 응답 스키마. 이 모듈이 러너의 공개 계약이다."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel

# 단어 하나: [텍스트, 시작(초), 끝(초)]
Word = tuple[str, float, float]

JobState = Literal["queued", "running", "done", "error"]


class Segment(BaseModel):
    start: float
    end: float
    text: str
    speaker: str | None = None
    words: list[Word] = []


class JobResult(BaseModel):
    language: str | None
    segments: list[Segment]
    meta: dict[str, Any] = {}


class JobStatusResponse(BaseModel):
    status: JobState
    result: JobResult | None = None
    error: str | None = None


class JobCreateResponse(BaseModel):
    job_id: str


class HealthResponse(BaseModel):
    device: str
    model: str
    versions: dict[str, str]

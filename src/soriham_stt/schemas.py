"""HTTP 잡 API 응답 스키마. 이 모듈이 러너의 공개 계약이다."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel

# 단어 하나: [텍스트, 시작(초), 끝(초)]
Word = tuple[str, float, float]

JobState = Literal["queued", "running", "done", "error"]

# 진행 중인 단계. 화자분리는 진행률을 낼 수 없어 단계 이름만 보고한다
JobStage = Literal["transcribe", "diarize"]


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
    # 아래 둘은 선택 필드다 — 예전 호출자는 무시해도 동작이 같다
    stage: JobStage | None = None
    progress: float | None = None  # 0~1, 알 수 없으면 None


class JobCreateResponse(BaseModel):
    job_id: str


class HealthResponse(BaseModel):
    device: str
    model: str
    versions: dict[str, str]

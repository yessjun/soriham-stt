from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from soriham_stt.backends.base import RawSegment, RawTranscript


class FakeBackend:
    """모델 없이 고정 결과를 돌려주는 테스트용 백엔드."""

    name = "fake"
    device = "cpu"

    def __init__(self, fail: bool = False, progress: tuple[float, ...] = ()) -> None:
        self.fail = fail
        self.calls: list[Path] = []
        self.progress = progress

    def transcribe(
        self,
        audio_path: Path,
        *,
        model: str,
        language: str | None,
        on_progress: Callable[[float], None] | None = None,
    ) -> RawTranscript:
        self.calls.append(audio_path)
        if on_progress is not None:
            for ratio in self.progress:
                on_progress(ratio)
        if self.fail:
            raise RuntimeError("가짜 백엔드 실패")
        return RawTranscript(
            language=language or "ko",
            segments=[
                RawSegment(
                    start=0.0,
                    end=1.2,
                    text="안녕하세요",
                    words=[("안녕하세요", 0.0, 1.2)],
                ),
                RawSegment(
                    start=1.5,
                    end=2.0,
                    text="반갑습니다",
                    words=[("반갑습니다", 1.5, 2.0)],
                ),
            ],
        )

    def versions(self) -> dict[str, str]:
        return {"fake": "0"}

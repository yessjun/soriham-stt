"""Apple Silicon용 mlx-whisper 백엔드. 이 모듈은 선택된 경우에만 import된다."""

from __future__ import annotations

from importlib.metadata import version
from pathlib import Path

from soriham_stt.backends.base import RawSegment, RawTranscript

# whisper 모델 별칭 → mlx-community 변환 모델 레포 (레포마다 이름 규칙이 달라 표로 고정)
_MLX_REPOS = {
    "tiny": "mlx-community/whisper-tiny",
    "base": "mlx-community/whisper-base-mlx",
    "small": "mlx-community/whisper-small-mlx",
    "medium": "mlx-community/whisper-medium-mlx",
    "large-v3": "mlx-community/whisper-large-v3-mlx",
    "large-v3-turbo": "mlx-community/whisper-large-v3-turbo",
}


def resolve_repo(model: str) -> str:
    if "/" in model:  # HF 레포를 직접 지정한 경우
        return model
    try:
        return _MLX_REPOS[model]
    except KeyError:
        raise ValueError(f"mlx 백엔드가 모르는 모델입니다: {model}") from None


class MlxWhisperBackend:
    name = "mlx-whisper"
    device = "mlx"

    def transcribe(self, audio_path: Path, *, model: str, language: str | None) -> RawTranscript:
        import mlx_whisper  # 모델 로드는 mlx_whisper가 레포 단위로 내부 캐시

        result = mlx_whisper.transcribe(
            str(audio_path),
            path_or_hf_repo=resolve_repo(model),
            language=language,
            word_timestamps=True,
        )
        segments = [
            RawSegment(
                start=float(seg["start"]),
                end=float(seg["end"]),
                text=str(seg["text"]).strip(),
                words=[
                    (str(w["word"]).strip(), float(w["start"]), float(w["end"]))
                    for w in seg.get("words", [])
                ],
            )
            for seg in result["segments"]
        ]
        return RawTranscript(language=result.get("language"), segments=segments)

    def versions(self) -> dict[str, str]:
        return {"mlx-whisper": version("mlx-whisper")}

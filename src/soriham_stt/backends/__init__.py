"""백엔드 자동 선택. 무거운 모듈은 실제 선택된 경우에만 import한다."""

from __future__ import annotations

import importlib.util
import platform

from soriham_stt.backends.base import TranscribeBackend
from soriham_stt.config import Settings


def detect_device(forced: str | None = None) -> str:
    """실행 환경에서 백엔드 종류(mlx | cuda | cpu)를 고른다."""
    if forced:
        return forced
    if (
        platform.system() == "Darwin"
        and platform.machine() == "arm64"
        and importlib.util.find_spec("mlx_whisper") is not None
    ):
        return "mlx"
    if importlib.util.find_spec("torch") is not None:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    return "cpu"


def select_backend(settings: Settings) -> TranscribeBackend:
    device = detect_device(settings.device)
    try:
        if device == "mlx":
            from soriham_stt.backends.mlx import MlxWhisperBackend

            return MlxWhisperBackend()
        from soriham_stt.backends.fwhisper import FasterWhisperBackend

        return FasterWhisperBackend(device=device)
    except ModuleNotFoundError as exc:
        extra = "mlx" if device == "mlx" else "cpu"
        raise RuntimeError(
            f"{device} 백엔드 의존성이 없습니다. `uv sync --extra {extra}`로 설치하세요"
        ) from exc

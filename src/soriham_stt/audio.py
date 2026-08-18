"""오디오 선변환. 전사와 화자분리가 같은 파형을 보게 한다."""

from __future__ import annotations

import subprocess
from pathlib import Path

SAMPLE_RATE = 16000


class DecodeFailed(Exception):
    """ffmpeg이 오디오를 읽지 못했다."""


def decode_to_wav(src: Path, dest_dir: Path) -> Path:
    """16kHz 모노 wav로 변환해 그 경로를 돌려준다.

    라이브러리마다 디코더가 달라서(pyannote는 torchcodec을 쓴다) 같은 파일을 하나는
    읽고 하나는 못 읽는 일이 생긴다. 실제로 정상적인 mp3에서 화자분리만 죽었다.
    ffmpeg으로 한 번 펴서 넘기면 그 차이가 사라지고, whisper 쪽 재디코딩도 없어진다.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / "audio.16k.wav"
    try:
        result = subprocess.run(
            [
                "ffmpeg",
                "-nostdin",
                "-v",
                "error",
                "-i",
                str(src),
                "-ar",
                str(SAMPLE_RATE),
                "-ac",
                "1",
                "-c:a",
                "pcm_s16le",
                "-y",
                str(dest),
            ],
            capture_output=True,
            text=True,
        )
    except OSError as exc:  # ffmpeg 미설치 등 — 호출자가 원본으로 진행할 수 있게 한다
        raise DecodeFailed(f"ffmpeg 실행 실패: {exc}") from exc
    if result.returncode != 0 or not dest.is_file():
        raise DecodeFailed(result.stderr.strip()[:400] or "ffmpeg 변환 실패")
    return dest

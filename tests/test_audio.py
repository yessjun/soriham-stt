from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from soriham_stt.audio import SAMPLE_RATE, DecodeFailed, decode_to_wav

ffmpeg_required = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg 없음")


@ffmpeg_required
def test_16k_모노_wav로_변환한다(tmp_path: Path):
    src = tmp_path / "src.wav"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=1",
            "-ar",
            "44100",
            "-ac",
            "2",
            "-y",
            str(src),
        ],
        check=True,
    )

    dest = decode_to_wav(src, tmp_path / "out")

    assert dest.is_file()
    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "stream=sample_rate,channels",
            "-of",
            "csv=p=0",
            str(dest),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert probe.stdout.strip().startswith(str(SAMPLE_RATE))
    assert probe.stdout.strip().endswith("1")


@ffmpeg_required
def test_오디오가_아니면_DecodeFailed(tmp_path: Path):
    src = tmp_path / "garbage.wav"
    src.write_bytes(b"not audio at all")

    with pytest.raises(DecodeFailed):
        decode_to_wav(src, tmp_path / "out")


def test_ffmpeg이_없으면_DecodeFailed(tmp_path: Path, monkeypatch):
    """미설치 호스트에서 잡 전체가 죽지 않고 원본으로 진행할 수 있어야 한다."""

    def missing(*a, **k):
        raise FileNotFoundError(2, "No such file or directory", "ffmpeg")

    monkeypatch.setattr(subprocess, "run", missing)
    src = tmp_path / "a.wav"
    src.write_bytes(b"x")

    with pytest.raises(DecodeFailed):
        decode_to_wav(src, tmp_path / "out")

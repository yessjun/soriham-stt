from __future__ import annotations

from pathlib import Path

import pytest

from fakes import FakeBackend
from soriham_stt import pipeline
from soriham_stt.jobs import Job, JobParams
from soriham_stt.pipeline import run_job


def make_job(diarize: bool) -> Job:
    return Job(
        id="j1",
        audio_path=Path("/nonexistent/a.wav"),
        params=JobParams(model="tiny", language="ko", diarize=diarize),
    )


def test_diarizer_turns_are_merged_into_result():
    turns = [(0.0, 1.2, "SPEAKER_00"), (1.5, 2.0, "SPEAKER_01")]
    result = run_job(make_job(diarize=True), FakeBackend(), diarizer=lambda p, t: turns)
    assert result.meta["diarized"] is True
    assert [s.speaker for s in result.segments] == ["SPEAKER_00", "SPEAKER_01"]


def test_diarize_false_skips_diarizer():
    called = []

    def diarizer(path: Path, token: str | None):
        called.append(path)
        return []

    result = run_job(make_job(diarize=False), FakeBackend(), diarizer=diarizer)
    assert called == []
    assert result.meta["diarized"] is False
    assert all(s.speaker is None for s in result.segments)


def test_diarizer_none_means_graceful_skip():
    result = run_job(make_job(diarize=True), FakeBackend(), diarizer=lambda p, t: None)
    assert result.meta["diarized"] is False
    assert all(s.speaker is None for s in result.segments)


def test_토큰이_없어_건너뛴_화자분리도_사유를_남긴다():
    """화면에 흔적이 없으면 설정이 빠진 것과 화자가 한 명인 것을 구분할 수 없다."""
    result = run_job(make_job(diarize=True), FakeBackend(), diarizer=lambda p, t: None)
    assert result.meta["diarize_error"] == "HF_TOKEN 미설정"


def test_화자분리_실패는_사유를_남기고_녹취록은_살린다():
    """조용히 화자 없는 결과가 나오면 몇 시간 뒤에야 알게 된다."""

    def broken(path, token):
        raise RuntimeError("디코더가 파일을 못 읽음")

    result = run_job(make_job(diarize=True), FakeBackend(), diarizer=broken)

    assert result.meta["diarized"] is False
    assert "디코더가 파일을 못 읽음" in result.meta["diarize_error"]
    assert len(result.segments) == 2  # 전사 결과는 그대로


def test_선변환_실패해도_원본으로_전사한다(tmp_path: Path):
    """ffmpeg이 못 읽는 입력이라도 전사까지 막지는 않는다."""
    src = tmp_path / "a.wav"
    src.write_bytes(b"not audio")
    job = Job(id="j2", audio_path=src, params=JobParams(model="tiny", language="ko", diarize=False))
    backend = FakeBackend()

    result = run_job(job, backend)

    assert len(result.segments) == 2
    assert "decode_error" in result.meta
    # 변환에 실패했으면 원본 경로를 그대로 넘긴다
    assert backend.calls == [src]


def test_소음_구간은_자리표시로_남는다():
    """지워버리면 녹취록에 설명 없는 구멍이 남는다."""
    from soriham_stt.backends.base import RawSegment, RawTranscript

    class NoisyBackend(FakeBackend):
        def transcribe(self, audio_path, *, model, language, on_progress=None):
            return RawTranscript(
                language="ko",
                segments=[
                    RawSegment(start=0.0, end=2.0, text="정상 발언", compression_ratio=1.2),
                    RawSegment(start=2.0, end=4.0, text="반복 반복", compression_ratio=18.6),
                    RawSegment(start=4.0, end=6.0, text="반복 반복", compression_ratio=18.6),
                    RawSegment(start=8.0, end=9.0, text="다시 발언", compression_ratio=1.3),
                ],
            )

    result = run_job(make_job(diarize=False), NoisyBackend())

    kinds = [(s.kind, s.text) for s in result.segments]
    assert kinds == [
        ("speech", "정상 발언"),
        ("noise", ""),
        ("speech", "다시 발언"),
    ]
    # 맞닿은 소음 구간은 하나로 합쳐진다
    noise = next(s for s in result.segments if s.kind == "noise")
    assert (noise.start, noise.end) == (2.0, 6.0)
    assert result.meta["noise_sec"] == 4.0


def test_받아적지_못한_구간을_잘라_다시_돌린다(tmp_path: Path, monkeypatch):
    """통째로 돌릴 때 오염됐던 구간이 떼어내면 읽히는 경우가 있다."""
    from soriham_stt.backends.base import RawSegment, RawTranscript

    src = tmp_path / "a.wav"
    src.write_bytes(b"x")
    # 실제 오디오가 아니므로 자르기는 흉내만 낸다 — 여기서 볼 것은 파이프라인 흐름이다
    monkeypatch.setattr(pipeline, "cut_wav", lambda src, start, dur, dest: dest)

    class RetryBackend(FakeBackend):
        def __init__(self):
            super().__init__()
            self.retries: list[Path] = []

        def transcribe(self, audio_path, *, model, language, on_progress=None):
            if "retry-" in audio_path.name:
                self.retries.append(audio_path)
                # 잘라서 돌리니 제대로 읽혔다 (구간 기준 시각으로 나온다)
                return RawTranscript(
                    language="ko",
                    segments=[
                        RawSegment(start=0.5, end=2.0, text="되살아난 발언", compression_ratio=1.2)
                    ],
                )
            return RawTranscript(
                language="ko",
                segments=[
                    RawSegment(start=0.0, end=2.0, text="정상 발언", compression_ratio=1.2),
                    RawSegment(
                        start=10.0, end=14.0, text="반복 반복 반복 반복", compression_ratio=18.6
                    ),
                ],
            )

    backend = RetryBackend()
    job = Job(id="j", audio_path=src, params=JobParams(model="tiny", language="ko", diarize=False))
    result = run_job(job, backend)

    assert len(backend.retries) == 1
    kinds = [(s.kind, s.text, round(s.start, 1)) for s in result.segments]
    # 재전사 결과는 원본 시각으로 옮겨 붙는다 (10.0 + 0.5)
    assert kinds == [("speech", "정상 발언", 0.0), ("speech", "되살아난 발언", 10.5)]
    assert result.meta["recovered_segments"] == 1
    assert result.meta["noise_sec"] == 0.0


def test_재전사도_실패하면_구간_표시로_남는다(tmp_path: Path, monkeypatch):
    from soriham_stt.backends.base import RawSegment, RawTranscript

    src = tmp_path / "a.wav"
    src.write_bytes(b"x")
    # 실제 오디오가 아니므로 자르기는 흉내만 낸다 — 여기서 볼 것은 파이프라인 흐름이다
    monkeypatch.setattr(pipeline, "cut_wav", lambda src, start, dur, dest: dest)

    class StubbornBackend(FakeBackend):
        def transcribe(self, audio_path, *, model, language, on_progress=None):
            if "retry-" in audio_path.name:
                return RawTranscript(
                    language="ko",
                    segments=[
                        RawSegment(
                            start=0.0, end=4.0, text="반복 반복 반복 반복", compression_ratio=18.6
                        )
                    ],
                )
            return RawTranscript(
                language="ko",
                segments=[
                    RawSegment(start=0.0, end=2.0, text="정상 발언", compression_ratio=1.2),
                    RawSegment(
                        start=10.0, end=14.0, text="반복 반복 반복 반복", compression_ratio=18.6
                    ),
                ],
            )

    result = run_job(
        Job(id="j", audio_path=src, params=JobParams(model="tiny", language="ko", diarize=False)),
        StubbornBackend(),
    )

    assert [(s.kind, s.text) for s in result.segments] == [("speech", "정상 발언"), ("noise", "")]
    assert result.meta["recovered_segments"] == 0


def test_화자분리_입력은_파형으로_올린다(tmp_path: Path) -> None:
    """경로를 넘기면 pyannote가 화자 구간마다 파일을 다시 디코딩한다.

    175분 실측에서 그 크롭이 CPU 한 코어를 잡고 전체 배속이 100배에서 9배로 떨어졌다.
    """
    pytest.importorskip("torch")
    import wave

    from soriham_stt.diarize import _as_input

    src = tmp_path / "audio.16k.wav"
    with wave.open(str(src), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"\x00\x10" * 16000)

    payload = _as_input(src)
    assert isinstance(payload, dict)
    assert payload["sample_rate"] == 16000
    assert payload["waveform"].shape == (1, 16000)


def test_못_읽는_오디오는_경로_그대로_넘긴다(tmp_path: Path) -> None:
    """선변환이 실패해 원본이 온 경우다. 화자분리까지 막을 이유는 없다."""
    from soriham_stt.diarize import _as_input

    src = tmp_path / "audio.mp3"
    src.write_bytes(b"not a wav")
    assert _as_input(src) == str(src)

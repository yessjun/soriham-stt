from __future__ import annotations

from pathlib import Path

from fakes import FakeBackend
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

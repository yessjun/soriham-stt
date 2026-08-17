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

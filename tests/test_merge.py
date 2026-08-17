from __future__ import annotations

from soriham_stt.backends.base import RawSegment, RawTranscript
from soriham_stt.merge import merge_transcript


def raw(*segments: RawSegment) -> RawTranscript:
    return RawTranscript(language="ko", segments=list(segments))


def words_of(text: str, start: float, step: float = 1.0) -> list[tuple[str, float, float]]:
    out = []
    t = start
    for w in text.split():
        out.append((w, t, t + step))
        t += step
    return out


def test_no_turns_keeps_original_segments():
    transcript = raw(RawSegment(0.0, 2.0, "안녕하세요 여러분", words_of("안녕하세요 여러분", 0.0)))
    segments = merge_transcript(transcript, [])
    assert len(segments) == 1
    assert segments[0].speaker is None
    assert segments[0].text == "안녕하세요 여러분"


def test_speaker_change_splits_segment():
    text = "네 안녕하세요 반갑습니다 시작하죠"
    transcript = raw(RawSegment(0.0, 4.0, text, words_of(text, 0.0)))
    turns = [(0.0, 2.0, "SPEAKER_00"), (2.0, 4.0, "SPEAKER_01")]
    segments = merge_transcript(transcript, turns)
    assert [(s.text, s.speaker) for s in segments] == [
        ("네 안녕하세요", "SPEAKER_00"),
        ("반갑습니다 시작하죠", "SPEAKER_01"),
    ]
    assert segments[0].start == 0.0
    assert segments[0].end == 2.0
    assert segments[1].start == 2.0


def test_original_segment_boundary_is_kept():
    transcript = raw(
        RawSegment(0.0, 2.0, "첫 문장", words_of("첫 문장", 0.0)),
        RawSegment(2.0, 4.0, "둘째 문장", words_of("둘째 문장", 2.0)),
    )
    turns = [(0.0, 4.0, "SPEAKER_00")]
    segments = merge_transcript(transcript, turns)
    assert [(s.text, s.speaker) for s in segments] == [
        ("첫 문장", "SPEAKER_00"),
        ("둘째 문장", "SPEAKER_00"),
    ]


def test_word_without_overlap_snaps_to_nearest_turn():
    transcript = raw(RawSegment(0.0, 3.0, "하나 둘", [("하나", 0.0, 1.0), ("둘", 2.2, 2.6)]))
    # 둘째 단어는 어떤 구간과도 안 겹치지만 0.2초 거리라 스냅된다
    turns = [(0.0, 2.0, "SPEAKER_00")]
    segments = merge_transcript(transcript, turns)
    assert len(segments) == 1
    assert segments[0].speaker == "SPEAKER_00"


def test_word_far_from_turns_gets_no_speaker():
    transcript = raw(RawSegment(0.0, 10.0, "하나 멀다", [("하나", 0.0, 1.0), ("멀다", 8.0, 9.0)]))
    turns = [(0.0, 1.0, "SPEAKER_00")]
    segments = merge_transcript(transcript, turns)
    assert [(s.text, s.speaker) for s in segments] == [
        ("하나", "SPEAKER_00"),
        ("멀다", None),
    ]


def test_short_island_absorbed_by_surrounding_speaker():
    words = [
        ("하나", 0.0, 1.0),
        ("둘", 1.0, 2.0),
        ("셋", 2.0, 2.3),
        ("넷", 3.0, 4.0),
        ("다섯", 4.0, 5.0),
    ]
    transcript = raw(RawSegment(0.0, 5.0, "하나 둘 셋 넷 다섯", words))
    # 셋(0.3초)만 다른 화자로 배정되는 상황 — 양옆이 같으므로 흡수
    turns = [
        (0.0, 2.0, "SPEAKER_00"),
        (2.0, 2.3, "SPEAKER_01"),
        (3.0, 5.0, "SPEAKER_00"),
    ]
    segments = merge_transcript(transcript, turns)
    assert len(segments) == 1
    assert segments[0].speaker == "SPEAKER_00"


def test_long_island_is_kept():
    words = words_of("하나 둘 셋 넷 다섯 여섯", 0.0)
    transcript = raw(RawSegment(0.0, 6.0, "하나 둘 셋 넷 다섯 여섯", words))
    turns = [
        (0.0, 2.0, "SPEAKER_00"),
        (2.0, 5.0, "SPEAKER_01"),
        (5.0, 6.0, "SPEAKER_00"),
    ]
    segments = merge_transcript(transcript, turns)
    assert [s.speaker for s in segments] == ["SPEAKER_00", "SPEAKER_01", "SPEAKER_00"]


def test_segment_without_words_assigned_as_whole():
    transcript = raw(RawSegment(0.0, 2.0, "단어 정보 없는 문장", []))
    turns = [(0.0, 2.0, "SPEAKER_01")]
    segments = merge_transcript(transcript, turns)
    assert len(segments) == 1
    assert segments[0].speaker == "SPEAKER_01"
    assert segments[0].text == "단어 정보 없는 문장"
    assert segments[0].words == []

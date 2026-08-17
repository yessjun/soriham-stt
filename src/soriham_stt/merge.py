"""단어 타임스탬프와 화자 구간의 병합. 순수 함수만 둔다 — 무거운 의존성 금지.

입력은 whisper 원출력(세그먼트 + 단어)과 화자분리 구간 목록이고, 출력은 화자가
배정되고 화자 전환점에서 분할된 세그먼트 목록이다.
"""

from __future__ import annotations

from dataclasses import dataclass

from soriham_stt.backends.base import RawTranscript
from soriham_stt.schemas import Segment

# 단어가 어떤 화자 구간과도 겹치지 않을 때 가장 가까운 구간을 인정할 허용 오차(초)
SNAP_TOLERANCE = 1.0
# 화자 튐 흡수 임계: 양옆이 같은 화자일 때 이 개수·길이 이하의 run은 주변에 흡수
ISLAND_MAX_WORDS = 2
ISLAND_MAX_SEC = 0.5

Turn = tuple[float, float, str]  # (시작, 끝, 화자 라벨)


@dataclass
class _TimedWord:
    text: str
    start: float
    end: float
    segment_index: int
    speaker: str | None = None


def merge_transcript(raw: RawTranscript, turns: list[Turn]) -> list[Segment]:
    """화자 구간을 단어에 배정하고 화자 전환점 기준으로 세그먼트를 재구성한다.

    whisper 원세그먼트 경계는 분할점으로 유지한다(문장 단위 보존). 단어가 없는
    세그먼트는 세그먼트 전체를 단어 하나처럼 취급해 배정한다.
    """
    if not turns:
        return [
            Segment(start=s.start, end=s.end, text=s.text, speaker=None, words=s.words)
            for s in raw.segments
        ]

    sorted_turns = sorted(turns)
    words = _collect_words(raw)
    for word in words:
        word.speaker = _assign_speaker(word, sorted_turns)
    _absorb_islands(words)
    return _rebuild_segments(words, raw)


def _collect_words(raw: RawTranscript) -> list[_TimedWord]:
    words: list[_TimedWord] = []
    for i, seg in enumerate(raw.segments):
        if seg.words:
            words.extend(
                _TimedWord(text=w, start=start, end=end, segment_index=i)
                for w, start, end in seg.words
            )
        else:
            # 단어 타임스탬프가 없는 세그먼트는 통째로 하나의 단위로 배정
            words.append(_TimedWord(text=seg.text, start=seg.start, end=seg.end, segment_index=i))
    return words


def _assign_speaker(word: _TimedWord, turns: list[Turn]) -> str | None:
    best_overlap = 0.0
    best_speaker: str | None = None
    nearest_gap = float("inf")
    nearest_speaker: str | None = None
    for t_start, t_end, speaker in turns:
        overlap = min(word.end, t_end) - max(word.start, t_start)
        if overlap > best_overlap:
            best_overlap = overlap
            best_speaker = speaker
        gap = max(t_start - word.end, word.start - t_end)
        if gap < nearest_gap:
            nearest_gap = gap
            nearest_speaker = speaker
    if best_speaker is not None:
        return best_speaker
    if nearest_gap <= SNAP_TOLERANCE:
        return nearest_speaker
    return None


def _absorb_islands(words: list[_TimedWord]) -> None:
    """양옆이 같은 화자인 짧은 run(오분할로 흔한 튐)을 주변 화자로 합친다."""
    runs = _speaker_runs(words)
    for k in range(1, len(runs) - 1):
        prev_speaker = words[runs[k - 1][0]].speaker
        next_speaker = words[runs[k + 1][0]].speaker
        start, end = runs[k]
        run_words = words[start:end]
        duration = run_words[-1].end - run_words[0].start
        if (
            prev_speaker is not None
            and prev_speaker == next_speaker
            and run_words[0].speaker != prev_speaker
            and len(run_words) <= ISLAND_MAX_WORDS
            and duration <= ISLAND_MAX_SEC
        ):
            for word in run_words:
                word.speaker = prev_speaker


def _speaker_runs(words: list[_TimedWord]) -> list[tuple[int, int]]:
    """연속 동일 화자 구간을 [시작, 끝) 인덱스 쌍으로 돌려준다."""
    runs: list[tuple[int, int]] = []
    start = 0
    for i in range(1, len(words) + 1):
        if i == len(words) or words[i].speaker != words[start].speaker:
            runs.append((start, i))
            start = i
    return runs


def _rebuild_segments(words: list[_TimedWord], raw: RawTranscript) -> list[Segment]:
    segments: list[Segment] = []
    group: list[_TimedWord] = []
    for word in words:
        if group and (
            word.speaker != group[0].speaker or word.segment_index != group[0].segment_index
        ):
            segments.append(_to_segment(group, raw))
            group = []
        group.append(word)
    if group:
        segments.append(_to_segment(group, raw))
    return segments


def _to_segment(group: list[_TimedWord], raw: RawTranscript) -> Segment:
    source = raw.segments[group[0].segment_index]
    # 단어 타임스탬프가 없어 세그먼트 통째로 배정된 경우 원문 그대로 유지
    whole_segment = len(group) == 1 and group[0].text == source.text and not source.words
    return Segment(
        start=group[0].start,
        end=group[-1].end,
        text=source.text if whole_segment else " ".join(w.text for w in group),
        speaker=group[0].speaker,
        words=[] if whole_segment else [(w.text, w.start, w.end) for w in group],
    )

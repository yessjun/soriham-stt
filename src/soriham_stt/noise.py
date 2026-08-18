"""환청 구간 판정. whisper는 무음·잡음에도 그럴듯한 문장을 지어낸다.

키보드 소리가 "감사합니다"로, 바람소리가 "올라"로 받아적히는 식이다. 신뢰도 지표로는
안 걸린다 — 환청 세그먼트의 avg_logprob이 오히려 더 높게 나온다. 실제로 갈리는 것은
**압축률**이다(같은 말을 되풀이하면 치솟는다)와 **연속 반복**이다.

실측(175분 회의 녹음, 잡음 구간 2분):

| 텍스트 | compression_ratio |
|---|---|
| 치킨인데? / 먹고 한번 더 하셔야 할 것 같아 | 1.23 |
| 오늘 저녁은 장갑을 입고 왔어요 | 2.32 |
| 이곳은 한국의 한 식당입니다 (7회) | 5.30 |
| 곱창 곱창 (15회) | 18.62 |
| ㅋㅋㅋㅋㅋ... | 89.20 |
"""

from __future__ import annotations

import re

from soriham_stt.backends.base import RawSegment

# 창 단위 압축률. 실측에서 정상 발화 창은 2.3 이하, 환청 창은 5.3 이상으로 사이가
# 넓다. 이 신호는 창 전체를 버리므로 그 간격의 위쪽에 붙여 보수적으로 잡는다
COMPRESSION_LIMIT = 4.0
# 같은 말이 이만큼 연달아 나오면 사람이 한 말로 보지 않는다
REPEAT_LIMIT = 3
# 한 세그먼트 안에서 같은 낱말이 이만큼 되풀이되면 그 자체로 환청이다
WORD_REPEAT_LIMIT = 3.0
# 한 글자만 늘어놓은 것("ㅋㅋㅋㅋ...")을 잡는다
CHAR_REPEAT_LIMIT = 8.0

# 유튜브 자막으로 학습한 흔적. 무음 구간에서 통째로 튀어나오고, 반복도 아니고 문장도
# 멀쩡해서 다른 규칙에 걸리지 않는다. 낱말이 아니라 문구 전체로 맞춘다 — "영상"이나
# "자막"은 실제 회의에서도 쓰는 말이라 낱말로 걸면 진짜 발언을 지운다
BOILERPLATE = [
    re.compile(p)
    for p in (
        r"구독(과|과 함께)?\s*(좋아요|알림)",
        r"시청\s*해\s*주(셔서|서서)?\s*감사",
        r"끝까지\s*(봐|시청)",
        r"다음\s*(영상|시간)에\s*(만나|뵙)",
        r"자막\s*(제공|by|:)",
        r"영상\s*편집\s*및\s*자막",
    )
]

_STRIP = re.compile(r"[\s.,!?…·~]+")


def _key(text: str) -> str:
    return _STRIP.sub("", text)


def classify(segments: list[RawSegment]) -> list[bool]:
    """세그먼트별로 환청(=소음 구간)인지 판정한다.

    whisper의 `compression_ratio`는 세그먼트가 아니라 **30초 창** 단위 값이다. 그걸
    그대로 쓰면 창 하나가 환청으로 오염됐을 때 그 안의 멀쩡한 발화까지 버린다. 그래서
    창 지표는 의심 신호로만 쓰고, 실제 판정은 세그먼트 자신을 보고 한다.

    판정은 네 갈래다: 한 세그먼트 안의 되풀이, 여러 세그먼트에 걸친 반복, 유튜브 자막
    상투구, 그리고 창 단위 압축률이다. 상투구는 문장이 멀쩡해서 반복 규칙에 안 걸리고,
    창 압축률은 반대로 한 번만 나온 환청을 잡아준다.
    """
    noise = [
        _is_degenerate(seg.text)
        or _is_boilerplate(seg.text)
        or (seg.compression_ratio is not None and seg.compression_ratio >= COMPRESSION_LIMIT)
        for seg in segments
    ]
    # 같은 문구가 연달아 REPEAT_LIMIT회 이상이면 그 묶음을 통째로 소음으로 본다
    run_start = 0
    for i in range(1, len(segments) + 1):
        same = (
            i < len(segments)
            and _key(segments[i].text)
            and _key(segments[i].text) == _key(segments[run_start].text)
        )
        if same:
            continue
        if i - run_start >= REPEAT_LIMIT and _key(segments[run_start].text):
            for j in range(run_start, i):
                noise[j] = True
        run_start = i
    return noise


def _is_boilerplate(text: str) -> bool:
    return any(p.search(text) for p in BOILERPLATE)


def _is_degenerate(text: str) -> bool:
    words = text.split()
    if len(words) >= 3 and len(words) / len(set(words)) >= WORD_REPEAT_LIMIT:
        return True
    chars = _key(text)
    return len(chars) >= 10 and len(chars) / len(set(chars)) >= CHAR_REPEAT_LIMIT

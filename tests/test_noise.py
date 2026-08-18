from __future__ import annotations

from soriham_stt.backends.base import RawSegment
from soriham_stt.noise import classify


def seg(text: str, ratio: float | None = 1.2, start: float = 0.0) -> RawSegment:
    return RawSegment(start=start, end=start + 1, text=text, compression_ratio=ratio)


def test_압축률이_높은_창은_소음으로_본다():
    """한 번만 나온 환청은 반복 규칙에 안 걸려서 이 신호가 필요하다.

    실측에서 정상 발화 창은 2.3 이하, 환청 창은 5.3 이상이었다.
    """
    segments = [seg("정상적인 회의 발언입니다", 1.23), seg("그럴듯하지만 실제로는 없던 문장", 5.30)]

    assert classify(segments) == [False, True]


def test_애매한_압축률은_남긴다():
    """창 규칙은 창 전체를 버리므로 실측 간격의 위쪽에 붙인다."""
    segments = [seg("네 그렇죠 네 그렇죠 맞습니다", 3.5)]

    assert classify(segments) == [False]


def test_같은_말이_연달아_반복되면_압축률과_무관하게_소음이다():
    segments = [seg("반복 반복", 1.0, i) for i in range(4)]

    assert classify(segments) == [True] * 4


def test_두_번_반복은_남긴다():
    """맞장구나 되묻기는 실제로 두 번 나올 수 있다."""
    segments = [seg("네, 맞습니다", 1.0, 0), seg("네, 맞습니다", 1.0, 1), seg("다음으로", 1.0, 2)]

    assert classify(segments) == [False, False, False]


def test_구두점_차이는_같은_말로_본다():
    segments = [seg("같은말", 1.0, 0), seg("같은말.", 1.0, 1), seg("같은말!", 1.0, 2)]

    assert classify(segments) == [True] * 3


def test_지표가_없으면_압축률로는_판정하지_않는다():
    """지표를 안 주는 백엔드에서도 반복 규칙은 계속 동작해야 한다."""
    segments = [seg("무언가", None, 0), seg("무언가", None, 1), seg("무언가", None, 2)]

    assert classify(segments) == [True] * 3


def test_빈_텍스트_연속은_반복으로_치지_않는다():
    segments = [seg("", 1.0, i) for i in range(5)]

    assert classify(segments) == [False] * 5


def test_유튜브_자막_상투구는_소음이다():
    """반복도 아니고 문장도 멀쩡해서 다른 규칙에 안 걸린다."""
    segments = [
        seg("영상편집 및 자막이 도움이 되셨다면 구독과 좋아요 부탁드립니다.", 1.5),
        seg("시청해주셔서 감사합니다", 1.4),
    ]

    assert classify(segments) == [True, True]


def test_영상이나_자막이라는_낱말만으로는_지우지_않는다():
    """실제 회의에서도 쓰는 말이다. 낱말이 아니라 문구 전체로 맞춰야 한다."""
    segments = [
        seg("우리가 영상을 탑재해서 링크로 올리면 될 것 같아요", 1.2),
        seg("자막 다는 작업은 누가 하나요", 1.2),
        seg("좋아요 그러면 그렇게 하시죠", 1.2),
    ]

    assert classify(segments) == [False, False, False]


def test_한_세그먼트_안의_낱말_되풀이도_잡는다():
    segments = [seg("같은말 같은말 같은말 같은말 같은말 같은말", 1.0)]

    assert classify(segments) == [True]


def test_한_글자_늘어놓기를_잡는다():
    segments = [seg("ㅋ" * 40, 1.0)]

    assert classify(segments) == [True]

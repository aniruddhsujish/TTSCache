import pytest
from tts_cache.keys import VoiceProfile
from tts_cache.storage import FileTier, MemoryTier, TieredStorage
from tts_cache.strategies.segment import SegmentStrategy
from tts_cache.tts.fake import FakeTTS
from tts_cache.counter import RequestCounter
from tts_cache.coalescing import Coalescer

PROFILE = VoiceProfile(language="en", voice="v1", model="m1", output_format="pcm_16000")


def make_strategy(tmp_path):
    tts = FakeTTS(delay=0)
    storage = TieredStorage(MemoryTier(), FileTier(root=str(tmp_path)))
    counter = RequestCounter(secret=b"test", threshold=1)
    coalescer = Coalescer()
    return SegmentStrategy(tts, storage, counter, coalescer), tts


@pytest.mark.asyncio
async def test_second_request_is_a_cache_hit(tmp_path):
    strategy, tts = make_strategy(tmp_path)

    first = await strategy.get_audio("Your order has shipped.", PROFILE, "user1")
    assert tts.calls == 1

    second = await strategy.get_audio("Your order has shipped.", PROFILE, "user2")
    assert tts.calls == 1
    assert second == first


@pytest.mark.asyncio
async def test_only_new_sentences_are_synthesized_in_order(tmp_path):
    strategy, tts = make_strategy(tmp_path)

    await strategy.get_audio(
        "Your order has shipped. Anything else you would like assistance with?",
        PROFILE,
        "user1",
    )
    assert tts.calls == 2

    results = await strategy.get_audio(
        "Your order has shipped. It arrives on Monday. Anything else you would like assistance with?",
        PROFILE,
        "user2",
    )
    assert tts.calls == 3
    assert [r.timestamps[0].word for r in results] == ["Your", "It", "Anything"]


@pytest.mark.asyncio
async def test_cached_only_after_five_distinct_user(tmp_path):
    tts = FakeTTS(delay=0)
    storage = TieredStorage(MemoryTier(), FileTier(root=str(tmp_path)))
    counter = RequestCounter(secret=b"test", threshold=5)
    coalescer = Coalescer()
    strategy = SegmentStrategy(tts, storage, counter, coalescer)

    for i in range(5):
        await strategy.get_audio("Your order has been shipped.", PROFILE, f"user{i}")
    assert tts.calls == 5

    await strategy.get_audio("Your order has been shipped.", PROFILE, f"user5")
    assert tts.calls == 5

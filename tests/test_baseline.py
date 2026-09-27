import pytest

from tts_cache.coalescing import Coalescer
from tts_cache.counter import RequestCounter
from tts_cache.keys import VoiceProfile
from tts_cache.metrics import Metrics
from tts_cache.storage import FileTier, MemoryTier, TieredStorage
from tts_cache.strategies.baseline import BaselineStrategy
from tts_cache.strategies.segment import SegmentStrategy
from tts_cache.tts.fake import FakeTTS

PROFILE = VoiceProfile(language="en", voice="v1", model="m1", output_format="pcm_16000")

FIRST = "Your order has been shipped. It arrives by Monday."
SECOND = "Your order has been shipped. It arrives by Tuesday."


def build(strategy, tmp_path):
    return strategy(
        FakeTTS(delay=0),
        TieredStorage(MemoryTier(), FileTier(root=str(tmp_path))),
        RequestCounter(secret=b"test", threshold=1),
        Coalescer(),
        Metrics(),
    )


@pytest.mark.asyncio
async def test_baseline_saves_nothing_when_one_sentence_of_the_message_changes(
    tmp_path,
):
    strategy = build(BaselineStrategy, tmp_path)
    await strategy.get_audio(FIRST, PROFILE, "user1")
    await strategy.get_audio(SECOND, PROFILE, "user2")
    assert strategy.metrics.savings_ratio() == 0.0


@pytest.mark.asyncio
async def test_segment_reuses_repeated_sentence(tmp_path):
    strategy = build(SegmentStrategy, tmp_path)
    await strategy.get_audio(FIRST, PROFILE, "user1")
    await strategy.get_audio(SECOND, PROFILE, "user2")
    assert strategy.metrics.savings_ratio() > 0.0

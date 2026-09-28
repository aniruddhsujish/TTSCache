import pytest
from tts_cache.keys import VoiceProfile
from tts_cache.storage import FileTier, MemoryTier, TieredStorage
from tts_cache.strategies.segment import SegmentStrategy
from tts_cache.tts.fake import FakeTTS
from tts_cache.counter import RequestCounter
from tts_cache.coalescing import Coalescer
from tts_cache.tts.base import TTSBackend, TTSResult
from tts_cache.metrics import Metrics, Outcome

PROFILE = VoiceProfile(language="en", voice="v1", model="m1", output_format="pcm_16000")


def make_strategy(tmp_path):
    tts = FakeTTS(delay=0)
    storage = TieredStorage(MemoryTier(), FileTier(root=str(tmp_path)))
    counter = RequestCounter(secret=b"test", threshold=1)
    coalescer = Coalescer()
    metrics = Metrics()
    return SegmentStrategy(tts, storage, counter, coalescer, metrics), tts


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
    metrics = Metrics()
    strategy = SegmentStrategy(tts, storage, counter, coalescer, metrics)

    for i in range(5):
        await strategy.get_audio("Your order has been shipped.", PROFILE, f"user{i}")
    assert tts.calls == 5

    await strategy.get_audio("Your order has been shipped.", PROFILE, "user5")
    assert tts.calls == 5


# --------- Empty Audio blocked by QC gate ---------------#
class EmptyAudioTTS(TTSBackend):
    def __init__(self):
        self.calls = 0

    async def synthesize(self, text, profile):
        self.calls += 1
        return TTSResult(audio=b"", sample_rate=16000, timestamps=[])


@pytest.mark.asyncio
async def test_bad_audio_is_never_cached(tmp_path):
    tts = EmptyAudioTTS()
    storage = TieredStorage(MemoryTier(), FileTier(root=str(tmp_path)))
    counter = RequestCounter(secret=b"test", threshold=1)
    strategy = SegmentStrategy(tts, storage, counter, Coalescer(), Metrics())

    await strategy.get_audio("Your order has shipped.", PROFILE, "user1")
    await strategy.get_audio("Your order has shipped.", PROFILE, "user2")

    assert (
        tts.calls == 2
    )  # second request should be a miss since first entry fails QC gate


@pytest.mark.asyncio
async def test_metrics_track_savings(tmp_path):
    tts = FakeTTS(delay=0)
    storage = TieredStorage(MemoryTier(), FileTier(root=str(tmp_path)))
    metrics = Metrics()
    strategy = SegmentStrategy(
        tts, storage, RequestCounter(secret=b"test", threshold=1), Coalescer(), metrics
    )

    await strategy.get_audio("Your order has shipped.", PROFILE, "user1")
    await strategy.get_audio("Your order has shipped.", PROFILE, "user2")

    assert metrics.outcomes[Outcome.MISS_STORED] == 1
    assert metrics.outcomes[Outcome.HIT] == 1
    assert metrics.savings_ratio() == 0.5

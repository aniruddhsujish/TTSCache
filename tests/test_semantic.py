import pytest

from tts_cache.coalescing import Coalescer
from tts_cache.counter import RequestCounter
from tts_cache.keys import VoiceProfile
from tts_cache.metrics import Metrics, Outcome
from tts_cache.storage import FileTier, MemoryTier, TieredStorage
from tts_cache.strategies.semantic import SemanticTemplateStrategy
from tts_cache.tts.fake import FakeTTS

PROFILE = VoiceProfile(language="en", voice="v1", model="m1", output_format="pcm_16000")


class FakeMatcher:
    """Declares specific pairs equivalent; counts how often it's asked."""

    def __init__(self, *pairs):
        self.pairs = {frozenset(p) for p in pairs}
        self.calls = 0

    def best_match(self, query, candidates):
        self.calls += 1
        for c in candidates:
            if frozenset((query, c)) in self.pairs:
                return c, 1.0, 1.0
        return None


def build(tmp_path, matcher):
    tts, metrics = FakeTTS(delay=0), Metrics()
    strategy = SemanticTemplateStrategy(
        tts,
        TieredStorage(MemoryTier(), FileTier(root=str(tmp_path))),
        RequestCounter(secret=b"test", threshold=1),
        Coalescer(),
        metrics,
        matcher=matcher,
    )
    return strategy, tts, metrics


@pytest.mark.asyncio
async def test_paraphrase_is_served_from_the_stored_phrasing(tmp_path):
    matcher = FakeMatcher(
        ("We have received your request.", "Your request has been received.")
    )
    strategy, tts, metrics = build(tmp_path, matcher)

    await strategy.get_audio("Your request has been received.", PROFILE, "u1")
    await strategy.get_audio("We have received your request.", PROFILE, "u2")

    assert tts.calls == 1
    assert metrics.outcomes[Outcome.SEMANTIC_HIT] == 1


@pytest.mark.asyncio
async def test_semantic_template_keeps_this_requests_number(tmp_path):
    matcher = FakeMatcher(
        ("We have shipped your order {NUM}.", "Your order {NUM} has been shipped.")
    )
    strategy, tts, _ = build(tmp_path, matcher)

    await strategy.get_audio("Your order 4521 has been shipped.", PROFILE, "u1")
    [result] = await strategy.get_audio(
        "We have shipped your order 7788.", PROFILE, "u2"
    )

    assert tts.calls == 2  # only "7788" synthesized
    assert [t.word for t in result.timestamps] == [
        "Your",
        "order",
        "7788",  # mid-sentence slot: the query's full stop does not come along
        "has",
        "been",
        "shipped.",
    ]


@pytest.mark.asyncio
async def test_semantic_value_takes_the_punctuation_of_its_new_slot(tmp_path):
    matcher = FakeMatcher(
        ("Your order {NUM} has been shipped.", "We have shipped your order {NUM}.")
    )
    strategy, _, _ = build(tmp_path, matcher)

    await strategy.get_audio("We have shipped your order 4521.", PROFILE, "u1")
    [result] = await strategy.get_audio(
        "Your order 7788 has been shipped.", PROFILE, "u2"
    )

    # the slot is sentence-final in the stored wording, so the number is spoken with the full stop
    assert [t.word for t in result.timestamps][-1] == "7788."


@pytest.mark.asyncio
async def test_repeated_slot_types_never_match_semantically(tmp_path):
    # same meaning, but the two {NUM}s swap roles: positional filling would swap the numbers
    matcher = FakeMatcher(
        (
            "Your order {NUM} will arrive in {NUM} days.",
            "In {NUM} days your order {NUM} will arrive.",
        )
    )
    strategy, tts, metrics = build(tmp_path, matcher)

    await strategy.get_audio("Your order 4521 will arrive in 3 days.", PROFILE, "u1")
    [result] = await strategy.get_audio(
        "In 5 days your order 7788 will arrive.", PROFILE, "u2"
    )

    assert metrics.outcomes[Outcome.SEMANTIC_HIT] == 0
    assert tts.calls == 2  # synthesized in full, as written
    assert [t.word for t in result.timestamps][:6] == [
        "In",
        "5",
        "days",
        "your",
        "order",
        "7788",
    ]


@pytest.mark.asyncio
async def test_exact_hit_never_asks_the_matcher(tmp_path):
    matcher = FakeMatcher()
    strategy, _, _ = build(tmp_path, matcher)

    await strategy.get_audio("Thank you for calling.", PROFILE, "u1")
    calls_after_first = matcher.calls
    await strategy.get_audio("Thank you for calling.", PROFILE, "u2")

    assert matcher.calls == calls_after_first


@pytest.mark.asyncio
async def test_bare_numbers_are_never_indexed(tmp_path):
    strategy, _, _ = build(tmp_path, FakeMatcher())
    await strategy.get_audio("Your order 4521 has been shipped.", PROFILE, "u1")
    await strategy.get_audio("Your order 7788 has been shipped.", PROFILE, "u2")

    indexed = [t for texts in strategy.index.values() for t in texts]
    assert not any(t.rstrip(".").isdigit() for t in indexed)

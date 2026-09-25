from tts_cache.storage import MemoryTier
from tts_cache.tts.base import TTSResult


def make_result(label: str) -> TTSResult:
    return TTSResult(audio=label.encode(), sample_rate=16000, timestamps=[])


def test_put_then_get_returns_same_result():
    tier = MemoryTier()
    result = make_result("hello")
    tier.put("k1", result)
    assert tier.get("k1") == result


def test_get_unknown_key_returns_none():
    tier = MemoryTier()
    assert tier.get("missing") is None


def test_evicts_least_recently_used():
    tier = MemoryTier(max_items=2)
    tier.put("a", make_result("a"))
    tier.put("b", make_result("b"))
    tier.get("a")                      # "a" is now the most recently used
    tier.put("c", make_result("c"))    # over capacity → evict one

    assert tier.get("b") is None       # least recently used → evicted
    assert tier.get("a") == make_result("a")
    assert tier.get("c") == make_result("c")
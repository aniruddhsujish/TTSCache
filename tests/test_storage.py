from tts_cache.storage import MemoryTier, FileTier, TieredStorage
from tts_cache.tts.base import TTSResult, WordTimestamp


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
    tier.get("a")  # "a" is now the most recently used
    tier.put("c", make_result("c"))  # over capacity → evict one

    assert tier.get("b") is None  # least recently used → evicted
    assert tier.get("a") == make_result("a")
    assert tier.get("c") == make_result("c")


def make_storage(tmp_path):
    return TieredStorage(MemoryTier(), FileTier(root=str(tmp_path)))


def test_file_tier_round_trip_keeps_timestamps(tmp_path):
    files = FileTier(root=str(tmp_path))
    result = TTSResult(
        audio=b"abc",
        sample_rate=16000,
        timestamps=[WordTimestamp(word="नमस्ते", start=0.0, end=0.3)],
    )
    files.put("k1", result)
    assert files.get("k1") == result


def test_found_in_files_after_memory_emptied(tmp_path):
    storage = make_storage(tmp_path)
    storage.put("k1", make_result("hello"))
    storage.memory = MemoryTier()

    assert storage.get("k1") == make_result("hello")
    assert storage.memory.get("k1") is not None


class BrokenTier:
    def get(self, key):
        raise OSError("disk on fire")

    def put(self, key, value):
        raise OSError("disk on fire")


def test_storage_errors_become_misses(tmp_path):
    storage = TieredStorage(MemoryTier(), BrokenTier())
    storage.put("k1", make_result("hello"))
    assert storage.get("k1") is None

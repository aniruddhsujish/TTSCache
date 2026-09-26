from tts_cache.storage import CacheEntry, MemoryTier, FileTier, TieredStorage
from tts_cache.tts.base import TTSResult, WordTimestamp

DAY = 24 * 60 * 60


class FakeClock:
    def __init__(self, start: float = 1_000_000.0):
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def make_result(label: str) -> TTSResult:
    return TTSResult(audio=label.encode(), sample_rate=16000, timestamps=[])


def make_entry(label: str, created_at: float = 1_000_000.0) -> CacheEntry:
    return CacheEntry(result=make_result(label), created_at=created_at)


def test_put_then_get_returns_same_result():
    tier = MemoryTier(clock=FakeClock())
    entry = make_entry("hello")
    tier.put("k1", entry)
    assert tier.get("k1") == entry


def test_get_unknown_key_returns_none():
    tier = MemoryTier()
    assert tier.get("missing") is None


def test_evicts_least_recently_used():
    tier = MemoryTier(max_items=2, clock=FakeClock())
    tier.put("a", make_entry("a"))
    tier.put("b", make_entry("b"))
    tier.get("a")  # "a" is now the most recently used
    tier.put("c", make_entry("c"))  # over capacity → evict one

    assert tier.get("b") is None  # least recently used → evicted
    assert tier.get("a") == make_entry("a")
    assert tier.get("c") == make_entry("c")


def test_memory_entry_expires():
    clock = FakeClock()
    tier = MemoryTier(max_age_days=30, clock=clock)
    tier.put("k1", make_entry("hello", created_at=clock()))
    clock.advance(31 * DAY)
    assert tier.get("k1") is None


def make_storage(tmp_path, clock):
    return TieredStorage(
        MemoryTier(clock=clock), FileTier(root=str(tmp_path), clock=clock), clock=clock
    )


def test_file_tier_round_trip_keeps_timestamps(tmp_path):
    files = FileTier(root=str(tmp_path), clock=FakeClock())
    entry = CacheEntry(
        result=TTSResult(
            audio=b"abc",
            sample_rate=16000,
            timestamps=[WordTimestamp(word="नमस्ते", start=0.0, end=0.3)],
        ),
        created_at=1_000_000.0,
    )
    files.put("k1", entry)
    assert files.get("k1") == entry


def test_file_entry_expires_and_is_deleted(tmp_path):
    clock = FakeClock()
    files = FileTier(root=str(tmp_path), max_age_days=30, clock=clock)
    files.put("k1", make_entry("hello", created_at=clock()))
    clock.advance(31 * DAY)
    assert files.get("k1") is None
    assert list(tmp_path.iterdir()) == []


def test_prune_expired_removes_never_read_files(tmp_path):
    clock = FakeClock()
    files = FileTier(root=str(tmp_path), max_age_days=30, clock=clock)
    files.put("old", make_entry("old", created_at=clock()))
    clock.advance(31 * DAY)
    files.put("new", make_entry("new", created_at=clock()))

    files.prune_expired()

    assert files.get("new") is not None
    assert not (tmp_path / "old.json").exists()


def test_found_in_files_after_memory_emptied(tmp_path):
    clock = FakeClock()
    storage = make_storage(tmp_path, clock)
    storage.put("k1", make_result("hello"))
    storage.memory = MemoryTier(clock=clock)

    assert storage.get("k1") == make_result("hello")
    assert storage.memory.get("k1") is not None


def test_promotion_keeps_original_age(tmp_path):
    clock = FakeClock()
    storage = make_storage(tmp_path, clock)
    storage.put("k1", make_result("hello"))

    clock.advance(25 * DAY)
    storage.memory = MemoryTier(clock=clock)  # Restart memory on day 25
    storage.get("k1")

    clock.advance(6 * DAY)  # advance 6 more days
    assert storage.get("k1") is None  # verify that entry is expired


class BrokenTier:
    def get(self, key):
        raise OSError("disk on fire")

    def put(self, key, value):
        raise OSError("disk on fire")


def test_storage_errors_become_misses(tmp_path):
    clock = FakeClock()
    storage = TieredStorage(MemoryTier(clock=clock), BrokenTier(), clock=clock)
    storage.put("k1", make_result("hello"))
    assert storage.get("k1") is None

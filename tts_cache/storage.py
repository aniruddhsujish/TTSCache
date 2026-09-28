import json
import logging
from dataclasses import asdict
from pathlib import Path
from collections import OrderedDict
import time
from typing import Callable
from tts_cache.tts.base import TTSResult, WordTimestamp
from dataclasses import dataclass

log = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    result: TTSResult
    created_at: float


class MemoryTier:

    def __init__(
        self,
        max_items: int = 500,
        max_age_days: float = 30,
        clock: Callable[[], float] = time.time,
    ):
        self.max_items = max_items
        self.max_age_days = max_age_days
        self.clock = clock
        self.items: OrderedDict[str, CacheEntry] = OrderedDict()

    def get(self, key: str) -> CacheEntry | None:
        """Gets a cache entry from Memory tier. This is quick access for hot clips

        This tier is LRU with a max number of entries. Get refreshes the entry"""
        entry = self.items.get(key)
        if entry is None:
            return None

        if self.clock() - entry.created_at > self.max_age_days * 24 * 60 * 60:
            del self.items[key]
            return None

        self.items.move_to_end(key)
        return self.items[key]

    def put(self, key: str, entry: CacheEntry) -> None:
        """Saves an entry to memory tier, Evicts the Least recently used entry if memory is full"""
        self.items[key] = entry
        self.items.move_to_end(key)
        if len(self.items) > self.max_items:
            self.items.popitem(last=False)


class FileTier:

    def __init__(
        self,
        root: str = ".cache_store",
        max_age_days=30,
        clock: Callable[[], float] = time.time,
    ):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.max_age_days = max_age_days
        self.clock = clock

    def _paths(self, key: str) -> tuple[Path, Path]:
        """Returns the audio file path and metadata file path"""
        return self.root / f"{key}.pcm", self.root / f"{key}.json"

    def _is_expired(self, meta) -> bool:
        """Checks if the file is expired (aka. more than max_age_days old)"""
        return self.clock() - meta["created_at"] > self.max_age_days * 24 * 60 * 60

    def put(self, key: str, entry: CacheEntry) -> None:
        """Saves entry to file storage. Virtually limitless so no eviction"""
        audio_path, meta_path = self._paths(key)
        audio_path.write_bytes(entry.result.audio)
        meta = {
            "sample_rate": entry.result.sample_rate,
            "timestamps": [asdict(ts) for ts in entry.result.timestamps],
            "created_at": entry.created_at,
        }
        meta_path.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")

    def get(self, key: str) -> CacheEntry | None:
        """Fetches entry from file storage"""
        audio_path, meta_path = self._paths(key)
        if not meta_path.exists():
            return None
        meta = json.loads(meta_path.read_text(encoding="utf-8"))

        if self._is_expired(meta):
            meta_path.unlink(missing_ok=True)
            audio_path.unlink(missing_ok=True)
            return None

        timestamps = [WordTimestamp(**d) for d in meta["timestamps"]]
        return CacheEntry(
            result=TTSResult(
                audio=audio_path.read_bytes(),
                sample_rate=meta["sample_rate"],
                timestamps=timestamps,
            ),
            created_at=meta["created_at"],
        )

    def prune_expired(self) -> None:
        """Removes entries from file storage that are expired"""
        for meta_path in self.root.glob("*.json"):
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if self._is_expired(meta):
                audio_path = meta_path.with_suffix(".pcm")
                meta_path.unlink(missing_ok=True)
                audio_path.unlink(missing_ok=True)


class TieredStorage:

    def __init__(
        self,
        memory: MemoryTier,
        files: FileTier,
        clock: Callable[[], float] = time.time,
    ):
        self.memory = memory
        self.files = files
        self.clock = clock

    def get(self, key: str) -> TTSResult | None:
        """First tries to fetch from memory, if not found falls back to File Storage"""
        try:
            entry = self.memory.get(key)
            if entry is not None:
                return entry.result

            entry = self.files.get(key)
            if entry is not None:
                self.memory.put(key, entry)  # promote to the memory
                return entry.result
            return None
        except Exception:
            log.exception("storage get failed for key %s", key)
            return None

    def put(self, key: str, value: TTSResult) -> None:
        """Saves an entry to Memory and File storage"""
        try:
            entry = CacheEntry(result=value, created_at=self.clock())
            self.files.put(key, entry)
            self.memory.put(key, entry)
        except Exception:
            log.exception("storage put failed for key %s", key)

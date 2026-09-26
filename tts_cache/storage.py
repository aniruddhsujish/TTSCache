import json
import logging
from dataclasses import asdict
from pathlib import Path
from collections import OrderedDict
from tts_cache.tts.base import TTSResult, WordTimestamp

log = logging.getLogger(__name__)


class MemoryTier:

    def __init__(self, max_items: int = 500):
        self.max_items = max_items
        self.items = OrderedDict()

    def get(self, key: str) -> TTSResult | None:
        if key not in self.items:
            return None
        self.items.move_to_end(key)
        return self.items[key]

    def put(self, key: str, value: TTSResult) -> None:
        self.items[key] = value
        self.items.move_to_end(key)
        if len(self.items) > self.max_items:
            self.items.popitem(last=False)


class FileTier:

    def __init__(self, root: str = ".cache_store"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _paths(self, key: str) -> tuple[Path, Path]:
        return self.root / f"{key}.pcm", self.root / f"{key}.json"

    def put(self, key: str, value: TTSResult) -> None:
        audio_path, meta_path = self._paths(key)
        audio_path.write_bytes(value.audio)
        meta = {
            "sample_rate": value.sample_rate,
            "timestamps": [asdict(ts) for ts in value.timestamps],
        }
        meta_path.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")

    def get(self, key: str) -> TTSResult | None:
        audio_path, meta_path = self._paths(key)
        if not meta_path.exists():
            return None
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        timestamps = [WordTimestamp(**d) for d in meta["timestamps"]]
        return TTSResult(
            audio=audio_path.read_bytes(),
            sample_rate=meta["sample_rate"],
            timestamps=timestamps,
        )


class TieredStorage:

    def __init__(self, memory: MemoryTier, files: FileTier):
        self.memory = memory
        self.files = files

    def get(self, key: str) -> TTSResult | None:
        try:
            result = self.memory.get(key)
            if result is not None:
                return result

            result = self.files.get(key)
            if result is not None:
                self.memory.put(key, result)  # promote to the memory
            return result
        except Exception:
            log.exception("storage get failed for key %s", key)
            return None

    def put(self, key: str, value: TTSResult) -> None:
        try:
            self.files.put(key, value)
            self.memory.put(key, value)
        except Exception:
            log.exception("storage put failed for key %s", key)

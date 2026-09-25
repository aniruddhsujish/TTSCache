from collections import OrderedDict
from tts_cache.tts.base import TTSResult

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
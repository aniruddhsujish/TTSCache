from tts_cache.keys import VoiceProfile, build_key
from tts_cache.normalize import normalize
from tts_cache.splitter import split_sentences
from tts_cache.storage import TieredStorage
from tts_cache.tts.base import TTSBackend, TTSResult
from tts_cache.counter import RequestCounter
from tts_cache.coalescing import Coalescer


class SegmentStrategy:

    def __init__(
        self,
        tts: TTSBackend,
        storage: TieredStorage,
        counter: RequestCounter,
        coalescer: Coalescer,
    ):
        self.tts = tts
        self.storage = storage
        self.counter = counter
        self.coalescer = coalescer

    async def get_audio(
        self, text: str, profile: VoiceProfile, user_id: str
    ) -> list[TTSResult]:
        normalized = normalize(text, profile.language)
        sentences = split_sentences(normalized, profile.language)

        results = []

        for sentence in sentences:
            key = build_key(sentence, profile)
            self.counter.record(key, user_id)
            found = self.storage.get(key)
            if found is not None:
                results.append(found)
            else:
                result = await self.coalescer.run(
                    key, lambda: self.tts.synthesize(sentence, profile)
                )
                if self.counter.should_admit(key):
                    self.storage.put(key, result)
                results.append(result)

        return results

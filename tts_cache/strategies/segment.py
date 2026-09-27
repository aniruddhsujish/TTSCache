from tts_cache.keys import VoiceProfile, build_key
from tts_cache.normalize import normalize
from tts_cache.splitter import split_sentences
from tts_cache.storage import TieredStorage
from tts_cache.tts.base import TTSBackend, TTSResult
from tts_cache.counter import RequestCounter
from tts_cache.coalescing import Coalescer
from tts_cache.quality import passes_quality
from tts_cache.metrics import Metrics, Outcome


class SegmentStrategy:

    def __init__(
        self,
        tts: TTSBackend,
        storage: TieredStorage,
        counter: RequestCounter,
        coalescer: Coalescer,
        metrics: Metrics,
    ):
        self.tts = tts
        self.storage = storage
        self.counter = counter
        self.coalescer = coalescer
        self.metrics = metrics

    def _split_into_cache_units(self, normalized: str, language: str) -> list[str]:
        return split_sentences(normalized, language)

    async def get_audio(
        self, text: str, profile: VoiceProfile, user_id: str
    ) -> list[TTSResult]:
        normalized = normalize(text, profile.language)
        sentences = self._split_into_cache_units(normalized, profile.language)

        results = []

        for sentence in sentences:
            key = build_key(sentence, profile)
            self.counter.record(key, user_id)
            found = self.storage.get(key)
            if found is not None:
                self.metrics.record(Outcome.HIT, len(sentence))
                results.append(found)
            else:
                result = await self.coalescer.run(
                    key, lambda: self.tts.synthesize(sentence, profile)
                )
                if not self.counter.should_admit(key):
                    self.metrics.record(Outcome.MISS_BELOW_THRESHOLD, len(sentence))
                elif not passes_quality(result, sentence):
                    self.metrics.record(Outcome.MISS_QUALITY_REJECTED, len(sentence))
                else:
                    self.storage.put(key, result)
                    self.metrics.record(Outcome.MISS_STORED, len(sentence))
                results.append(result)

        return results

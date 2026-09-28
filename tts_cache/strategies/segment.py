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
        """For segment caching, splits into list of sentences"""

        return split_sentences(normalized, language)

    def _on_stored(self, text: str, profile: VoiceProfile) -> None:
        """Called after a unit is stored. For a subclass to access"""

    async def _get_or_synthesize(
        self, text: str, profile: VoiceProfile, user_id: str
    ) -> TTSResult:
        """Serve from cache if available, else synthesize the audio"""
        key = build_key(text, profile)
        self.counter.record(key, user_id)
        found = self.storage.get(key)
        if found is not None:
            self.metrics.record(Outcome.HIT, len(text))
            return found
        else:
            result = await self.coalescer.run(
                key, lambda: self.tts.synthesize(text, profile)
            )
            if not self.counter.should_admit(key):
                self.metrics.record(Outcome.MISS_BELOW_THRESHOLD, len(text))
            elif not passes_quality(result, text):
                self.metrics.record(Outcome.MISS_QUALITY_REJECTED, len(text))
            else:
                self.storage.put(key, result)
                self._on_stored(text, profile)
                self.metrics.record(Outcome.MISS_STORED, len(text))
            return result

    async def get_audio(
        self, text: str, profile: VoiceProfile, user_id: str
    ) -> list[TTSResult]:
        """The main function that is exposed that gets the audio by cache/ synthesis.

        Returns a list of TTSResult objects that reprasent the audio for each sentence.
        """
        normalized = normalize(text, profile.language)
        sentences = self._split_into_cache_units(normalized, profile.language)

        return [
            await self._get_or_synthesize(sentence, profile, user_id)
            for sentence in sentences
        ]

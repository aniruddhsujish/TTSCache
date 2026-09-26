from tts_cache.keys import VoiceProfile, build_key
from tts_cache.normalize import normalize
from tts_cache.splitter import split_sentences
from tts_cache.storage import TieredStorage
from tts_cache.tts.base import TTSBackend, TTSResult


class SegmentStrategy:

    def __init__(self, tts: TTSBackend, storage: TieredStorage):
        self.tts = tts
        self.storage = storage

    async def synthesize_response(
        self, text: str, profile: VoiceProfile
    ) -> list[TTSResult]:
        normalized = normalize(text, profile.language)
        sentences = split_sentences(normalized, profile.language)

        results = []

        for sentence in sentences:
            key = build_key(sentence, profile)
            found = self.storage.get(key)
            if found is not None:
                results.append(found)
            else:
                result = await self.tts.synthesize(sentence, profile)
                self.storage.put(key, result)
                results.append(result)

        return results

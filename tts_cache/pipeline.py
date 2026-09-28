import logging

from tts_cache.keys import VoiceProfile
from tts_cache.strategies.segment import SegmentStrategy
from tts_cache.tts.base import TTSBackend, TTSError, TTSResult

log = logging.getLogger(__name__)


class CachePipeline:

    def __init__(self, strategy: SegmentStrategy, tts: TTSBackend):
        self.strategy = strategy
        self.tts = tts

    async def speak(
        self, text: str, profile: VoiceProfile, user_id: str
    ) -> list[TTSResult]:
        """The function that runs the entire flow for a particular strategy. Error handling around strategy.get_audio function"""
        try:
            return await self.strategy.get_audio(text, profile, user_id)
        except TTSError:
            raise
        except Exception:
            log.exception("cache path failed - falling back to plain synthesis")
            result = await self.tts.synthesize(text, profile)
            return [result]

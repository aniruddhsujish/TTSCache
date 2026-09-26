import pytest

from tts_cache.keys import VoiceProfile
from tts_cache.pipeline import CachePipeline
from tts_cache.tts.base import TTSError
from tts_cache.tts.fake import FakeTTS

PROFILE = VoiceProfile(language="en", voice="v1", model="m1", output_format="pcm_16000")


class BrokenStrategy:

    async def get_audio(self, text, profile, user_id):
        raise ValueError("bug somewhere in the cache path")


class TTSFailingStrategy:

    async def get_audio(self, text, profile, user_id):
        raise TTSError("provider down")


@pytest.mark.asyncio
async def test_cache_failure_falls_back_to_plain_synthesis():
    tts = FakeTTS(delay=0)
    pipeline = CachePipeline(BrokenStrategy(), tts)

    results = await pipeline.speak("Your order has shipped.", PROFILE, "user1")

    assert len(results) == 1
    assert tts.calls == 1


@pytest.mark.asyncio
async def test_tts_failure_is_not_hidden():
    tts = FakeTTS(delay=0)
    pipeline = CachePipeline(TTSFailingStrategy(), tts)

    with pytest.raises(TTSError):
        await pipeline.speak("Your order has shipped", PROFILE, "user1")

    assert tts.calls == 0

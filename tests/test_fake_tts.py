import pytest
import asyncio
from tts_cache.tts.fake import FakeTTS

@pytest.mark.asyncio
async def test_longer_text_gives_longer_audio():
    tts = FakeTTS(delay=0)
    short = await tts.synthesize("hello", "en", "v1", "m1", {})
    long = await tts.synthesize("hello there my friend", "en", "v1", "m1", {})
    assert len(long.audio) > len(short.audio)
    assert len(long.timestamps) == 4

@pytest.mark.asyncio
async def test_count_increments_after_each_run():
    tts = FakeTTS(delay=0)
    await tts.synthesize("hello", "en", "v1", "m1", {})
    assert tts.calls == 1
    await tts.synthesize("hello there", "en", "v1", "m1", {})
    assert tts.calls == 2

@pytest.mark.asyncio
async def test_synthesize_fails_when_failure_rate_is_1():
    tts = FakeTTS(delay=0, failure_rate=1.0)
    with pytest.raises(RuntimeError, match="fake TTS failure"):
        await tts.synthesize("hello", "en", "v1", "m1", {})

@pytest.mark.asyncio
async def test_different_text_gives_different_audio():
    tts = FakeTTS(delay=0)
    result1 = await tts.synthesize("hello", "en", "v1", "m1", {})
    result2 = await tts.synthesize("world", "en", "v1", "m1", {})
    assert result1.audio != result2.audio

@pytest.mark.asyncio
async def test_can_be_cancelled():
    tts = FakeTTS(delay=1.0)
    task = asyncio.create_task(tts.synthesize("hello", "en", "v1", "m1", {}))
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
import asyncio

import pytest

from tts_cache.coalescing import Coalescer
from tts_cache.keys import VoiceProfile
from tts_cache.tts.base import TTSError
from tts_cache.tts.fake import FakeTTS

PROFILE = VoiceProfile(language="en", voice="v1", model="m1", output_format="pcm_16000")


@pytest.mark.asyncio
async def test_concurrent_misses_make_one_tts_call():
    tts = FakeTTS(delay=0.1)
    coalescer = Coalescer(wait_timeout=1.0)

    results = await asyncio.gather(
        *[
            coalescer.run("k1", lambda: tts.synthesize("hello", PROFILE))
            for _ in range(50)
        ]
    )

    assert tts.calls == 1
    assert all(r == results[0] for r in results)
    assert coalescer.in_flight == {}


@pytest.mark.asyncio
async def test_slow_leader_waiters_fall_back_after_timeout():
    tts = FakeTTS(delay=0.5)
    coalescer = Coalescer(wait_timeout=0.1)

    await asyncio.gather(
        coalescer.run("k1", lambda: tts.synthesize("hello", PROFILE)),
        coalescer.run("k1", lambda: tts.synthesize("hello", PROFILE)),
    )

    assert tts.calls == 2


@pytest.mark.asyncio
async def test_tts_failures_reach_waiters_without_retries():
    tts = FakeTTS(delay=0.1, failure_rate=1.0)
    coalescer = Coalescer(wait_timeout=1.0)

    results = await asyncio.gather(
        *[
            coalescer.run("k1", lambda: tts.synthesize("hello", PROFILE))
            for _ in range(5)
        ],
        return_exceptions=True
    )

    assert all(isinstance(r, TTSError) for r in results)
    assert tts.calls == 1
    assert coalescer.in_flight == {}


@pytest.mark.asyncio
async def test_cancelled_leader_lets_waiter_synthesize():
    tts = FakeTTS(delay=0.3)
    coalescer = Coalescer(wait_timeout=1.0)

    leader = asyncio.create_task(
        coalescer.run("k1", lambda: tts.synthesize("hello", PROFILE))
    )
    await asyncio.sleep(0.05)
    waiter = asyncio.create_task(
        coalescer.run("k1", lambda: tts.synthesize("hello", PROFILE))
    )
    await asyncio.sleep(0.05)

    leader.cancel()

    result = await waiter
    assert result is not None
    assert tts.calls == 2
    with pytest.raises(asyncio.CancelledError):
        await leader

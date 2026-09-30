import asyncio
from typing import Awaitable, Callable

from tts_cache.tts.base import TTSResult


class LeaderCancelled(Exception):
    """The request we were waiting on was cancelled; synthesize ourselves."""


class Coalescer:

    def __init__(self, wait_timeout: float = 1.0):
        self.wait_timeout = wait_timeout
        self.in_flight: dict[str, asyncio.Future] = {}
        self.coalesced = 0

    async def run(
        self, key: str, make_call: Callable[[], Awaitable[TTSResult]]
    ) -> TTSResult:
        """Runs the coalescing Leader/ Waiter logic"""

        # Waiter path: look for the same call in flight and wait for its result
        existing = self.in_flight.get(key)
        if existing is not None:
            try:
                result = await asyncio.wait_for(
                    asyncio.shield(existing), timeout=self.wait_timeout
                )
                self.coalesced += 1
                return result
            except (asyncio.TimeoutError, LeaderCancelled):
                return await make_call()

        # Leader path: start the synth call that others can wait for
        future = asyncio.get_running_loop().create_future()
        self.in_flight[key] = future
        try:
            result = await make_call()
            future.set_result(result)
            return result
        except asyncio.CancelledError:
            future.set_exception(LeaderCancelled())
            raise
        except Exception as exc:
            future.set_exception(exc)
            raise
        finally:
            if future.done() and not future.cancelled():
                future.exception()
            del self.in_flight[key]

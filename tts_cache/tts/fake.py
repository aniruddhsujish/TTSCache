import random
import asyncio
import hashlib
from tts_cache.tts.base import TTSBackend, TTSResult, WordTimestamp


class FakeTTS(TTSBackend):

    def __init__(self, delay=0.2, failure_rate=0.0, sample_rate=16000, seed=42):
        self.delay = delay
        self.failure_rate = failure_rate
        self.sample_rate = sample_rate
        self.rng = random.Random(seed)
        self.calls = 0

    async def synthesize(self, text, profile):
        self.calls += 1

        await asyncio.sleep(self.delay)

        if self.rng.random() < self.failure_rate:
            raise RuntimeError("fake TTS failure")

        words = text.split()

        WORD_DURATION = 0.3
        timestamps = []
        current = 0.0

        for word in words:
            start = current
            end = start + WORD_DURATION
            timestamps.append(WordTimestamp(word=word, start=start, end=end))
            current = end

        bytes_per_word = int(WORD_DURATION * self.sample_rate) * 2
        chunks = []

        for word in words:
            fill = hashlib.sha256(word.encode()).digest()[0]
            chunks.append(bytes([fill]) * bytes_per_word)

        audio = b"".join(chunks)

        return TTSResult(
            audio=audio, sample_rate=self.sample_rate, timestamps=timestamps
        )

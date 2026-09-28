from tts_cache.tts.base import TTSResult

MAX_CHARS_PER_SECOND = 40.0
MIN_CHARS_PER_SECOND = 2.0
BYTES_PER_SAMPLE = 2


def duration_seconds(result: TTSResult) -> float:
    """Returns duration of the audio sample"""
    return len(result.audio) / (result.sample_rate * 2)


def passes_quality(result: TTSResult, text: str) -> bool:
    """Basic quality pass before admission into the cache - checks that the audio isn't empty and the characters per second sits within a safe interval"""
    if not result.audio:
        return False

    seconds = duration_seconds(result)
    chars_per_second = len(text) / seconds

    return MIN_CHARS_PER_SECOND <= chars_per_second <= MAX_CHARS_PER_SECOND

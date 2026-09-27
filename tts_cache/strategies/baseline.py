from tts_cache.strategies.segment import SegmentStrategy


class BaselineStrategy(SegmentStrategy):
    """Standard audio caching as the baseline. The whole response is one cache unit."""

    def _split_into_cache_units(self, normalized, language) -> list[str]:
        return [normalized]

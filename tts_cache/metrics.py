from collections import Counter
from enum import Enum


class Outcome(str, Enum):
    HIT = "hit"
    MISS_STORED = "miss_stored"
    MISS_BELOW_THRESHOLD = "miss_below_threshold"
    MISS_QUALITY_REJECTED = "miss_quality_rejected"
    SEMANTIC_HIT = "semantic_hit"


class Metrics:

    def __init__(self):
        self.outcomes: Counter[Outcome] = Counter()
        self.chars_requested = 0
        self.chars_saved = 0

    def record(self, outcome: Outcome, chars: int) -> None:
        self.outcomes[outcome] += 1
        self.chars_requested += chars
        if outcome in (Outcome.HIT, Outcome.SEMANTIC_HIT):
            self.chars_saved += chars

    def savings_ratio(self) -> float:
        if self.chars_requested == 0:
            return 0.0
        return self.chars_saved / self.chars_requested

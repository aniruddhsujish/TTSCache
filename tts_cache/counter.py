import hashlib
import hmac
import time
from typing import Callable

WEEK_SECONDS = 7 * 24 * 60 * 60


def private_hash(value: str, secret: bytes) -> str:
    return hmac.new(secret, value.encode("utf-8"), hashlib.sha256).hexdigest()


class RequestCounter:

    def __init__(
        self,
        secret: bytes,
        threshold: int = 5,
        window: float = WEEK_SECONDS,
        clock: Callable[[], float] = time.time,
    ):
        self.secret = secret
        self.threshold = threshold
        self.window = window
        self.clock = clock
        self.seen: dict[str, dict[str, float]] = {}

    def _prune(self, k: str) -> None:
        """Deletes an entry under key k if it is older than the given window from the counter"""

        users = self.seen.get(k)
        if users is None:
            return
        cutoff = self.clock() - self.window

        for u, last_seen in list(users.items()):
            if last_seen < cutoff:
                del users[u]
        if not users:
            del self.seen[k]

    def record(self, key: str, user_id: str) -> None:
        """Records a lookup for a key, noting user_id so it can track distinct ones"""
        k = private_hash(key, self.secret)
        u = private_hash(user_id, self.secret)
        self._prune(k)
        users = self.seen.setdefault(k, {})
        if len(users) >= self.threshold and u not in users:
            return
        users[u] = self.clock()

    def should_admit(self, key: str) -> bool:
        """Returns if a result should be added to cache aka. if it has been seen by {threshold} number of  distinct users in the time window"""
        k = private_hash(key, self.secret)
        self._prune(k)
        return len(self.seen.get(k, {})) >= self.threshold

    def prune_all(self) -> None:
        """Deletes all entries that are older than the given window"""
        for k in list(self.seen.keys()):
            self._prune(k)

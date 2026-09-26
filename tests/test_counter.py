from tts_cache.counter import RequestCounter, WEEK_SECONDS

SECRET = b"test-secret"


class FakeClock:

    def __init__(self, start: float = 1_000_000.0):
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def make_counter():
    clock = FakeClock()
    return RequestCounter(secret=SECRET, clock=clock), clock


def test_admits_after_five_distinct_users():
    counter, _ = make_counter()
    for i in range(4):
        counter.record("k1", f"user{i}")
    assert not counter.should_admit("k1")
    counter.record("k1", "user4")
    assert counter.should_admit("k1")


def test_same_user_repeat_not_counted():
    counter, _ = make_counter()
    for i in range(5):
        counter.record("k1", "user1")
    assert not counter.should_admit("k1")


def test_old_visits_cleared_from_window():
    counter, clock = make_counter()
    for i in range(4):
        counter.record("k1", f"user{i}")
    clock.advance(WEEK_SECONDS + 10)
    counter.record("k1", "user4")
    assert not counter.should_admit("k1")


def test_prune_all_removes_forgotten_keys():
    counter, clock = make_counter()
    counter.record("k1", "user0")
    clock.advance(WEEK_SECONDS + 10)
    counter.prune_all()
    assert counter.seen == {}


def test_raw_values_are_not_stored():
    counter, _ = make_counter()
    counter.record("Your balance is 45000 rupees", "user1234")
    stored = str(counter.seen)
    assert "45000" not in stored
    assert "1234" not in stored

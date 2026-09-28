import re

from tts_cache.tts.base import TTSResult, WordTimestamp
from tts_cache.keys import VoiceProfile, build_key
from tts_cache.metrics import Outcome
from tts_cache.normalize import normalize
from tts_cache.quality import passes_quality
from tts_cache.strategies.segment import SegmentStrategy

SLOT_PATTERNS = [
    (
        "DATE",
        re.compile(r"^\d{1,4}[/.-]\d{1,2}[/.-]\d{1,4}$"),
    ),  # 12/03/2026, 2026-03-12, 12.03.2026
    ("TIME", re.compile(r"^\d{1,2}:\d{2}$")),  # 5:30, 17:45
    ("NUM", re.compile(r"^[\d,]+(?:\.\d+)?$")),  # 4521, 45,230.50
]
SLOT_PLACEHOLDER = re.compile(r"\{(?:NUM|DATE|TIME)\}")
TRAILING_PUNCTUATION = ".,!?।"
MAX_VARIABLES = 2
BYTES_PER_SAMPLE = 2


def slot_type(word: str) -> str | None:
    """ "'DATE', 'TIME' or 'Num' if this word is a variable"""
    core = word.rstrip(TRAILING_PUNCTUATION)
    for name, pattern in SLOT_PATTERNS:
        if pattern.match(core):
            return name
    return None


def slots(template: str) -> list[str]:
    """The placeholders in a template, in order. eg.g ['{NUM}', '{DATE}']"""
    return SLOT_PLACEHOLDER.findall(template)


def find_variables(words: list[str]) -> list[int] | None:
    """Positions of variable words, or None if there are none or too many."""
    positions = [i for i, w in enumerate(words) if slot_type(w)]

    if not positions or len(positions) > MAX_VARIABLES:
        return None
    return positions


def make_template(words: list[str], positions: list[int]) -> tuple[str, list[str]]:
    """Replace variable words with typed placeholders, keeping trailing punctuation."""
    template_words = list(words)
    values = []
    for i in positions:
        word = words[i]
        core = word.rstrip(TRAILING_PUNCTUATION)
        punctuation = word[len(core) :]
        template_words[i] = "{" + slot_type(word) + "}" + punctuation
        values.append(word)
    return " ".join(template_words), values


def to_byte(seconds: float, sample_rate: int) -> int:
    return round(seconds * sample_rate) * BYTES_PER_SAMPLE


def duration(result: TTSResult) -> float:
    return len(result.audio) / (result.sample_rate * BYTES_PER_SAMPLE)


def _cut(
    result: TTSResult, start: float, end: float, first_word: int, last_word: int
) -> TTSResult:
    """The audio between start and end, with the timestamps of words first_word to last_word"""
    audio = result.audio[
        to_byte(start, result.sample_rate) : to_byte(end, result.sample_rate)
    ]
    timestamps = [
        WordTimestamp(
            word=ts.word, start=ts.start - start, end=ts.end - start
        )  # Shifts the timestamps to start at 0
        for ts in result.timestamps[first_word:last_word]
    ]
    return TTSResult(audio=audio, sample_rate=result.sample_rate, timestamps=timestamps)


def slice_fixed_parts(result: TTSResult, positions: list[int]) -> list[TTSResult]:
    """Cut out the variable words; return the fixed pieces in order"""
    parts = []
    start = 0.0
    first_word = 0

    for p in positions:
        var_start = result.timestamps[p].start
        parts.append(_cut(result, start, var_start, first_word, p))
        start = result.timestamps[p].end
        first_word = p + 1

    parts.append(
        _cut(result, start, duration(result), first_word, len(result.timestamps))
    )
    return parts


def join(pieces: list[TTSResult]) -> TTSResult:
    """Glue clips end to end, shifting each clip's timestamps by the offset before it."""
    audio = b""
    timestamps = []
    offset = 0.0
    for piece in pieces:
        audio += piece.audio
        timestamps += [
            WordTimestamp(word=ts.word, start=ts.start + offset, end=ts.end + offset)
            for ts in piece.timestamps
        ]
        offset += duration(piece)
    return TTSResult(
        audio=audio, sample_rate=pieces[0].sample_rate, timestamps=timestamps
    )


def interleave(fixed: list[TTSResult], variables: list[TTSResult]) -> list[TTSResult]:
    """Put the pieces back in sentence in order: fixed0, var0, fixed1, var1, fixed2."""
    pieces = []
    for i, var in enumerate(variables):
        pieces.append(fixed[i])
        pieces.append(var)
    pieces.append(fixed[-1])
    return pieces


class TemplateStrategy(SegmentStrategy):

    def _load_parts(self, template_key: str, count: int) -> list[TTSResult] | None:
        """All fixed parts, or None if any one is missing."""
        parts = [self.storage.get(f"{template_key}:{i}") for i in range(count)]
        return None if any(p is None for p in parts) else parts

    async def get_audio(
        self, text: str, profile: VoiceProfile, user_id: str
    ) -> list[TTSResult]:
        normalized = normalize(text, profile.language)
        sentences = self._split_into_cache_units(normalized, profile.language)
        return [await self._resolve_sentence(s, profile, user_id) for s in sentences]

    async def _resolve_sentence(
        self, sentence: str, profile: VoiceProfile, user_id: str
    ) -> TTSResult:
        words = sentence.split()
        positions = find_variables(words)

        # 1. No variables found, default sentence level caching
        if positions is None:
            return await self._get_or_synthesize(sentence, profile, user_id)

        # 2. Variable found but exact sentence cached before
        exact = self.storage.get(build_key(sentence, profile))
        if exact is not None:
            self.metrics.record(Outcome.HIT, len(sentence))
            return exact

        # Template level caching
        template_text, values = make_template(words, positions)
        template_key = build_key("[template] " + template_text, profile)
        self.counter.record(template_key, user_id)

        # 3. Template hit -> reuse the fixed parts and run the numbers thru the cache
        fixed = self._load_parts(template_key, len(positions) + 1)
        if fixed is not None:
            fixed_chars = len(sentence) - sum(len(v) for v in values)
            self.metrics.record(Outcome.HIT, fixed_chars)
            variables = [
                await self._get_or_synthesize(v, profile, user_id) for v in values
            ]
            return join(interleave(fixed, variables))

        # 4. template miss -> synthesize the full sentence and store fixed parts in the cache if crossing admission threshold
        result = await self._get_or_synthesize(sentence, profile, user_id)
        if self.counter.should_admit(template_key) and passes_quality(result, sentence):
            for i, part in enumerate(slice_fixed_parts(result, positions)):
                self.storage.put(f"{template_key}:{i}", part)
        return result

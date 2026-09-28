import re

from tts_cache.tts.fake import TTSResult, WordTimestamp

NUMBER = re.compile(r"^[\d,]+(?:\.\d+)?$")
TRAILING_PUNCTUATION = ".,!?।"
MAX_VARIABLES = 2
BYTES_PER_SAMPLE = 2


def find_variables(words: list[str]) -> list[int] | None:
    """Return the positions of numeric words, or None if there are none or too many."""
    positions = []
    for i, word in enumerate(words):
        core = word.rstrip(TRAILING_PUNCTUATION)
        if NUMBER.match(core):
            positions.append(i)

    if not positions or len(positions) > MAX_VARIABLES:
        return None
    return positions


def make_template(words: list[str], positions: list[int]) -> tuple[str, list[str]]:
    """Replace variable words with {NUM}, keeping trailing punctuation."""
    template_words = list(words)
    values = []
    for i in positions:
        word = words[i]
        core = word.rstrip(TRAILING_PUNCTUATION)
        punctuation = word[len(core) :]
        template_words[i] = "{NUM}" + punctuation
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

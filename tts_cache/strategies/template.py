import re

NUMBER = re.compile(r"^[\d,]+(?:\.\d+)?$")
TRAILING_PUNCTUATION = ".,!?।"
MAX_VARIABLES = 2


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

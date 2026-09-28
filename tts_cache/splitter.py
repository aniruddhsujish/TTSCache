from tts_cache.normalize import load_lang_rules

DEFAULT_TERMINATORS = [".", "?", "!", "।"]
DEFAULT_ABBREVIATIONS = []


def split_sentences(text: str, language: str) -> list[str]:
    """Splits the text into sentences for Segment caching. Supports different terminator characters based on language configs and abbreviation recognition support."""
    rules = load_lang_rules(language) or {}
    terminators = rules.get("terminators", DEFAULT_TERMINATORS)
    abbreviations = rules.get("abbreviations", DEFAULT_ABBREVIATIONS)

    words = text.split()
    sentences = []
    current = []

    for word in words:
        current.append(word)

        ends_with_terminator = word.endswith(tuple(terminators))
        is_abbreviation = word in abbreviations

        if ends_with_terminator and not is_abbreviation:
            sentences.append(" ".join(current))
            current = []

    if current:
        sentences.append(" ".join(current))

    return sentences

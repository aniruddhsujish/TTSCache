from tts_cache.splitter import split_sentences


def test_splits_two_sentences():
    result = split_sentences("Your bill is 500 rupees. Pay by Friday.", "en")
    assert result == ["Your bill is 500 rupees.", "Pay by Friday."]

def test_split_ignores_abbreviations():
    result = split_sentences("Please visit Dr. X. He is available.", "en")
    assert result == ["Please visit Dr. X.", "He is available."]

def test_splits_hindi_sentences():
    result = split_sentences("आपका ऑर्डर आ गया। धन्यवाद।", "hi")
    assert result == ["आपका ऑर्डर आ गया।", "धन्यवाद।"]

def test_split_no_terminator():
    result = split_sentences("Hello there", "en")
    assert result == ["Hello there"]

def test_empty_string():
    result = split_sentences("", "en")
    assert result == []

def test_default_lang_no_rules():
    result = split_sentences("Your bill is 500 rupees. Pay by Friday.", "ta")
    assert result == ["Your bill is 500 rupees.", "Pay by Friday."]

def test_space_before_terminator():
    result = split_sentences("आपका ऑर्डर आ गया । धन्यवाद।", "hi")
    assert result == ["आपका ऑर्डर आ गया ।", "धन्यवाद।"]
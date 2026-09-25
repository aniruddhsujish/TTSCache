from dataclasses import replace
import pytest
from tts_cache.keys import VoiceProfile, build_key

BASE = VoiceProfile(language="hi", voice="v1", model="m1", output_format="pcm_16000")
TEXT = "नमस्ते"

def test_same_input_gives_same_key():
    assert build_key(TEXT, BASE) == build_key(TEXT, BASE)

def test_different_text_gives_different_key():
    assert build_key("नमस्ते", BASE) != build_key("धन्यवाद", BASE)

@pytest.mark.parametrize(
    "changes",
    [
        {"language": "mr"},
        {"voice": "v2"},
        {"model": "m2"},
        {"output_format": "mp3_44100"},
        {"settings": {"speed": 1.2}},
        {"namespace": "tenant_acme"},
    ],
)
def test_changing_any_field_changes_the_key(changes):
    changed = replace(BASE, **changes)
    assert build_key(TEXT, changed) != build_key(TEXT, BASE)

def test_hindi_and_marathi_get_different_keys():
    hindi = replace(BASE, language="hi")
    marathi = replace(BASE, language="mr")
    assert build_key(TEXT, hindi) != build_key(TEXT, marathi)

def test_settings_order_does_not_matter():
    a = replace(BASE, settings={"speed": 1, "style": 0})
    b = replace(BASE, settings={"style": 0, "speed": 1})
    assert build_key(TEXT, a) == build_key(TEXT, b)
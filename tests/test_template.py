import pytest

from tts_cache.strategies.template import (
    find_variables,
    interleave,
    join,
    make_template,
    slice_fixed_parts,
    to_byte,
)
from tts_cache.tts.fake import FakeTTS
from tts_cache.keys import VoiceProfile

PROFILE = VoiceProfile(language="en", voice="v1", model="m1", output_format="pcm_16000")


def test_finds_one_variable_in_the_middle():
    words = "Your order 4521 has been shipped.".split()
    assert find_variables(words) == [2]


def test_find_one_variable_in_the_end():
    words = "Your ticket number is 4521.".split()
    assert find_variables(words) == [4]


def test_find_two_variables():
    words = "Your order 4521 will arrive in 3 days.".split()
    assert find_variables(words) == [2, 6]


def test_rejects_three_variables():
    words = "Order 1 of 2 costs 45,230.50 rupees.".split()
    assert find_variables(words) is None


def test_no_variables():
    words = "Thank you for calling.".split()
    assert find_variables(words) is None


def test_find_one_variable_hindi():
    words = "आपका ऑर्डर 4521 भेज दिया गया है।".split()
    assert find_variables(words) == [2]


def test_converts_to_template_with_single_variable():
    words = "Your ticket number is 4521.".split()
    assert make_template(words, [4]) == (
        "Your ticket number is {NUM}.",
        ["4521."],
    )


def test_converts_to_template_double_variable():
    words = "Your order 4521 will arrive in 3 days.".split()
    assert make_template(words, [2, 6]) == (
        "Your order {NUM} will arrive in {NUM} days.",
        ["4521", "3"],
    )


@pytest.mark.asyncio
async def test_slicing_keeps_the_fixed_words_only():
    tts = FakeTTS(delay=0)
    result = await tts.synthesize("Your order 4521 has shipped.", PROFILE)

    parts = slice_fixed_parts(result, [2])

    assert [ts.word for ts in parts[0].timestamps] == ["Your", "order"]
    assert [ts.word for ts in parts[1].timestamps] == ["has", "shipped."]
    assert parts[1].timestamps[0].start == 0.0
    assert len(parts[0].audio) == to_byte(0.6, result.sample_rate)


@pytest.mark.asyncio
async def test_slice_then_join_rebuilds_the_original_sentence():
    tts = FakeTTS(delay=0)
    result = await tts.synthesize("Your order 4521 has shipped.", PROFILE)

    fixed = slice_fixed_parts(result, [2])
    number = await tts.synthesize("4521", PROFILE)
    rebuilt = join(interleave(fixed, [number]))

    assert rebuilt.audio == result.audio
    assert [ts.word for ts in rebuilt.timestamps] == [
        "Your",
        "order",
        "4521",
        "has",
        "shipped.",
    ]
    assert rebuilt.timestamps[3].start == pytest.approx(0.9)

from tts_cache.quality import passes_quality
from tts_cache.tts.base import TTSResult

RATE = 16000


def audio_of(seconds: float) -> TTSResult:
    num_bytes = int(seconds * RATE) * 2
    return TTSResult(audio=b"\x01" * num_bytes, sample_rate=RATE, timestamps=[])


TEXT = "Your order has shipped."  # 23 characters


def test_normal_audio_passes():
    assert passes_quality(audio_of(1.5), TEXT)


def test_empty_audio_fails():
    empty = TTSResult(audio=b"", sample_rate=RATE, timestamps=[])
    assert not passes_quality(empty, TEXT)


def test_cut_off_audio_fails():
    assert not passes_quality(audio_of(0.2), TEXT)


def test_runaway_audio_fails():
    assert not passes_quality(audio_of(60.0), TEXT)

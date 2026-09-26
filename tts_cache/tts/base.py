from abc import ABC, abstractmethod
from dataclasses import dataclass
from tts_cache.keys import VoiceProfile


@dataclass
class WordTimestamp:
    word: str
    start: float
    end: float


@dataclass
class TTSResult:
    audio: bytes
    sample_rate: int
    timestamps: list[WordTimestamp]


class TTSBackend(ABC):

    @abstractmethod
    async def synthesize(self, text: str, profile: VoiceProfile) -> TTSResult: ...


class TTSError(Exception):
    """Raised when the TTS provider fails"""

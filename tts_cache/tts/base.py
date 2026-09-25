from abc import ABC, abstractmethod
from dataclasses import dataclass

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
    async def synthesize(
        self,
        text: str,
        language: str,
        voice: str,
        model: str,
        settings: dict[str, object]
    ) -> TTSResult:
        ...


import hashlib
import json
from dataclasses import dataclass, field


@dataclass(frozen=True)
class VoiceProfile:
    language: str
    voice: str
    model: str
    output_format: str
    settings: dict = field(default_factory=dict)
    namespace: str = "shared"


def build_key(text: str, profile: VoiceProfile) -> str:
    """Builds the key from text, profile and namespace"""

    payload = {
        "text": text,
        "namespace": profile.namespace,
        "language": profile.language,
        "voice": profile.voice,
        "model": profile.model,
        "output_format": profile.output_format,
        "settings": profile.settings,
    }
    serialized = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

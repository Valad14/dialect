"""Data models for dialect speech transcription."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any


@dataclass(slots=True)
class SpeechSegment:
    """A single ASR segment with timing."""

    start: float
    end: float
    text: str

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass(slots=True)
class PhoneticOutput:
    """Phonetic transcription in two notations plus applied features."""

    ipa: str
    cyrillic: str
    features: dict[str, int] = field(default_factory=dict)


@dataclass(slots=True)
class TranscriptionResult:
    """Complete transcription result for JSON/TXT export."""

    audio_file: str | None
    orthographic: str
    phonetic_ipa: str
    phonetic_cyrillic: str
    profile: str
    model_name: str | None = None
    language: str = "ru"
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    segments: list[SpeechSegment] = field(default_factory=list)
    features: dict[str, int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["segments"] = [asdict(segment) for segment in self.segments]
        return payload

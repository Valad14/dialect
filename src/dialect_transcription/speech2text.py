"""Speech-to-text module backed by OpenAI Whisper."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import importlib.util
import re

from .models import SpeechSegment
from .runtime import FFmpegUnavailableError, ensure_ffmpeg


class WhisperNotInstalledError(RuntimeError):
    """Raised when openai-whisper is not installed in the environment."""


class WhisperSpeech2Text:
    """Thin wrapper around the local `openai-whisper` package."""

    def __init__(self, model_name: str = "base", device: str | None = None) -> None:
        self.model_name = model_name
        self.device = device
        ensure_ffmpeg()
        if importlib.util.find_spec("whisper") is None:
            raise WhisperNotInstalledError(
                "Пакет openai-whisper не установлен. Установите зависимости из requirements.txt."
            )
        import whisper  # type: ignore

        load_kwargs: dict[str, Any] = {}
        if device and device != "auto":
            load_kwargs["device"] = device
        self._model = whisper.load_model(model_name, **load_kwargs)

    def transcribe(
        self,
        audio_path: str | Path,
        *,
        language: str = "ru",
        initial_prompt: str | None = None,
        word_timestamps: bool = False,
    ) -> tuple[str, list[SpeechSegment], dict[str, Any]]:
        """Transcribe an audio file and return normalized text and segments."""
        ensure_ffmpeg()
        path = Path(audio_path)
        if not path.exists():
            raise FileNotFoundError(f"Аудиофайл не найден: {path}")

        options: dict[str, Any] = {
            "language": language or None,
            "task": "transcribe",
            "verbose": False,
            "word_timestamps": word_timestamps,
            "fp16": False,
        }
        if initial_prompt:
            options["initial_prompt"] = initial_prompt

        result = self._model.transcribe(str(path), **options)
        text = normalize_orthography(str(result.get("text", "")))
        segments = [
            SpeechSegment(
                start=float(segment.get("start", 0.0)),
                end=float(segment.get("end", 0.0)),
                text=normalize_orthography(str(segment.get("text", ""))),
            )
            for segment in result.get("segments", [])
            if str(segment.get("text", "")).strip()
        ]
        return text, segments, result


def normalize_orthography(text: str) -> str:
    """Normalize ASR text while keeping punctuation needed for pauses."""
    text = text.strip().lower().replace("ё", "е")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([,.!?;:])", r"\1", text)
    text = re.sub(r"([,.!?;:])(?=\S)", r"\1 ", text)
    return text.strip()

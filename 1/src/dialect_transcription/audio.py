"""Audio helpers for Streamlit and CLI."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import os
import tempfile
from typing import BinaryIO


@dataclass(slots=True)
class AudioInfo:
    path: str
    filename: str
    size_mb: float
    duration_sec: float | None = None
    sample_rate: int | None = None
    channels: int | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def save_upload_to_temp(uploaded_file: BinaryIO, original_name: str) -> Path:
    """Save a Streamlit UploadedFile into a temporary file and return its path."""
    suffix = Path(original_name).suffix or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        return Path(tmp.name)


def get_audio_info(audio_path: str | Path) -> AudioInfo:
    """Read basic metadata. Works without librosa, but gives more with it."""
    path = Path(audio_path)
    size_mb = path.stat().st_size / (1024 * 1024) if path.exists() else 0.0
    info = AudioInfo(path=str(path), filename=path.name, size_mb=size_mb)

    if not path.exists():
        info.error = f"Файл не найден: {path}"
        return info

    try:
        import soundfile as sf  # type: ignore

        with sf.SoundFile(str(path)) as snd:
            info.duration_sec = len(snd) / float(snd.samplerate)
            info.sample_rate = int(snd.samplerate)
            info.channels = int(snd.channels)
            return info
    except Exception:
        pass

    try:
        import librosa  # type: ignore

        audio, sample_rate = librosa.load(str(path), sr=None, mono=False)
        if hasattr(audio, "ndim") and audio.ndim > 1:
            channels = int(audio.shape[0])
            samples = int(audio.shape[1])
        else:
            channels = 1
            samples = int(len(audio))
        info.duration_sec = samples / float(sample_rate)
        info.sample_rate = int(sample_rate)
        info.channels = channels
        return info
    except Exception as exc:
        info.error = f"Не удалось прочитать метаданные аудио: {exc}"
        return info


def cleanup_temp_file(path: str | Path | None) -> None:
    if not path:
        return
    try:
        os.unlink(path)
    except OSError:
        pass

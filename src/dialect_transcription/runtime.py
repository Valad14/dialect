"""Runtime helpers for local launch and Streamlit Community Cloud.

The main purpose of this module is to make ffmpeg available before Whisper
tries to call it. Streamlit Cloud should install ffmpeg from packages.txt, but
this fallback also exposes the bundled executable from imageio-ffmpeg.
"""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import stat
import tempfile


class FFmpegUnavailableError(RuntimeError):
    """Raised when ffmpeg can't be found or prepared."""


def _prepend_to_path(directory: Path) -> None:
    """Prepend a directory to PATH once."""
    directory_str = str(directory)
    current_parts = os.environ.get("PATH", "").split(os.pathsep)
    if directory_str not in current_parts:
        os.environ["PATH"] = directory_str + os.pathsep + os.environ.get("PATH", "")


def _make_command_alias(source: Path, target_dir: Path) -> Path:
    """Create an executable named ffmpeg/ffmpeg.exe pointing to imageio's binary."""
    target_dir.mkdir(parents=True, exist_ok=True)
    command_name = "ffmpeg.exe" if os.name == "nt" else "ffmpeg"
    target = target_dir / command_name

    if target.exists() or target.is_symlink():
        try:
            if target.resolve() == source.resolve():
                _prepend_to_path(target_dir)
                return target
        except OSError:
            pass
        try:
            target.unlink()
        except OSError:
            pass

    try:
        target.symlink_to(source)
    except OSError:
        shutil.copy2(source, target)

    try:
        mode = target.stat().st_mode
        target.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    except OSError:
        pass

    _prepend_to_path(target_dir)
    return target


def ensure_ffmpeg() -> str:
    """Return a usable ffmpeg executable path or raise a helpful error.

    Order of attempts:
    1. Use ffmpeg already available in PATH, for example from packages.txt.
    2. Use the bundled binary supplied by imageio-ffmpeg and create a stable
       command alias named ``ffmpeg`` so libraries that call subprocesses work.
    """
    existing = shutil.which("ffmpeg")
    if existing:
        return existing

    try:
        import imageio_ffmpeg  # type: ignore

        bundled = Path(imageio_ffmpeg.get_ffmpeg_exe())
        if bundled.exists():
            alias_dir = Path(tempfile.gettempdir()) / "dialect_transcription_ffmpeg"
            alias = _make_command_alias(bundled, alias_dir)
            found = shutil.which("ffmpeg")
            return found or str(alias)
    except Exception as exc:  # pragma: no cover - depends on deployment image
        raise FFmpegUnavailableError(_ffmpeg_help()) from exc

    raise FFmpegUnavailableError(_ffmpeg_help())


def ffmpeg_status() -> tuple[bool, str]:
    """Return a status tuple suitable for a friendly UI check."""
    try:
        path = ensure_ffmpeg()
        return True, path
    except FFmpegUnavailableError as exc:
        return False, str(exc)


def _ffmpeg_help() -> str:
    return (
        "Не найден ffmpeg — компонент, который нужен для чтения MP3, M4A, OGG, WEBM "
        "и для работы Whisper. В проект уже добавлены packages.txt и imageio-ffmpeg. "
        "Если ошибка появилась на Streamlit Cloud, убедитесь, что app.py, requirements.txt "
        "и packages.txt лежат в корне GitHub-репозитория, затем удалите старое приложение "
        "и разверните его заново."
    )

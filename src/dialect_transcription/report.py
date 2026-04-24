"""Export helpers for transcription results."""

from __future__ import annotations

import json
from pathlib import Path

from .models import TranscriptionResult


def result_to_txt(result: TranscriptionResult) -> str:
    lines = [
        "ОТЧЕТ О ТРАНСКРИПЦИИ ДИАЛЕКТНОЙ РЕЧИ",
        "=" * 72,
        f"Время обработки: {result.timestamp}",
        f"Аудиофайл: {result.audio_file or 'текстовый режим'}",
        f"Модель Speech2Text: {result.model_name or 'не использовалась'}",
        f"Язык: {result.language}",
        f"Профиль фонетики: {result.profile}",
        "",
        "ОРФОГРАФИЧЕСКАЯ ЗАПИСЬ",
        "-" * 72,
        result.orthographic or "<пусто>",
        "",
        "ФОНЕТИЧЕСКАЯ ТРАНСКРИПЦИЯ: КИРИЛЛИЦА",
        "-" * 72,
        result.phonetic_cyrillic or "<пусто>",
        "",
        "ФОНЕТИЧЕСКАЯ ТРАНСКРИПЦИЯ: IPA",
        "-" * 72,
        result.phonetic_ipa or "<пусто>",
        "",
    ]
    if result.features:
        lines.extend(["ПРИМЕНЕННЫЕ ПРАВИЛА", "-" * 72])
        for key, value in sorted(result.features.items()):
            lines.append(f"{key}: {value}")
        lines.append("")
    if result.segments:
        lines.extend(["СЕГМЕНТЫ", "-" * 72])
        for segment in result.segments:
            lines.append(f"[{segment.start:8.2f} - {segment.end:8.2f}] {segment.text}")
        lines.append("")
    if result.warnings:
        lines.extend(["ПРЕДУПРЕЖДЕНИЯ", "-" * 72])
        lines.extend(result.warnings)
        lines.append("")
    return "\n".join(lines)


def save_result(result: TranscriptionResult, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".json":
        path.write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        path.write_text(result_to_txt(result), encoding="utf-8")
    return path

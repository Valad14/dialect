"""Command line interface for dialect speech transcription."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dialect_transcription import DIALECT_PROFILES, RussianDialectPhonemizer, TranscriptionResult
from dialect_transcription.report import result_to_txt, save_result
from dialect_transcription.speech2text import WhisperSpeech2Text, normalize_orthography


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Автоматическая транскрипция диалектной речи")
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--audio", help="Путь к аудиофайлу")
    input_group.add_argument("--text", help="Готовый текст для speech2phoneme без распознавания аудио")

    parser.add_argument("--model", default="base", choices=["tiny", "base", "small", "medium", "large"], help="Модель Whisper")
    parser.add_argument("--language", default="ru", help="Код языка для Whisper")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="Устройство для Whisper")
    parser.add_argument("--profile", default="literary", choices=list(DIALECT_PROFILES.keys()), help="Диалектный профиль")
    parser.add_argument("--stress", default="penultimate", choices=["penultimate", "first", "last"], help="Эвристика ударения")
    parser.add_argument("--no-prothetic-v", action="store_true", help="Не добавлять протетическое в-/w-")
    parser.add_argument("--fricative-g", action="store_true", help="Принудительно включить фрикативное г")
    parser.add_argument("--okanye", action="store_true", help="Принудительно включить оканье")
    parser.add_argument("--no-reduction", action="store_true", help="Отключить редукцию гласных")
    parser.add_argument("--no-final-devoicing", action="store_true", help="Отключить конечное оглушение")
    parser.add_argument("--prompt", default=None, help="Начальная подсказка для Whisper")
    parser.add_argument("--output", default=None, help="Куда сохранить результат: .json или .txt")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    segments = []
    model_name = None
    audio_file = None

    if args.audio:
        device = None if args.device == "auto" else args.device
        asr = WhisperSpeech2Text(model_name=args.model, device=device)
        orthographic, segments, _raw = asr.transcribe(args.audio, language=args.language, initial_prompt=args.prompt)
        model_name = args.model
        audio_file = str(args.audio)
    else:
        orthographic = normalize_orthography(args.text)

    base_profile = DIALECT_PROFILES[args.profile]
    phonemizer = RussianDialectPhonemizer(
        args.profile,
        prothetic_v=(False if args.no_prothetic_v else base_profile.prothetic_v),
        fricative_g=(True if args.fricative_g else base_profile.fricative_g),
        okanye=(True if args.okanye else base_profile.okanye),
        reduce_vowels=(False if args.no_reduction else base_profile.reduce_vowels),
        final_devoicing=(False if args.no_final_devoicing else base_profile.final_devoicing),
        stress_strategy=args.stress,
    )
    phonetic = phonemizer.phonemize(orthographic, segments=segments or None)
    result = TranscriptionResult(
        audio_file=audio_file,
        orthographic=orthographic,
        phonetic_ipa=phonetic.ipa,
        phonetic_cyrillic=phonetic.cyrillic,
        profile=phonemizer.profile.title,
        model_name=model_name,
        language=args.language,
        segments=segments,
        features=phonetic.features,
        warnings=[
            "Фонетическая часть является правиловым модулем; ударения и диалектные признаки желательно проверять вручную."
        ],
    )

    if args.output:
        out = save_result(result, args.output)
        print(f"Результат сохранен: {out}")
    else:
        print(result_to_txt(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

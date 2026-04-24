"""Streamlit application for automatic dialect speech transcription."""

from __future__ import annotations

from pathlib import Path
import json
import sys
import tempfile

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dialect_transcription import DIALECT_PROFILES, RussianDialectPhonemizer, TranscriptionResult
from dialect_transcription.audio import cleanup_temp_file, get_audio_info
from dialect_transcription.models import SpeechSegment
from dialect_transcription.report import result_to_txt
from dialect_transcription.speech2text import WhisperNotInstalledError, WhisperSpeech2Text, normalize_orthography


st.set_page_config(
    page_title="Диалектная транскрипция",
    page_icon="🎙️",
    layout="wide",
)


@st.cache_resource(show_spinner=False)
def get_asr(model_name: str, device: str) -> WhisperSpeech2Text:
    selected_device = None if device == "auto" else device
    return WhisperSpeech2Text(model_name=model_name, device=selected_device)


def save_uploaded_file(uploaded_file) -> Path:
    suffix = Path(uploaded_file.name).suffix or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getvalue())
        return Path(tmp.name)


def build_result(
    *,
    orthographic: str,
    segments: list[SpeechSegment],
    audio_file: str | None,
    model_name: str | None,
    language: str,
    profile_key: str,
    prothetic_v: bool,
    fricative_g: bool,
    okanye: bool,
    reduce_vowels: bool,
    final_devoicing: bool,
    stress_strategy: str,
) -> TranscriptionResult:
    phonemizer = RussianDialectPhonemizer(
        profile_key,
        prothetic_v=prothetic_v,
        fricative_g=fricative_g,
        okanye=okanye,
        reduce_vowels=reduce_vowels,
        final_devoicing=final_devoicing,
        stress_strategy=stress_strategy,
    )
    phonetic = phonemizer.phonemize(orthographic, segments=segments or None)
    profile = phonemizer.profile
    warnings: list[str] = []
    if not orthographic.strip():
        warnings.append("Speech2Text не вернул распознанный текст. Проверьте качество аудио или выберите другую модель.")
    warnings.append(
        "Фонетическая часть является правиловым speech2phoneme-модулем. Для научной разметки желательно сверять ударения и диалектные признаки вручную."
    )
    return TranscriptionResult(
        audio_file=audio_file,
        orthographic=orthographic,
        phonetic_ipa=phonetic.ipa,
        phonetic_cyrillic=phonetic.cyrillic,
        profile=profile.title,
        model_name=model_name,
        language=language,
        segments=segments,
        features=phonetic.features,
        warnings=warnings,
    )


def display_result(result: TranscriptionResult) -> None:
    st.success("Транскрипция готова")
    left, right = st.columns(2)
    with left:
        st.subheader("Орфографическая запись")
        st.text_area("Орфографическая запись", result.orthographic, height=210, label_visibility="collapsed")
    with right:
        st.subheader("Фонетическая транскрипция: кириллица")
        st.text_area("Фонетическая транскрипция: кириллица", result.phonetic_cyrillic, height=210, label_visibility="collapsed")

    with st.expander("IPA-транскрипция", expanded=True):
        st.code(result.phonetic_ipa or "<пусто>", language="text")

    if result.features:
        with st.expander("Примененные фонетические правила"):
            features_df = pd.DataFrame(
                [{"правило": key, "количество": value} for key, value in sorted(result.features.items())]
            )
            st.dataframe(features_df, use_container_width=True, hide_index=True)

    if result.segments:
        with st.expander("Сегменты Speech2Text с таймкодами"):
            segment_df = pd.DataFrame(
                [
                    {
                        "начало": round(segment.start, 2),
                        "конец": round(segment.end, 2),
                        "длительность": round(segment.duration, 2),
                        "текст": segment.text,
                    }
                    for segment in result.segments
                ]
            )
            st.dataframe(segment_df, use_container_width=True, hide_index=True)

    txt_report = result_to_txt(result)
    json_report = json.dumps(result.to_dict(), ensure_ascii=False, indent=2)
    download_left, download_right = st.columns(2)
    with download_left:
        st.download_button(
            "Скачать отчет TXT",
            data=txt_report,
            file_name="dialect_transcription_report.txt",
            mime="text/plain",
            use_container_width=True,
        )
    with download_right:
        st.download_button(
            "Скачать JSON",
            data=json_report,
            file_name="dialect_transcription_result.json",
            mime="application/json",
            use_container_width=True,
        )


st.title("🎙️ Автоматическая транскрипция диалектной речи")
st.markdown(
    "Приложение делает **орфографическую запись** через Speech2Text и затем строит "
    "**фонетическую транскрипцию** через speech2phoneme / speech@phonetics."
)

with st.sidebar:
    st.header("Speech2Text")
    model_name = st.selectbox("Модель Whisper", ["tiny", "base", "small", "medium", "large"], index=1)
    language = st.text_input("Язык распознавания", value="ru", max_chars=8)
    device = st.selectbox("Устройство", ["auto", "cpu", "cuda"], index=0)
    word_timestamps = st.checkbox("Запрашивать word timestamps", value=False)
    initial_prompt = st.text_area(
        "Подсказка для Whisper",
        value="Русская диалектная речь, разговорная запись, возможны областные формы.",
        height=80,
    )

    st.header("Speech2Phoneme")
    profile_key = st.selectbox(
        "Диалектный профиль",
        list(DIALECT_PROFILES.keys()),
        format_func=lambda key: DIALECT_PROFILES[key].title,
    )
    base_profile = DIALECT_PROFILES[profile_key]
    st.caption(base_profile.description)
    prothetic_v = st.checkbox("Протетическое в-/w- перед у/о/ю в начале фразы", value=base_profile.prothetic_v)
    okanye = st.checkbox("Оканье: сохранять безударное о", value=base_profile.okanye)
    fricative_g = st.checkbox("Фрикативное г: г → ɣ", value=base_profile.fricative_g)
    reduce_vowels = st.checkbox("Редукция безударных гласных", value=base_profile.reduce_vowels)
    final_devoicing = st.checkbox("Конечное оглушение звонких согласных", value=base_profile.final_devoicing)
    stress_strategy = st.selectbox("Эвристика ударения без словаря", ["penultimate", "first", "last"], index=0)

uploaded_file = st.file_uploader(
    "Выберите аудиофайл",
    type=["wav", "mp3", "flac", "m4a", "ogg", "webm"],
    help="Для MP3/M4A/OGG/WebM нужен установленный ffmpeg.",
)

manual_text = st.text_area(
    "Или вставьте уже готовую орфографическую запись для проверки speech2phoneme",
    value="",
    placeholder="Например: у нас строились куряни, в пять комнатей две комнати их и не строили.",
    height=90,
)

run = st.button("Выполнить транскрипцию", type="primary", use_container_width=True)

if uploaded_file is not None:
    st.audio(uploaded_file)
    with st.expander("Информация о выбранном файле", expanded=True):
        st.write({"имя": uploaded_file.name, "размер_MB": round(uploaded_file.size / (1024 * 1024), 3)})

if run:
    temp_path: Path | None = None
    try:
        if uploaded_file is not None:
            temp_path = save_uploaded_file(uploaded_file)
            info = get_audio_info(temp_path)
            with st.expander("Метаданные аудио", expanded=False):
                st.json(info.to_dict())
            with st.spinner("Speech2Text: распознаю аудио через Whisper..."):
                asr = get_asr(model_name, device)
                orthographic, segments, _raw = asr.transcribe(
                    temp_path,
                    language=language,
                    initial_prompt=initial_prompt.strip() or None,
                    word_timestamps=word_timestamps,
                )
            result = build_result(
                orthographic=orthographic,
                segments=segments,
                audio_file=uploaded_file.name,
                model_name=model_name,
                language=language,
                profile_key=profile_key,
                prothetic_v=prothetic_v,
                fricative_g=fricative_g,
                okanye=okanye,
                reduce_vowels=reduce_vowels,
                final_devoicing=final_devoicing,
                stress_strategy=stress_strategy,
            )
            display_result(result)
        elif manual_text.strip():
            orthographic = normalize_orthography(manual_text)
            result = build_result(
                orthographic=orthographic,
                segments=[],
                audio_file=None,
                model_name=None,
                language=language,
                profile_key=profile_key,
                prothetic_v=prothetic_v,
                fricative_g=fricative_g,
                okanye=okanye,
                reduce_vowels=reduce_vowels,
                final_devoicing=final_devoicing,
                stress_strategy=stress_strategy,
            )
            display_result(result)
        else:
            st.warning("Загрузите аудио или вставьте текст для фонетической транскрипции.")
    except WhisperNotInstalledError as exc:
        st.error(str(exc))
    except FileNotFoundError as exc:
        st.error(str(exc))
    except Exception as exc:  # Streamlit should show a friendly error instead of crashing.
        st.exception(exc)
    finally:
        cleanup_temp_file(temp_path)

st.divider()
with st.expander("Краткая инструкция"):
    st.markdown(
        """
1. Загрузите аудиофайл WAV/MP3/FLAC/M4A/OGG/WebM или вставьте готовый текст.
2. Выберите модель Whisper и диалектный профиль.
3. Нажмите **Выполнить транскрипцию**.
4. Скачайте TXT-отчет или JSON с сегментами и примененными правилами.

Обозначения: `/` — короткая пауза, `//` — длинная пауза или граница высказывания, `'` / `ʲ` — мягкость согласного, `ъ` / `ə` — редуцированный гласный.
        """
    )

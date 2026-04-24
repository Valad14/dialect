"""Minimal Streamlit app for automatic dialect speech transcription."""

from __future__ import annotations

from pathlib import Path
import json
import sys
import tempfile

import streamlit as st

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dialect_transcription import DIALECT_PROFILES, RussianDialectPhonemizer, TranscriptionResult
from dialect_transcription.audio import cleanup_temp_file, get_audio_info
from dialect_transcription.models import SpeechSegment
from dialect_transcription.report import result_to_txt
from dialect_transcription.runtime import FFmpegUnavailableError, ffmpeg_status
from dialect_transcription.speech2text import WhisperNotInstalledError, WhisperSpeech2Text, normalize_orthography


APP_TITLE = "Транскрипция диалектной речи"
LANGUAGE = "ru"
DEFAULT_PROFILE = "literary"
DEFAULT_PROMPT = "Русская диалектная речь, разговорная запись, возможны областные формы."
MODEL_LABELS = {
    "tiny": "tiny — быстрее",
    "base": "base — оптимально",
    "small": "small — точнее, медленнее",
}


st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🎙️",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    #MainMenu, footer, header {visibility: hidden;}
    .block-container {max-width: 980px; padding-top: 2.2rem; padding-bottom: 3rem;}
    h1 {font-size: 2.45rem !important; line-height: 1.05 !important; letter-spacing: -0.04em; margin-bottom: .3rem;}
    h2, h3 {letter-spacing: -0.02em;}
    .lead {font-size: 1.08rem; color: #525866; margin-bottom: 1.2rem;}
    .soft-card {border: 1px solid #e8e8ee; border-radius: 18px; padding: 1.1rem 1.2rem; background: #ffffff;}
    .muted {color: #6b7280; font-size: .95rem;}
    .tiny {color: #6b7280; font-size: .84rem;}
    div[data-testid="stSidebar"] {background: #fafafa; border-right: 1px solid #eeeeee;}
    div[data-testid="stSidebar"] h2, div[data-testid="stSidebar"] h3 {font-size: 1.05rem !important;}
    .stButton > button {border-radius: 14px; min-height: 3rem; font-weight: 600;}
    .stDownloadButton > button {border-radius: 12px;}
    div[data-baseweb="input"], textarea {border-radius: 14px !important;}
    div[data-testid="stFileUploader"] section {border-radius: 18px; border-style: solid;}
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def get_asr(model_name: str) -> WhisperSpeech2Text:
    return WhisperSpeech2Text(model_name=model_name, device=None)


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
) -> TranscriptionResult:
    profile = DIALECT_PROFILES[DEFAULT_PROFILE]
    phonemizer = RussianDialectPhonemizer(
        DEFAULT_PROFILE,
        prothetic_v=profile.prothetic_v,
        fricative_g=profile.fricative_g,
        okanye=profile.okanye,
        reduce_vowels=profile.reduce_vowels,
        final_devoicing=profile.final_devoicing,
        stress_strategy=profile.stress_strategy,
    )
    phonetic = phonemizer.phonemize(orthographic, segments=segments or None)
    warnings: list[str] = []
    if not orthographic.strip():
        warnings.append("Speech2Text не вернул текст. Попробуйте запись с меньшим шумом или другую модель.")
    warnings.append(
        "Фонетическая транскрипция создается правиловым speech2phoneme-модулем. "
        "Для научной публикации ударения и спорные диалектные признаки нужно проверить вручную."
    )
    return TranscriptionResult(
        audio_file=audio_file,
        orthographic=orthographic,
        phonetic_ipa=phonetic.ipa,
        phonetic_cyrillic=phonetic.cyrillic,
        profile=phonemizer.profile.title,
        model_name=model_name,
        language=LANGUAGE,
        segments=segments,
        features=phonetic.features,
        warnings=warnings,
    )


def render_downloads(result: TranscriptionResult) -> None:
    txt_report = result_to_txt(result)
    json_report = json.dumps(result.to_dict(), ensure_ascii=False, indent=2)
    left, right = st.columns(2)
    with left:
        st.download_button(
            "Скачать TXT",
            data=txt_report,
            file_name="dialect_transcription_report.txt",
            mime="text/plain",
            use_container_width=True,
        )
    with right:
        st.download_button(
            "Скачать JSON",
            data=json_report,
            file_name="dialect_transcription_result.json",
            mime="application/json",
            use_container_width=True,
        )


def display_result(result: TranscriptionResult) -> None:
    st.success("Готово. Ниже можно проверить и скачать результат.")

    with st.container(border=True):
        st.subheader("Орфографическая запись")
        st.text_area(
            "Орфографическая запись",
            result.orthographic or "",
            height=180,
            label_visibility="collapsed",
        )

    with st.container(border=True):
        st.subheader("Фонетическая транскрипция")
        tab_cyr, tab_ipa = st.tabs(["Кириллица", "IPA"])
        with tab_cyr:
            st.text_area(
                "Фонетическая транскрипция кириллицей",
                result.phonetic_cyrillic or "",
                height=180,
                label_visibility="collapsed",
            )
        with tab_ipa:
            st.code(result.phonetic_ipa or "<пусто>", language="text")

    render_downloads(result)

    with st.expander("Подробности обработки"):
        st.write(
            {
                "файл": result.audio_file or "текстовый режим",
                "модель": result.model_name or "не использовалась",
                "язык": result.language,
                "фонетический профиль": result.profile,
            }
        )
        if result.segments:
            st.markdown("**Фрагменты с таймкодами**")
            for segment in result.segments:
                st.write(f"{segment.start:.2f}–{segment.end:.2f}: {segment.text}")
        if result.warnings:
            st.markdown("**Важно**")
            for warning in result.warnings:
                st.caption(warning)


def render_ffmpeg_error(exc: Exception | str) -> None:
    st.error("Не удалось подготовить аудиообработку ffmpeg.")
    with st.expander("Как исправить", expanded=True):
        st.markdown(
            """
В обновленной версии уже добавлены два способа решить проблему:

1. `packages.txt` с одной строкой `ffmpeg` — Streamlit Cloud устанавливает системный пакет.
2. `imageio-ffmpeg` — запасной вариант, который дает приложению собственный ffmpeg.

Если ошибка сохраняется на Streamlit Cloud, почти всегда причина в том, что файлы лежат не в корне репозитория. В GitHub рядом должны находиться:

```text
app.py
requirements.txt
packages.txt
src/
```

После загрузки этих файлов лучше удалить старое приложение в Streamlit Cloud и создать его заново, указав `app.py` как main file path.
            """
        )
        st.caption(str(exc))


def transcribe_audio(uploaded_file, model_name: str) -> None:
    temp_path: Path | None = None
    try:
        temp_path = save_uploaded_file(uploaded_file)
        info = get_audio_info(temp_path)
        if info.duration_sec:
            st.caption(f"Длительность: {info.duration_sec:.1f} сек. Размер: {info.size_mb:.2f} MB")

        with st.spinner("Распознаю речь. Первая обработка может занять несколько минут..."):
            asr = get_asr(model_name)
            orthographic, segments, _raw = asr.transcribe(
                temp_path,
                language=LANGUAGE,
                initial_prompt=DEFAULT_PROMPT,
                word_timestamps=False,
            )
        result = build_result(
            orthographic=orthographic,
            segments=segments,
            audio_file=uploaded_file.name,
            model_name=model_name,
        )
        display_result(result)
    except FFmpegUnavailableError as exc:
        render_ffmpeg_error(exc)
    except WhisperNotInstalledError as exc:
        st.error(str(exc))
    except FileNotFoundError as exc:
        if "ffmpeg" in str(exc).lower():
            render_ffmpeg_error(exc)
        else:
            st.error(str(exc))
    except Exception as exc:
        message = str(exc)
        if "ffmpeg" in message.lower():
            render_ffmpeg_error(exc)
        else:
            st.error("Не удалось выполнить транскрипцию. Попробуйте другой аудиофайл или модель tiny/base.")
            with st.expander("Технические подробности"):
                st.exception(exc)
    finally:
        cleanup_temp_file(temp_path)


def transcribe_text(manual_text: str) -> None:
    orthographic = normalize_orthography(manual_text)
    result = build_result(
        orthographic=orthographic,
        segments=[],
        audio_file=None,
        model_name=None,
    )
    display_result(result)


with st.sidebar:
    st.markdown("### Настройка")
    model_name = st.selectbox(
        "Модель распознавания",
        list(MODEL_LABELS.keys()),
        index=1,
        format_func=lambda value: MODEL_LABELS[value],
        help="tiny работает быстрее, small обычно точнее. Для Streamlit Cloud лучше начинать с tiny или base.",
    )
    st.markdown("---")
    st.caption("Другие параметры скрыты, чтобы приложение было простым для пользователя.")

st.title(APP_TITLE)
st.markdown(
    '<p class="lead">Загрузите аудиозапись или вставьте текст. Приложение сделает орфографическую запись и фонетическую транскрипцию.</p>',
    unsafe_allow_html=True,
)

ok, ffmpeg_message = ffmpeg_status()
if not ok:
    st.warning("ffmpeg пока не найден. Если обработка аудио не запустится, проверьте packages.txt и пересоздайте приложение в Streamlit Cloud.")

work_tab, guide_tab, about_tab = st.tabs(["Транскрибировать", "Руководство", "О проекте"])

with work_tab:
    st.markdown("### 1. Выберите источник")
    input_mode = st.radio(
        "Источник",
        ["Аудиофайл", "Готовый текст"],
        horizontal=True,
        label_visibility="collapsed",
    )

    if input_mode == "Аудиофайл":
        uploaded_file = st.file_uploader(
            "Перетащите аудиофайл сюда",
            type=["wav", "mp3", "flac", "m4a", "ogg", "webm"],
            help="Поддерживаются WAV, MP3, FLAC, M4A, OGG и WEBM.",
        )
        if uploaded_file is not None:
            st.audio(uploaded_file)
            st.caption(f"Выбран файл: {uploaded_file.name} · {uploaded_file.size / (1024 * 1024):.2f} MB")

        st.markdown("### 2. Запустите обработку")
        run_audio = st.button(
            "Начать транскрипцию",
            type="primary",
            use_container_width=True,
            disabled=uploaded_file is None,
        )
        if run_audio and uploaded_file is not None:
            transcribe_audio(uploaded_file, model_name)

    else:
        manual_text = st.text_area(
            "Вставьте орфографическую запись",
            value="",
            placeholder="Например: у нас строились куряни, в пять комнатей две комнати их и не строили.",
            height=140,
        )
        run_text = st.button(
            "Сделать фонетическую транскрипцию",
            type="primary",
            use_container_width=True,
            disabled=not manual_text.strip(),
        )
        if run_text and manual_text.strip():
            transcribe_text(manual_text)

with guide_tab:
    st.markdown(
        """
### Руководство пользователя

**1. Выберите модель в левом меню.**  
Для первой проверки используйте `base`. Если приложение работает медленно на Streamlit Cloud, выберите `tiny`.

**2. Загрузите аудио.**  
Подойдут форматы WAV, MP3, FLAC, M4A, OGG и WEBM. Чем меньше шумов и посторонних голосов, тем лучше распознавание.

**3. Нажмите «Начать транскрипцию».**  
Первый запуск может быть долгим: сервер загружает модель Whisper. Следующие запуски обычно быстрее.

**4. Проверьте результат.**  
Вы получите два основных блока: орфографическую запись и фонетическую транскрипцию. Вкладка IPA нужна для международной фонетической записи.

**5. Скачайте отчет.**  
TXT удобен для чтения и вставки в работу, JSON — для дальнейшей обработки данных.

**Что делать, если результат неточный:** попробуйте модель `small`, укоротите аудио, уберите музыку и шум, запишите речь ближе к микрофону.
        """
    )

with about_tab:
    st.markdown(
        """
### О проекте

Это приложение помогает быстро получить первичную транскрипцию звучащей русской диалектной речи.

Процесс состоит из двух этапов:

1. **Speech2Text** — аудио распознается локальной моделью Whisper, результатом становится обычная орфографическая запись.
2. **speech2phoneme / speech@phonetics** — орфографический текст проходит через правиловый фонетический модуль, который формирует транскрипцию кириллицей и в IPA.

Приложение рассчитано на учебную, демонстрационную и предварительную исследовательскую работу. Оно не заменяет экспертную ручную расшифровку: автоматическая фонетика использует правила и эвристики, поэтому ударение, редукцию и специфические диалектные признаки нужно проверять вручную.

В интерфейсе оставлена только одна настройка — модель распознавания. Это сделано специально, чтобы приложением мог пользоваться человек без технической подготовки.
        """
    )

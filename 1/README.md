# Автоматическая транскрипция диалектной речи

Проект повторяет функциональность демонстрационного приложения: загружает аудио, получает орфографическую запись через Speech2Text и строит фонетическую транскрипцию в двух вариантах — кириллица и IPA.

## Что умеет программа

- загружать аудио: WAV, MP3, FLAC, M4A, OGG, WebM;
- распознавать русскую речь локально через OpenAI Whisper;
- выводить орфографическую запись;
- строить фонетическую транскрипцию:
  - кириллическая запись для лингвистической расшифровки;
  - IPA-запись;
- переключать диалектные профили:
  - среднерусский / базовый литературный;
  - севернорусский профиль с оканьем;
  - южнорусский профиль с фрикативным `г`;
  - режим без редукции для отладки;
- показывать сегменты Speech2Text с таймкодами;
- скачивать результат в TXT и JSON.

## Установка

Нужен Python 3.10 или новее. Для Whisper также нужен `ffmpeg`.

### Linux / Ubuntu

```bash
sudo apt update
sudo apt install ffmpeg python3-venv
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

### macOS

```bash
brew install ffmpeg
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

### Windows

Установите Python 3.10+ и ffmpeg, затем в PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install -U pip
pip install -r requirements.txt
```

## Запуск веб-приложения

```bash
streamlit run app.py
```

После запуска откройте адрес, который покажет Streamlit, обычно `http://localhost:8501`.

## Запуск из командной строки

Распознавание аудио:

```bash
python cli.py --audio sample.wav --model base --profile literary --output result.txt
```

Только фонетическая транскрипция готового текста:

```bash
python cli.py --text "у нас строились куряни, в пять комнатей" --profile south
```

Сохранение JSON:

```bash
python cli.py --audio sample.wav --model small --profile north --output result.json
```

## Структура проекта

```text
app.py                              # Streamlit-интерфейс
cli.py                              # CLI-режим
requirements.txt                    # зависимости
src/dialect_transcription/
  audio.py                          # метаданные и временные файлы аудио
  models.py                         # dataclass-модели результата
  report.py                         # экспорт TXT/JSON
  speech2text.py                    # Speech2Text через Whisper
  speech2phoneme.py                 # speech2phoneme + speech@phonetics правила
```

## Обозначения в транскрипции

- `/` — короткая пауза;
- `//` — длинная пауза или граница фразы;
- `'` или `ʲ` — мягкость согласного;
- `ъ` или `ə` — редуцированный гласный;
- `в-` или `w-` — протетический начальный призвук перед округленными гласными;
- `ɣ` / `г̞` — фрикативный вариант `г` в южнорусском профиле.

## Важное ограничение

Whisper дает орфографический текст, а фонетическая часть в этом проекте — правиловый модуль. Для полноценной научной диалектологической разметки нужны проверка ударений, ручная корректировка и/или обучение отдельной speech-to-phoneme модели на размеченных диалектных аудиоданных.

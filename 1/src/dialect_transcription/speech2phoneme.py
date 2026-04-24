"""Rule-based Russian text-to-phoneme module with several dialect presets.

The module is deliberately dependency-free. It is not meant to replace a
full linguistic transcriber with stress dictionaries; it provides a practical
baseline for a Streamlit demo and can later be swapped for a trained
speech-to-phoneme model.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
import unicodedata

from .models import PhoneticOutput, SpeechSegment


VOWELS = "аеёиоуыэюя"
SOFT_VOWELS = "еёиюя"
HARD_ALWAYS = set("жшц")
SOFT_ALWAYS = set("чщй")
CONSONANTS = set("бвгджзйклмнпрстфхцчшщ")
WORD_RE = re.compile(r"[а-яёА-ЯЁa-zA-Z+\u0301]+|[.,!?;:]+|[-—]+", re.UNICODE)


@dataclass(frozen=True, slots=True)
class DialectProfile:
    """Settings controlling phonetic rendering."""

    key: str
    title: str
    description: str
    okanye: bool = False
    fricative_g: bool = False
    prothetic_v: bool = True
    final_devoicing: bool = True
    reduce_vowels: bool = True
    stress_strategy: str = "penultimate"  # first, penultimate, last


DIALECT_PROFILES: dict[str, DialectProfile] = {
    "literary": DialectProfile(
        key="literary",
        title="Среднерусский / литературный базовый",
        description="Аканье, редукция безударных гласных, конечное оглушение.",
        okanye=False,
        fricative_g=False,
        prothetic_v=True,
        final_devoicing=True,
        reduce_vowels=True,
        stress_strategy="penultimate",
    ),
    "north": DialectProfile(
        key="north",
        title="Севернорусский профиль: оканье",
        description="Сохраняет безударное о как [o]; подходит для демонстрации оканья.",
        okanye=True,
        fricative_g=False,
        prothetic_v=True,
        final_devoicing=True,
        reduce_vowels=True,
        stress_strategy="penultimate",
    ),
    "south": DialectProfile(
        key="south",
        title="Южнорусский профиль: фрикативное г + аканье",
        description="Передает г как [ɣ] и применяет аканье/редукцию.",
        okanye=False,
        fricative_g=True,
        prothetic_v=True,
        final_devoicing=True,
        reduce_vowels=True,
        stress_strategy="penultimate",
    ),
    "no_reduction": DialectProfile(
        key="no_reduction",
        title="Без редукции",
        description="Буквенная фонемизация без редукции гласных; полезно для отладки.",
        okanye=False,
        fricative_g=False,
        prothetic_v=False,
        final_devoicing=False,
        reduce_vowels=False,
        stress_strategy="penultimate",
    ),
}


class RussianDialectPhonemizer:
    """Convert Russian orthographic text into IPA and Cyrillic transcription.

    The converter supports optional acute stress marks in text. You may write
    ``молоко`` or ``молоко́``. For text produced by ASR, where stress is usually
    absent, the class uses a simple configurable stress heuristic.
    """

    ipa_consonants = {
        "б": "b",
        "в": "v",
        "г": "g",
        "д": "d",
        "ж": "ʐ",
        "з": "z",
        "й": "j",
        "к": "k",
        "л": "l",
        "м": "m",
        "н": "n",
        "п": "p",
        "р": "r",
        "с": "s",
        "т": "t",
        "ф": "f",
        "х": "x",
        "ц": "t͡s",
        "ч": "t͡ɕ",
        "ш": "ʂ",
        "щ": "ɕː",
    }

    cyr_consonants = {
        "б": "б",
        "в": "в",
        "г": "г",
        "д": "д",
        "ж": "ж",
        "з": "з",
        "й": "й",
        "к": "к",
        "л": "л",
        "м": "м",
        "н": "н",
        "п": "п",
        "р": "р",
        "с": "с",
        "т": "т",
        "ф": "ф",
        "х": "х",
        "ц": "ц",
        "ч": "ч",
        "ш": "ш",
        "щ": "ш'",
    }

    ipa_final_devoice = {
        "b": "p",
        "v": "f",
        "g": "k",
        "ɣ": "x",
        "d": "t",
        "z": "s",
        "ʐ": "ʂ",
    }

    cyr_final_devoice = {
        "б": "п",
        "в": "ф",
        "г": "к",
        "ɣ": "х",
        "г̞": "х",
        "д": "т",
        "з": "с",
        "ж": "ш",
    }

    def __init__(
        self,
        profile: str | DialectProfile = "literary",
        *,
        prothetic_v: bool | None = None,
        fricative_g: bool | None = None,
        okanye: bool | None = None,
        reduce_vowels: bool | None = None,
        final_devoicing: bool | None = None,
        stress_strategy: str | None = None,
    ) -> None:
        if isinstance(profile, DialectProfile):
            base = profile
        else:
            base = DIALECT_PROFILES.get(profile, DIALECT_PROFILES["literary"])

        self.profile = DialectProfile(
            key=base.key,
            title=base.title,
            description=base.description,
            okanye=base.okanye if okanye is None else okanye,
            fricative_g=base.fricative_g if fricative_g is None else fricative_g,
            prothetic_v=base.prothetic_v if prothetic_v is None else prothetic_v,
            final_devoicing=base.final_devoicing if final_devoicing is None else final_devoicing,
            reduce_vowels=base.reduce_vowels if reduce_vowels is None else reduce_vowels,
            stress_strategy=base.stress_strategy if stress_strategy is None else stress_strategy,
        )
        self.features: dict[str, int] = {}

    def phonemize(self, text: str, segments: list[SpeechSegment] | None = None) -> PhoneticOutput:
        """Return IPA and Cyrillic phonetic transcription."""
        self.features = {}
        if segments:
            ipa = self._segments_to_phonemes(segments, alphabet="ipa")
        else:
            ipa = self._text_to_phonemes(text, alphabet="ipa", append_final_pause=True)
        features = dict(self.features)

        # Render the Cyrillic view with the same rules, but do not double-count
        # the feature statistics.
        self.features = {}
        if segments:
            cyr = self._segments_to_phonemes(segments, alphabet="cyr")
        else:
            cyr = self._text_to_phonemes(text, alphabet="cyr", append_final_pause=True)
        self.features = features
        return PhoneticOutput(ipa=ipa, cyrillic=cyr, features=features)

    def _segments_to_phonemes(self, segments: list[SpeechSegment], *, alphabet: str) -> str:
        rendered: list[str] = []
        previous_end: float | None = None
        for segment in segments:
            if not segment.text.strip():
                previous_end = segment.end
                continue
            if previous_end is not None:
                gap = max(0.0, segment.start - previous_end)
                if gap >= 0.95:
                    rendered.append("//")
                elif gap >= 0.35:
                    rendered.append("/")
            rendered_text = self._text_to_phonemes(segment.text, alphabet=alphabet, append_final_pause=False)
            if rendered_text:
                rendered.append(rendered_text)
            previous_end = segment.end
        return self._cleanup_output(" ".join(rendered), append_final_pause=True)

    def _text_to_phonemes(self, text: str, *, alphabet: str, append_final_pause: bool) -> str:
        tokens = WORD_RE.findall(text.lower())
        output: list[str] = []
        phrase_start = True

        for token in tokens:
            if self._is_word(token):
                word = self._word_to_phonemes(token, alphabet=alphabet)
                if phrase_start and self.profile.prothetic_v and self._starts_with_rounded_vowel(token):
                    prefix = "w-" if alphabet == "ipa" else "в-"
                    word = f"{prefix}{word}"
                    self._feature("prothetic_v")
                output.append(word)
                phrase_start = False
            elif token in {".", "!", "?"} or any(char in token for char in ".!?"):
                output.append("//")
                phrase_start = True
            elif token in {",", ";", ":"} or any(char in token for char in ",;:"):
                output.append("/")
                phrase_start = False
            # hyphens are ignored: they only separate orthographic parts

        return self._cleanup_output(" ".join(output), append_final_pause=append_final_pause)

    @staticmethod
    def _is_word(token: str) -> bool:
        return any("а" <= char <= "я" or char == "ё" for char in token.lower())

    @staticmethod
    def _starts_with_rounded_vowel(token: str) -> bool:
        stripped = token.lower().lstrip("+\u0301")
        return stripped.startswith(("у", "о", "ю"))

    def _word_to_phonemes(self, word: str, *, alphabet: str) -> str:
        normalized, explicit_stress = self._extract_stress(word)
        vowel_indices = [idx for idx, char in enumerate(normalized) if char in VOWELS]
        stress_idx = self._guess_stress_index(normalized, vowel_indices, explicit_stress)

        out: list[str] = []
        letters_for_out: list[str] = []
        previous_letter = ""

        for idx, char in enumerate(normalized):
            if char not in VOWELS and char not in CONSONANTS and char not in {"ь", "ъ"}:
                continue

            if char == "ь":
                self._soften_previous(out, letters_for_out, alphabet)
                previous_letter = char
                continue

            if char == "ъ":
                previous_letter = char
                continue

            if char in CONSONANTS:
                phone = self._consonant_phone(char, alphabet=alphabet)
                out.append(phone)
                letters_for_out.append(char)
                previous_letter = char
                continue

            # Vowel.
            if char in SOFT_VOWELS and previous_letter in CONSONANTS:
                self._soften_previous(out, letters_for_out, alphabet)
                iotated = False
            else:
                iotated = previous_letter in {"", "ь", "ъ"} or previous_letter in VOWELS
            out.extend(self._vowel_phones(char, idx, stress_idx, iotated=iotated, alphabet=alphabet))
            letters_for_out.extend([char] * (len(out) - len(letters_for_out)))
            previous_letter = char

        if self.profile.final_devoicing:
            last_spoken_letter = next((char for char in reversed(normalized) if char in VOWELS or char in CONSONANTS), "")
            if last_spoken_letter in CONSONANTS:
                self._apply_final_devoicing(out, letters_for_out, alphabet=alphabet)
        return "".join(out)

    def _extract_stress(self, word: str) -> tuple[str, int | None]:
        """Return normalized word and explicit stressed vowel index if marked."""
        normalized_chars: list[str] = []
        stressed_index: int | None = None
        stress_next = False

        # NFC preserves the letter ё and keeps Russian acute stress as a
        # separate combining mark, which is exactly what we need here.
        for char in unicodedata.normalize("NFC", word.lower()):
            if char == "+":
                stress_next = True
                continue
            if char == "\u0301":
                # Combining acute: stress the previous vowel if any.
                for idx in range(len(normalized_chars) - 1, -1, -1):
                    if normalized_chars[idx] in VOWELS:
                        stressed_index = idx
                        break
                continue
            if unicodedata.category(char).startswith("M"):
                continue
            normalized_chars.append(char)
            if char == "ё":
                stressed_index = len(normalized_chars) - 1
            elif stress_next and char in VOWELS:
                stressed_index = len(normalized_chars) - 1
                stress_next = False

        return "".join(normalized_chars), stressed_index

    def _guess_stress_index(
        self,
        word: str,
        vowel_indices: list[int],
        explicit_stress: int | None,
    ) -> int | None:
        if explicit_stress is not None:
            return explicit_stress
        if not vowel_indices:
            return None
        if len(vowel_indices) == 1:
            return vowel_indices[0]
        if "ё" in word:
            return word.index("ё")
        if self.profile.stress_strategy == "first":
            return vowel_indices[0]
        if self.profile.stress_strategy == "last":
            return vowel_indices[-1]
        # Penultimate is only a heuristic for ASR text without stress marks.
        return vowel_indices[-2]

    def _consonant_phone(self, char: str, *, alphabet: str) -> str:
        if char == "г" and self.profile.fricative_g:
            self._feature("fricative_g")
            return "ɣ" if alphabet == "ipa" else "г̞"
        if alphabet == "ipa":
            return self.ipa_consonants[char]
        return self.cyr_consonants[char]

    def _vowel_phones(
        self,
        char: str,
        idx: int,
        stress_idx: int | None,
        *,
        iotated: bool,
        alphabet: str,
    ) -> list[str]:
        stressed = idx == stress_idx
        base_ipa, base_cyr = self._reduced_or_full_vowel(char, stressed=stressed, stress_idx=stress_idx, idx=idx)
        if alphabet == "ipa":
            result = [base_ipa]
            if iotated and char in {"е", "ё", "ю", "я"}:
                result.insert(0, "j")
            return result

        result = [base_cyr]
        if iotated and char in {"е", "ё", "ю", "я"}:
            result.insert(0, "й")
        return result

    def _reduced_or_full_vowel(
        self,
        char: str,
        *,
        stressed: bool,
        stress_idx: int | None,
        idx: int,
    ) -> tuple[str, str]:
        full = {
            "а": ("a", "а"),
            "о": ("o", "о"),
            "э": ("e", "э"),
            "е": ("e", "э"),
            "ё": ("o", "о"),
            "и": ("i", "и"),
            "ы": ("ɨ", "ы"),
            "у": ("u", "у"),
            "ю": ("u", "у"),
            "я": ("a", "а"),
        }
        if stressed or not self.profile.reduce_vowels:
            return full[char]

        if char == "о" and self.profile.okanye:
            self._feature("okanye_o_preserved")
            return ("o", "о")

        if char in {"а", "о", "я"}:
            if stress_idx is not None and idx < stress_idx:
                self._feature("akanye_pretonic")
                return ("ɐ", "а")
            self._feature("vowel_reduction")
            return ("ə", "ъ")

        if char in {"е", "и"}:
            self._feature("vowel_reduction")
            return ("ɪ", "и")

        if char in {"э"}:
            self._feature("vowel_reduction")
            return ("ɪ", "и")

        return full[char]

    def _soften_previous(self, out: list[str], letters_for_out: list[str], alphabet: str) -> None:
        if not out:
            return
        if not letters_for_out:
            return
        previous_letter = letters_for_out[-1]
        if previous_letter in HARD_ALWAYS:
            return
        if previous_letter not in CONSONANTS:
            return
        marker = "ʲ" if alphabet == "ipa" else "'"
        if out[-1].endswith(marker):
            return
        out[-1] = f"{out[-1]}{marker}"
        self._feature("palatalization")

    def _apply_final_devoicing(self, out: list[str], letters_for_out: list[str], *, alphabet: str) -> None:
        for idx in range(len(out) - 1, -1, -1):
            letter = letters_for_out[idx] if idx < len(letters_for_out) else ""
            if letter not in CONSONANTS:
                continue
            marker = "ʲ" if alphabet == "ipa" else "'"
            phone = out[idx]
            soft = phone.endswith(marker)
            base = phone[: -len(marker)] if soft else phone
            mapping = self.ipa_final_devoice if alphabet == "ipa" else self.cyr_final_devoice
            if base in mapping:
                out[idx] = mapping[base] + (marker if soft else "")
                self._feature("final_devoicing")
            return

    def _cleanup_output(self, text: str, *, append_final_pause: bool) -> str:
        raw_parts = re.sub(r"\s+", " ", text).strip().split()
        parts: list[str] = []
        for part in raw_parts:
            if part in {"/", "//"}:
                if not parts:
                    continue
                if parts[-1] in {"/", "//"}:
                    # Long pause dominates short pause.
                    if part == "//" or parts[-1] == "//":
                        parts[-1] = "//"
                    else:
                        parts[-1] = "/"
                else:
                    parts.append(part)
            else:
                parts.append(part)

        text = " ".join(parts).strip()
        if append_final_pause and text and not text.endswith("//"):
            if text.endswith("/"):
                text = text[:-1].strip()
            text = f"{text} //"
        return text

    def _feature(self, key: str) -> None:
        self.features[key] = self.features.get(key, 0) + 1

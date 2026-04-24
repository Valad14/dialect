import unittest

from dialect_transcription.speech2phoneme import RussianDialectPhonemizer
from dialect_transcription.models import SpeechSegment


class RussianDialectPhonemizerTests(unittest.TestCase):
    def test_basic_outputs_are_not_empty(self):
        phonemizer = RussianDialectPhonemizer("literary")
        result = phonemizer.phonemize("у нас строились куряни.")
        self.assertIn("//", result.ipa)
        self.assertIn("в-", result.cyrillic)
        self.assertTrue(result.features)

    def test_north_profile_preserves_unstressed_o(self):
        north = RussianDialectPhonemizer("north")
        result = north.phonemize("молоко")
        self.assertIn("о", result.cyrillic)
        self.assertIn("okanye_o_preserved", result.features)

    def test_south_profile_has_fricative_g(self):
        south = RussianDialectPhonemizer("south")
        result = south.phonemize("гора")
        self.assertIn("ɣ", result.ipa)
        self.assertIn("fricative_g", result.features)

    def test_segment_pauses(self):
        phonemizer = RussianDialectPhonemizer("literary")
        segments = [SpeechSegment(0.0, 1.0, "у нас"), SpeechSegment(1.6, 2.0, "дом")]
        result = phonemizer.phonemize("у нас дом", segments=segments)
        self.assertIn("/", result.cyrillic)


if __name__ == "__main__":
    unittest.main()

import unittest
from app import seconds_to_srt, wrap_text_for_srt, split_segment_by_words


class TimestampTests(unittest.TestCase):
    def test_zero(self):
        self.assertEqual(seconds_to_srt(0), "00:00:00,000")

    def test_simple(self):
        self.assertEqual(seconds_to_srt(1.234), "00:00:01,234")
        self.assertEqual(seconds_to_srt(65.001), "00:01:05,001")
        self.assertEqual(seconds_to_srt(3661.789), "01:01:01,789")


class SubtitleWrapTests(unittest.TestCase):
    def test_wrap_text_for_srt(self):
        self.assertEqual(wrap_text_for_srt("one two three four five", 2), "one two\nthree four\nfive")
        self.assertEqual(wrap_text_for_srt("  one   two   three  ", 2), "one two\nthree")


class SegmentSplitTests(unittest.TestCase):
    def test_split_by_words_native(self):
        # Mock a segment with 5 words
        seg = {
            "start": 0.0,
            "end": 5.0,
            "text": "one two three four five",
            "words": [
                {"word": "one", "start": 0.0, "end": 1.0},
                {"word": "two", "start": 1.0, "end": 2.0},
                {"word": "three", "start": 2.0, "end": 3.0},
                {"word": "four", "start": 3.0, "end": 4.0},
                {"word": "five", "start": 4.0, "end": 5.0},
            ]
        }
        
        # Test split with words_per_line = 2
        res = split_segment_by_words(seg, current_start=0.0, total_duration=10.0, words_per_line=2, selected_output_script="Auto")
        self.assertEqual(len(res), 3)
        self.assertEqual(res[0], {"start": 0.0, "end": 2.0, "text": "one two"})
        self.assertEqual(res[1], {"start": 2.0, "end": 4.0, "text": "three four"})
        self.assertEqual(res[2], {"start": 4.0, "end": 5.0, "text": "five"})

    def test_split_by_words_fallback_interpolation(self):
        # Mock a segment with no "words" key
        seg = {
            "start": 0.0,
            "end": 6.0,
            "text": "one two three four"
        }
        
        # Test split with words_per_line = 2 (linear interpolation)
        res = split_segment_by_words(seg, current_start=0.0, total_duration=10.0, words_per_line=2, selected_output_script="Auto")
        self.assertEqual(len(res), 2)
        # 4 words, words_per_line = 2 -> 2 groups. Total duration = 6.0.
        # Group 1: 0.0 to 3.0
        # Group 2: 3.0 to 6.0
        self.assertEqual(res[0], {"start": 0.0, "end": 3.0, "text": "one two"})
        self.assertEqual(res[1], {"start": 3.0, "end": 6.0, "text": "three four"})

    def test_split_by_words_no_split(self):
        seg = {
            "start": 0.0,
            "end": 5.0,
            "text": "one two three",
            "words": [
                {"word": "one", "start": 0.0, "end": 1.0},
                {"word": "two", "start": 1.0, "end": 2.0},
                {"word": "three", "start": 2.0, "end": 3.0},
            ]
        }
        # words_per_line = 0 or negative should not split
        res = split_segment_by_words(seg, current_start=0.0, total_duration=10.0, words_per_line=0, selected_output_script="Auto")
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0]["text"], "one two three")
        self.assertEqual(res[0]["start"], 0.0)
        self.assertEqual(res[0]["end"], 5.0)


if __name__ == "__main__":
    unittest.main()

import unittest
from app import seconds_to_srt


class TimestampTests(unittest.TestCase):
    def test_zero(self):
        self.assertEqual(seconds_to_srt(0), "00:00:00,000")

    def test_simple(self):
        self.assertEqual(seconds_to_srt(1.234), "00:00:01,234")
        self.assertEqual(seconds_to_srt(65.001), "00:01:05,001")
        self.assertEqual(seconds_to_srt(3661.789), "01:01:01,789")


if __name__ == "__main__":
    unittest.main()

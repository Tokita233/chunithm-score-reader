from __future__ import annotations

import os
import unittest
from pathlib import Path

import app


SAMPLE_1_SCORES = """
1003831 1006551 1005465 998597 995355 997631 1001828 1006743 1006506 1002578
985739 997106 984288 1001947 999196 995307 1003748 999362 1004551 976208
1000032 1000797 984314 1003659 1004496 1003275 997617 997420 1003925 997263
1007812 980357 991164 1006298 987452 998931 1003118 997516 988301 999200
1005620 989841 1008357 990199 981133 997166 1007082 999943 992256 1003798
""".split()

SAMPLE_2_SCORES = """
1007411 1007913 1008780 1007608 1007528 1007485 1008368 1008364 1008279 1008210
1008957 1008934 1008921 1007191 1007860 1005174 1007810 1007110 1007684 1008658
1007649 1007647 1008640 1007501 1008360 1006841 1002644 1006777 1008978 1007976
1007702 1008645 1008459 1007533 1008528 1008258 1007851 1007108 1007524 1004965
1008269 1005832 1008148 1008923 1005603 1006470 1008423 1006367 1008078 1009023
""".split()


class TippySampleTests(unittest.TestCase):
    def test_tippy_template_uses_a_constant_row_pitch(self) -> None:
        import numpy as np

        cards = app.template_cards(np.zeros((3202, 2160, 3), dtype=np.uint8), "tippy")
        row_ys = [cards[index].y for index in range(0, 30, 5)]
        pitches = [right - left for left, right in zip(row_ys, row_ys[1:])]
        self.assertLessEqual(max(pitches) - min(pitches), 1)

    def test_noisy_titles_are_corrected_from_local_catalog(self) -> None:
        self.assertEqual(app.canonical_song_title("Airdn"), "Air")
        self.assertEqual(app.canonical_song_title("Trackless wildermness"), "Trackless wilderness")
        self.assertEqual(app.canonical_song_title("ホ一リ一サソバラソド"), "ホーリーサンバランド")
        self.assertEqual(app.canonical_song_title("チラツアーー"), "キミツアー→")
        self.assertEqual(app.canonical_song_title("Xpovos"), "χρόνος")
        self.assertEqual(app.canonical_song_title("勘滅"), "勦滅")

    def test_score_plausibility(self) -> None:
        self.assertTrue(app.plausible_report_score("998931"))
        self.assertTrue(app.plausible_report_score("1007411"))
        self.assertFalse(app.plausible_report_score("1998931"))
        self.assertFalse(app.plausible_report_score("97263"))

    def test_direct_title_can_restore_a_lost_prefix(self) -> None:
        self.assertTrue(app.prefer_direct_title(".Fracture.", "水晶世界Fracture"))
        self.assertTrue(app.prefer_direct_title("Paradox of", "献身Paradoxof"))
        self.assertFalse(app.prefer_direct_title("Air", "Airdorkee"))

    def recognize_scores(self, variable: str) -> list[str]:
        value = os.getenv(variable)
        if not value:
            self.skipTest(f"set {variable} to run the OCR regression sample")
        path = Path(value)
        image = app.decode_image(path.read_bytes())
        cards = app.reading_order(app.detect_cards(image, "tippy"))[:50]
        return [item["score"] for item in app.parse_tippy_report(image, cards)]

    def test_low_resolution_report_scores(self) -> None:
        self.assertEqual(self.recognize_scores("TIPPY_SAMPLE_1"), SAMPLE_1_SCORES)

    def test_high_resolution_report_scores(self) -> None:
        self.assertEqual(self.recognize_scores("TIPPY_SAMPLE_2"), SAMPLE_2_SCORES)


if __name__ == "__main__":
    unittest.main()

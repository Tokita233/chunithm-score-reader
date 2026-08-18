from __future__ import annotations

import base64
import io
import re
import threading
import webbrowser
from dataclasses import dataclass

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request
from PIL import Image
from rapidocr import EngineType, LangDet, LangRec, ModelType, OCRVersion, RapidOCR


app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024
# Use the dedicated Japanese recognizer rather than a generic CJK model.  Its
# dictionary includes kana, kanji, Latin letters and digits, which matches
# CHUNITHM titles better (especially tiny hiragana/katakana in report cards).
ocr = RapidOCR(
    params={
        "Det.engine_type": EngineType.ONNXRUNTIME,
        "Det.lang_type": LangDet.CH,
        "Det.model_type": ModelType.MOBILE,
        "Det.ocr_version": OCRVersion.PPOCRV4,
        "Rec.engine_type": EngineType.ONNXRUNTIME,
        "Rec.lang_type": LangRec.JAPAN,
        "Rec.model_type": ModelType.MOBILE,
        "Rec.ocr_version": OCRVersion.PPOCRV4,
    }
)


@dataclass
class Card:
    x: int
    y: int
    w: int
    h: int


def reading_order(cards: list[Card]) -> list[Card]:
    """Book-like order: group nearby cards into rows, then read each row L→R."""
    rows: list[list[Card]] = []
    for card in sorted(cards, key=lambda item: item.y + item.h / 2):
        center_y = card.y + card.h / 2
        matching_row = None
        for row in rows:
            row_center = sum(item.y + item.h / 2 for item in row) / len(row)
            if abs(center_y - row_center) <= max(card.h, row[0].h) * 0.45:
                matching_row = row
                break
        if matching_row is None:
            rows.append([card])
        else:
            matching_row.append(card)
    rows.sort(key=lambda row: sum(item.y for item in row) / len(row))
    return [card for row in rows for card in sorted(row, key=lambda item: item.x)]


def decode_image(raw: bytes) -> np.ndarray:
    image = Image.open(io.BytesIO(raw)).convert("RGB")
    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


def merge_intervals(values: list[int], tolerance: int) -> list[int]:
    groups: list[list[int]] = []
    for value in sorted(values):
        if not groups or value - groups[-1][-1] > tolerance:
            groups.append([value])
        else:
            groups[-1].append(value)
    return [round(sum(group) / len(group)) for group in groups]


def projection_centers(gray: np.ndarray, axis: int, expected: int) -> list[int]:
    """Find repeated dark card bands without assuming the input resolution."""
    dark = (gray < 118).astype(np.uint8)
    density = dark.mean(axis=axis)
    threshold = max(0.13, float(np.percentile(density, 72)))
    indices = np.flatnonzero(density > threshold).tolist()
    tolerance = max(3, int(len(density) * 0.004))
    groups: list[list[int]] = []
    for value in indices:
        if not groups or value - groups[-1][-1] > tolerance:
            groups.append([value])
        else:
            groups[-1].append(value)
    bands = [(g[0], g[-1]) for g in groups if g[-1] - g[0] > len(density) * 0.015]
    centers = [round((a + b) / 2) for a, b in bands]
    if len(centers) > expected:
        # The five widest/darkest column bands or the repeated row bands win.
        scored = sorted(
            centers,
            key=lambda c: density[max(0, c - tolerance):c + tolerance + 1].mean(),
            reverse=True,
        )[:expected]
        centers = sorted(scored)
    return centers


def layout_name(image: np.ndarray) -> str:
    """Distinguish the tall Tippy report from the squarer CHUNITHM MATE UI."""
    h, w = image.shape[:2]
    return "mate" if w / h >= 0.82 else "tippy"


def template_cards(image: np.ndarray) -> list[Card]:
    """Normalized coordinates for both supported five-column report UIs."""
    h, w = image.shape[:2]
    if layout_name(image) == "mate":
        # CHUNITHM MATE Best 50 Information: 30 cards above, 20 below.
        xs = [0.0167, 0.2111, 0.4056, 0.6000, 0.7944]
        top_ys = [0.16625, 0.24125, 0.31625, 0.39125, 0.46625, 0.54125]
        current_ys = [0.6406, 0.7156, 0.7906, 0.8656]
        cw, ch = round(w * 0.1875), round(h * 0.0675)
        return [
            Card(round(w * x), round(h * y), cw, ch)
            for y in top_ys + current_ys
            for x in xs
        ]

    xs = [0.020, 0.215, 0.410, 0.605, 0.800]
    top_ys = [0.252, 0.316, 0.381, 0.446, 0.511, 0.575]
    current_ys = [0.726, 0.790, 0.855, 0.919]
    cw, ch = round(w * 0.184), round(h * 0.054)
    return [
        Card(round(w * x), round(h * y), cw, ch)
        for y in top_ys + current_ys
        for x in xs
    ]


def detect_cards(image: np.ndarray) -> list[Card]:
    """Detect the 5-column layout; fall back to normalized template coordinates."""
    h, w = image.shape[:2]
    # The MATE layout is highly regular and has colored cards on a pale
    # background, where dark-pixel contour detection is less reliable.
    if layout_name(image) == "mate":
        return template_cards(image)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Ignore header/banner regions when finding the five repeated columns.
    body = gray[int(h * 0.22):, :]
    x_centers = projection_centers(body, axis=0, expected=5)
    if len(x_centers) != 5:
        return template_cards(image)

    # Card rows are short, dark horizontal bands. Locate connected components
    # after a wide horizontal close, then merge nearly identical y positions.
    dark = cv2.inRange(gray, 0, 130)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(9, w // 70), max(3, h // 800)))
    joined = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(joined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    ys: list[int] = []
    for contour in contours:
        x, y, cw, ch = cv2.boundingRect(contour)
        if y > h * 0.20 and w * 0.12 < cw < w * 0.24 and h * 0.025 < ch < h * 0.075:
            ys.append(y)
    row_ys = merge_intervals(ys, max(8, int(h * 0.012)))
    if len(row_ys) < 7:
        return template_cards(image)

    cw, ch = round(w * 0.184), round(h * 0.054)
    lefts = [max(0, round(c - cw / 2)) for c in x_centers]
    cards = [Card(x, y, cw, ch) for y in row_ys for x in lefts]
    # Reading order is always row-major: left to right, then top to bottom.
    return reading_order(cards)[:55]


def text_lines(crop: np.ndarray) -> list[tuple[str, float, float]]:
    result = ocr(crop)
    lines: list[tuple[str, float, float]] = []
    if result.boxes is None:
        return lines
    for box, text, confidence in zip(result.boxes, result.txts, result.scores):
        y = float(np.mean([p[1] for p in box]))
        lines.append((str(text).strip(), float(confidence), y))
    return sorted(lines, key=lambda row: row[2])


def classify_difficulty(crop: np.ndarray) -> str:
    h, w = crop.shape[:2]
    # Difficulty/constant pill sits on the top band, right of the #rank badge.
    pill = crop[0:max(8, int(h * 0.22)), int(w * 0.47):int(w * 0.79)]
    hsv = cv2.cvtColor(pill, cv2.COLOR_BGR2HSV)
    hue, sat, val = cv2.split(hsv)
    vivid = (sat > 70) & (val > 70)
    purple = vivid & (hue >= 125) & (hue <= 170)
    red = vivid & ((hue <= 12) | (hue >= 172))
    dark = val < 75
    if purple.mean() > 0.07:
        return "Master"
    if red.mean() > 0.018 and dark.mean() > 0.22:
        return "Ultima"
    if red.mean() > 0.06:
        return "Expert"
    green = vivid & (hue >= 35) & (hue <= 90)
    if green.mean() > 0.07:
        return "Advanced"
    return "Master"


def build_card_result(
    crop: np.ndarray,
    index: int,
    lines: list[tuple[str, float, float]],
    panel_height: int,
) -> dict:
    raw_text = [text for text, confidence, _ in lines if confidence >= 0.35]

    score = ""
    score_candidates: list[str] = []
    for text in raw_text:
        digits = re.sub(r"\D", "", text)
        if 6 <= len(digits) <= 8:
            score_candidates.append(digits)
    if score_candidates:
        candidate = max(score_candidates, key=lambda value: (len(value), value))
        # Reports display e.g. 100,7649, meaning score 1,007,649.
        if len(candidate) == 7:
            score = candidate
        elif len(candidate) == 8 and candidate.startswith("100"):
            score = candidate[:3] + candidate[4:]

    ignored = re.compile(
        r"^(#?\d+|ID\s*\d+|SSS\+?|FULL\s*COMBO|HARD|CLEAR|"
        r"MASTER|ULTIMA|EXPERT|ADVANCED|BASIC|\d+[.,]\d+)$",
        re.I,
    )
    names = []
    for text, confidence, y in lines:
        cleaned = re.sub(r"\s+", " ", text).strip(" |")
        if confidence < 0.35 or ignored.match(cleaned):
            continue
        if len(re.sub(r"[^A-Za-z\u3040-\u30ff\u3400-\u9fff]", "", cleaned)) < 2:
            continue
        if y < panel_height * 0.12 or y > panel_height * 0.52:
            continue
        names.append(cleaned)
    name = max(names, key=len) if names else ""

    ok, encoded = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 82])
    preview = "data:image/jpeg;base64," + base64.b64encode(encoded).decode() if ok else ""
    return {
        "index": index,
        "score": score,
        "name": name,
        "difficulty": classify_difficulty(crop),
        "preview": preview,
        "raw": raw_text,
    }


def parse_card(crop: np.ndarray, index: int, layout: str = "tippy") -> dict:
    h, w = crop.shape[:2]
    panel_start = 0.275 if layout == "mate" else 0.34
    panel = crop[:, int(w * panel_start):]
    panel = cv2.resize(panel, None, fx=2.4, fy=2.4, interpolation=cv2.INTER_CUBIC)
    return build_card_result(crop, index, text_lines(panel), panel.shape[0])


def parse_cards_batched(image: np.ndarray, cards: list[Card], layout: str) -> list[dict]:
    """OCR five cards together so cloud CPUs finish well inside proxy limits."""
    crops: list[np.ndarray] = []
    panels: list[np.ndarray] = []
    panel_start = 0.275 if layout == "mate" else 0.34
    for card in cards:
        crop = image[
            card.y:min(card.y + card.h, image.shape[0]),
            card.x:min(card.x + card.w, image.shape[1]),
        ]
        if not crop.size:
            continue
        crops.append(crop)
        panel = crop[:, int(crop.shape[1] * panel_start):]
        panels.append(cv2.resize(panel, None, fx=2.4, fy=2.4, interpolation=cv2.INTER_CUBIC))

    all_lines: list[list[tuple[str, float, float]]] = [[] for _ in panels]
    batch_size, cols, gap = 5, 2, 16
    for start in range(0, len(panels), batch_size):
        batch = panels[start:start + batch_size]
        tile_w = max(panel.shape[1] for panel in batch)
        tile_h = max(panel.shape[0] for panel in batch)
        rows = (len(batch) + cols - 1) // cols
        canvas = np.full(
            (rows * tile_h + (rows + 1) * gap, cols * tile_w + (cols + 1) * gap, 3),
            255,
            dtype=np.uint8,
        )
        placements: list[tuple[int, int, int, int]] = []
        for local_index, panel in enumerate(batch):
            row, col = divmod(local_index, cols)
            x0 = gap + col * (tile_w + gap)
            y0 = gap + row * (tile_h + gap)
            canvas[y0:y0 + panel.shape[0], x0:x0 + panel.shape[1]] = panel
            placements.append((x0, y0, panel.shape[1], panel.shape[0]))

        output = ocr(canvas)
        if output.boxes is None:
            continue
        for box, text, confidence in zip(output.boxes, output.txts, output.scores):
            cx = float(np.mean([point[0] for point in box]))
            cy = float(np.mean([point[1] for point in box]))
            for local_index, (x0, y0, pw, ph) in enumerate(placements):
                if x0 <= cx <= x0 + pw and y0 <= cy <= y0 + ph:
                    all_lines[start + local_index].append(
                        (str(text).strip(), float(confidence), cy - y0)
                    )
                    break

    return [
        build_card_result(crop, index, sorted(lines, key=lambda row: row[2]), panel.shape[0])
        for index, (crop, panel, lines) in enumerate(zip(crops, panels, all_lines), 1)
    ]


def parse_cards_direct(image: np.ndarray, cards: list[Card], layout: str) -> list[dict]:
    """Read fixed title/score bands directly; avoids costly text detection."""
    crops: list[np.ndarray] = []
    title_images: list[np.ndarray] = []
    score_images: list[np.ndarray] = []
    if layout == "mate":
        title_box = (0.275, 0.045, 0.995, 0.34)
        score_box = (0.28, 0.40, 0.75, 0.70)
    else:
        # The old Tippy Bot card has a separate rank header above the title.
        # Keep below that header, and include the full score width so its last
        # digit is not clipped off.
        title_box = (0.355, 0.18, 0.995, 0.39)
        score_box = (0.35, 0.35, 0.995, 0.71)

    def band(crop: np.ndarray, box: tuple[float, float, float, float]) -> np.ndarray:
        h, w = crop.shape[:2]
        x1, y1, x2, y2 = box
        roi = crop[int(h * y1):max(int(h * y2), int(h * y1) + 1),
                   int(w * x1):max(int(w * x2), int(w * x1) + 1)]
        return cv2.resize(roi, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)

    for card in cards:
        crop = image[
            card.y:min(card.y + card.h, image.shape[0]),
            card.x:min(card.x + card.w, image.shape[1]),
        ]
        if not crop.size:
            continue
        crops.append(crop)
        title_images.append(band(crop, title_box))
        score_image = band(crop, score_box)
        score_images.append(cv2.copyMakeBorder(
            score_image, 12, 12, 20, 20, cv2.BORDER_CONSTANT, value=(255, 255, 255)
        ))

    title_output = ocr.recognize_txt(title_images)
    score_output = ocr.recognize_txt(score_images)
    results: list[dict] = []
    for index, (crop, raw_name, raw_score) in enumerate(
        zip(crops, title_output.txts, score_output.txts), 1
    ):
        name = re.sub(r"\s+(MASTER|ULTIMA|EXPERT|ADVANCED|BASIC).*$", "", str(raw_name), flags=re.I)
        name = re.sub(r"\s+", " ", name).strip(" .|_-\u3000")
        name = re.sub(r"ー{2,}$", "ー", name)
        digits = re.sub(r"\D", "", str(raw_score))
        score = ""
        if len(digits) in (6, 7):
            score = digits
        elif len(digits) == 8 and digits.startswith("100"):
            score = digits[:3] + digits[4:]
        elif len(digits) > 7:
            candidates = re.findall(r"\d{7,8}", digits)
            if candidates:
                value = candidates[0]
                score = value if len(value) == 7 else value[:3] + value[4:]

        score_is_valid = score.isdigit() and 900000 <= int(score) <= 1010000
        name_letters = re.sub(r"[^A-Za-z\u3040-\u30ff\u3400-\u9fff]", "", name)
        if not score_is_valid or len(name_letters) < 2:
            # A few old cards use artwork/contrast that confuses direct line
            # recognition. Run the slower detector only for those cards.
            fallback = parse_card(crop, index, layout)
            fallback_score = fallback["score"]
            if (
                fallback_score.isdigit()
                and 900000 <= int(fallback_score) <= 1010000
            ):
                score = fallback_score
            if fallback["name"]:
                name = fallback["name"]

        ok, encoded = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 82])
        preview = "data:image/jpeg;base64," + base64.b64encode(encoded).decode() if ok else ""
        results.append({
            "index": index,
            "score": score,
            "name": name,
            "difficulty": classify_difficulty(crop),
            "preview": preview,
            "raw": [str(raw_name), str(raw_score)],
        })
    return results


@app.get("/")
def home():
    return render_template("index.html")


@app.post("/api/recognize")
def recognize():
    upload = request.files.get("image")
    if upload is None:
        return jsonify({"error": "请选择图片"}), 400
    try:
        image = decode_image(upload.read())
        expected = int(request.form.get("expected", "45"))
        cards = reading_order(detect_cards(image))
        if expected in (45, 50):
            cards = cards[:expected]
        results = parse_cards_direct(image, cards, layout_name(image))
        return jsonify({"count": len(results), "layout": layout_name(image), "items": results})
    except Exception as exc:
        return jsonify({"error": f"识别失败：{exc}"}), 500


if __name__ == "__main__":
    threading.Timer(1.2, lambda: webbrowser.open("http://127.0.0.1:7860")).start()
    app.run(host="127.0.0.1", port=7860, debug=False)

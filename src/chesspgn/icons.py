"""Find figurine icons on page images and attach their shape group to the moves in the reader's text.

No AI and no per-book setup: Tesseract marks a few icons (a character sitting right before a square), those
are grouped by shape, each real group becomes a template, and template matching finds every other icon on the
page. The groups have no names; the decoder learns which piece each group is.
"""

import re
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pytesseract

from chesspgn.notation import GROUP_SYMBOLS

GROUP_THRESHOLD = 0.75  # normalized correlation to join a shape group
MATCH_THRESHOLD = 0.72  # template-matching score for an icon occurrence
OVERLAP_LIMIT = 0.3  # IoU above which two hits are the same icon
MIN_GROUP_SIZE = 2  # single seeds are mostly noise (letters, diagram shading)
SQUARE_CROP_WIDTH = 3.6  # icon widths to the right that hold the move text

# Alignment costs (order-preserving, text moves vs icons)
SAME_TEXT = 0.0
SAME_SQUARE = 0.5  # "d4" vs "xd4"
MISREAD = 1.0
TEXT_WITHOUT_ICON = 0.3  # pawn moves have no icon
ICON_WITHOUT_TEXT = 1.0

_NOT_ICON = set("abcdefgh012345678.[]();:!?+=-,x ½'\"")
_SQUARE_AHEAD = re.compile(r"x?([a-h][1-8])")
_MOVE_TEXT = re.compile(r"[a-h]?x?[a-h][1-8]")
_SQUARE_IN_TEXT = re.compile(r"(?<![A-Za-z])[a-h]?x?[a-h][1-8]")


@dataclass(frozen=True)
class IconHit:
    image_index: int
    box: tuple[int, int, int, int]  # x1, y1, x2, y2 in image pixels
    group: int
    square_text: str | None  # move text printed right after the icon, e.g. "f3", "xd4", "de2"
    order: tuple[int, ...]  # reading order: image, block, paragraph, line, x


def find_icons(images: list[Path]) -> list[IconHit]:
    pages = [cv2.imread(str(p)) for p in images]
    seeds = [crop for page in pages for crop in _seed_icons(page)]
    templates = _templates(seeds)
    if not templates:
        return []
    hits: list[IconHit] = []
    for index, page in enumerate(pages):
        words = pytesseract.image_to_data(page, output_type=pytesseract.Output.DICT)
        gray = cv2.cvtColor(page, cv2.COLOR_BGR2GRAY)
        for x1, y1, x2, y2, group in _match(gray, templates):
            square = _read_square(gray, x1, y1, x2, y2)
            order = (index, *_reading_position(words, x1, y1, y2))
            hits.append(IconHit(index, (x1, y1, x2, y2), group, square, order))
    return sorted(hits, key=lambda h: h.order)


# ---------------------------------------------------------------- finding


def _seed_icons(page: np.ndarray) -> list[np.ndarray]:
    """Crops of characters that sit directly before a move's square (Tesseract's letters for icons vary)."""
    height = page.shape[0]
    boxes = []
    for line in pytesseract.image_to_boxes(page).splitlines():
        parts = line.split(" ")
        if len(parts) < 5:
            continue
        ch, x1, y1, x2, y2 = parts[0], int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4])
        boxes.append((ch, x1, height - y2, x2, height - y1))  # top-left origin
    crops = []
    for i, (ch, x1, y1, x2, y2) in enumerate(boxes[:-1]):
        if ch in _NOT_ICON:
            continue
        ahead = "".join(b[0] for b in boxes[i + 1:i + 4])
        nxt = boxes[i + 1]
        if _SQUARE_AHEAD.match(ahead) and abs(nxt[2] - y1) <= 25 and nxt[1] - x2 <= 15:
            crops.append(page[max(0, y1 - 2):y2 + 2, max(0, x1 - 2):x2 + 2])
    return [c for c in crops if c.size]


def _vector(crop: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    v = cv2.resize(gray, (24, 24), interpolation=cv2.INTER_AREA).astype(np.float32).ravel()
    v -= v.mean()
    norm = np.linalg.norm(v)
    return v / norm if norm else v


def _templates(seeds: list[np.ndarray]) -> list[np.ndarray]:
    groups: list[list[np.ndarray]] = []
    centroids: list[np.ndarray] = []
    for crop in seeds:
        v = _vector(crop)
        scores = [float(np.dot(v, c / np.linalg.norm(c))) for c in centroids]
        best = int(np.argmax(scores)) if scores else -1
        if best >= 0 and scores[best] >= GROUP_THRESHOLD:
            groups[best].append(crop)
            centroids[best] = centroids[best] + v
        else:
            groups.append([crop])
            centroids.append(v.copy())
    templates = []
    for members in groups:
        if len(members) < MIN_GROUP_SIZE:
            continue
        h = int(np.median([m.shape[0] for m in members]))
        w = int(np.median([m.shape[1] for m in members]))
        stack = [cv2.resize(cv2.cvtColor(m, cv2.COLOR_BGR2GRAY), (w, h)).astype(np.float32) for m in members]
        templates.append(np.mean(stack, axis=0).astype(np.uint8))
    return templates


def _match(gray: np.ndarray, templates: list[np.ndarray]) -> list[tuple[int, int, int, int, int]]:
    hits = []
    for group, template in enumerate(templates):
        if template.shape[0] > gray.shape[0] or template.shape[1] > gray.shape[1]:
            continue
        scores = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
        ys, xs = np.where(scores >= MATCH_THRESHOLD)
        for x, y in zip(xs, ys):
            hits.append((float(scores[y, x]), int(x), int(y), int(x + template.shape[1]), int(y + template.shape[0]), group))
    hits.sort(key=lambda h: -h[0])
    kept: list[tuple[int, int, int, int, int]] = []
    for _score, x1, y1, x2, y2, group in hits:
        if all(_iou((x1, y1, x2, y2), k[:4]) < OVERLAP_LIMIT for k in kept):
            kept.append((x1, y1, x2, y2, group))
    return kept


def _iou(a: tuple[int, ...], b: tuple[int, ...]) -> float:
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union else 0.0


def _read_square(gray: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> str | None:
    crop = gray[max(0, y1 - 3):y2 + 3, x2:x2 + int(SQUARE_CROP_WIDTH * (x2 - x1))]
    if crop.size == 0:
        return None
    text = pytesseract.image_to_string(crop, config="--psm 7 -c tessedit_char_whitelist=abcdefghx12345678").strip()
    match = _MOVE_TEXT.match(text)
    return match.group(0) if match else None


def _reading_position(words: dict, x1: int, y1: int, y2: int) -> tuple[int, int, int, int]:
    """Tesseract's reading order (block, paragraph, line) of the word under this icon, then x."""
    middle = (y1 + y2) / 2
    best, best_distance = None, float("inf")
    for i, text in enumerate(words["text"]):
        if not text.strip():
            continue
        left, top, width, height = words["left"][i], words["top"][i], words["width"][i], words["height"][i]
        if not top - 5 <= middle <= top + height + 5:
            continue
        distance = 0 if left - 10 <= x1 <= left + width else min(abs(x1 - left), abs(x1 - left - width))
        if distance < best_distance:
            best, best_distance = i, distance
    if best is None:
        return (9999, 0, 0, x1)
    return (words["block_num"][best], words["par_num"][best], words["line_num"][best], x1)


# ---------------------------------------------------------------- linking


def annotate(text: str, hits: list[IconHit]) -> str:
    """Put each icon's group symbol in front of the move it belongs to in the reader's text."""
    if not hits:
        return text
    spots = [(m.start(), m.group(0)) for m in _SQUARE_IN_TEXT.finditer(text)]
    pairs = _align([s for _p, s in spots], [h.square_text or "??" for h in hits])
    inserts = {spots[i][0]: GROUP_SYMBOLS[hits[j].group % len(GROUP_SYMBOLS)] for i, j in pairs}
    out, last = [], 0
    for position in sorted(inserts):
        out.append(text[last:position])
        out.append(inserts[position])
        last = position
    out.append(text[last:])
    return "".join(out)


def _align(text_moves: list[str], icon_moves: list[str]) -> list[tuple[int, int]]:
    """Order-preserving alignment (dynamic programming); returns matched (text index, icon index) pairs."""
    n, m = len(text_moves), len(icon_moves)
    cost = [[0.0] * (m + 1) for _ in range(n + 1)]
    step = [[""] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        cost[i][0], step[i][0] = cost[i - 1][0] + TEXT_WITHOUT_ICON, "text"
    for j in range(1, m + 1):
        cost[0][j], step[0][j] = cost[0][j - 1] + ICON_WITHOUT_TEXT, "icon"
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost[i][j], step[i][j] = min(
                (cost[i - 1][j - 1] + _pair_cost(text_moves[i - 1], icon_moves[j - 1]), "pair"),
                (cost[i - 1][j] + TEXT_WITHOUT_ICON, "text"),
                (cost[i][j - 1] + ICON_WITHOUT_TEXT, "icon"),
            )
    pairs, i, j = [], n, m
    while i > 0 or j > 0:
        if step[i][j] == "pair":
            pairs.append((i - 1, j - 1))
            i, j = i - 1, j - 1
        elif step[i][j] == "text":
            i -= 1
        else:
            j -= 1
    return pairs[::-1]


def _pair_cost(text_move: str, icon_move: str) -> float:
    if text_move == icon_move:
        return SAME_TEXT
    if icon_move != "??" and text_move[-2:] == icon_move[-2:]:
        return SAME_SQUARE
    return MISREAD

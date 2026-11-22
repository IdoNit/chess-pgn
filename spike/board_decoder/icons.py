"""Find figurine icons with Tesseract char boxes, cut them out, group them by shape."""
import re
import cv2
import numpy as np
import pytesseract

_NOT_ICON = set("abcdefgh012345678.[]();:!?+=-,x ½'\"")


def find_icons(img):
    """[(crop, (x1,y1,x2,y2), square_after)] for every char that sits right before a move's square."""
    h = img.shape[0]
    boxes = []
    for line in pytesseract.image_to_boxes(img).splitlines():
        ch, x1, y1, x2, y2, _ = line.split(" ")
        boxes.append((ch, int(x1), h - int(y2), int(x2), h - int(y1)))  # top-left origin
    icons = []
    for i, (ch, x1, y1, x2, y2) in enumerate(boxes):
        if ch in _NOT_ICON:
            continue
        rest = "".join(b[0] for b in boxes[i + 1:i + 4])
        m = re.match(r"x?([a-h][1-8])", rest)
        if m is None:
            continue
        nxt = boxes[i + 1]
        if abs(nxt[2] - y1) > 25 or nxt[1] - x2 > 15:  # must be on the same line, right next to the square
            continue
        icons.append((img[max(0, y1 - 2):y2 + 2, max(0, x1 - 2):x2 + 2], (x1, y1, x2, y2), m.group(1)))
    return icons


def _vec(crop):
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, (24, 24), interpolation=cv2.INTER_AREA).astype(np.float32)
    g = g - g.mean()
    n = np.linalg.norm(g)
    return g.ravel() / n if n else g.ravel()


def group(icons, threshold=0.75):
    """Greedy grouping by normalized correlation; returns a group id per icon."""
    reps: list[list[np.ndarray]] = []
    ids = []
    for crop, _box, _sq in icons:
        v = _vec(crop)
        scores = [float(np.dot(v, np.mean(r, axis=0) / np.linalg.norm(np.mean(r, axis=0)))) for r in reps]
        best = int(np.argmax(scores)) if scores else -1
        if best >= 0 and scores[best] >= threshold:
            reps[best].append(v)
            ids.append(best)
        else:
            reps.append([v])
            ids.append(len(reps) - 1)
    return ids


def templates(icons, ids, min_members=2):
    """Average image of each real group (singletons are mostly noise: letters, diagram shading)."""
    out = []
    for g in sorted(set(ids)):
        members = [icons[i][0] for i in range(len(icons)) if ids[i] == g]
        if len(members) < min_members:
            continue
        hs = int(np.median([m.shape[0] for m in members])); ws = int(np.median([m.shape[1] for m in members]))
        stack = [cv2.resize(cv2.cvtColor(m, cv2.COLOR_BGR2GRAY), (ws, hs)).astype(np.float32) for m in members]
        out.append(np.mean(stack, axis=0).astype(np.uint8))
    return out


def match_all(img, temps, threshold=0.72):
    """Every occurrence of every template on the page: [(x1, y1, x2, y2, group, score)], overlaps removed."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hits = []
    for g, t in enumerate(temps):
        res = cv2.matchTemplate(gray, t, cv2.TM_CCOEFF_NORMED)
        ys, xs = np.where(res >= threshold)
        for x, y in zip(xs, ys):
            hits.append((int(x), int(y), int(x + t.shape[1]), int(y + t.shape[0]), g, float(res[y, x])))
    hits.sort(key=lambda h: -h[5])
    kept = []
    for h in hits:
        if all(_iou(h, k) < 0.3 for k in kept):
            kept.append(h)
    return kept


def _iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter)


CLASS_SYMBOLS = "①②③④⑤⑥⑦⑧"
_SQUARE_IN_TEXT = re.compile(r"(?<![A-Za-z])[a-h]?x?[a-h][1-8]")


def read_squares(img, hits):
    """For each icon, the move text printed right after it (e.g. 'f3', 'xd4', 'de2'), in reading order."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
    words = [i for i in range(len(data["text"])) if data["text"][i].strip()]
    out = []
    for x1, y1, x2, y2, g, _s in hits:
        w = x2 - x1
        crop = gray[max(0, y1 - 3):y2 + 3, x2:x2 + int(3.6 * w)]
        txt = pytesseract.image_to_string(
            crop, config="--psm 7 -c tessedit_char_whitelist=abcdefghx12345678"
        ).strip()
        m = re.match(r"[a-h]?x?[a-h][1-8]", txt)
        order = _reading_position(data, words, x1, y1, x2, y2)
        out.append((order, m.group(0) if m else None, g))
    out.sort(key=lambda t: t[0])
    return [(sq, g) for _o, sq, g in out]


def _reading_position(data, words, x1, y1, x2, y2):
    """Tesseract's own reading order (block, paragraph, line) of the word under this icon, then x."""
    cy = (y1 + y2) / 2
    best, best_d = None, 1e9
    for i in words:
        l, t, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
        if t - 5 <= cy <= t + h + 5:
            d = 0 if l - 10 <= x1 <= l + w else min(abs(x1 - l), abs(x1 - (l + w)))
            if d < best_d:
                best, best_d = i, d
    if best is None:
        return (9999, 0, 0, x1)
    return (data["block_num"][best], data["par_num"][best], data["line_num"][best], x1)


def _core(s):
    return s[-2:]


GAP_TEXT = 0.3  # a move in the text without an icon: normal (pawn moves)
GAP_ICON = 1.0  # an icon that matches no move in the text
SAME_SQUARE = 0.5
MISREAD = 1.0  # icon's square read differently from the text's (one of the two OCRs misread it)


def _align(a: list[str], b: list[str]) -> list[tuple[int, int]]:
    """Order-preserving alignment of text moves (a) with icon squares (b); returns matched (i, j) pairs."""
    n, m = len(a), len(b)
    cost = [[0.0] * (m + 1) for _ in range(n + 1)]
    back = [[None] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        cost[i][0], back[i][0] = cost[i - 1][0] + GAP_TEXT, "a"
    for j in range(1, m + 1):
        cost[0][j], back[0][j] = cost[0][j - 1] + GAP_ICON, "b"
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if a[i - 1] == b[j - 1]:
                match = 0.0
            elif b[j - 1] != "??" and _core(a[i - 1]) == _core(b[j - 1]):
                match = SAME_SQUARE  # "d4" vs "xd4": same square, prefix differs
            else:
                match = MISREAD
            options = [
                (cost[i - 1][j - 1] + match, "m"),
                (cost[i - 1][j] + GAP_TEXT, "a"),
                (cost[i][j - 1] + GAP_ICON, "b"),
            ]
            cost[i][j], back[i][j] = min(options)
    pairs, i, j = [], n, m
    while i > 0 or j > 0:
        step = back[i][j]
        if step == "m":
            pairs.append((i - 1, j - 1)); i -= 1; j -= 1
        elif step == "a":
            i -= 1
        else:
            j -= 1
    return pairs[::-1]


def annotate_text(text, icon_squares):
    """Put a group symbol in front of each move in `text` that had an icon (order-preserving alignment)."""
    spots = [(m.start(), m.group(0)) for m in _SQUARE_IN_TEXT.finditer(text)]
    pairs = _align([s for _p, s in spots], [sq if sq else "??" for sq, _g in icon_squares])
    inserts = {spots[i][0]: CLASS_SYMBOLS[icon_squares[j][1]] for i, j in pairs}
    out, last = [], 0
    for pos in sorted(inserts):
        out.append(text[last:pos]); out.append(inserts[pos]); last = pos
    out.append(text[last:])
    return "".join(out), len(inserts), len(icon_squares)

import json
import re
from pathlib import Path

STATIC = Path(__file__).parent.parent / "src" / "chesspgn" / "web" / "static"
REASONS = ["ocr", "icon", "unreadable", "missing", "close_call", "time_budget", "skipped", "number", "pin_dropped", "ai_read", "depends"]


def strings() -> dict[str, str]:
    return json.loads((STATIC / "strings.he.json").read_text(encoding="utf-8"))


def used_keys() -> set[str]:
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    js = (STATIC / "app.js").read_text(encoding="utf-8")
    return set(re.findall(r'data-i18n="([^"]+)"', html)) | set(re.findall(r'\bt\("([^"]+)"', js))


def test_every_used_key_exists():
    missing = used_keys() - set(strings())
    assert not missing, f"missing strings: {sorted(missing)}"


def test_every_flag_reason_has_text():
    s = strings()
    for reason in REASONS:
        assert f"reason.{reason}" in s
        assert f"reason.side:{reason}" in s


def test_error_messages():
    s = strings()
    assert s["error.no_game"] == "לא נמצא משחק בתמונה"
    assert s["error.engine_down"] == "מנוע הקריאה לא פועל – הפעילו את Docker ונסו שוב"


def test_page_is_rtl_hebrew():
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    assert 'lang="he"' in html and 'dir="rtl"' in html


def test_moves_and_board_are_ltr():
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    js = (STATIC / "app.js").read_text(encoding="utf-8")
    assert 'id="moves" dir="ltr"' in html and 'id="board" dir="ltr"' in html
    assert "<bdi" in js  # Latin names inside Hebrew text are isolated


def test_review_fix_features_present():
    js = (STATIC / "app.js").read_text(encoding="utf-8")
    assert "AbortController" in js  # a hung conversion ends with a message (I5)
    assert "/api/download?ids=" in js  # combined download (I4)
    assert "/dismiss" in js  # confirm a flagged move (I3)
    s = strings()
    assert "error.timeout" in s and "error.illegal_pin" in s


def test_page_incomplete_message():
    assert "error.page_incomplete" in strings()

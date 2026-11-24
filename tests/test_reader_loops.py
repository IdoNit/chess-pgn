"""GLM-OCR can get stuck repeating itself on a page (dotted leaders, diagrams); Ollama then aborts with a 500."""

import json

import httpx
import pytest

from chesspgn.reader import OllamaReader, PageIncomplete
from tests.conftest import fixture_path

LOOP = httpx.Response(500, json={"error": "prediction aborted, token repeat limit reached"})


def reader_with(handler) -> OllamaReader:
    return OllamaReader("http://ollama.test", transport=httpx.MockTransport(handler))


def test_a_page_that_loops_is_read_again_with_a_repeat_penalty():
    sent = []

    def handler(request):
        sent.append(json.loads(request.content)["options"])
        if len(sent) == 1:
            return LOOP
        return httpx.Response(200, json={"message": {"content": "23. P-KR4"}, "done_reason": "stop"})

    assert reader_with(handler).read(fixture_path("stein.jpg")) == "23. P-KR4"
    assert "repeat_penalty" not in sent[0] and sent[1]["repeat_penalty"] > 1


def test_a_page_that_keeps_looping_is_incomplete_not_engine_down():
    with pytest.raises(PageIncomplete):
        reader_with(lambda request: LOOP).read(fixture_path("stein.jpg"))

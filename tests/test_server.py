import json
from contextlib import contextmanager
from http.server import ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from context_behavior import checkpoint, infer
from context_behavior.demo import app
from context_behavior.dialogue import DialogueRouter, FixedReply

SCENE = json.loads((Path(__file__).resolve().parents[1] / "demo" / "scene.json").read_text())


@contextmanager
def serve(handler):
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()


def post(base, path, body):
    request = Request(base + path, data=json.dumps(body).encode(), method="POST",
                      headers={"Content-Type": "application/json"})
    try:
        return json.load(urlopen(request))
    except HTTPError as error:
        return {"status": error.code, **json.load(error)}


class StubDialogue:
    name = "stub"

    def __init__(self):
        self.calls = []

    def reply(self, text, context=None):
        self.calls.append(text)
        return "stub reply"


def test_model_is_loaded_once_and_reused(tiny_checkpoint, monkeypatch):
    loads = []
    original = checkpoint.load_checkpoint

    def counting(path):
        loads.append(path)
        return original(path)

    monkeypatch.setattr(infer, "load_checkpoint", counting)
    infer._CACHE.clear()
    with serve(app(SCENE, tiny_checkpoint[0])) as base:
        for text in ("open the drawer", "hello there", "sit on the chair"):
            result = post(base, "/api/interpret", {"text": text})
            assert result["backend"].startswith("trained BERT") and "dispatch" in result
    assert len(loads) == 1
    infer.predict(tiny_checkpoint[0], "close the window")
    assert len(loads) == 1


def test_dialogue_routing_and_state_endpoints():
    stub = StubDialogue()
    with serve(app(SCENE, None, dialogue=DialogueRouter(stub, FixedReply()))) as base:
        scene = json.load(urlopen(base + "/api/scene"))
        assert scene["backend"].startswith("labelled rule fallback") and len(scene["ontology"]["action"]) == 14
        chat = post(base, "/api/interpret", {"text": "How are you today?"})
        assert chat["dispatch"]["route"] == "conversation" and chat["dialogue"]["reply"] == "stub reply"
        assert stub.calls == ["How are you today?"]
        negated = post(base, "/api/interpret", {"text": "Don't open the window"})
        assert negated["dispatch"]["route"] == "conversation" and negated["dialogue"]["reply"].startswith("Okay")
        assert negated["states"]["objects"]["window"]["open"] is False and stub.calls == ["How are you today?"]
        sit = post(base, "/api/interpret", {"text": "Please sit on the chair"})
        assert sit["dispatch"]["command"]["behavior"] == "sit_on" and sit["states"]["agent"]["posture"] == "sit"
        reset = post(base, "/api/reset", {})
        assert reset["states"]["agent"]["posture"] == "stand"
        assert post(base, "/api/interpret", {"text": ""})["status"] == 400
        # Conversation replies keep the shared co-speech route available to the page.
        request = Request(base + "/api/beat-query", data=b'{"text":"hello","mode":"automatic"}', method="POST")
        try:
            urlopen(request)
        except HTTPError as error:
            assert error.code != 404
        module = urlopen(base + "/static/context-room.js").read().decode()
        assert "prepareApplicationMotion" in module and "stage.sit" in module

"""Browser room: Table 1 classification -> scene-grounded behaviour or dialogue reply."""
from __future__ import annotations

import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock
from urllib.parse import urlsplit

from .avatar_http import serve_avatar_asset
from .dialogue import build_dialogue
from .ontology import CONVERSATION, ENTITY_HEADS, Ontology
from .rules import RuleInterpreter
from .scene import BehaviorPlanner
from .speech_backend import SpeechBackend, speech_route

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from beat_runtime import serve_beat  # noqa: E402

LOCAL_MODULES = {"/static/voice-input.js", "/static/context-room.js"}


def parse_rules(text: str, scene: dict | None = None) -> tuple[str, dict[str, str]]:
    """Earlier helper kept for callers: (``conversation``|``action``, non-None class entities)."""
    prediction = RuleInterpreter().predict(text)
    intent = "conversation" if prediction["subject"] == CONVERSATION else "action"
    return intent, {head: prediction[head] for head in ENTITY_HEADS if prediction[head] != "None"}


def make_interpreter(model: Path | None):
    if model:
        from .infer import cached_interpreter
        return cached_interpreter(model)
    return RuleInterpreter()


def app(scene: dict, model: Path | None = None, *, interpreter=None, dialogue=None):
    interpreter = interpreter or make_interpreter(model)
    dialogue = dialogue or build_dialogue("fixed")
    ontology = getattr(interpreter, "ontology", None) or Ontology.load()
    initial = json.loads(json.dumps(scene))
    world = {"planner": BehaviorPlanner(initial, ontology)}
    lock = Lock()
    speech = SpeechBackend()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def send_json(self, value, status=200):
            body = json.dumps(value).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def send_file(self, path: Path, kind: str):
            body = path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", kind)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def scene_payload(self):
            planner = world["planner"]
            return {"scene": planner.public_scene(), "states": planner.states(), "backend": interpreter.backend,
                    "dialogue": dialogue.name, "ontology": {k: list(v) for k, v in ontology.classes.items()}}

        def do_GET(self):
            path = urlsplit(self.path).path
            if serve_beat(self, ROOT, "automatic"):
                return
            if path in LOCAL_MODULES:
                self.send_file(ROOT / path.lstrip("/"), "text/javascript")
            elif serve_avatar_asset(self, ROOT / "static"):
                return
            elif path == "/api/scene":
                with lock:
                    self.send_json(self.scene_payload())
            elif path == "/api/speech":
                self.send_json(speech.status())
            elif path == "/":
                self.send_file(ROOT / "demo" / "index.html", "text/html; charset=utf-8")
            else:
                self.send_error(404)

        def do_POST(self):
            if serve_beat(self, ROOT, "automatic"):
                return
            if speech_route(self, speech):
                return
            path = urlsplit(self.path).path
            if path == "/api/reset":
                with lock:
                    world["planner"] = BehaviorPlanner(json.loads(json.dumps(initial)), ontology)
                    self.send_json(self.scene_payload())
                return
            if path != "/api/interpret":
                self.send_error(404)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 16_000:
                    raise ValueError("request body must be 1-16000 bytes")
                text = str(json.loads(self.rfile.read(length))["text"]).strip()
                if not text or len(text) > 500:
                    raise ValueError("text must be 1-500 characters")
                prediction = interpreter.predict(text)
                with lock:
                    decision = world["planner"].plan(prediction)
                    states = world["planner"].states()
                reply = None
                if decision.route == "conversation":
                    reply = dialogue.respond(text, prediction)
                    decision.reply = reply["reply"]
                intent = "conversation" if prediction["subject"] == CONVERSATION else "action"
                self.send_json({"text": text, "prediction": prediction, "intent": intent,
                                "entities": {h: prediction[h] for h in ENTITY_HEADS},
                                "dispatch": decision.to_dict(), "dialogue": reply, "states": states,
                                "backend": interpreter.backend})
            except (ValueError, KeyError, RuntimeError, json.JSONDecodeError) as exc:
                self.send_json({"error": str(exc)}, 400)

    return Handler


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", default=str(ROOT / "demo" / "scene.json"))
    parser.add_argument("--model", type=Path, help="Checkpoint from context-behavior-train; omitted selects the labelled rule fallback")
    parser.add_argument("--port", type=int, default=8762)
    parser.add_argument("--dialogue", choices=["fixed", "openai", "command"], help="Conversation client (default: CONTEXT_DIALOGUE or fixed)")
    parser.add_argument("--dialogue-url", help="OpenAI-compatible base URL, e.g. http://127.0.0.1:11434/v1")
    parser.add_argument("--dialogue-model", help="Model name for the OpenAI-compatible endpoint")
    parser.add_argument("--dialogue-command", help="Local command that reads the utterance on stdin and prints a reply")
    args = parser.parse_args(argv)
    try:
        interpreter = make_interpreter(args.model)
        dialogue = build_dialogue(args.dialogue, args.dialogue_url, args.dialogue_model, args.dialogue_command)
    except ValueError as exc:
        parser.error(str(exc))
    scene = json.loads(Path(args.scene).read_text(encoding="utf-8"))
    print(f"Interpreter: {interpreter.backend}; dialogue: {dialogue.name}", flush=True)
    print(f"Context room at http://127.0.0.1:{args.port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), app(scene, interpreter=interpreter, dialogue=dialogue)).serve_forever()


if __name__ == "__main__":
    main()

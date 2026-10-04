"""Browser room for checked context-to-action dispatch."""
from __future__ import annotations

import argparse
import json
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .dispatcher import ActionDispatcher
from .speech_backend import SpeechBackend, speech_route

ROOT = Path(__file__).resolve().parents[2]


def parse_rules(text: str, scene: dict) -> tuple[str, dict[str, str]]:
    """Explicit example backend: phrase matching over the declared room inventory."""
    words = re.findall(r"[a-z]+", text.casefold())
    target = next((item["id"] for item in scene["objects"] if item["id"].replace("_", " ") in text.casefold()), None)
    action = next((name for name, tokens in {"open": ("open",), "close": ("close",),
                    "turn_on": ("turn", "on"), "turn_off": ("turn", "off"),
                    "move_to": ("go", "to"), "sit_on": ("sit",),
                    "lie_on": ("lie",), "bring": ("bring",)}.items() if all(token in words for token in tokens)), None)
    if not action and not target:
        return "conversation", {}
    entities = {}
    if action:
        entities["action"] = action
    if target:
        entities["target"] = target
    return "action", entities


def app(scene: dict, model: Path | None):
    dispatcher = ActionDispatcher(scene)
    speech = SpeechBackend()
    states = {item["id"]: item.get("initial_state", "idle") for item in scene["objects"]}

    class Handler(BaseHTTPRequestHandler):
        def send_json(self, value, status=200):
            body = json.dumps(value).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/api/scene":
                self.send_json({"scene": scene, "states": states, "backend": "trained BERT checkpoint" if model else "explicit authored rules"})
            elif self.path == "/api/speech":
                self.send_json(speech.status())
            elif self.path in {"/static/avatar.js", "/static/speech.js", "/static/voice-input.js", "/static/vendor/three.module.js"}:
                body = (ROOT / self.path.lstrip("/")).read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/javascript")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif self.path == "/":
                body = (ROOT / "demo" / "index.html").read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            else:
                self.send_error(404)

        def do_POST(self):
            if speech_route(self, speech):
                return
            if self.path != "/api/interpret":
                self.send_error(404)
                return
            try:
                text = str(json.loads(self.rfile.read(int(self.headers["Content-Length"]))) ["text"])
                if model:
                    from .infer import predict
                    intent, entities = predict(model, text)
                else:
                    intent, entities = parse_rules(text, scene)
                decision = dispatcher.dispatch(intent, entities)
                if decision.accepted and decision.command:
                    states[decision.command["target"]] = decision.command["action"]
                self.send_json({"intent": intent, "entities": entities, "dispatch": vars(decision), "states": states,
                                "backend": "trained BERT checkpoint" if model else "explicit authored rules"})
            except (ValueError, KeyError, RuntimeError) as exc:
                self.send_json({"error": str(exc)}, 400)

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", default=str(ROOT / "demo" / "scene.json"))
    parser.add_argument("--model", type=Path, help="Trained checkpoint; omitted selects transparent rules")
    parser.add_argument("--port", type=int, default=8762)
    args = parser.parse_args()
    if args.model and not (args.model / "model.pt").exists():
        parser.error("--model must contain a trained model.pt checkpoint")
    scene = json.loads(Path(args.scene).read_text(encoding="utf-8"))
    print(f"Context room at http://127.0.0.1:{args.port}")
    ThreadingHTTPServer(("127.0.0.1", args.port), app(scene, args.model)).serve_forever()


if __name__ == "__main__":
    main()

"""Pluggable dialogue clients for conversation-class (Subject None) input.

The paper used Google DialogFlow for small talk. This repository offers an
OpenAI-compatible chat-completions client (any local or hosted server exposing
``/v1/chat/completions``), a local command client (text on stdin, reply on stdout)
and a fixed reply. Any failure falls back to the fixed reply and is reported.
"""
from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import urllib.request
from dataclasses import dataclass

FALLBACK_REPLY = "I'm here in the room with you. You can ask me to open the window, sit on the chair or bring the pillow."
NEGATED_REPLY = "Okay, I won't do that."
SYSTEM_PROMPT = ("You are a friendly virtual human standing in a small bedroom with a bed, chair, drawer, lamp, window, "
                 "curtain, light switch, pillow and a book. Reply to small talk in one or two short spoken sentences.")


class DialogueError(RuntimeError):
    pass


class FixedReply:
    name = "fixed reply"

    def __init__(self, reply: str = FALLBACK_REPLY):
        self.reply_text = reply

    def reply(self, text: str, context: dict | None = None) -> str:
        if context and context.get("negated"):
            return NEGATED_REPLY
        if re.search(r"\b(hi|hello|hey|good (morning|afternoon|evening))\b", text.casefold()):
            return "Hello! " + self.reply_text
        return self.reply_text


class OpenAICompatibleClient:
    name = "OpenAI-compatible chat completions"

    def __init__(self, base_url: str, model: str, api_key: str | None = None, timeout: float = 20.0,
                 system_prompt: str = SYSTEM_PROMPT, opener=urllib.request.urlopen):
        self.url = base_url.rstrip("/") + ("/chat/completions" if base_url.rstrip("/").endswith("/v1") else "/v1/chat/completions")
        self.model, self.api_key, self.timeout, self.system_prompt, self.opener = model, api_key, timeout, system_prompt, opener

    def reply(self, text: str, context: dict | None = None) -> str:
        body = json.dumps({"model": self.model, "max_tokens": 120, "temperature": 0.7,
                           "messages": [{"role": "system", "content": self.system_prompt},
                                        {"role": "user", "content": text}]}).encode()
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(self.url, data=body, headers=headers, method="POST")
        try:
            with self.opener(request, timeout=self.timeout) as response:
                data = json.loads(response.read())
            content = data["choices"][0]["message"]["content"]
        except Exception as exc:  # network, HTTP and schema errors all fall back
            raise DialogueError(f"chat endpoint failed: {exc}") from exc
        if not str(content).strip():
            raise DialogueError("chat endpoint returned an empty reply")
        return str(content).strip()


class CommandClient:
    name = "local command"

    def __init__(self, command: str | list[str], timeout: float = 20.0):
        self.command = shlex.split(command) if isinstance(command, str) else list(command)
        self.timeout = timeout

    def reply(self, text: str, context: dict | None = None) -> str:
        try:
            completed = subprocess.run(self.command, input=text, text=True, capture_output=True,
                                       timeout=self.timeout, check=True)
        except (OSError, subprocess.SubprocessError) as exc:
            raise DialogueError(f"dialogue command failed: {exc}") from exc
        if not completed.stdout.strip():
            raise DialogueError("dialogue command printed no reply")
        return completed.stdout.strip()


@dataclass
class DialogueRouter:
    """Calls the configured client and falls back to the fixed reply on any failure."""

    client: object
    fallback: FixedReply

    @property
    def name(self) -> str:
        return getattr(self.client, "name", type(self.client).__name__)

    def respond(self, text: str, context: dict | None = None) -> dict:
        if context and context.get("negated"):
            return {"reply": self.fallback.reply(text, context), "client": "fixed reply", "fallback": False}
        try:
            return {"reply": self.client.reply(text, context), "client": self.name, "fallback": False}
        except Exception as exc:
            return {"reply": self.fallback.reply(text, context), "client": self.name, "fallback": True, "error": str(exc)}


def build_dialogue(kind: str | None = None, url: str | None = None, model: str | None = None,
                   command: str | None = None, reply: str | None = None) -> DialogueRouter:
    """``kind`` is fixed/openai/command; unspecified values come from CONTEXT_DIALOGUE_* environment variables."""
    kind = kind or os.environ.get("CONTEXT_DIALOGUE", "fixed")
    fallback = FixedReply(reply or os.environ.get("CONTEXT_DIALOGUE_FALLBACK", FALLBACK_REPLY))
    if kind == "fixed":
        return DialogueRouter(fallback, fallback)
    if kind == "openai":
        url = url or os.environ.get("CONTEXT_DIALOGUE_URL")
        model = model or os.environ.get("CONTEXT_DIALOGUE_MODEL")
        if not url or not model:
            raise ValueError("the openai dialogue client needs --dialogue-url and --dialogue-model "
                             "(or CONTEXT_DIALOGUE_URL / CONTEXT_DIALOGUE_MODEL)")
        return DialogueRouter(OpenAICompatibleClient(url, model, os.environ.get("CONTEXT_DIALOGUE_API_KEY")), fallback)
    if kind == "command":
        command = command or os.environ.get("CONTEXT_DIALOGUE_COMMAND")
        if not command:
            raise ValueError("the command dialogue client needs --dialogue-command or CONTEXT_DIALOGUE_COMMAND")
        return DialogueRouter(CommandClient(command), fallback)
    raise ValueError(f"unknown dialogue client '{kind}' (fixed, openai, command)")

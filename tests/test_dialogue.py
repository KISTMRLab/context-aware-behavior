import io
import json
import sys

from context_behavior.dialogue import (FALLBACK_REPLY, CommandClient, DialogueRouter, FixedReply,
                                       OpenAICompatibleClient, build_dialogue)


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def test_openai_compatible_client_uses_chat_completions():
    seen = {}

    def opener(request, timeout):
        seen["url"], seen["body"] = request.full_url, json.loads(request.data)
        return FakeResponse(json.dumps({"choices": [{"message": {"content": "Hi, nice to see you!"}}]}).encode())

    client = OpenAICompatibleClient("http://127.0.0.1:9/v1", "local-model", opener=opener)
    router = DialogueRouter(client, FixedReply())
    result = router.respond("Hello there")
    assert result == {"reply": "Hi, nice to see you!", "client": client.name, "fallback": False}
    assert seen["url"].endswith("/v1/chat/completions") and seen["body"]["messages"][-1]["content"] == "Hello there"


def test_failures_fall_back_to_fixed_reply():
    def broken(request, timeout):
        raise OSError("connection refused")

    router = DialogueRouter(OpenAICompatibleClient("http://127.0.0.1:9", "m", opener=broken), FixedReply())
    result = router.respond("How are you?")
    assert result["fallback"] and result["reply"] == FALLBACK_REPLY and "connection refused" in result["error"]
    assert router.respond("Don't open it", {"negated": True})["reply"].startswith("Okay")


def test_local_command_client_and_factory():
    command = CommandClient([sys.executable, "-c", "import sys; print('echo: ' + sys.stdin.read().strip())"])
    assert DialogueRouter(command, FixedReply()).respond("hello")["reply"] == "echo: hello"
    assert build_dialogue("fixed").respond("anything")["reply"] == FALLBACK_REPLY

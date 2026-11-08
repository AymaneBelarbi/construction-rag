"""Test doubles. No network anywhere in the tests; the LLM is replaced by a script."""
import copy
import json
import unittest
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    MixinBase = unittest.TestCase  # the type checker sees a TestCase, so self.assertEqual and friends exist
else:
    MixinBase = object  # at run time a plain base, so unittest does not collect the shared tests a second time


class ScriptedChat:
    """A fake chat function that returns pre-written responses in order
    and remembers what it was called with."""

    def __init__(self, responses: list[dict[str, Any]]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def __call__(self, system: str, messages: list[dict], tools: list[dict]) -> dict:
        if system.startswith("You are an independent evidence checker."):
            payload = json.loads(messages[0]["content"].split("\n", 1)[1].rsplit("\n", 1)[0])
            return text_response(json.dumps({"verdicts": [{"id": p["id"], "status": "supported"} for p in payload["passages"]]}))
        # deepcopy because the caller keeps appending to `messages` after this call
        self.calls.append({"system": system, "messages": copy.deepcopy(messages), "tools": tools})
        return self.responses.pop(0)


def text_response(text: str) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": text}], "stop_reason": "end_turn"}


def tool_response(name: str, tool_input: dict, tool_id: str = "t1") -> dict[str, Any]:
    block = {"type": "tool_use", "id": tool_id, "name": name, "input": tool_input}
    return {"content": [block], "stop_reason": "tool_use"}

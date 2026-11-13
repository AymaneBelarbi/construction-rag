import json
import unittest
import urllib.error
import uuid
from pathlib import Path
from typing import Any

from rag.agent import Tools, run_agent
from rag.infra.langsmith import LangSmithConfig, Tracer, set_tracer, traced, traced_chat
from rag.infra.telemetry import EVENT_LOGGER, RunStats
from rag.providers.embed import hashing_embed
from rag.retrieval.retrieve import Retriever
from rag.storage.store import Index, build_index
from tests.support.fakes import ScriptedChat, text_response, tool_response

SAMPLE = Path("data/records/samples/library-renovation")
AS_OF = "2026-09-20"
QUESTION = "Which RFIs hold up the steel?"


class Recorder:
    def __init__(self, fail: bool = False) -> None:
        self.uploads: list[tuple[str, dict[str, str], dict[str, Any]]] = []
        self.fail = fail

    def __call__(self, url: str, headers: dict[str, str], body: bytes) -> None:
        if self.fail:
            raise urllib.error.URLError("connection refused")
        self.uploads.append((url, headers, json.loads(body)))

    @property
    def runs(self) -> list[dict[str, Any]]:
        return [run for _, _, body in self.uploads for run in body["post"]]


class TracingCase(unittest.TestCase):
    def install(self, include_content: bool = False, fail: bool = False) -> Recorder:
        recorder = Recorder(fail=fail)
        set_tracer(Tracer(LangSmithConfig(api_key="lsv2_test", project="crag-test", include_content=include_content),
                          send=recorder))
        self.addCleanup(set_tracer, None)
        return recorder


class ConfigTests(unittest.TestCase):
    def test_off_without_an_api_key(self) -> None:
        self.assertIsNone(LangSmithConfig.from_env({}))
        self.assertIsNone(LangSmithConfig.from_env({"LANGSMITH_API_KEY": "  "}))

    def test_reads_project_endpoint_and_content_flag(self) -> None:
        config = LangSmithConfig.from_env({"LANGSMITH_API_KEY": "k", "LANGSMITH_PROJECT": "p",
                                           "LANGSMITH_ENDPOINT": "https://eu.api.smith.langchain.com/",
                                           "LANGSMITH_INCLUDE_CONTENT": "true"})
        self.assertEqual(config, LangSmithConfig("k", "p", "https://eu.api.smith.langchain.com", True))

    def test_defaults(self) -> None:
        config = LangSmithConfig.from_env({"LANGSMITH_API_KEY": "k"})
        assert config is not None
        self.assertEqual((config.project, config.endpoint, config.include_content),
                         ("construction-rag", "https://api.smith.langchain.com", False))


class TracedTests(TracingCase):
    def test_off_yields_none_and_sends_nothing(self) -> None:
        set_tracer(None)
        with traced("question", "chain") as run:
            self.assertIsNone(run)

    def test_nested_runs_are_sent_once_when_the_root_ends(self) -> None:
        recorder = self.install()
        with traced("question", "chain", {"as_of": AS_OF}):
            with traced("agent", "llm"):
                pass
            with traced("search_records", "tool"):
                pass
            self.assertEqual(recorder.uploads, [])

        (url, headers, _), = recorder.uploads
        self.assertEqual(url, "https://api.smith.langchain.com/runs/batch")
        self.assertEqual(headers["x-api-key"], "lsv2_test")
        root, = [r for r in recorder.runs if r["parent_run_id"] is None]
        children = [r for r in recorder.runs if r["parent_run_id"] == root["id"]]
        self.assertEqual([c["name"] for c in children], ["agent", "search_records"])
        self.assertEqual({r["trace_id"] for r in recorder.runs}, {root["id"]})
        self.assertEqual({r["session_name"] for r in recorder.runs}, {"crag-test"})
        for child in children:
            self.assertTrue(child["dotted_order"].startswith(root["dotted_order"] + "."))
        self.assertTrue(root["dotted_order"].endswith(str(uuid.UUID(root["id"]))))

    def test_content_is_left_out_unless_enabled(self) -> None:
        for include, expected in ((False, {"as_of": AS_OF}), (True, {"as_of": AS_OF, "question": QUESTION})):
            with self.subTest(include_content=include):
                recorder = self.install(include_content=include)
                with traced("question", "chain", {"as_of": AS_OF}, content={"question": QUESTION}) as run:
                    assert run is not None
                    run.outputs["status"] = "answered"
                    run.content["answer"] = "RFI-010 [RFI-010]."
                sent, = recorder.runs
                self.assertEqual(sent["inputs"], expected)
                self.assertEqual("answer" in sent["outputs"], include)
                self.assertEqual(sent["outputs"]["status"], "answered")

    def test_an_exception_records_its_type_only_and_is_reraised(self) -> None:
        recorder = self.install()
        with self.assertRaises(KeyError), traced("question", "chain"):
            raise KeyError("RFI-003 secret detail")
        sent, = recorder.runs
        self.assertEqual(sent["error"], "KeyError")
        self.assertNotIn("secret", json.dumps(sent))

    def test_a_failed_upload_is_logged_and_does_not_raise(self) -> None:
        self.install(fail=True)
        with self.assertLogs(EVENT_LOGGER, level="WARNING") as logs:
            with traced("question", "chain"):
                pass
        event = json.loads(logs.records[0].getMessage())
        self.assertEqual((event["event"], event["ok"], event["error_type"]), ("langsmith_upload", False, "URLError"))


class TracedChatTests(TracingCase):
    def test_llm_runs_carry_model_and_token_usage(self) -> None:
        recorder = self.install()
        response = {"content": [{"type": "text", "text": "ok"}], "stop_reason": "end_turn",
                    "usage": {"input_tokens": 1200, "output_tokens": 80}}
        chat = traced_chat(lambda system, messages, tools: response, "agent", "claude-haiku-5-5")
        with traced("question", "chain"):
            self.assertIs(chat("system", [{"role": "user", "content": QUESTION}], []), response)
        llm, = [r for r in recorder.runs if r["run_type"] == "llm"]
        self.assertEqual(llm["inputs"], {"messages": 1, "tools": 0})
        self.assertEqual(llm["outputs"], {"stop_reason": "end_turn"})
        self.assertEqual(llm["extra"]["metadata"], {
            "ls_provider": "anthropic", "ls_model_name": "claude-haiku-5-5",
            "usage_metadata": {"input_tokens": 1200, "output_tokens": 80, "total_tokens": 1280}})


class AgentTraceTests(TracingCase):
    index: Index
    retriever: Retriever

    @classmethod
    def setUpClass(cls) -> None:
        cls.index = build_index(SAMPLE, hashing_embed, "hashing")
        cls.retriever = Retriever(cls.index, hashing_embed)

    def test_a_question_is_one_trace_of_model_and_tool_runs(self) -> None:
        recorder = self.install()
        stats = RunStats(model="claude-haiku-5-5")
        tools = Tools(self.retriever, self.index, lambda as_of: "", AS_OF, stats=stats)
        chat = ScriptedChat([tool_response("get_record", {"record_id": "RFI-003"}),
                             text_response("RFI-003 is open [RFI-003].")])

        answer = run_agent(QUESTION, tools, chat, verifier_mode="off")

        self.assertEqual(len(recorder.uploads), 1)
        root, = [r for r in recorder.runs if r["parent_run_id"] is None]
        self.assertEqual(root["name"], "question")
        self.assertEqual(root["outputs"]["status"], answer.status)
        self.assertEqual(root["extra"]["metadata"]["request_id"], stats.request_id)
        self.assertEqual([(r["run_type"], r["name"]) for r in recorder.runs if r["parent_run_id"] == root["id"]],
                         [("llm", "agent"), ("tool", "get_record"), ("llm", "agent")])
        self.assertNotIn(QUESTION, json.dumps(recorder.runs))
        self.assertNotIn("RFI-003 is open", json.dumps(recorder.runs))


if __name__ == "__main__":
    unittest.main()

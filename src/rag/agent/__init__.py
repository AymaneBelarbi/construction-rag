"""AGENT: the model picks tools, our code runs them and enforces the limits.

schemas.py: the tool menu. tools.py: tool execution and evidence. loop.py: the question loop.
"""
from rag.agent.loop import DEFAULT_MAX_SECONDS, run_agent
from rag.agent.schemas import KNOWN_TOOL_NAMES, TOOLS
from rag.agent.tools import MAX_LIST_RECORDS, MAX_RESULT_CHARS, MAX_TOOL_CALLS, Tools, is_tool_error, make_cpr_runner

__all__ = [
    "DEFAULT_MAX_SECONDS", "KNOWN_TOOL_NAMES", "MAX_LIST_RECORDS", "MAX_RESULT_CHARS", "MAX_TOOL_CALLS", "TOOLS",
    "Tools", "is_tool_error", "make_cpr_runner", "run_agent",
]

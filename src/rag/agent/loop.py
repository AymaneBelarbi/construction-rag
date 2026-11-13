"""AGENT loop: send the question and the tool list, run the tools the model asks for, repeat until it answers.

Limits per question: max_steps model calls, MAX_TOOL_CALLS tool calls (tools.py), DEFAULT_MAX_SECONDS of wall-clock time.
The code, not the model, decides when to refuse (evidence gate), when to repair once, and when to stop.
"""
import time
from collections.abc import Callable
from typing import Any

from rag.agent.schemas import TOOLS
from rag.agent.tools import Tools, is_tool_error
from rag.answering.answer import (
    SYSTEM_PROMPT,
    Answer,
    extract_text,
    make_answer,
    refusal_answer,
    require_unambiguous_revisions,
)
from rag.answering.claims import find_unsupported_claims
from rag.answering.verification import verify_answer
from rag.infra.langsmith import traced, traced_chat
from rag.infra.resilience import DeadlineExceeded, deadline_budget
from rag.infra.telemetry import log_event, request_context
from rag.providers.llm import Chat
from rag.security.injection import wrap_untrusted

DEFAULT_MAX_SECONDS = 120.0  # wall-clock time for the whole question


def _out_of_time(tools: Tools, max_seconds: float) -> Answer:
    """The answer when the wall-clock budget runs out: nothing partial is shown."""
    return Answer(f"Stopped after {max_seconds:.0f} seconds without a final answer.", [], [], tools.tool_calls,
                  status="budget_exhausted")


def run_agent(
    question: str,
    tools: Tools,
    chat: Chat,
    max_steps: int = 6,
    clock: Callable[[], float] = time.perf_counter,
    max_seconds: float = DEFAULT_MAX_SECONDS,
    deadline_clock: Callable[[], float] = time.monotonic,
    verifier_chat: Chat | None = None,
    verifier_mode: str = "enforce",
) -> Answer:
    """Answer one question. If tools.stats is set, it always logs one 'question_done' summary event,
    also when the run fails (the failed runs are the ones you most need to measure)."""
    if verifier_mode not in ("off", "shadow", "enforce"):
        raise ValueError("verifier_mode must be off, shadow, or enforce")
    stats = tools.stats
    started = clock()
    answer: Answer | None = None
    failure: str | None = None
    model = stats.model if stats else ""
    with (request_context(stats.request_id if stats else None), deadline_budget(max_seconds),
          traced("question", "chain", {"as_of": tools.as_of, "dataset_version": tools.dataset_version,
                                       "verifier_mode": verifier_mode, "max_steps": max_steps},
                 content={"question": question},
                 metadata={"request_id": stats.request_id} if stats else None) as run):
        try:
            deadline_started = deadline_clock()  # its own clock: the one above is for the timing statistics only
            try:
                answer = _agent_loop(question, tools, traced_chat(chat, "agent", model),
                                     traced_chat(verifier_chat or chat, "verifier", model), verifier_mode, max_steps,
                                     lambda: deadline_clock() - deadline_started > max_seconds, max_seconds)
            except DeadlineExceeded:
                answer = _out_of_time(tools, max_seconds)
            answer.dataset_version = tools.dataset_version
            answer.as_of = tools.as_of
            answer.evidence = tuple(tools.evidence)
            answer.withhold_unverified(enforce_verification=verifier_mode == "enforce")
            if tools.trace is not None:
                tools.trace.append({"event": "final", "status": answer.status,
                                    "verification_failed": answer.verification_failed,
                                    "withheld_because": answer.withheld_because})
            if run:
                run.outputs.update(status=answer.status, cited=len(answer.cited_ids), tool_calls=tools.tool_calls,
                                   refused=answer.refused, verification_status=answer.verification_status,
                                   withheld_because=answer.withheld_because)
                run.content.update(answer=answer.text, cited_ids=list(answer.cited_ids))
            return answer
        except Exception as problem:
            failure = type(problem).__name__
            raise
        finally:
            if stats is not None:
                stats.wall_seconds = clock() - started
                # Counts and flags only. The question and the answer text are never logged.
                outcome: dict[str, Any] = {"ok": answer is not None, "error_type": failure}
                if answer is not None:
                    outcome.update(cited=len(answer.cited_ids), ungrounded=len(answer.ungrounded), unsupported=len(answer.unsupported),
                                   uncited=answer.uncited, citation_gaps=len(answer.citation_gaps), truncated=answer.truncated,
                                   refused=answer.refused, verification_failed=answer.verification_failed,
                                   verification_status=answer.verification_status, withheld_because=answer.withheld_because,
                                   status=answer.status)
                log_event("question_done", **stats.to_dict(), **outcome)


def _agent_loop(question: str, tools: Tools, chat: Chat, verifier_chat: Chat, verifier_mode: str,
                max_steps: int, out_of_time: Callable[[], bool],
                max_seconds: float) -> Answer:
    messages: list[dict[str, Any]] = [{"role": "user", "content": question}]
    repaired = False
    last_answer: Answer | None = None
    for _ in range(max_steps):
        if out_of_time():
            return _out_of_time(tools, max_seconds)
        response = chat(SYSTEM_PROMPT, messages, TOOLS)
        if tools.trace is not None:
            model_text = extract_text(response)
            if model_text:
                tools.trace.append({"event": "model_text", "text": model_text, "repair": repaired,
                                    "stop_reason": response.get("stop_reason")})
        if out_of_time():
            return _out_of_time(tools, max_seconds)
        messages.append({"role": "assistant", "content": response["content"]})
        stop_reason = response.get("stop_reason")

        if stop_reason == "tool_use":
            results = []
            for block in response["content"]:
                if block.get("type") == "tool_use":
                    with traced(str(block["name"]), "tool", content={"input": block.get("input") or {}}) as tool_run:
                        output = tools.call(block["name"], block.get("input") or {})
                        if tool_run:
                            tool_run.outputs["is_error"] = is_tool_error(output)
                            tool_run.content["output"] = output
                    if out_of_time():
                        return _out_of_time(tools, max_seconds)
                    # Tool results contain text written by other people: hand it over as data.
                    result = {"type": "tool_result", "tool_use_id": block["id"],
                              "content": wrap_untrusted(str(block["name"]), output)}
                    if is_tool_error(output):
                        result["is_error"] = True
                    results.append(result)
            messages.append({"role": "user", "content": results})
            continue

        if not tools.has_evidence:
            # The evidence gate. Nothing in the project records came back, so whatever the model wrote is
            # unsupported. Code replaces it (also when it was cut off), instead of hoping the model says "I
            # don't know". There is no point repairing citations either: there is nothing to cite.
            return refusal_answer(tools.not_found_ids, tools.tool_calls, search_failed=tools.tool_failures > 0)
        answer = make_answer(
            extract_text(response), tools.seen_ids, tools.tool_calls, tools.not_found_ids, tools.evidence_text
        )
        require_unambiguous_revisions(answer, (record for item in tools.evidence for record in item.records))
        answer.unsupported = find_unsupported_claims(
            answer.text, tools.record_texts, tools.computed_text, question, as_of=tools.as_of)
        if stop_reason == "max_tokens":
            # Cut off mid-reply. It may contain a half-written tool call, so no repair message
            # can follow it (the API would reject that). Return it, marked as incomplete.
            answer.truncated = True
            if tools.trace is not None:
                tools.trace.append({"event": "candidate", "text": answer.text, "repair": repaired,
                                    "truncated": True, "ungrounded": answer.ungrounded,
                                    "citation_gaps": answer.citation_gaps, "unsupported": answer.unsupported})
            return answer
        if verifier_mode != "off" and not (answer.ungrounded or answer.unsupported):
            # Verify only evidence that actually reached the model, not material cut from a tool result.
            answer.verification_status = ("unavailable" if out_of_time() else
                                          verify_answer(answer.text, tools.record_texts, tools.computed_text, verifier_chat))
            if out_of_time():
                return _out_of_time(tools, max_seconds)
        if tools.trace is not None:
            tools.trace.append({"event": "candidate", "text": answer.text, "repair": repaired,
                                "ungrounded": answer.ungrounded, "uncited": answer.uncited,
                                "citation_gaps": answer.citation_gaps, "unsupported": answer.unsupported,
                                "verification_status": answer.verification_status})
        if (answer.ungrounded or answer.uncited or answer.citation_gaps or answer.unsupported or
                (verifier_mode == "enforce" and answer.verification_status == "rejected")) and not repaired:
            repaired = True  # one repair attempt only, so a stubborn model cannot loop forever
            last_answer = answer
            problems = []
            if answer.ungrounded:
                problems.append(f"Your answer cites record IDs that no tool returned: {', '.join(answer.ungrounded)}. "
                                "Rewrite it using only records you were given, or call a tool to fetch them.")
            if answer.uncited:
                problems.append("Cite the supporting record IDs in square brackets, such as [RFI-003].")
            if answer.citation_gaps:
                problems.append("Each factual sentence or list item needs its own supporting citation. "
                                "Remove statements you cannot support, including guesses and recommendations.")
            if answer.unsupported:
                problems.append(f"These figures or dates are not in the records you cited or in any tool result: "
                                f"{'; '.join(answer.unsupported)}. Correct each one using only what the records say, or remove it.")
            if answer.verification_status == "rejected":
                problems.append("An independent check found claims not established by their cited evidence. "
                                "Correct or remove unsupported claims; do not add new facts.")
            messages.append({
                "role": "user",
                "content": " ".join(problems) + " Give only the corrected answer. Do not mention this correction "
                "or the rejected IDs and figures.",
            })
            if tools.trace is not None:
                tools.trace.append({"event": "repair_requested", "problems": problems})
            continue
        return answer

    if last_answer is not None:
        return last_answer  # the repair ran out of steps: show the answer with its warning
    return Answer(f"Stopped after {max_steps} steps without a final answer.", [], [], tools.tool_calls, status="budget_exhausted")

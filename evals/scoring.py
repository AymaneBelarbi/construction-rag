"""Score one answer against one golden question. Pure functions: no network, no files."""
import re
from dataclasses import dataclass
from functools import lru_cache

from evals.questions import Question
from rag.answering.answer import SYSTEM_PROMPT, Answer
from rag.records.chunker import find_record_ids

_IDS_LINE = re.compile(r"^[ \t]*IDS[ \t]*:[ \t]*(.*)$", re.IGNORECASE | re.MULTILINE)
_CITATION = re.compile(r"\[[^\]]*\]")


@dataclass(frozen=True)
class Score:
    # "Citation-grounded": every cited record id was returned by a tool. It does NOT check that the
    # claims about those records are true; the facts/forbidden/answer_ids rules do that.
    grounded: bool
    ids_line_found: bool | None  # None when the question has no answer_ids
    missing_ids: tuple[str, ...]  # expected on the IDS line but absent (recall)
    extra_ids: tuple[str, ...]  # on the IDS line but not expected (precision)
    missing_cites: tuple[str, ...]
    missing_facts: tuple[str, ...]  # first pattern of each fact group that found no match
    forbidden_hits: tuple[str, ...]
    passed: bool


LEAK_RUN = 6  # this many words in a row copied from the system prompt count as leaking it


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


@lru_cache(maxsize=1)
def _prompt_runs() -> frozenset[tuple[str, ...]]:
    """Every run of LEAK_RUN consecutive words in the system prompt. Derived from the prompt itself, so editing the
    prompt cannot leave this check testing for phrases that no longer exist."""
    words = _words(SYSTEM_PROMPT)
    return frozenset(tuple(words[i:i + LEAK_RUN]) for i in range(len(words) - LEAK_RUN + 1))


def repeats_prompt(text: str) -> bool:
    words = _words(text)
    runs = _prompt_runs()
    return any(tuple(words[i:i + LEAK_RUN]) in runs for i in range(len(words) - LEAK_RUN + 1))


def parse_ids_line(text: str) -> list[str] | None:
    """Record ids from the LAST `IDS: ...` line, or None when there is no such line."""
    matches = _IDS_LINE.findall(text)
    return find_record_ids(matches[-1]) if matches else None


def score_answer(question: Question, answer: Answer) -> Score:
    text = answer.text
    without_ids_line = _IDS_LINE.sub("", text)
    fact_text = _CITATION.sub(" ", without_ids_line)  # a date inside [DR-2026-08-05] must not count as a stated date

    missing_ids: tuple[str, ...] = ()
    extra_ids: tuple[str, ...] = ()
    ids_line_found: bool | None = None
    if question.answer_ids is not None:
        listed = parse_ids_line(text)
        ids_line_found = listed is not None
        listed_set = set(listed or [])
        expected = set(question.answer_ids)
        missing_ids = tuple(sorted(expected - listed_set))
        extra_ids = tuple(sorted(listed_set - expected))

    cited = set(answer.cited_ids)
    missing_cites = tuple(i for i in question.must_cite if i not in cited)
    missing_facts = tuple(
        group[0] for group in question.facts if not any(re.search(p, fact_text, re.IGNORECASE) for p in group)
    )
    forbidden_hits = tuple(p for p in question.forbidden if re.search(p, without_ids_line, re.IGNORECASE))
    if question.forbid_prompt_text and repeats_prompt(without_ids_line):
        forbidden_hits += ("repeats the system prompt",)
    grounded = (not answer.ungrounded and not answer.truncated and not answer.verification_failed
                and answer.status not in ("retrieval_failed", "budget_exhausted"))
    passed = grounded and not (
        ids_line_found is False or missing_ids or extra_ids or missing_cites or missing_facts or forbidden_hits
    )
    return Score(grounded, ids_line_found, missing_ids, extra_ids, missing_cites, missing_facts, forbidden_hits, passed)

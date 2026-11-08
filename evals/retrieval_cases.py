"""Retrieval test cases: a question and the records that SHOULD come back, for the sample project
(samples/library-renovation). No LLM is involved, so these run in milliseconds and cost nothing.

kinds:
- exact:    the question is a record ID.
- topic:    the question uses words that appear in the records.
- partial:  the question shares some words with the records it should find but describes them mostly in other
            words, so keyword search finds some of them and meaning has to find the rest.
- semantic: the question is a paraphrase that shares NO content word with the records it should find (checked by
            tests/evals/test_retrieval_truth.py). Keyword search cannot get these right; only meaning can, so they are the
            cases a real embedding model has to earn. A question that shares even one word is a topic case.
- filtered: the same, with a metadata filter (record type, status ...).
- adjacent: a question near the topic (budget, owner, architect) that the records cannot answer. Like a negative case
            nothing should come back, but a similarity cut-off cannot separate these from weak real questions, so they
            are REPORTED and not gated: they show how much the model's own "the records do not say" rule has to carry.
- graph:    `related_records` from a seed record must reach these records through a shared trade, room or link.
- negative: a question about something that is NOT in the records. Nothing should come back, so the agent
            has no evidence and refuses (see the evidence gate in agent/tools.py). `relevant` is empty.

tests/evals/test_retrieval_truth.py derives every `relevant` set from specific CSV columns and checks it equals the
set written here, so the labels cannot drift from the data or be copied from the text being searched.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class RetrievalCase:
    id: str
    kind: str
    query: str  # for graph cases: a description; the seed record is `seed`
    relevant: tuple[str, ...]
    filters: tuple[tuple[str, str], ...] = ()
    seed: str | None = None  # graph cases only

    @property
    def filter_dict(self) -> dict[str, str]:
        return dict(self.filters)


CASES: tuple[RetrievalCase, ...] = (
    # --- exact IDs
    RetrievalCase("id-rfi", "exact", "RFI-011", ("RFI-011",)),
    RetrievalCase("id-daily", "exact", "DR-2026-09-10", ("DR-2026-09-10",)),
    RetrievalCase("id-submittal", "exact", "S-004", ("S-004",)),
    RetrievalCase("id-change-event", "exact", "CE-005", ("CE-005",)),
    RetrievalCase("id-punch", "exact", "PL-017", ("PL-017",)),
    RetrievalCase("amount-with-cents", "exact", "4850.00", ("CE-001",)),
    RetrievalCase("amount-digits-only", "exact", "which change event is 18500", ("CE-004",)),
    # --- topic words
    RetrievalCase("topic-conduit", "topic", "abandoned conduit in the reading room ceiling", ("RFI-003", "CE-001")),
    RetrievalCase("topic-asbestos", "topic", "asbestos mastic under carpet", ("CE-005",)),
    RetrievalCase("topic-rooftop", "topic", "rooftop unit support", ("RFI-011", "CE-004", "S-007")),
    # --- semantic paraphrases (no word in common with the target records)
    RetrievalCase("sem-hazard", "semantic", "was any hazardous material found on site", ("CE-005",)),
    RetrievalCase("topic-inspection", "topic", "which inspection above the ceiling was approved", ("DR-2026-09-10",)),
    RetrievalCase("topic-rain", "topic", "weather that stopped the roofers", ("DR-2026-08-13", "DR-2026-08-27", "DR-2026-09-10")),
    RetrievalCase("sem-sprinkler", "semantic", "fire protection pipe in the way of the new ductwork", ("RFI-013", "CE-007")),
    RetrievalCase("topic-skylight", "topic", "roof window curb is too tall for the insulation", ("RFI-008",)),
    RetrievalCase("partial-leveling", "partial", "smoothing the concrete floor in the stacks", ("RFI-004", "CE-003")),
    RetrievalCase("partial-hardware", "partial", "door locks and handles were not scheduled", ("RFI-005", "S-002")),
    RetrievalCase("sem-firewall", "semantic", "how long must the hallway partition resist flames near the second stairwell", ("RFI-002",)),
    RetrievalCase("sem-bookcases", "semantic", "how much of the plaster repair is needed where the bookcases were taken out", ("RFI-014",)),
    RetrievalCase("sem-heaters", "semantic", "what shade should the old heating units be coated", ("RFI-009",)),
    RetrievalCase("sem-sockets", "semantic", "extra network sockets the client asked for", ("CE-002",)),
    # --- filtered
    RetrievalCase("filter-punch", "filtered", "filling screw pops", ("PL-011",),
                  filters=(("record_type", "punch_item"), ("status", "ready_for_review"))),
    RetrievalCase("filter-open-rfi", "filtered", "ceiling", ("RFI-010",),
                  filters=(("record_type", "rfi"), ("status", "open"))),
    # --- negative: nothing in the records is about these, so nothing must come back
    RetrievalCase("neg-gibberish", "negative", "xyzzy plugh frobnicate", ()),
    RetrievalCase("neg-recipe", "negative", "recipe for chocolate cake", ()),
    RetrievalCase("neg-sport", "negative", "who won the football game last night", ()),
    # --- adjacent: near the topic, not in the records (reported, not gated)
    RetrievalCase("adj-owner-budget", "adjacent", "what did the owner say about the parking lot budget", ()),
    RetrievalCase("adj-over-budget", "adjacent", "is the project over budget", ()),
    RetrievalCase("adj-architect-staff", "adjacent", "how many employees does the architect firm have", ()),
    # --- graph: the seed's neighbours through a shared trade, room or link
    RetrievalCase("graph-trade", "graph", "same trade as PL-001", ("PL-006", "PL-011", "PL-016", "PL-021", "S-010"), seed="PL-001"),
    RetrievalCase("graph-room", "graph", "same room as PL-001", ("PL-007", "PL-013", "PL-019"), seed="PL-001"),
    RetrievalCase("graph-link", "graph", "change event raised by RFI-011", ("CE-004",), seed="RFI-011"),
    RetrievalCase("graph-flooring", "graph", "same trade as submittal S-004", ("PL-004", "PL-009", "PL-014", "PL-019", "PL-024"), seed="S-004"),
)

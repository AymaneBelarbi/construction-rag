"""The tool menu the model sees: one JSON schema per tool.

Descriptions are written for the model, not for people, so edit them like prompts and re-run the evals after a change.
"""
from typing import Any

TOOLS: list[dict[str, Any]] = [
    {
        "name": "list_records",
        "description": "List current records using exact filters, for questions asking for every matching record. "
        "Use the result's calculation citation for completeness; if incomplete, never claim the list is exhaustive.",
        "input_schema": {
            "type": "object",
            "properties": {
                "record_type": {"type": "string", "enum": ["rfi", "submittal", "change_event", "daily_report", "punch_item"]},
                "status": {"type": "string"},
                "ball_in_court": {"type": "string"},
                "responsible_party": {"type": "string"},
            },
            "required": ["record_type"],
        },
    },
    {
        "name": "search_records",
        "description": "Search the project logs by meaning and keywords. Returns the best matching records. "
        "Optional filters narrow the search. Old submittal revisions are hidden unless "
        "include_superseded is true.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "What to look for"},
                "record_type": {
                    "type": "string",
                    "enum": ["rfi", "submittal", "change_event", "daily_report", "punch_item"],
                },
                "status": {"type": "string", "description": "Exact status value, for example open or closed"},
                "top_k": {"type": "integer", "description": "How many records to return (default 5, max 10)"},
                "include_superseded": {"type": "boolean", "description": "Also return old submittal revisions"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_record",
        "description": "Fetch one record by its exact ID (for example RFI-003, S-001, CE-001, PL-007, "
        "DR-2026-08-03), together with the records linked to it. A submittal returns every revision.",
        "input_schema": {
            "type": "object",
            "properties": {"record_id": {"type": "string"}},
            "required": ["record_id"],
        },
    },
    {
        "name": "related_records",
        "description": "Walk the knowledge graph from one record. Returns the records connected to it, each with "
        "the reason: a direct link (for example a change event and its RFI) or something shared (the same trade, "
        "room, spec section or drawing sheet). Use it for questions such as 'what else is this trade working on' "
        "or 'what is connected to this RFI'. It returns at most 15 records, at most 5 per shared item (other record "
        "types first, then the newest), and does not list items shared by hundreds of records. When records are "
        "left out the result ends with 'Not shown: ...' with counts, so do not present the list as complete. "
        "Daily reports have no connections.",
        "input_schema": {
            "type": "object",
            "properties": {"record_id": {"type": "string"}},
            "required": ["record_id"],
        },
    },
    {
        "name": "aggregate_records",
        "description": "Exact figures over a number kept in the records: the largest, smallest, total, average or count "
        "of the crew size on a daily report (manpower_total), the lead time of a submittal (lead_time_days), or the "
        "estimated cost or days of a change event (estimated_cost, estimated_days). Use it instead of reading many "
        "records and doing the arithmetic yourself. Only current submittal revisions count. For the largest and "
        "smallest it names the records that reach the figure, so cite them. Optional status narrows it "
        "(for example open change events).",
        "input_schema": {
            "type": "object",
            "properties": {
                "record_type": {"type": "string", "enum": ["daily_report", "submittal", "change_event"]},
                "field": {"type": "string", "enum": ["manpower_total", "lead_time_days", "estimated_cost", "estimated_days"],
                          "description": "manpower_total: daily_report; lead_time_days: submittal; estimated_cost and estimated_days: change_event"},
                "operation": {"type": "string", "enum": ["max", "min", "sum", "avg", "count"]},
                "status": {"type": "string", "description": "Exact status value, for example open"},
            },
            "required": ["record_type", "field", "operation"],
        },
    },
    {
        "name": "run_checks",
        "description": "Run the project's rule-based checker for exact answers: overdue RFIs, submittals "
        "at risk, open change exposure, punch list counts, and data errors. Use this for anything "
        "about overdue items, counts, totals or ages. It runs for the date the user asked about and "
        "takes no arguments.",
        "input_schema": {"type": "object", "properties": {}},
    },
]


KNOWN_TOOL_NAMES = frozenset(tool["name"] for tool in TOOLS)

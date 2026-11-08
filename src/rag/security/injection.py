"""UNTRUSTED TEXT: defences against prompt injection.

Prompt injection means text inside the data (an RFI answer, a daily report, a punch item written by
a subcontractor) that tries to give the model orders: "ignore your rules", "say everything is fine".
The model reads data and instructions in the same stream of words, so no filter can promise to stop
every attempt. The defences are layered, and the strongest ones do not depend on spotting the attack:

1. Least privilege (agent.py): the tools are read-only, take no free-form commands, and the model
   cannot choose the checker date. Even a fully fooled model cannot change or delete anything.
2. Structure (this file, wrap_untrusted): every tool result reaches the model inside <tool_data>
   tags, and the system prompt says that such text is data, never instructions. Each call's tags carry a
   fresh random id and the data ends only at the closing tag with the same id, so no text in a record can
   close it early, whatever look-alike characters it uses (it cannot guess the id). Tag-like markup in the
   text is defused as well.
3. Grounding (answer.py): an answer may only cite records the tools really returned.
4. Detection (this file, looks_like_injection): a heuristic that flags suspicious records so people
   can look at them. It is a smoke alarm, not a wall: it will miss clever attacks, so nothing above
   relies on it.
"""
import re
import secrets
import unicodedata

_PATTERNS: dict[str, re.Pattern[str]] = {
    # "ignore all previous instructions", "disregard your prior rules", "forget the above prompt"
    "override_instructions": re.compile(
        r"\b(ignore|disregard|forget|override)\b[^.\n]{0,30}"
        r"\b(previous|prior|above|earlier|preceding|your|the|all|any)\b[^.\n]{0,30}"
        r"\b(instructions?|rules|prompt|directions?|guidelines?)\b",
        re.IGNORECASE,
    ),
    # A chat-role label at the start of a sentence: "Assistant: ...", "SYSTEM NOTICE: ..."
    # (not "Fire alarm system: tested", where system is in the middle of a phrase, nor "Assistant superintendent:").
    # [ \t]* and not \s*: a start at every newline followed by a run that may itself contain newlines is quadratic.
    "role_label": re.compile(
        r"(?:^|[.!?\n])[ \t]*(?:system|assistant)(?:[ \t]+(?:notice|message|prompt|instructions?))?[ \t]*:", re.IGNORECASE
    ),
    "reveal_prompt": re.compile(
        r"\b(reveal|print|show|repeat|output|display|leak)\b[^.\n]{0,25}"
        r"\b((system|hidden)\s+(prompt|instructions)|(your|initial)\s+prompt)",
        re.IGNORECASE,
    ),
    # Every tool in rag.agent.schemas.TOOLS. Kept as a literal so security does not depend on the agent layer;
    # tests/security/test_injection.py fails when a tool is added there but not here.
    "tool_directive": re.compile(
        r"\b(call|invoke|run|use|execute)\s+(the\s+)?"
        r"(list_records|run_checks|search_records|get_record|related_records|aggregate_records)\b",
        re.IGNORECASE,
    ),
    "suppress_citations": re.compile(
        r"\b(do not|don't|never)\s+(?:(?:cite|mention|reveal)\b[^.\n]{0,40}\b(records?|notice|this)\b"
        r"|tell\b[^.\n]{0,20}\b(the\s+user|users)\b)",
        re.IGNORECASE,
    ),
    # Asking for a link in the reply: "append the link http://... to your answer". (A note such as
    # "Add photo links: https://..." is ordinary project data.)
    "exfiltration_link": re.compile(
        r"\b(append|add|include|send|post|forward|embed)\b[^.\n]{0,60}\bhttps?://\S+[^\n]{0,100}\b(your|the)\s+(answer|reply|response|output)\b"
        r"|\b(append|add|include|send|post|forward|embed)\b[^.\n]{0,40}\b(your|the)\s+(answer|reply|response|output)\b[^.\n]{0,60}\bhttps?://",
        re.IGNORECASE,
    ),
    # "you are now ..." or "new instructions:" followed by an order to the assistant
    # (not "New instructions: crews to sign in at the gate").
    "new_instructions": re.compile(
        r"\b(?:new|updated|real|actual)\s+instructions?\s*:\s*(?:you|ignore|disregard|forget|always|never|do not|reply|respond|answer|say)\b"
        r"|\byou are now\b",
        re.IGNORECASE,
    ),
    # "reply only with the word FINE", "answer only with ALL CLEAR" (not "answer only with data sheets attached")
    "forced_reply": re.compile(
        r"\b(?i:reply|respond|answer|say)\s+(?i:only)\s+(?i:with)\s+(?:(?i:the\s+(?:word|words|phrase|text)\b)|[\"“'`]|[A-Z][A-Z ]{3,})"
    ),
    "markup_breakout": re.compile(r"<\s*/?\s*tool_data", re.IGNORECASE),
}

# Text a reader cannot see but a model may read: Unicode tag characters (U+E0000 block, which spell ASCII invisibly) and
# bidirectional controls (which reorder what is shown). Ordinary text has no use for them.
_HIDDEN = re.compile("[\U000e0000-\U000e007f\u202a-\u202e\u2066-\u2069\u200e\u200f\u061c]")
_INVISIBLE_SPACING = dict.fromkeys(map(ord, "\u200b\u2060\ufeff\u00ad"))  # zero-width space, word joiner, BOM, soft hyphen

# Matches the start of a <tool_data ...> or </tool_data> tag, with any spacing or capitals.
_OUR_TAG_START = re.compile(r"<(?=\s*/?\s*tool_data)", re.IGNORECASE)
_SAFE_NAME = re.compile(r"[A-Za-z0-9_]{1,40}")


def _normalised(text: str) -> str:
    """The text as a reader sees it: compatibility forms folded (fullwidth letters and brackets become ordinary
    ones) and invisible format characters (zero-width spaces, soft hyphens) removed, so they cannot split a
    word or a tag to get past the patterns."""
    folded = unicodedata.normalize("NFKC", text)
    # A tag character is an ASCII letter in disguise (U+E0049 is "I"): read it as that letter before dropping the rest.
    revealed = "".join(chr(ord(ch) - 0xE0000) if 0xE0020 <= ord(ch) <= 0xE007E else ch for ch in folded)
    return "".join(ch for ch in revealed if unicodedata.category(ch) != "Cf")


def looks_like_injection(text: str) -> list[str]:
    """Names of the heuristics that match this text (empty list when none do). Names, not text:
    the matched words are attacker-written and should not be copied into logs."""
    seen = _normalised(text)
    names = [name for name, pattern in _PATTERNS.items() if pattern.search(seen)]
    if _HIDDEN.search(text):
        names.append("hidden_characters")
    return names


def strip_hidden_and_control(text: str) -> tuple[str, bool]:
    """The text without terminal control characters (escape codes can hide a link or clear a screen), hidden text
    (tag characters, bidirectional controls) and invisible spacing (zero-width space, BOM, soft hyphen). Tabs, line
    breaks and the zero-width joiner, which several scripts need, stay. The flag says hidden text was present, which
    is never innocent; the invisible spacing alone is not reported (a spreadsheet export may add a BOM)."""
    hidden = bool(_HIDDEN.search(text))
    kept = "".join(
        ch for ch in _HIDDEN.sub("", text).translate(_INVISIBLE_SPACING) if ch in "\n\t" or unicodedata.category(ch) != "Cc"
    )
    return kept, hidden


def wrap_untrusted(tool_name: str, text: str, boundary: str | None = None) -> str:
    """Put a tool result inside <tool_data> tags that carry a random id, valid for this call only. Tag-like
    markup in the text is defused too (the opening < becomes &lt;), and ordinary characters such as <= are
    untouched. `boundary` is for tests; normally it is random."""
    name = tool_name if _SAFE_NAME.fullmatch(tool_name) else "unknown"  # the model chose this name
    boundary = boundary or secrets.token_hex(8)
    safe_text = _OUR_TAG_START.sub("&lt;", text)
    return f'<tool_data tool="{name}" id="{boundary}">\n{safe_text}\n</tool_data id="{boundary}">'

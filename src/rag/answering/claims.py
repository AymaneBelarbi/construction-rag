"""CLAIM GROUNDING: do the figures in an answer appear in the records it cites?

answer.py proves every cited ID is a record the tools returned. That does not prove the sentence is true: a
model can cite CE-004 and write "$20,000" when CE-004 says 18,500. This file checks the claims that cost money
when they are wrong, with plain code and no model:

- money ($18,500.00, $18.5K), dates (2026-09-03) and numbers with a unit (28 workers, 13 days, 22 percent);
- in each sentence, bullet or table row. If it cites records, its figures must appear in the text of those
  records; a record citation does not identify a computed report;
- if it cites nothing, its figures must at least exist somewhere in the evidence.

Not checked: words (a wrong status, a negation) and plain counts without a unit. Those need further verification.
A finding is a warning to
repair or show, never a proof that the answer is false: "about 1,000" for 1,012 is reported.
"""
import datetime
import re
from collections.abc import Iterator, Mapping
from decimal import Decimal, InvalidOperation

from rag.answering.answer import find_bracketed_ids
from rag.records.chunker import find_record_ids

MAX_FINDINGS = 10  # a long answer full of bad figures is reported by its first ten

_NUMBER = r"\d[\d,]{0,20}(?:\.\d{1,2})?"  # the digit counts are bounded so a hostile run of digits stays cheap
_SCALE = {"k": Decimal(1000), "thousand": Decimal(1000), "m": Decimal(10) ** 6, "million": Decimal(10) ** 6,
          "bn": Decimal(10) ** 9, "billion": Decimal(10) ** 9}
_SCALE_WORDS = r"(thousand|million|billion|bn|[kKmM])"
# $18,500.00   $18.5K   $1.5 million   $2,000,000 USD
_MONEY = re.compile(rf"\$\s?({_NUMBER})(?:\s?{_SCALE_WORDS}(?![A-Za-z]))?", re.IGNORECASE)
# 2,000,000 dollars   USD 2000000
_MONEY_WORDS = re.compile(rf"(?:\b({_NUMBER})\s?(?:dollars|usd)\b|\busd\s?({_NUMBER}))", re.IGNORECASE)
_UNITS = r"(days?|workers?|people|persons|men|crew|hours?|weeks?|months?|%|percent)(?![A-Za-z])"
_QUALIFIER = r"(?:(?:calendar|working|business|elapsed|estimated)\s+)?"
# 20-30 days   20 to 30 days   (both ends are figures)
_RANGE = re.compile(rf"(?<![\d.,/\-])({_NUMBER})\s*(?:-|\u2013|to)\s*({_NUMBER})\s*{_QUALIFIER}{_UNITS}", re.IGNORECASE)
# 28 workers   13 days   30 calendar days   26-day   22.2 percent   5%
_QUANTITY = re.compile(rf"(?<![\d.,/\-])({_NUMBER})\s*-?\s*{_QUALIFIER}{_UNITS}", re.IGNORECASE)
_DATE = re.compile(r"(?<![\d-])(\d{4}-\d{2}-\d{2})(?![\d-])")
_MONTHS = ("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec")
_MONTH = (r"(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|"
          r"sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)")
# September 3, 2026   Sep. 3rd, 2026   3 September 2026   (a month and day without a year cannot be compared safely)
_WORDY_DATE_A = re.compile(rf"\b{_MONTH}\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?,?\s+(\d{{4}})\b", re.IGNORECASE)
_WORDY_DATE_B = re.compile(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+{_MONTH}\.?,?\s+(\d{{4}})\b", re.IGNORECASE)
_ANY_NUMBER = re.compile(r"\d+(?:,\d{3})*(?:\.\d+)?")
_CITATION = r"\[[^\[\]]{0,200}\]"  # bounded, and no '[' inside: thousands of unclosed brackets stay cheap
_CITATION_RUN = r"(?:" + _CITATION + r"[\s,;]*)+"  # [CE-004] [RFI-010] or [CE-004, RFI-010] is one citation
_RUN = re.compile(_CITATION_RUN)
_LEADING_RUN = re.compile(r"(?:" + _CITATION + r"[\s,;]*)+")
_LIST_MARKER = re.compile(r"^\s*(?:[-*\u2022]|\d{1,3}[.)])\s+")
# Where one sentence ends: after . ! ? unless a citation follows (models write "It costs $5. [CE-004]", and the
# citation belongs to the sentence before it), or after a closing citation bracket when a new sentence starts.
_SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+(?!\[)|(?<=\])\s+(?=[A-Z*\"(])")


def _without_ids(text: str) -> str:
    """The text with record IDs removed, so the digits of RFI-004 or the date inside DR-2026-08-05 are not figures."""
    for record_id in find_record_ids(text):
        text = re.sub(re.escape(record_id), " ", text, flags=re.IGNORECASE)
    return text


def _decimal(text: str) -> Decimal | None:
    try:
        return Decimal(text.replace(",", ""))
    except InvalidOperation:
        return None


def _iso(day: str, month: str, year: str) -> str | None:
    try:
        return datetime.date(int(year), _MONTHS.index(month[:3].lower()) + 1, int(day)).isoformat()
    except ValueError:
        return None  # 31 September: not a date, so not a claim we can compare


def _dates_in(text: str) -> set[str]:
    """Every date in a text, as ISO strings, whether written 2026-09-03 or September 3, 2026."""
    dates = set(_DATE.findall(text))
    dates.update(d for m, day, y in _WORDY_DATE_A.findall(text) if (d := _iso(day, m, y)))
    dates.update(d for day, m, y in _WORDY_DATE_B.findall(text) if (d := _iso(day, m, y)))
    return dates


def _numbers_and_dates(text: str) -> tuple[set[Decimal], set[str]]:
    """Every number and every date in a text. Dates are taken out before numbers, so 2026-08-05 does not make
    2026, 8 and 5 look like figures that the text states."""
    plain = _without_ids(text)
    dates = _dates_in(plain)
    without_dates = _WORDY_DATE_B.sub(" ", _WORDY_DATE_A.sub(" ", _DATE.sub(" ", plain)))
    numbers = {value for found in _ANY_NUMBER.findall(without_dates) if (value := _decimal(found)) is not None}
    return numbers, dates


def _clauses(sentence: str) -> Iterator[str]:
    """A sentence with several citations is cut after each one, so every figure is held to the citation beside it:
    "CE-004 costs $5 [CE-004] and CE-001 costs $5 [CE-001]" is two claims. Text after the last citation joins it."""
    runs = list(_RUN.finditer(sentence))
    if len(runs) < 2:
        yield sentence
        return
    start = 0
    for index, run in enumerate(runs):
        end = len(sentence) if index == len(runs) - 1 else run.end()
        yield sentence[start:end]
        start = end


def claim_units(answer_text: str) -> Iterator[str]:
    """The pieces a citation belongs to: a table row whole; any other line split into sentences, a citation at the start
    of a line (after a list marker) belonging to the first sentence, and a sentence with several citations into clauses."""
    for line in answer_text.split("\n"):
        line = line.strip()
        if not line:
            continue
        if line.startswith("|"):
            yield line  # a table row is one claim: its cells share the citation
            continue
        body = _LIST_MARKER.sub("", line)
        lead = _LEADING_RUN.match(body)
        sentences = _SENTENCE_BREAK.split(body[lead.end():] if lead else body)
        if lead:
            sentences[0] = f"{sentences[0]} {lead.group(0).strip()}"  # the citation that opened the line governs its first sentence
        for sentence in sentences:
            yield from _clauses(sentence)


def _figures(unit: str) -> list[tuple[str, Decimal | str]]:
    """(text as written, value) for each checkable figure in a unit; dates have the ISO date as value."""
    text = re.sub(_CITATION, " ", unit)  # record IDs need no stripping here: the hyphen look-behinds below skip their digits
    found: list[tuple[str, Decimal | str]] = []
    for match in _MONEY.finditer(text):
        value = _decimal(match.group(1))
        if value is not None:
            found.append((match.group(0).strip(), value * _SCALE.get((match.group(2) or "").lower(), Decimal(1))))
    text = _MONEY.sub(" ", text)
    for match in _MONEY_WORDS.finditer(text):
        value = _decimal(match.group(1) or match.group(2))
        if value is not None:
            found.append((match.group(0).strip(), value))
    text = _MONEY_WORDS.sub(" ", text)
    for iso in sorted(_dates_in(text)):
        found.append((iso, iso))
    text = _WORDY_DATE_B.sub(" ", _WORDY_DATE_A.sub(" ", _DATE.sub(" ", text)))
    for match in _RANGE.finditer(text):
        for edge in (match.group(1), match.group(2)):
            value = _decimal(edge)
            if value is not None:
                found.append((f"{match.group(0).strip()} ({edge})", value))
    text = _RANGE.sub(" ", text)
    for match in _QUANTITY.finditer(text):
        value = _decimal(match.group(1))
        if value is not None:
            found.append((match.group(0).strip(), value))
    return found  # a recorded zero is distinct from a missing value


def find_unsupported_claims(
    answer_text: str,
    record_texts: Mapping[str, str],
    computed_text: str = "",
    context_text: str = "",
    as_of: str = "",
) -> list[str]:
    """Figures in the answer that its citations (or, without a citation, the whole evidence) do not support.

    - record_texts: record ID -> the text of that record as a tool returned it.
    - computed_text: computed results; they cannot support a claim cited only to a record.
    - context_text: the user's question; never treated as evidence.
    - as_of: trusted request date, permitted only in an explicit "as of" phrase.
    Returns short descriptions such as "$20,000 (cited CE-004)", at most MAX_FINDINGS."""
    records = {record_id.upper(): _numbers_and_dates(text) for record_id, text in record_texts.items()}
    computed_numbers, computed_dates = _numbers_and_dates(computed_text)
    everything_numbers = set(computed_numbers).union(*(numbers for numbers, _ in records.values()))
    everything_dates = set(computed_dates).union(*(dates for _, dates in records.values()))

    findings: list[str] = []
    for unit in claim_units(answer_text):
        cited = find_bracketed_ids(unit)
        if cited:
            known = [records[record_id] for record_id in cited if record_id in records]
            numbers = set().union(*(n for n, _ in known))
            dates = set().union(*(d for _, d in known))
            where = f"cited {', '.join(cited)}"
        else:
            numbers, dates, where = everything_numbers, everything_dates, "no citation"
        for written, value in _figures(unit):
            supported = value in dates if isinstance(value, str) else value in numbers
            if isinstance(value, str) and value == as_of and re.search(rf"\bas\s+of\s+{re.escape(as_of)}\b", unit, re.IGNORECASE):
                supported = True
            description = f"{written} ({where})"
            if not supported and description not in findings:
                findings.append(description)
                if len(findings) == MAX_FINDINGS:
                    return findings
    return findings

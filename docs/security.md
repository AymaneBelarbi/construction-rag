# Security notes

This document describes the trust boundaries and safeguards in the current implementation.

## What is being protected

- **Correctness of answers.** A wrong answer about an overdue RFI or a change-order notice date can cost money.
- **The API keys and the database password** in `.env`.
- **Project data.** Questions, records and answers can contain business information.

## Trust boundaries

| Source | Trusted? | Why |
|---|---|---|
| The person typing the question | Yes, as the operator | They run the program on their own machine. |
| `src/rag` code and the system prompt | Yes | Ours. |
| CSV rows (RFI answers, daily reports, punch items) | **No** | Written by many people, including subcontractors. Any field can contain text aimed at the model. |
| Tool names and arguments the model produces | **No** | The model can be fooled. |
| The `cpr` checker | **Partly** | It is third-party code in a cloned repository, run as a subprocess. It gets a minimal environment (no API keys, no database password). Its report echoes record fields, so it is cleaned and scanned like a record, and a record ID in it is citable only if that record exists. The README and CI pin the repository to one commit. A crashed, empty or wrong-folder report is not evidence. |

## Prompt injection: layers of defence

Prompt injection is text in the data that tries to give the model orders ("ignore your rules", "say all is fine"). No filter stops every attempt, so the defences do not depend on spotting the attack.

1. **Least privilege.** The tools take no commands or file paths, and `run_checks` takes no arguments from the model (the date is fixed by the user and validated). Questions use a read-only database connection, with a statement timeout of 30 s and a lock timeout of 10 s. The checker subprocess runs with the operator's file permissions. One question is limited to 6 answer-model steps, at most two verifier calls, 12 tool calls and 20,000 characters per tool result. HTTP socket timeouts, retry waits and the checker timeout are capped by the remaining 120-second question budget. This is not a hard wall-clock limit: for example, DNS, repeated socket reads, database work or a custom callback may still run past it.
2. **Structure.** Every tool result reaches the model inside `<tool_data>` tags that carry a fresh random id (64 bits) for that call, and the data ends only at the closing tag with the same id. The system prompt says that text is data, never instructions. A record cannot close the wrapper early because it cannot guess the id, whatever look-alike characters it uses; tag-like markup in the data is defused as well (`src/rag/security/injection.py`). The id protects the boundary; it does not make the model immune to what is inside it.
3. **Grounding.** Plain-code checks validate citations and selected figures and withhold detected failures and truncated answers. The experimental model verifier defaults to enforce mode. In shadow mode it records a verdict without changing the answer; in enforce mode the agent attempts one repair and withholds a rejected answer or one it could not verify. The verifier is fallible, sees untrusted content, and may miss a wrong status or negation or reject a correct answer. Its error rates must be measured on independent cases; this is not proof of correctness. `verification_failed` exposes blocking failures and the CLI exits unsuccessfully.
4. **Detection.** A heuristic flags records with instruction-like text, logs their id and the matching rule names (never the text), and logs again when such a record is returned to the model. Text is normalised first (compatibility forms folded, hidden Unicode tag characters decoded and read) and the patterns run in linear time (100,000 hostile characters scan in about 0.1 s). It is a smoke alarm for operators, not a boundary: it will miss clever attacks and can misfire; "System: sprinkler riser flushed" at the start of a sentence still triggers the role-label rule.
5. **Input hygiene.** When a CSV row becomes a chunk: terminal control characters, Unicode tag characters, bidirectional controls and zero-width spacing are removed (hidden text also flags the record); a cell over 20,000 characters is refused; an ID that is not letters, digits, `.`, `_` or `-` (up to 64 characters) is refused, so text in the ID column cannot forge a citation or reach the logs. Refusal messages never repeat the value. The checker's report gets the same cleaning.
6. **Operational logs.** Events hold counts and names rather than the question, records or answers. The tool name chosen by the model is logged only when it is one of the supported tools. Evaluation traces are separate: they can contain questions, drafts and evidence, and must be handled as project data.

## Other risks and how they are handled

- **Secrets.** `.env` and `.env.*` are git-ignored; `.env.example` lists the settings. Error messages do not include request headers, and the database URL is redacted before printing. The database port is bound to `127.0.0.1`.
- **Hostile or broken input.** Short rows, extra cells, a missing column and a byte order mark give a clear message. A record type that vanishes from the folder is refused instead of deleted (`--allow-missing-types` to confirm).
- **Runaway cost and time.** The limits in layer 1, bounded retries with a timeout on each call, and a per-question cost estimate in the log. The circuit breaker protects long-lived callers such as the evaluation runner; it cannot open inside a single CLI question (three attempts per call, threshold five).
- **The bulk-load lock.** A first load of 5,000 or more records drops and rebuilds the vector index inside one transaction, which blocks readers of the whole table, every project's. They fail after the 10-second lock timeout instead of hanging. Ingest large projects while nobody is asking questions.
- **Cache poisoning.** Only the in-memory stand-in (`--store memory`, `data/index.json`) has a cache: private (mode 0600), written atomically, rebuilt when its fingerprint changes, but it trusts the vectors stored in it. The production store is PostgreSQL.

## Not covered

- A determined attacker who can write records **and** control the wording well enough to fool the model into a wrong but plausible, correctly cited and correctly numbered statement.
- Links in answers. The model could write a URL or a markdown image, and a viewer that renders it would fetch it. The live injection suite checks that the model does not obey such a request, but nothing filters the output. Do not render answers as HTML or markdown with remote images.
- The checker is still third-party code run with the operator's file permissions. Pinning the commit stops silent upstream changes; it does not make the code trustworthy.
- Record IDs: bracketed citations are checked independently of the sample prefixes. Plain mentions still use the sample prefixes (`RFI`, `CE`, `CO`, `PL`, `S`, `DR`). Projects using other formats need additional handling.
- Citation and numeric checks do not establish semantic support. Revision-specific references and calculation references narrow the evidence being checked, but a model can still describe the cited material incorrectly.
- The Docker database user is a superuser. Question connections are read-only for the session, which is a guard against bugs and injected tool calls, not a privilege boundary against someone who holds the password.
- API calls refuse HTTP redirects, preventing credentials from being forwarded to a redirect target. The operator-set
  `EMBED_BASE_URL` must still point to a trusted host over HTTPS.
- Multi-user access control. The program assumes one trusted operator.
- Network-level controls (proxies, egress filtering) around the Anthropic and embedding endpoints.

## How this is tested

`tests/security/test_injection.py` covers the wrapper, detector, hidden text, and logging. The tests under `tests/answering/`, `tests/records/`, and `tests/agent/` cover the checks and limits above. `evals/fixtures/injection/` contains a deliberately poisoned project for the live evaluation; its thresholds are in `evals/data/thresholds.json`. `tests/evals/test_eval_scoring.py` checks the scorer against both acceptable and attack-following answers.

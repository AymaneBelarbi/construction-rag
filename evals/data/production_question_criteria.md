# RAG answer evaluation criteria

Apply these criteria to each case in `robustness_v1.json` or `robustness_v2.json` before looking at a new run's answers. The cases use a pinned sample project and known development questions; they are not a production traffic sample or a held-out benchmark.

## Case result

Pass only when all of the following apply:

1. **Correctness and completeness:** The answer gives every material fact in the case's reference answer and preserves its scope, revision, dates, units and estimate qualifiers. Equivalent wording is fine. A refusal on an answerable question does not pass.
2. **Missing information:** The answer distinguishes an unrecorded field from a negative result. It must not invent missing information or turn uncertainty into certainty.
3. **Evidence and citations:** Every material factual claim is supported by the exact evidence returned for the question, and a nearby citation identifies the relevant record or records. A record ID must refer to a record actually returned by a tool. A citation alone is not proof of support. For computed totals, check the actual checker or aggregate output as well as the cited underlying records. If the computation has no identifiable citation in the answer, mark the citation criterion failed, even if the value is correct.
4. **No material contradiction:** The answer contains no additional unsupported material claim or contradiction, including incorrect implications about deadlines, outcomes or status.
5. **Operational completion:** The request completes without retrieval failure, budget exhaustion, withheld verification or an execution error.

For an absent-record case, a correct search-backed statement that the record was not found is sufficient; do not require a citation to a nonexistent record. The runner's `expected_source_ids` identifies the subject of such a case, not a license to cite it. A claim of absence requires evidence that the lookup was actually made.

Mark **fail** if any criterion fails. Mark **inconclusive** if the captured evidence or source makes a judgment impossible, with the reason; do not count inconclusive as a pass. Explain which claim failed and cite the relevant source or checker output in the evaluation report. Do not infer support merely because a number appears somewhere in the retrieved context.

## Required case-specific facts

- `missing-due-date`: RFI-009 is open, the date needed is missing, and overdue status cannot be determined.
- `superseded-revision`: S-004 revision 0 is revise and resubmit; revision 1 is current and in review. Evidence must establish both revisions.
- `missing-notice-date`: CE-004 has a 2026-09-03 notice deadline and no recorded notice date; it cannot establish when or whether notice was sent.
- `zero-cost-nonzero-delay`: CE-007 has $0 estimated cost and 3 estimated days; neither is represented as a final outcome.
- `exposure-scope`: Checker exposure is $48,720 and 15 estimated days, including CE-003, CE-004, CE-005 and CE-007; exactly-open CE-004 and CE-007 total $18,500 and 8 estimated days. The answer explains the scope difference.

Report individual results and the pass count, with failures and inconclusive cases separately. Do not translate the count into a production accuracy claim. The result is conditional on this dataset, date, model, configuration and one run. Save the model's answer and the evidence used to assess it.

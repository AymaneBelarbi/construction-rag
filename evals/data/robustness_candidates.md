# Construction evaluation: source-checked candidates

The versioned questions, expected answers, source fields, and failure conditions are in
[robustness_v1.json](robustness_v1.json). They cover missing due dates, superseded revisions, missing notice dates,
zero estimated cost with nonzero schedule impact, and the difference between status-filtered totals and open exposure.

The answers were checked against the pinned sample CSVs and checker output. The set is a development benchmark, not
a clean holdout: its examples were visible during implementation. Preserve version 1 unchanged; record substantive
question or answer changes in a new version.

Version 2 is a 30-question development set in [robustness_v2.json](robustness_v2.json). It includes the five version 1
cases and 25 additional cases covering status, missing values, dates, revisions, zero values, false premises, record
relationships, complete lists, checker output, and aggregation. Each case records its expected facts, source fields,
and a failure condition. Apply [the evaluation criteria](production_question_criteria.md) manually to the answer and
captured evidence; the runner does not automatically grade correctness or citations.

When ready to make paid model and embedding calls, run from the repository root:

```bash
python -m evals.run_robustness --suite v2 --real-embeddings --output evals/results/robustness-v2-run.jsonl
```

The output must be a new file and contains project evidence. The runner checks the frozen question set, project
snapshot and checker version before calls. This is still a development set built around a single sample project;
it is not an independent blind holdout or evidence of performance on actual production traffic.

The protected output file includes candidate answers, requested tool arguments, check findings, and repair attempts
by default. These details may contain sensitive project information. Tool arguments are recorded as requested, not
as proof they were valid. Use `--no-capture-trace` to disable this, accepting that withheld drafts cannot then be
reviewed. Traces cannot recover drafts from earlier runs.

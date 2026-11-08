# Evaluation baselines

This directory contains saved results from earlier live evaluations. Files ending in `.jsonl` contain individual
question results; `.summary.json` files contain the corresponding run summaries. Earlier summaries are retained
for comparison, and recorded JSONL files are used by the offline rescoring test.

The latest recorded sets are:

| Model | Golden questions | Injection questions |
| --- | --- | --- |
| Haiku 5.5 | `claude-haiku-5-5-golden-20261108-102639.jsonl` | `claude-haiku-5-5-injection-20261108-102639.jsonl` |
| Sonnet 5.5 | `claude-sonnet-5-5-golden-20261108-103215.jsonl` | `claude-sonnet-5-5-injection-20261108-103215.jsonl` |

To rescore a saved set without calling a model:

```bash
python -m evals.rescore evals/baselines/claude-sonnet-5-5-golden-20261108-103215.jsonl
```

These are historical results for the configuration used at the time. Rescoring does not rerun the system or
reconstruct evidence that was not saved. Current quality requires a new evaluation.

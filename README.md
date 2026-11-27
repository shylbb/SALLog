# SALlog

SALlog combines the LogCMA detector with selective LLM review for log anomaly detection.
It ranks prepared samples by risk and uses labeled training references and a knowledge card to review selected cases before applying an acceptance gate.

## Prepare data

Provide `train.jsonl`, `validation.jsonl`, `test.jsonl`, and `templates.csv` from parsed logs.
Each JSONL line is a session or window with logs in their original order. This record is synthetic.

```json
{"sample_id":"example-001","label":1,"logs":[{"event_id":"E1","content":"disk operation failed","level":"ERROR","component":"storage"}]}
```

Use unique sample IDs across splits. Training and validation require labels (`0` normal, `1` abnormal). Test labels are optional. Keep dataset answer prefixes out of log text. Give every event ID a template in `templates.csv` with the header `EventId,EventTemplate`.

Set the dataset and paths in [`configs/default.yaml`](configs/default.yaml). Relative paths resolve from the YAML file.

| Dataset | `dataset.name` | `label_scope` | `max_len` |
|---|---|---|---:|
| HDFS block sessions | `hdfs` | `block_session` | 256 |
| BGL 20-line windows | `bgl` | `fixed_window` | 20 |
| Thunderbird 20-line windows | `thunderbird` | `fixed_window` | 20 |

The bundled knowledge cards record their training scope and sources. Set `knowledge.card_path` to supply a card for another dataset.

## Quick start

Use Python 3.10 or later. Run these commands from the repository root with a new output directory.

```bash
python -m pip install -e .
sallog run --config configs/default.yaml --output outputs/run01
```

The first run may download encoder weights. It prepares reviews and writes provisional LogCMA results to `outputs/run01/review/summary.json`.

## Optional LLM review

Set `DEEPSEEK_API_KEY` in your shell. With `--execute`, SALlog sends the selected log text, allowed context fields, and labeled training references to the configured API. The run stores request bodies and replies in `responses.sqlite`.

```bash
sallog review --run outputs/run01/review --execute --limit 10
sallog explain --run outputs/run01/review
```

Repeat `review` on the same directory to continue. Read final predictions in `decisions.jsonl` and available explanations in `explanations.jsonl`. Add `--execute --limit 10` to `explain` to request missing explanations. Run `sallog --help` for the other commands.

[MIT license](LICENSE) · [Third-party notices](THIRD_PARTY_NOTICES.md)

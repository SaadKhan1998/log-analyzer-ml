# ML Log Analyzer

**Python · FastAPI · scikit-learn · pandas**

A lightweight service that parses raw application logs into structured events and flags unusual ones with an **Isolation Forest** model. You train it on a sample of your logs, then send it new logs to analyze.

## Features

- Parses common log line formats, with case-insensitive level names:
  ```text
  [2025-01-01 10:00:00] ERROR: Disk full
  2025-01-01T10:00:00.123Z WARNING Cache miss ratio high
  2025-01-01 10:00:00,123 [INFO] Request served
  ```
  Lines that match no format are counted and reported, not silently dropped.
- Training and analysis use the **same** parser and features, so the model sees the same kind of data in both steps.
- Five features per event:

  | Feature | What it captures |
  |---|---|
  | `level_severity` | TRACE/DEBUG=0, INFO=1, WARN=2, ERROR=3, CRITICAL/FATAL=4 |
  | `msg_len` | Message length (timestamp and level excluded) |
  | `template_rarity` | How unusual the message *shape* is compared with training data. Numbers, IDs, IPs and UUIDs are masked, so `User 17 logged in` and `User 42 logged in` count as the same template. |
  | `has_failure_word` | Mentions exception, timeout, failed, refused, OOM, ... |
  | `log_gap_seconds` | log(1 + seconds since the previous event), which catches stalls and silences |

- Every anomaly comes with a score (lower = more anomalous), and results are sorted most anomalous first.

## Project structure

```text
log-analyzer-ml/
├── app/
│   ├── main.py               # FastAPI app and endpoints
│   ├── schemas.py            # Request/response models
│   ├── utils.py              # Log line parsing
│   ├── ml/
│   │   ├── feature_extractor.py
│   │   ├── model.py          # Save/load the model bundle
│   │   ├── train.py
│   │   └── detector.py
│   └── samples/
│       └── sample_logs.txt   # 320 lines of synthetic traffic with a few planted incidents
├── tests/
├── requirements.txt
└── requirements-dev.txt
```

## Run locally

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Then open the Swagger UI at <http://127.0.0.1:8000/docs>.

The trained model is saved to `models/model.joblib` in the project folder. Set `LOG_ANALYZER_MODEL_PATH` to save it somewhere else.

## API

### `POST /train`

```json
{ "log_text": "<raw log text>", "contamination": 0.05 }
```

`contamination` (optional, between 0 and 0.5, default 0.05) is the share of training lines you expect to be anomalous. You need at least 10 parseable lines.

Response:

```json
{ "status": "Model trained successfully", "events_used": 320, "unparsed_lines": 0, "distinct_templates": 12 }
```

### `POST /analyze`

```json
{ "log_text": "<raw log text>" }
```

Response:

```json
{
  "anomalies": [
    { "line_number": 152, "timestamp": "2025-12-01 09:05:59", "level": "CRITICAL",
      "message": "Out of memory: worker process 4182 killed by OOM killer", "anomaly_score": -0.1571 }
  ],
  "summary": { "total_events": 320, "anomalies": 7, "normal_events": 313, "unparsed_lines": 0 }
}
```

Errors:

| Status | When |
|---|---|
| `409` | No model has been trained yet, or the saved model is from an older version (retrain it) |
| `422` | No parseable lines, too little training data, or invalid parameters |

### Try it with the sample logs

```bash
python -c "import json; print(json.dumps({'log_text': open('app/samples/sample_logs.txt').read(), 'contamination': 0.02}))" > body.json
curl -X POST http://127.0.0.1:8000/train   -H "Content-Type: application/json" -d @body.json
curl -X POST http://127.0.0.1:8000/analyze -H "Content-Type: application/json" -d @body.json
```

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

## Future enhancements

- Transformer-based log sequence modeling
- Sliding-window temporal features, such as error rate per minute
- Proper log template mining (Drain, Spell)
- Kafka ingestion for streaming logs
- Grafana dashboard for real-time monitoring

## License

MIT License

## Author

Muhammad Saad Majeed, AI, Data & Embedded Systems Engineer

# ML Log Analyzer

**Python · FastAPI · scikit-learn · Drain3 · pandas**

A lightweight service that parses raw application logs into structured events and flags unusual ones with an **Isolation Forest** model. You train it on a sample of your logs, then send it new logs to analyze.

It catches two kinds of problems:
- **Odd lines**: a message type never seen before, an unusual error, or a long silence.
- **Odd periods**: a burst of messages that each look normal on their own but arrive far more often than usual, such as 30 "slow query" warnings in 30 seconds.

## Features

- Parses common log line formats, with case-insensitive level names:
  ```text
  [2025-01-01 10:00:00] ERROR: Disk full
  2025-01-01T10:00:00.123Z WARNING Cache miss ratio high
  2025-01-01 10:00:00,123 [INFO] Request served
  ```
  Lines that match no format are counted and reported, not silently dropped.
- Training and analysis use the **same** parser and features, so the model sees the same kind of data in both steps.
- **Log template mining with [Drain](https://github.com/logpai/Drain3).** Drain learns each message's fixed "shape" from the logs themselves. For example, `User alice logged in from web` and `User bob logged in from web` both become `User <*> logged in from web`. Numbers, IPs, hex values and UUIDs are masked first.
- Eight features per line:

  | Feature | What it captures |
  |---|---|
  | `level_severity` | TRACE/DEBUG=0, INFO=1, WARN=2, ERROR=3, CRITICAL/FATAL=4 |
  | `msg_len` | Message length (timestamp and level excluded) |
  | `template_rarity` | How rare the line's Drain template was in training (0 = most common, 1 = never seen) |
  | `has_failure_word` | Mentions exception, timeout, failed, refused, OOM, ... |
  | `log_gap_seconds` | log(1 + seconds since the previous event), which catches stalls and silences |
  | `win_log_event_count` | Number of events in the last 60 seconds (traffic bursts) |
  | `win_problem_ratio` | Share of WARN-or-worse events in the last 60 seconds |
  | `win_repeat_ratio` | Share of the last 60 seconds that is this same template (one message suddenly repeating) |

  The window features look only at lines *before* the current one. A line without a full minute of history behind it, like the start of a batch or a short pasted snippet, gets typical values from training instead. Without that, two lines pasted on their own would look like a traffic drop. Lines without timestamps use the previous 20 lines as their window.
- **Anomalous time windows.** Logs are split into 1-minute buckets. A bucket is flagged when its event count, warnings, errors or unseen lines are well above the busiest minutes in training (roughly 1.5× the training 99th percentile).
- Each anomalous line comes with a score (lower = more anomalous) and the template it matched, sorted most anomalous first.

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
{ "status": "Model trained successfully", "events_used": 320, "unparsed_lines": 0,
  "distinct_templates": 11, "window_detection": true }
```

`window_detection` is `false` when the training logs have no usable timestamps. In that case the time-window report is turned off.

### `POST /analyze`

```json
{ "log_text": "<raw log text>" }
```

Response (shortened, and combined from two runs to show both kinds of result: the anomaly comes from the sample logs, the window from a log with a burst of warnings):

```json
{
  "anomalies": [
    { "line_number": 152, "timestamp": "2025-12-01 09:05:59", "level": "CRITICAL",
      "message": "Out of memory: worker process 4182 killed by OOM killer",
      "template": "Out of memory: worker process <NUM> killed by OOM killer",
      "anomaly_score": -0.1214 }
  ],
  "windows": [
    { "start": "2025-12-03 10:02:00", "end": "2025-12-03 10:03:00",
      "events": 45, "warnings": 30, "errors": 0, "unseen_lines": 0,
      "reasons": ["events: 45 (normal max ~40)", "warnings: 30 (normal max ~5)"] }
  ],
  "summary": { "total_events": 320, "anomalies": 7, "normal_events": 313,
               "unparsed_lines": 0, "anomalous_windows": 1 }
}
```

`template` is `null` for a line that matches nothing seen in training. Window times are in UTC; timestamps without a timezone are treated as UTC.

Errors:

| Status | When |
|---|---|
| `409` | No model has been trained yet, or the saved model is from an older version or unreadable (retrain it) |
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
- Kafka ingestion for streaming logs
- Grafana dashboard for real-time monitoring

## License

Released under the [MIT License](LICENSE).

## Author

Muhammad Saad Majeed, AI, Data & Embedded Systems Engineer

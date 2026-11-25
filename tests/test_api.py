from datetime import datetime, timedelta

from joblib import dump

PLANTED_INCIDENTS = {
    "Out of memory",
    "Unhandled exception",
    "Database connection refused",
    "TLS certificate",
}


def test_root(client):
    assert client.get("/").status_code == 200


def test_analyze_before_training_returns_409(client):
    resp = client.post("/analyze", json={"log_text": "[2025-01-01 10:00:00] INFO: ok"})
    assert resp.status_code == 409
    assert "/train" in resp.json()["detail"]


def test_analyze_with_no_parseable_lines_returns_422(client, sample_logs):
    client.post("/train", json={"log_text": sample_logs})
    resp = client.post("/analyze", json={"log_text": "garbage\nmore garbage"})
    assert resp.status_code == 422


def test_train_with_too_little_data_returns_422(client):
    resp = client.post("/train", json={"log_text": "[2025-01-01 10:00:00] INFO: ok"})
    assert resp.status_code == 422


def test_train_rejects_bad_contamination(client, sample_logs):
    resp = client.post("/train", json={"log_text": sample_logs, "contamination": 0.9})
    assert resp.status_code == 422


def test_old_model_file_asks_for_retrain(client, isolated_model_path):
    dump({"version": 2, "model": None, "template_freq": {}}, isolated_model_path)
    resp = client.post("/analyze", json={"log_text": "[2025-01-01 10:00:00] INFO: ok"})
    assert resp.status_code == 409


def test_corrupt_model_file_asks_for_retrain(client, isolated_model_path):
    isolated_model_path.write_bytes(b"not a joblib file")
    resp = client.post("/analyze", json={"log_text": "[2025-01-01 10:00:00] INFO: ok"})
    assert resp.status_code == 409


def test_train_then_analyze_finds_planted_incidents(client, sample_logs, isolated_model_path):
    train = client.post("/train", json={"log_text": sample_logs, "contamination": 0.03})
    assert train.status_code == 200
    body = train.json()
    assert body["events_used"] == 320
    assert body["window_detection"] is True
    assert isolated_model_path.exists()

    resp = client.post("/analyze", json={"log_text": sample_logs})
    assert resp.status_code == 200
    body = resp.json()
    summary = body["summary"]
    assert summary["total_events"] == 320
    assert summary["anomalies"] + summary["normal_events"] == 320
    assert summary["unparsed_lines"] == 0
    assert summary["anomalous_windows"] == len(body["windows"]) == 0

    scores = [a["anomaly_score"] for a in body["anomalies"]]
    assert scores == sorted(scores) and all(s < 0 for s in scores)

    flagged = " | ".join(a["message"] for a in body["anomalies"])
    for incident in PLANTED_INCIDENTS:
        assert incident in flagged
    # The 25-minute stall before line 261 is the most anomalous event.
    assert body["anomalies"][0]["line_number"] == 261


def test_anomalies_include_their_template(client, sample_logs):
    client.post("/train", json={"log_text": sample_logs, "contamination": 0.03})
    anomalies = client.post("/analyze", json={"log_text": sample_logs}).json()["anomalies"]
    oom = next(a for a in anomalies if "Out of memory" in a["message"])
    assert oom["template"] == "Out of memory: worker process <NUM> killed by OOM killer"


def test_novel_error_in_short_snippet_is_flagged_but_routine_line_is_not(client, sample_logs):
    client.post("/train", json={"log_text": sample_logs, "contamination": 0.03})
    text = (
        "[2025-12-02 09:00:00] INFO: GET /api/orders/4521 200 22ms\n"
        "[2025-12-02 09:00:02] ERROR: Segfault in image resizer, core dumped\n"
    )
    anomalies = client.post("/analyze", json={"log_text": text}).json()["anomalies"]
    assert [a["line_number"] for a in anomalies] == [2]
    assert anomalies[0]["template"] is None


def test_burst_of_known_warnings_is_flagged(client, sample_logs):
    """Each 'Slow query' line is normal on its own; 30 of them in 30 seconds is not."""
    client.post("/train", json={"log_text": sample_logs, "contamination": 0.03})
    t, lines = datetime(2025, 12, 3, 10, 0, 0), []
    for i in range(60):
        t += timedelta(seconds=2)
        lines.append(f"[{t:%Y-%m-%d %H:%M:%S}] INFO: GET /api/orders/{2000 + i} 200 {10 + i % 40}ms")
    burst_start = len(lines) + 1
    for i in range(30):
        t += timedelta(seconds=1)
        lines.append(f"[{t:%Y-%m-%d %H:%M:%S}] WARN: Slow query on orders table took {600 + i}ms")
    for i in range(30):
        t += timedelta(seconds=2)
        lines.append(f"[{t:%Y-%m-%d %H:%M:%S}] INFO: GET /api/orders/{3000 + i} 200 {10 + i % 40}ms")

    body = client.post("/analyze", json={"log_text": "\n".join(lines)}).json()
    flagged = {a["line_number"] for a in body["anomalies"]}
    burst_lines = set(range(burst_start, burst_start + 30))
    assert len(flagged & burst_lines) >= 20
    assert len(flagged - burst_lines) <= 3

    assert body["summary"]["anomalous_windows"] == 1
    window = body["windows"][0]
    assert window["warnings"] == 30
    assert any(r.startswith("warnings") for r in window["reasons"])

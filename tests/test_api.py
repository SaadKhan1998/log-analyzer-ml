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
    dump(object(), isolated_model_path)  # the pre-fix format saved a bare model
    resp = client.post("/analyze", json={"log_text": "[2025-01-01 10:00:00] INFO: ok"})
    assert resp.status_code == 409


def test_train_then_analyze_finds_planted_incidents(client, sample_logs, isolated_model_path):
    train = client.post("/train", json={"log_text": sample_logs, "contamination": 0.02})
    assert train.status_code == 200
    assert train.json()["events_used"] == 320
    assert isolated_model_path.exists()

    resp = client.post("/analyze", json={"log_text": sample_logs})
    assert resp.status_code == 200
    body = resp.json()
    summary = body["summary"]
    assert summary["total_events"] == 320
    assert summary["anomalies"] + summary["normal_events"] == 320
    assert summary["unparsed_lines"] == 0

    scores = [a["anomaly_score"] for a in body["anomalies"]]
    assert scores == sorted(scores) and all(s < 0 for s in scores)

    flagged = " | ".join(a["message"] for a in body["anomalies"])
    for incident in PLANTED_INCIDENTS:
        assert incident in flagged


def test_novel_error_is_flagged_but_routine_line_is_not(client, sample_logs):
    client.post("/train", json={"log_text": sample_logs, "contamination": 0.02})
    text = (
        "[2025-12-02 09:00:00] INFO: GET /api/orders/4521 200 22ms\n"
        "[2025-12-02 09:00:02] ERROR: Segfault in image resizer, core dumped\n"
    )
    anomalies = client.post("/analyze", json={"log_text": text}).json()["anomalies"]
    assert [a["line_number"] for a in anomalies] == [2]

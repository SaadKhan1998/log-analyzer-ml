from datetime import datetime, timedelta

import pytest

from app.ml.feature_extractor import FEATURE_COLUMNS, LogFeaturizer
from app.utils import extract_events


def make_log(entries, start=datetime(2025, 1, 1, 10, 0, 0)):
    """entries: list of (seconds_after_previous, level, message)."""
    t, lines = start, []
    for delta, level, message in entries:
        t += timedelta(seconds=delta)
        lines.append(f"[{t:%Y-%m-%d %H:%M:%S}] {level}: {message}")
    return extract_events("\n".join(lines)).events


@pytest.fixture
def steady_traffic():
    """Ten minutes of routine traffic, one line every 2 seconds."""
    return make_log(
        [(2, "INFO", f"User {i % 50} logged in from 10.0.0.{i % 200}") for i in range(150)]
        + [(2, "INFO", f"GET /api/orders/{1000 + i} 200 {10 + i % 40}ms") for i in range(150)]
    )


def test_drain_groups_messages_with_different_variables(steady_traffic):
    featurizer = LogFeaturizer().fit(steady_traffic)
    assert featurizer.n_templates == 2
    assert set(featurizer.templates().values()) == {
        "User <NUM> logged in from <IP>",
        "GET /api/orders/<NUM> <NUM> <NUM>ms",
    }


def test_drain_generalizes_words_not_covered_by_masks():
    # Usernames are not numbers, so regex masking alone would split these.
    events = make_log(
        [(1, "INFO", f"User {name} logged in from web") for name in ["alice", "bob", "carol", "dave"] * 5]
    )
    featurizer = LogFeaturizer().fit(events)
    assert list(featurizer.templates().values()) == ["User <*> logged in from web"]
    unseen = make_log([(1, "INFO", "User zed logged in from web")])
    assert featurizer.template_ids(unseen) == [1]


def test_unseen_message_has_full_rarity(steady_traffic):
    featurizer = LogFeaturizer().fit(steady_traffic)
    events = make_log(
        [(1, "INFO", "User 7 logged in from 10.0.0.9"), (1, "ERROR", "Disk array corrupted on node seven")]
    )
    df = featurizer.transform(events)
    assert list(df.columns) == FEATURE_COLUMNS
    assert featurizer.template_ids(events)[1] is None
    assert df.loc[1, "template_rarity"] == 1.0
    assert df.loc[0, "template_rarity"] < 0.1
    assert df.loc[1, "has_failure_word"] == 1
    assert df.loc[1, "level_severity"] > df.loc[0, "level_severity"]


def test_window_features_capture_a_burst(steady_traffic):
    featurizer = LogFeaturizer().fit(steady_traffic)
    events = steady_traffic[:100] + make_log(
        [(0.5, "WARN", f"Slow query took {500 + i}ms") for i in range(40)],
        start=datetime(2025, 1, 1, 10, 3, 21),
    )
    df = featurizer.transform(events)
    calm, burst = df.iloc[90], df.iloc[-1]
    assert burst["win_log_event_count"] > calm["win_log_event_count"]
    assert burst["win_problem_ratio"] > 0.5 and calm["win_problem_ratio"] == 0
    assert burst["win_repeat_ratio"] > 0.5


def test_first_lines_do_not_look_like_a_traffic_drop(steady_traffic):
    featurizer = LogFeaturizer().fit(steady_traffic)
    df = featurizer.transform(steady_traffic)
    assert df.loc[0, "win_log_event_count"] == df.loc[10, "win_log_event_count"]


def test_short_snippet_gets_neutral_traffic_volume(steady_traffic):
    featurizer = LogFeaturizer().fit(steady_traffic)
    snippet = make_log([(1, "INFO", "User 3 logged in from 10.0.0.3")] * 2)
    df = featurizer.transform(snippet)
    assert (df["win_log_event_count"] == featurizer.neutral_log_event_count).all()


def test_missing_timestamps_fall_back_to_line_count_windows(steady_traffic):
    featurizer = LogFeaturizer().fit(steady_traffic)
    events = [{"timestamp": "yesterday", "level": "INFO", "message": "hi"}] * 30
    df = featurizer.transform(events)
    assert df["log_gap_seconds"].eq(0).all()
    assert df["win_log_event_count"].iloc[-1] == pytest.approx(3.0445, abs=1e-3)  # log1p(20)
    assert featurizer.window_report(events) == []


def test_window_report_flags_error_burst_minute(steady_traffic):
    featurizer = LogFeaturizer().fit(steady_traffic)
    events = steady_traffic[:60] + make_log(
        [(1, "ERROR", "Upstream payment gateway timed out") for _ in range(20)],
        start=datetime(2025, 1, 1, 10, 2, 0),
    )
    windows = featurizer.window_report(events)
    assert len(windows) == 1
    assert windows[0]["start"] == "2025-01-01 10:02:00"
    assert windows[0]["errors"] == 20
    assert windows[0]["unseen_lines"] == 20  # the timeout message never appeared in training
    reasons = " | ".join(windows[0]["reasons"])
    assert "errors: 20" in reasons and "unseen lines: 20" in reasons


def test_training_without_timestamps_disables_window_report():
    events = [{"timestamp": "", "level": "INFO", "message": f"tick {i}"} for i in range(30)]
    featurizer = LogFeaturizer().fit(events)
    assert featurizer.window_thresholds is None
    assert featurizer.window_report(events) == []

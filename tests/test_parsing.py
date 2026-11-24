from app.ml.feature_extractor import FEATURE_COLUMNS, build_template_frequencies, extract_features, to_template
from app.utils import extract_events


def test_parses_bracketed_format():
    result = extract_events("[2025-01-01 10:00:00] ERROR: Disk full")
    assert result.unparsed == 0
    assert result.events == [
        {"line_number": 1, "timestamp": "2025-01-01 10:00:00", "level": "ERROR", "message": "Disk full"}
    ]


def test_parses_iso_formats_and_normalizes_levels():
    text = "\n".join(
        [
            "2025-01-01T10:00:00.123Z WARNING Cache miss ratio high",
            "2025-01-01 10:00:01,500 [info] Request served",
            "[2025-01-01 10:00:02] fatal: Kernel panic",
            "[2025-01-01 10:00:03] Err: Bad thing",
        ]
    )
    levels = [e["level"] for e in extract_events(text).events]
    assert levels == ["WARN", "INFO", "CRITICAL", "ERROR"]


def test_counts_unparsed_lines_and_skips_blank_lines():
    text = "[2025-01-01 10:00:00] INFO: ok\n\n   \nnot a log line\r\nanother junk line"
    result = extract_events(text)
    assert len(result.events) == 1
    assert result.unparsed == 2


def test_line_numbers_point_at_original_lines():
    text = "junk\n\n[2025-01-01 10:00:00] INFO: ok"
    assert extract_events(text).events[0]["line_number"] == 3


def test_template_masks_variable_parts():
    assert to_template("User 17 logged in from 10.0.0.1") == to_template("User 42 logged in from 10.0.3.9")
    assert to_template("id 550e8400-e29b-41d4-a716-446655440000 at 0xdeadbeef") == "id <*> at <*>"


def test_features_shape_and_values():
    train = extract_events(
        "[2025-01-01 10:00:00] INFO: User 1 logged in\n"
        "[2025-01-01 10:00:10] INFO: User 2 logged in\n"
    ).events
    freq = build_template_frequencies(train)
    events = extract_events(
        "[2025-01-01 10:00:00] INFO: User 3 logged in\n"
        "[2025-01-01 11:00:00] ERROR: Connection refused by upstream\n"
    ).events
    df = extract_features(events, freq)
    assert list(df.columns) == FEATURE_COLUMNS
    known, novel = df.iloc[0], df.iloc[1]
    assert known["template_rarity"] == 0.0 and novel["template_rarity"] == 1.0
    assert known["has_failure_word"] == 0 and novel["has_failure_word"] == 1
    assert novel["level_severity"] > known["level_severity"]
    assert known["log_gap_seconds"] == 0.0 and novel["log_gap_seconds"] > 8  # log1p(3600) ~ 8.19


def test_features_handle_empty_and_unparseable_timestamps():
    assert extract_features([], {}).empty
    events = [{"timestamp": "yesterday", "level": "INFO", "message": "hi"}] * 2
    assert extract_features(events, {}).log_gap_seconds.tolist() == [0.0, 0.0]

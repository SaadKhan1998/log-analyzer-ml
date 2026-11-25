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

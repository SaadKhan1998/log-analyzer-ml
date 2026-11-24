"""Parsing raw log text into structured events.

Supported line formats (level names are case-insensitive):

    [2025-01-01 10:00:00] ERROR: Disk full
    2025-01-01T10:00:00.123Z WARNING Cache miss ratio high
    2025-01-01 10:00:00,123 [INFO] Request served

Lines that match none of these are counted as ``unparsed`` rather than
silently disappearing, so callers can tell when the format is wrong.
"""

import re
from dataclasses import dataclass, field
from datetime import datetime

# Longer alternatives first so e.g. WARNING is not cut short to WARN.
_LEVELS = r"WARNING|WARN|ERROR|ERR|CRITICAL|CRIT|FATAL|NOTICE|INFO|DEBUG|TRACE"
_ISO_TS = (
    r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:[.,]\d+)?"
    r"(?:Z|[+-]\d{2}:?\d{2})?"
)

_BRACKETED = re.compile(
    rf"^\[(?P<timestamp>[^\]]*)\]\s+\[?(?P<level>{_LEVELS})\]?\b:?\s*(?P<message>.*)$",
    re.IGNORECASE,
)
_ISO = re.compile(
    rf"^(?P<timestamp>{_ISO_TS})\s+\[?(?P<level>{_LEVELS})\]?\b:?\s*(?P<message>.*)$",
    re.IGNORECASE,
)

_LEVEL_ALIASES = {
    "WARNING": "WARN",
    "ERR": "ERROR",
    "CRIT": "CRITICAL",
    "FATAL": "CRITICAL",
}


@dataclass
class ParseResult:
    events: list[dict] = field(default_factory=list)
    unparsed: int = 0


def normalize_level(level: str) -> str:
    level = level.upper()
    return _LEVEL_ALIASES.get(level, level)


def parse_timestamp(value: str) -> datetime | None:
    """Best-effort ISO-8601 parsing; returns None when the timestamp is unusable."""
    value = value.strip().replace(",", ".")
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def extract_events(log_text: str) -> ParseResult:
    """Split raw log text into events. Blank lines are ignored."""
    result = ParseResult()
    for line_number, raw in enumerate(log_text.splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        match = _BRACKETED.match(line) or _ISO.match(line)
        if not match:
            result.unparsed += 1
            continue
        result.events.append(
            {
                "line_number": line_number,
                "timestamp": match.group("timestamp").strip(),
                "level": normalize_level(match.group("level")),
                "message": match.group("message").strip(),
            }
        )
    return result

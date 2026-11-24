"""Turning parsed events into numeric features for the anomaly model.

Features per event:
  * level_severity    - TRACE/DEBUG=0, INFO=1, WARN=2, ERROR=3, CRITICAL=4
  * msg_len           - length of the message (timestamp/level excluded)
  * template_rarity   - how unusual the message *shape* was in the training
                        data (numbers, IDs, IPs masked out); unseen = 1.0
  * has_failure_word  - message mentions exception/timeout/failed/...
  * log_gap_seconds   - log(1 + seconds since the previous event in the batch)
"""

import math
import re
from collections import Counter

import pandas as pd

from app.utils import parse_timestamp

FEATURE_COLUMNS = [
    "level_severity",
    "msg_len",
    "template_rarity",
    "has_failure_word",
    "log_gap_seconds",
]

LEVEL_SEVERITY = {
    "TRACE": 0,
    "DEBUG": 0,
    "INFO": 1,
    "NOTICE": 1,
    "WARN": 2,
    "ERROR": 3,
    "CRITICAL": 4,
}

_FAILURE_WORDS = re.compile(
    r"\b(exception|traceback|fail(?:ed|ure|s)?|timeout|timed out|refused|denied|"
    r"panic|segfault|out of memory|oom|unreachable|corrupt(?:ed)?)\b",
    re.IGNORECASE,
)

# Order matters: mask the most specific shapes first.
_MASKS = [
    re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b", re.I),
    re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}(?::\d+)?\b"),
    re.compile(r"\b0x[0-9a-f]+\b", re.I),
    re.compile(r"\b[0-9a-f]{12,}\b", re.I),
    re.compile(r"\d+(?:\.\d+)?"),
]


def to_template(message: str) -> str:
    """Mask variable parts so 'user 17 logged in' and 'user 42 logged in' match."""
    for pattern in _MASKS:
        message = pattern.sub("<*>", message)
    return message.lower()


def build_template_frequencies(events: list[dict]) -> dict[str, float]:
    counts = Counter(to_template(e["message"]) for e in events)
    total = sum(counts.values())
    return {template: n / total for template, n in counts.items()}


def _gaps_seconds(events: list[dict]) -> list[float]:
    gaps, previous = [], None
    for event in events:
        current = parse_timestamp(event["timestamp"])
        if current is not None and previous is not None:
            try:
                gap = max((current - previous).total_seconds(), 0.0)
            except TypeError:  # mixing timezone-aware and naive timestamps
                gap = 0.0
        else:
            gap = 0.0
        gaps.append(gap)
        if current is not None:
            previous = current
    return gaps


def extract_features(events: list[dict], template_freq: dict[str, float]) -> pd.DataFrame:
    if not events:
        return pd.DataFrame(columns=FEATURE_COLUMNS, dtype=float)

    max_freq = max(template_freq.values(), default=1.0)
    rows = []
    for event, gap in zip(events, _gaps_seconds(events)):
        message = event["message"]
        freq = template_freq.get(to_template(message), 0.0)
        rows.append(
            {
                "level_severity": LEVEL_SEVERITY.get(event["level"], 1),
                "msg_len": len(message),
                "template_rarity": 1.0 - freq / max_freq,
                "has_failure_word": int(bool(_FAILURE_WORDS.search(message))),
                "log_gap_seconds": math.log1p(gap),
            }
        )
    return pd.DataFrame(rows, columns=FEATURE_COLUMNS, dtype=float)

"""Turning parsed events into numeric features for the anomaly model.

A ``LogFeaturizer`` is fitted on training logs and saved with the model, so
/analyze computes features exactly the way /train did.

Per-line features:
  * level_severity      - TRACE/DEBUG=0, INFO=1, WARN=2, ERROR=3, CRITICAL=4
  * msg_len             - length of the message (timestamp/level excluded)
  * template_rarity     - how rare the line's Drain template was in training
                          (0 = most common template, 1 = never seen)
  * has_failure_word    - message mentions exception/timeout/failed/...
  * log_gap_seconds     - log(1 + seconds since the previous event)

Sliding-window features (trailing window ending at each line):
  * win_log_event_count - log(1 + events in the window): traffic bursts
  * win_problem_ratio   - share of WARN-or-worse events in the window
  * win_repeat_ratio    - share of window events with this line's template:
                          the same message suddenly repeating

The window is the last ``window_seconds`` of log time; for lines without a
usable timestamp it falls back to the previous ``fallback_window_events``
lines. Lines without a full window of history (the start of a batch, or a
short pasted snippet) get the typical training volume, and both ratios are
smoothed toward their training averages, so a handful of lines is not
mistaken for a traffic drop or a burst.

Separately, ``window_report`` splits logs into fixed time buckets and flags
buckets whose volume, warnings, errors or unseen lines are far above
anything seen in the training buckets.
"""

import math
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from drain3 import TemplateMiner
from drain3.masking import MaskingInstruction
from drain3.template_miner_config import TemplateMinerConfig

from app.utils import parse_timestamp

FEATURE_COLUMNS = [
    "level_severity",
    "msg_len",
    "template_rarity",
    "has_failure_word",
    "log_gap_seconds",
    "win_log_event_count",
    "win_problem_ratio",
    "win_repeat_ratio",
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
PROBLEM_SEVERITY = LEVEL_SEVERITY["WARN"]
ERROR_SEVERITY = LEVEL_SEVERITY["ERROR"]

_FAILURE_WORDS = re.compile(
    r"\b(exception|traceback|fail(?:ed|ure|s)?|timeout|timed out|refused|denied|"
    r"panic|segfault|out of memory|oom|unreachable|corrupt(?:ed)?)\b",
    re.IGNORECASE,
)

# Masked before Drain sees the message, so obvious variables never split a
# template. Order matters: most specific shapes first.
_MASKS = [
    (r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b", "UUID"),
    (r"\b\d{1,3}(?:\.\d{1,3}){3}(?::\d+)?\b", "IP"),
    (r"\b0x[0-9a-fA-F]+\b", "HEX"),
    (r"\b[0-9a-fA-F]{12,}\b", "HEX"),
    (r"\d+(?:\.\d+)?", "NUM"),
]

WINDOW_METRICS = ["events", "warnings", "errors", "unseen_lines"]

# Pseudo-count for smoothing window ratios toward their training averages:
# a 3-line window barely moves them, a 30-line window mostly speaks for itself.
RATIO_PRIOR_WEIGHT = 5


def _event_seconds(event: dict) -> float | None:
    """Event time as epoch seconds; naive timestamps are treated as UTC."""
    ts = parse_timestamp(event["timestamp"])
    if ts is None:
        return None
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts.timestamp()


def _new_template_miner() -> TemplateMiner:
    config = TemplateMinerConfig()
    config.profiling_enabled = False
    config.masking_instructions = [MaskingInstruction(p, name) for p, name in _MASKS]
    return TemplateMiner(config=config)


class LogFeaturizer:
    def __init__(
        self,
        window_seconds: int = 60,
        fallback_window_events: int = 20,
        bucket_seconds: int = 60,
    ):
        self.window_seconds = window_seconds
        self.fallback_window_events = fallback_window_events
        self.bucket_seconds = bucket_seconds
        self.miner = _new_template_miner()
        self.template_freq: dict[int, float] = {}
        self.window_thresholds: dict[str, float] | None = None
        # Typical traffic volume and problem share in training, used for lines
        # without enough history (a 2-line snippet is not a traffic drop).
        self.neutral_log_event_count: float | None = None
        self.problem_share: float = 0.0

    # ---- fitting -------------------------------------------------------

    def fit(self, events: list[dict]) -> "LogFeaturizer":
        self.miner = _new_template_miner()
        for event in events:
            self.miner.add_log_message(event["message"])

        clusters = self.miner.drain.clusters
        total = sum(c.size for c in clusters) or 1
        self.template_freq = {c.cluster_id: c.size / total for c in clusters}

        self.problem_share = sum(
            LEVEL_SEVERITY.get(e["level"], 1) >= PROBLEM_SEVERITY for e in events
        ) / len(events)

        # Median volume over lines that have a full window of history.
        seconds = [_event_seconds(e) for e in events]
        earliest = self._running_earliest(seconds)
        full = [
            math.log1p(len(self._window_indices(i, seconds)))
            for i in range(len(events))
            if self._has_full_history(i, seconds, earliest)
        ]
        self.neutral_log_event_count = (
            float(np.median(full)) if full else math.log1p(len(events))
        )

        buckets = self._bucket_metrics(events, self.template_ids(events))
        if buckets:
            self.window_thresholds = {}
            for metric in WINDOW_METRICS:
                q99 = float(np.quantile([b[metric] for b in buckets], 0.99))
                self.window_thresholds[metric] = max(q99 * 1.5, q99 + 3)
        else:
            self.window_thresholds = None  # no usable timestamps in training
        return self

    @property
    def n_templates(self) -> int:
        return len(self.template_freq)

    def templates(self) -> dict[int, str]:
        return {c.cluster_id: c.get_template() for c in self.miner.drain.clusters}

    def template_ids(self, events: list[dict]) -> list[int | None]:
        """Template (cluster) ID per event; None for messages unlike any training template."""
        ids = []
        for event in events:
            cluster = self.miner.match(event["message"], full_search_strategy="fallback")
            ids.append(cluster.cluster_id if cluster is not None else None)
        return ids

    # ---- per-line features ----------------------------------------------

    def transform(self, events: list[dict]) -> pd.DataFrame:
        if not events:
            return pd.DataFrame(columns=FEATURE_COLUMNS, dtype=float)

        seconds = [_event_seconds(e) for e in events]
        earliest = self._running_earliest(seconds)
        severities = [LEVEL_SEVERITY.get(e["level"], 1) for e in events]
        template_ids = self.template_ids(events)
        max_freq = max(self.template_freq.values(), default=1.0)

        rows = []
        previous_seconds = None
        for i, event in enumerate(events):
            message = event["message"]
            freq = self.template_freq.get(template_ids[i], 0.0)

            gap = 0.0
            if seconds[i] is not None and previous_seconds is not None:
                gap = max(seconds[i] - previous_seconds, 0.0)
            if seconds[i] is not None:
                previous_seconds = seconds[i]

            window = self._window_indices(i, seconds)
            n = len(window)
            problems = sum(1 for j in window if severities[j] >= PROBLEM_SEVERITY)
            repeats = sum(1 for j in window if template_ids[j] == template_ids[i])
            if self._has_full_history(i, seconds, earliest):
                log_count = math.log1p(n)
            else:
                log_count = self.neutral_log_event_count
            k = RATIO_PRIOR_WEIGHT

            rows.append(
                {
                    "level_severity": severities[i],
                    "msg_len": len(message),
                    "template_rarity": 1.0 - freq / max_freq,
                    "has_failure_word": int(bool(_FAILURE_WORDS.search(message))),
                    "log_gap_seconds": math.log1p(gap),
                    "win_log_event_count": log_count,
                    "win_problem_ratio": (problems + k * self.problem_share) / (n + k),
                    "win_repeat_ratio": (repeats + k * freq) / (n + k),
                }
            )
        return pd.DataFrame(rows, columns=FEATURE_COLUMNS, dtype=float)

    def _has_full_history(self, i: int, seconds: list[float | None], earliest: list[float | None]) -> bool:
        """Whether the batch reaches back a whole window before event i."""
        if seconds[i] is None:
            return i >= self.fallback_window_events - 1
        return seconds[i] - earliest[i] >= self.window_seconds

    @staticmethod
    def _running_earliest(seconds: list[float | None]) -> list[float | None]:
        earliest, current = [], None
        for s in seconds:
            if s is not None and (current is None or s < current):
                current = s
            earliest.append(current)
        return earliest

    def _window_indices(self, i: int, seconds: list[float | None]) -> list[int]:
        """Indices of events in the trailing window ending at event i (inclusive)."""
        if seconds[i] is None:
            return list(range(max(0, i - self.fallback_window_events + 1), i + 1))
        window = [i]
        j = i - 1
        while j >= 0:
            if seconds[j] is not None:
                age = seconds[i] - seconds[j]
                if age > self.window_seconds:
                    break
                if age >= 0:
                    window.append(j)
            j -= 1
        return window

    # ---- time-window report ---------------------------------------------

    def _bucket_metrics(self, events: list[dict], template_ids: list[int | None]) -> list[dict]:
        buckets: dict[int, dict] = defaultdict(lambda: dict.fromkeys(WINDOW_METRICS, 0))
        for event, template_id in zip(events, template_ids):
            secs = _event_seconds(event)
            if secs is None:
                continue
            bucket = buckets[int(secs // self.bucket_seconds)]
            severity = LEVEL_SEVERITY.get(event["level"], 1)
            bucket["events"] += 1
            bucket["warnings"] += int(severity == PROBLEM_SEVERITY)
            bucket["errors"] += int(severity >= ERROR_SEVERITY)
            bucket["unseen_lines"] += int(template_id is None)

        return [
            {"bucket": key, **metrics} for key, metrics in sorted(buckets.items())
        ]

    def window_report(self, events: list[dict]) -> list[dict]:
        """Time buckets whose metrics are far above the busiest training buckets."""
        if not self.window_thresholds:
            return []
        flagged = []
        for b in self._bucket_metrics(events, self.template_ids(events)):
            reasons = [
                f"{metric.replace('_', ' ')}: {b[metric]} (normal max ~{self.window_thresholds[metric]:.0f})"
                for metric in WINDOW_METRICS
                if b[metric] > self.window_thresholds[metric]
            ]
            if reasons:
                start = b["bucket"] * self.bucket_seconds
                flagged.append(
                    {
                        "start": _iso(start),
                        "end": _iso(start + self.bucket_seconds),
                        **{metric: b[metric] for metric in WINDOW_METRICS},
                        "reasons": reasons,
                    }
                )
        return flagged


def _iso(epoch_seconds: float) -> str:
    return datetime.fromtimestamp(epoch_seconds, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

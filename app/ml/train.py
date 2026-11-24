from sklearn.ensemble import IsolationForest

from app.utils import extract_events

from .feature_extractor import build_template_frequencies, extract_features
from .model import save_model

MIN_TRAINING_EVENTS = 10


class NotEnoughDataError(ValueError):
    pass


def train_model(log_text: str, contamination: float = 0.05) -> dict:
    """Fit an Isolation Forest on logs parsed the same way /analyze parses them."""
    parsed = extract_events(log_text)
    if len(parsed.events) < MIN_TRAINING_EVENTS:
        raise NotEnoughDataError(
            f"Need at least {MIN_TRAINING_EVENTS} parseable log lines to train, "
            f"got {len(parsed.events)} ({parsed.unparsed} lines did not match a known format)."
        )

    template_freq = build_template_frequencies(parsed.events)
    features = extract_features(parsed.events, template_freq)

    model = IsolationForest(
        n_estimators=150, contamination=contamination, random_state=42
    )
    model.fit(features)
    save_model(model, template_freq)

    return {
        "status": "Model trained successfully",
        "events_used": len(parsed.events),
        "unparsed_lines": parsed.unparsed,
        "distinct_templates": len(template_freq),
    }

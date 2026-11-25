from sklearn.ensemble import IsolationForest

from app.utils import extract_events

from .feature_extractor import LogFeaturizer
from .model import save_model

MIN_TRAINING_EVENTS = 10


class NotEnoughDataError(ValueError):
    pass


def train_model(log_text: str, contamination: float = 0.05) -> dict:
    """Fit the featurizer and an Isolation Forest on logs parsed the same way /analyze parses them."""
    parsed = extract_events(log_text)
    if len(parsed.events) < MIN_TRAINING_EVENTS:
        raise NotEnoughDataError(
            f"Need at least {MIN_TRAINING_EVENTS} parseable log lines to train, "
            f"got {len(parsed.events)} ({parsed.unparsed} lines did not match a known format)."
        )

    featurizer = LogFeaturizer().fit(parsed.events)
    features = featurizer.transform(parsed.events)

    model = IsolationForest(
        n_estimators=150, contamination=contamination, random_state=42
    )
    model.fit(features)
    save_model(model, featurizer)

    return {
        "status": "Model trained successfully",
        "events_used": len(parsed.events),
        "unparsed_lines": parsed.unparsed,
        "distinct_templates": featurizer.n_templates,
        "window_detection": featurizer.window_thresholds is not None,
    }

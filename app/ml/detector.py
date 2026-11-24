from .feature_extractor import extract_features
from .model import load_model


def detect_anomalies(events: list[dict]) -> list[dict]:
    """Return the events the model flags, each with an ``anomaly_score``.

    Scores come from IsolationForest.decision_function: negative means
    anomalous, and lower means more anomalous. Results are sorted most
    anomalous first. Raises ModelNotTrainedError if no model is saved.
    """
    bundle = load_model()
    if not events:
        return []

    model = bundle["model"]
    features = extract_features(events, bundle["template_freq"])
    preds = model.predict(features)
    scores = model.decision_function(features)

    anomalies = [
        {**event, "anomaly_score": round(float(score), 4)}
        for event, pred, score in zip(events, preds, scores)
        if pred == -1
    ]
    return sorted(anomalies, key=lambda e: e["anomaly_score"])

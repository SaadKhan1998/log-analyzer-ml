from .model import load_model


def detect_anomalies(events: list[dict]) -> tuple[list[dict], list[dict]]:
    """Return (anomalous lines, anomalous time windows).

    Each anomalous line carries an ``anomaly_score`` from
    IsolationForest.decision_function (negative = anomalous, lower = more
    anomalous) and the Drain ``template`` it matched (None if unseen in
    training). Lines are sorted most anomalous first.

    Raises ModelNotTrainedError if no usable model is saved.
    """
    bundle = load_model()
    if not events:
        return [], []

    model, featurizer = bundle["model"], bundle["featurizer"]
    features = featurizer.transform(events)
    preds = model.predict(features)
    scores = model.decision_function(features)
    templates = featurizer.templates()
    template_ids = featurizer.template_ids(events)

    anomalies = [
        {
            **event,
            "template": templates.get(template_id),
            "anomaly_score": round(float(score), 4),
        }
        for event, pred, score, template_id in zip(events, preds, scores, template_ids)
        if pred == -1
    ]
    anomalies.sort(key=lambda e: e["anomaly_score"])
    return anomalies, featurizer.window_report(events)

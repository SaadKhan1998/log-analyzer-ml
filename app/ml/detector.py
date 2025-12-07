from .feature_extractor import extract_features
from .model import load_model

def detect_anomalies(events):
    model = load_model()

    if model is None:
        raise Exception("❌ No model found. Train a model first using /train.")

    features = extract_features(events)
    preds = model.predict(features)

    anomalies = []
    for idx, p in enumerate(preds):
        if p == -1:
            anomalies.append(events[idx])

    return anomalies

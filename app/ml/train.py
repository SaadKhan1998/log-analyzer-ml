import pandas as pd
from sklearn.ensemble import IsolationForest
from .feature_extractor import extract_features
from .model import save_model

def train_model(log_text: str):
    logs = log_text.split("\n")
    
    # Convert logs into event structure
    events = []
    for line in logs:
        if line.strip():
            events.append({"timestamp":"", "level":"INFO", "message":line})
    
    features = extract_features(events)
    
    model = IsolationForest(n_estimators=150, contamination=0.05, random_state=42)
    model.fit(features)
    save_model(model)
    
    return "Model trained successfully"

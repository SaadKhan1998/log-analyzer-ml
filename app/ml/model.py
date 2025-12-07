from sklearn.ensemble import IsolationForest
from joblib import dump, load
import os

MODEL_PATH = "model.joblib"

def load_model():
    if os.path.exists(MODEL_PATH):
        return load(MODEL_PATH)
    return None

def save_model(model):
    dump(model, MODEL_PATH)

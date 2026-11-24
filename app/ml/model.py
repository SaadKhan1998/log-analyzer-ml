"""Saving and loading the trained model bundle.

The bundle holds the Isolation Forest *and* the template frequencies learned
at training time, since feature extraction needs both.

The file location defaults to ``<repo>/models/model.joblib`` (independent of
the current working directory) and can be overridden with the
``LOG_ANALYZER_MODEL_PATH`` environment variable.
"""

import os
from pathlib import Path

from joblib import dump, load

BUNDLE_VERSION = 2
DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "model.joblib"


class ModelNotTrainedError(RuntimeError):
    pass


def model_path() -> Path:
    return Path(os.environ.get("LOG_ANALYZER_MODEL_PATH", DEFAULT_MODEL_PATH))


def save_model(model, template_freq: dict[str, float]) -> Path:
    path = model_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    dump(
        {"version": BUNDLE_VERSION, "model": model, "template_freq": template_freq},
        path,
    )
    return path


def load_model() -> dict:
    path = model_path()
    if not path.exists():
        raise ModelNotTrainedError("No trained model found. Call POST /train first.")
    bundle = load(path)
    if not isinstance(bundle, dict) or bundle.get("version") != BUNDLE_VERSION:
        raise ModelNotTrainedError(
            "Saved model is from an older version of this tool. Retrain with POST /train."
        )
    return bundle

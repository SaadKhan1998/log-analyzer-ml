import pandas as pd

def extract_features(events):
    """
    Converts events into a numerical DataFrame for ML models.
    Feature ideas:
      • error counts
      • message length
      • event frequency
    """
    df = pd.DataFrame(events)
    df["msg_len"] = df["message"].apply(len)
    df["level_code"] = df["level"].map({"INFO":0, "WARN":1, "ERROR":2}).fillna(0)

    return df[["msg_len", "level_code"]]

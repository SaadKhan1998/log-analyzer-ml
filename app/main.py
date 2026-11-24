from fastapi import FastAPI, HTTPException

from app.ml.detector import detect_anomalies
from app.ml.model import ModelNotTrainedError
from app.ml.train import NotEnoughDataError, train_model
from app.schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    Summary,
    TrainRequest,
    TrainResponse,
)
from app.utils import extract_events

app = FastAPI(title="ML Log Analyzer", version="1.1")


@app.get("/")
def root():
    return {"message": "Log Anomaly Detection API is running"}


@app.post("/train", response_model=TrainResponse)
def train(req: TrainRequest):
    try:
        return train_model(req.log_text, contamination=req.contamination)
    except NotEnoughDataError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest):
    parsed = extract_events(req.log_text)
    if not parsed.events:
        raise HTTPException(
            status_code=422,
            detail=(
                f"No parseable log lines found ({parsed.unparsed} lines did not match "
                "a known format). Expected e.g. '[2025-01-01 10:00:00] ERROR: message'."
            ),
        )

    try:
        anomalies = detect_anomalies(parsed.events)
    except ModelNotTrainedError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    summary = Summary(
        total_events=len(parsed.events),
        anomalies=len(anomalies),
        normal_events=len(parsed.events) - len(anomalies),
        unparsed_lines=parsed.unparsed,
    )
    return AnalyzeResponse(anomalies=anomalies, summary=summary)

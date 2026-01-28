from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pathlib import Path
from app.schemas import AnalyzeRequest, AnalyzeResponse
from app.utils import extract_events
from app.ml.train import train_model
from app.ml.detector import detect_anomalies

app = FastAPI(title="ML Log Analyzer", version="1.0")

GAME_PAGE_PATH = Path(__file__).with_name("game.html")

@app.get("/", response_class=HTMLResponse)
def root():
    return HTMLResponse(GAME_PAGE_PATH.read_text(encoding="utf-8"))

@app.get("/status")
def status():
    return {"message": "Log Anomaly Detection API is running"}

@app.get("/game", response_class=HTMLResponse)
def game():
    return HTMLResponse(GAME_PAGE_PATH.read_text(encoding="utf-8"))

@app.post("/train")
def train(req: AnalyzeRequest):
    message = train_model(req.log_text)
    return {"status": message}

@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest):
    events = extract_events(req.log_text)
    anomalies = detect_anomalies(events)

    summary = {
        "total_events": len(events),
        "anomalies": len(anomalies),
        "normal_events": len(events) - len(anomalies)
    }

    return AnalyzeResponse(anomalies=anomalies, summary=summary)

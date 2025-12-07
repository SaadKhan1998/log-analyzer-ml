from pydantic import BaseModel

class AnalyzeRequest(BaseModel):
    log_text: str

class AnalyzeResponse(BaseModel):
    anomalies: list
    summary: dict

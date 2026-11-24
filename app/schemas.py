from pydantic import BaseModel, Field


class AnalyzeRequest(BaseModel):
    log_text: str = Field(..., description="Raw log text, one event per line.")


class TrainRequest(AnalyzeRequest):
    contamination: float = Field(
        0.05,
        gt=0,
        le=0.5,
        description="Expected share of anomalous lines in the training data.",
    )


class TrainResponse(BaseModel):
    status: str
    events_used: int
    unparsed_lines: int
    distinct_templates: int


class Anomaly(BaseModel):
    line_number: int
    timestamp: str
    level: str
    message: str
    anomaly_score: float


class Summary(BaseModel):
    total_events: int
    anomalies: int
    normal_events: int
    unparsed_lines: int


class AnalyzeResponse(BaseModel):
    anomalies: list[Anomaly]
    summary: Summary

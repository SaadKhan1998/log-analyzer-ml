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
    window_detection: bool = Field(
        ..., description="False when training logs had no usable timestamps."
    )


class Anomaly(BaseModel):
    line_number: int
    timestamp: str
    level: str
    message: str
    template: str | None = Field(
        None, description="Drain template this line matched; null if unseen in training."
    )
    anomaly_score: float


class AnomalousWindow(BaseModel):
    start: str
    end: str
    events: int
    warnings: int
    errors: int
    unseen_lines: int = Field(..., description="Lines matching no template seen in training.")
    reasons: list[str]


class Summary(BaseModel):
    total_events: int
    anomalies: int
    normal_events: int
    unparsed_lines: int
    anomalous_windows: int


class AnalyzeResponse(BaseModel):
    anomalies: list[Anomaly]
    windows: list[AnomalousWindow]
    summary: Summary

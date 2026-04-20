from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.predict import predict, load_model, get_model_feature_names


app = FastAPI(
    title="Diabetes Risk Prediction API",
    description="API for diabetes risk prediction using a trained ML model",
    version="0.1.0",
)


class PredictionRequest(BaseModel):
    preg: int = Field(..., ge=0, json_schema_extra={"example": 2})
    plas: int = Field(..., ge=0, json_schema_extra={"example": 130})
    pres: int = Field(..., ge=0, json_schema_extra={"example": 70})
    skin: int = Field(..., ge=0, json_schema_extra={"example": 25})
    insu: int = Field(..., ge=0, json_schema_extra={"example": 120})
    mass: float = Field(..., ge=0, json_schema_extra={"example": 28.5})
    pedi: float = Field(..., ge=0, json_schema_extra={"example": 0.35})
    age: int = Field(..., ge=0, json_schema_extra={"example": 33})

class PredictionResponse(BaseModel):
    prediction: int
    probability: float


class HealthResponse(BaseModel):
    status: str


class ModelInfoResponse(BaseModel):
    model_loaded: bool
    feature_names: list[str]


@app.get("/", tags=["root"])
def root() -> dict[str, str]:
    return {"message": "Diabetes Risk Prediction API is running"}


@app.get("/health", response_model=HealthResponse, tags=["health"])
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.get("/model-info", response_model=ModelInfoResponse, tags=["model"])
def model_info() -> ModelInfoResponse:
    try:
        model = load_model()
        feature_names = get_model_feature_names(model)
        return ModelInfoResponse(model_loaded=True, feature_names=feature_names)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/predict", response_model=PredictionResponse, tags=["prediction"])
def predict_one(payload: PredictionRequest) -> PredictionResponse:
    try:
        result = predict(payload.model_dump())[0]
        return PredictionResponse(**result)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
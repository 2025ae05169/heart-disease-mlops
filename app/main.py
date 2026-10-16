from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ConfigDict

from src.preprocess import FEATURES


BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "artifacts" / "heart_disease_model.joblib"

app = FastAPI(
    title="Heart Disease Prediction API",
    description="MLOps Assignment - Heart Disease Prediction",
    version="1.0.0",
)


class PatientInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    age: float = Field(ge=0, le=120)
    sex: int = Field(ge=0, le=1)
    cp: int
    trestbps: float
    chol: float
    fbs: int = Field(ge=0, le=1)
    restecg: int
    thalach: float
    exang: int = Field(ge=0, le=1)
    oldpeak: float
    slope: int
    ca: float
    thal: float


@app.on_event("startup")
def load_model():
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model not found: {MODEL_PATH}")

    app.state.model = joblib.load(MODEL_PATH)


@app.get("/")
def home():
    return {
        "message": "Heart Disease Prediction API is running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": hasattr(app.state, "model"),
    }


@app.post("/predict")
def predict(patient: PatientInput) -> dict[str, Any]:
    if not hasattr(app.state, "model"):
        raise HTTPException(
            status_code=503,
            detail="Model not loaded",
        )

    patient_data = patient.model_dump()
    input_df = pd.DataFrame([patient_data])

    # Ensure feature order matches model training
    input_df = input_df[FEATURES]

    prediction = int(
        app.state.model.predict(input_df)[0]
    )

    probability = float(
        app.state.model.predict_proba(input_df)[0][1]
    )

    return {
        "prediction": prediction,
        "heart_disease_probability": round(probability, 4),
        "interpretation": (
            "Heart disease predicted"
            if prediction == 1
            else "No heart disease predicted"
        ),
    }
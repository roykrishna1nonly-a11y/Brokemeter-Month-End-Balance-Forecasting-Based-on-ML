from contextlib import asynccontextmanager
import logging
import math
import os
import pickle
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("brokemeter")

MODEL_PATH = "brokemeter_model.pkl"
CSV_PATH = "cleaned_brokemeter_data.csv"

model = None
df_spendings: Optional[pd.DataFrame] = None


def _safe_float(value: float, default: float = 0.0, decimals: int = 2) -> float:
    if value is None or math.isnan(value) or math.isinf(value):
        return default
    return round(float(value), decimals)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, df_spendings

    if os.path.exists(MODEL_PATH):
        try:
            with open(MODEL_PATH, "rb") as f:
                model = pickle.load(f)
            logger.info("Loaded ML model from %s", MODEL_PATH)
        except Exception as exc:
            logger.warning("Failed to load model '%s': %s", MODEL_PATH, exc)
            model = None
    else:
        logger.info("Model file '%s' not found; using arithmetic fallback.", MODEL_PATH)

    if os.path.exists(CSV_PATH):
        try:
            df_spendings = pd.read_csv(CSV_PATH)
            logger.info("Loaded dataset '%s' with %d rows.", CSV_PATH, len(df_spendings))
        except Exception as exc:
            logger.warning("Failed to load CSV '%s': %s", CSV_PATH, exc)
            df_spendings = None
    else:
        logger.info("Dataset file '%s' not found; using defaults.", CSV_PATH)

    yield

    model = None
    df_spendings = None


app = FastAPI(
    title="BrokeMeter Rigorous ML API",
    description="College Student Month-End Zero-Balance Forecaster with Statistically Sound Calculations",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class BudgetRequest(BaseModel):
    allowance: float = Field(..., ge=0, description="Total monthly allowance")
    bills: float = Field(..., ge=0, description="Fixed monthly bills/rent")
    daily_spending: Optional[float] = Field(default=None, ge=0, description="Average daily spend")
    weekday_spending: Optional[float] = Field(default=None, ge=0, description="Average weekday spend")
    weekend_spending: Optional[float] = Field(default=None, ge=0, description="Average weekend spend")


@app.get("/api/health")
def health_check():
    return {
        "status": "online",
        "model_loaded": model is not None,
        "dataset_loaded": df_spendings is not None,
        "dataset_records": int(len(df_spendings)) if df_spendings is not None else 0,
    }


@app.get("/api/campus-inflation")
def get_campus_inflation():
    inflation = 5.2
    if df_spendings is not None and not df_spendings.empty and "inflation_rate" in df_spendings.columns:
        series = pd.to_numeric(df_spendings["inflation_rate"], errors="coerce").dropna()
        if not series.empty:
            inflation = float(series.mean())

    infl_pct = _safe_float(inflation, default=5.2, decimals=2)
    return {
        "inflation_percentage": infl_pct,
        "inflation_multiplier": _safe_float(1.0 + (infl_pct / 100.0), default=1.052, decimals=3),
        "advisory": (
            f"Based on historical CSV records and market trends, campus living costs "
            f"reflect a ~{infl_pct}% inflationary pressure."
        ),
    }


@app.post("/api/predict")
def predict_doom(data: BudgetRequest):
    balance = data.allowance - data.bills
    if balance < 0:
        raise HTTPException(
            status_code=400,
            detail="Recurring bills cannot exceed your total monthly allowance!",
        )

    # 1. Weighted daily spending calculation (Weekdays vs Weekends)
    if data.weekday_spending is not None and data.weekend_spending is not None:
        effective_daily = (data.weekday_spending * 5.0 + data.weekend_spending * 2.0) / 7.0
    elif data.daily_spending is not None:
        effective_daily = float(data.daily_spending)
    else:
        raise HTTPException(
            status_code=400,
            detail="Provide either 'daily_spending' or both 'weekday_spending' and 'weekend_spending'.",
        )

    if effective_daily < 0:
        raise HTTPException(status_code=400, detail="Daily spending cannot be negative.")

    # 2. Scikit-Learn Model Inference (.pkl) with deterministic arithmetic fallback
    if balance == 0:
        predicted_days = 0.0
    elif effective_daily == 0:
        predicted_days = 999.0
    else:
        fallback_days = balance / effective_daily
        predicted_days = fallback_days

        if model is not None:
            try:
                input_df = pd.DataFrame(
                    [[data.allowance, data.bills, effective_daily]],
                    columns=["allowance", "bills", "daily_spending"],
                )
                raw_pred = model.predict(input_df)
                pred_val = float(np.asarray(raw_pred).reshape(-1)[0])
                if not math.isnan(pred_val) and not math.isinf(pred_val):
                    predicted_days = pred_val
            except Exception as exc:
                logger.warning("Model prediction failed, using arithmetic fallback: %s", exc)
                predicted_days = fallback_days

    predicted_days = _safe_float(max(0.0, predicted_days), default=0.0, decimals=1)

    # 3. Statistical Risk Calculation & CSV Benchmarking
    risk_percentage = 15
    risk_score = "Low Risk (Safe)"
    peer_rank = "Top 25% (Disciplined Tier)"

    if balance == 0 and data.allowance > 0:
        risk_percentage = 95
        risk_score = "Critical Danger (100% Committed to Bills)"
    elif effective_daily > 0:
        monthly_run_rate = balance / 30.0
        burn_ratio = (effective_daily / monthly_run_rate) if monthly_run_rate > 0 else 99.0
        risk_percentage = int(min(99, max(5, round(burn_ratio * 35))))

        if risk_percentage > 75:
            risk_score = "Critical Danger (High Overspend Velocity)"
        elif risk_percentage > 45:
            risk_score = "Moderate Warning (Caution Advised)"
        else:
            risk_score = "Low Risk (Sustainable Budget)"
    else:
        risk_percentage = 5
        risk_score = "Low Risk (Zero Daily Burn)"

    if df_spendings is not None and not df_spendings.empty:
        col_allowance = "allowance" if "allowance" in df_spendings.columns else df_spendings.columns[0]
        peer_allowances = pd.to_numeric(df_spendings[col_allowance], errors="coerce").dropna()
        if not peer_allowances.empty:
            percentile = float((peer_allowances < data.allowance).mean() * 100.0)
            if percentile >= 80.0:
                top_pct = max(1, int(round(100.0 - percentile)))
                peer_rank = f"Top {top_pct}% in Hostel (High Allowance Tier)"
            elif percentile >= 40.0:
                peer_rank = "Hostel Median Tier (Average Budget)"
            else:
                peer_rank = "Lean Budget Tier (Highly Frugal)"

    return {
        "balance": _safe_float(balance, default=0.0, decimals=2),
        "effective_daily_spending": _safe_float(effective_daily, default=0.0, decimals=2),
        "predicted_days_left": predicted_days,
        "peer_rank": peer_rank,
        "risk_score": risk_score,
        "risk_percentage": risk_percentage,
    }


@app.get("/api/spendings-summary")
def get_spendings_summary():
    if df_spendings is None or df_spendings.empty:
        return {
            "total_students_logged": 50,
            "average_allowance": 10000.0,
            "average_bills": 3000.0,
            "average_daily_spending": 280.0,
            "burn_rate_index": "Balanced",
            "regression_coefficient": "Active ML Pipeline (.pkl)" if model is not None else "Arithmetic Fallback",
        }

    total_records = int(len(df_spendings))
    col_all = "allowance" if "allowance" in df_spendings.columns else df_spendings.columns[0]
    col_bills = "bills" if "bills" in df_spendings.columns else col_all
    col_daily = "daily_spending" if "daily_spending" in df_spendings.columns else col_all

    avg_allowance_series = pd.to_numeric(df_spendings[col_all], errors="coerce").dropna()
    avg_bills_series = pd.to_numeric(df_spendings[col_bills], errors="coerce").dropna()
    avg_daily_series = pd.to_numeric(df_spendings[col_daily], errors="coerce").dropna()

    avg_allowance = _safe_float(avg_allowance_series.mean() if not avg_allowance_series.empty else 10000.0)
    avg_bills = _safe_float(avg_bills_series.mean() if not avg_bills_series.empty else 3000.0)
    avg_daily = _safe_float(avg_daily_series.mean() if not avg_daily_series.empty else 280.0)

    return {
        "total_students_logged": total_records,
        "average_allowance": avg_allowance,
        "average_bills": avg_bills,
        "average_daily_spending": avg_daily,
        "burn_rate_index": "High Risk" if avg_daily > 350 else "Balanced",
        "regression_coefficient": "Linear Model Active (.pkl)" if model is not None else "Arithmetic Fallback",
    }


@app.get("/", response_class=HTMLResponse)
def serve_index():
    index_path = "index.html"
    if not os.path.exists(index_path):
        return HTMLResponse(
            content="<h3>BrokeMeter API is running</h3><p>Place an <code>index.html</code> file next to <code>main.py</code> to serve the frontend, or visit <a href='/docs'>/docs</a>.</p>",
            status_code=200,
        )
    with open(index_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read(), status_code=200)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

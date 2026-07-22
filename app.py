from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

app = Flask(__name__)

FEATURES = ["year", "month", "month_sin", "month_cos", "barangay", "trees", "farmers", "lag_1", "lag_12"]
CATEGORICAL = ["barangay"]
NUMERIC = [name for name in FEATURES if name not in CATEGORICAL]


def _error(message: str, status: int = 400):
    return jsonify({"error": message}), status


def _prepare_history(raw_history: list[dict[str, Any]], barangays: list[dict[str, Any]]) -> pd.DataFrame:
    if not raw_history:
        raise ValueError("history is required")

    history = pd.DataFrame(raw_history)
    required = {"year", "month", "barangay", "quantity"}
    if not required.issubset(history.columns):
        raise ValueError(f"history must contain: {', '.join(sorted(required))}")

    history["year"] = pd.to_numeric(history["year"], errors="coerce")
    history["month"] = pd.to_numeric(history["month"], errors="coerce")
    history["quantity"] = pd.to_numeric(history["quantity"], errors="coerce").fillna(0).clip(lower=0)
    history["barangay"] = history["barangay"].astype(str).str.strip().str.upper()
    history = history.dropna(subset=["year", "month"])
    history = history[history["month"].between(1, 12)]

    stats = {
        str(row.get("barangay", "")).strip().upper(): {
            "trees": max(0.0, float(row.get("trees", 0) or 0)),
            "farmers": max(0.0, float(row.get("farmers", 0) or 0)),
        }
        for row in barangays
    }

    grouped = history.groupby(["year", "month", "barangay"], as_index=False)["quantity"].sum()
    if grouped.empty:
        raise ValueError("history contains no valid observations")

    period_values = pd.PeriodIndex(
        year=grouped["year"].astype(int),
        month=grouped["month"].astype(int),
        freq="M",
    )
    min_period = period_values.min()
    max_period = period_values.max()
    periods = pd.period_range(min_period, max_period, freq="M")
    names = sorted(set(stats) | set(grouped["barangay"]))

    lookup = {(int(r.year), int(r.month), r.barangay): float(r.quantity) for r in grouped.itertuples()}
    rows: list[dict[str, Any]] = []
    for name in names:
        series: list[float] = []
        for period in periods:
            quantity = lookup.get((period.year, period.month, name), 0.0)
            lag_1 = series[-1] if series else 0.0
            lag_12 = series[-12] if len(series) >= 12 else 0.0
            details = stats.get(name, {"trees": 0.0, "farmers": 0.0})
            rows.append({
                "year": period.year,
                "month": period.month,
                "month_sin": math.sin(2 * math.pi * period.month / 12),
                "month_cos": math.cos(2 * math.pi * period.month / 12),
                "barangay": name,
                "trees": details["trees"],
                "farmers": details["farmers"],
                "lag_1": lag_1,
                "lag_12": lag_12,
                "quantity": quantity,
            })
            series.append(quantity)
    return pd.DataFrame(rows).sort_values(["year", "month", "barangay"]).reset_index(drop=True)


def _make_model() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("category", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
            ("number", "passthrough", NUMERIC),
        ]
    )
    forest = RandomForestRegressor(
        n_estimators=350,
        max_depth=14,
        min_samples_leaf=2,
        max_features=0.8,
        random_state=42,
        n_jobs=-1,
    )
    return Pipeline([("preprocessor", preprocessor), ("model", forest)])


def _predict_month(model: Pipeline, year: int, month: int, barangays: list[dict[str, Any]], history_by_name: dict[str, list[float]]) -> dict[str, float]:
    rows = []
    names = []
    for item in barangays:
        name = str(item.get("barangay", "")).strip().upper()
        if not name:
            continue
        previous = history_by_name[name]
        rows.append({
            "year": year,
            "month": month,
            "month_sin": math.sin(2 * math.pi * month / 12),
            "month_cos": math.cos(2 * math.pi * month / 12),
            "barangay": name,
            "trees": max(0.0, float(item.get("trees", 0) or 0)),
            "farmers": max(0.0, float(item.get("farmers", 0) or 0)),
            "lag_1": previous[-1] if previous else 0.0,
            "lag_12": previous[-12] if len(previous) >= 12 else 0.0,
        })
        names.append(name)

    predictions = np.maximum(0, model.predict(pd.DataFrame(rows)[FEATURES]))
    result = {}
    for name, value in zip(names, predictions):
        result[name] = float(value)
        history_by_name[name].append(float(value))
    return result


@app.get("/health")
def health():
    return {"status": "ok", "model": "RandomForestRegressor"}


@app.post("/forecast")
def forecast():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return _error("A JSON object is required.", 415)

    history = payload.get("history") or []
    barangays = payload.get("barangays") or []
    forecast_year = int(payload.get("forecast_year") or 0)
    five_years = [int(y) for y in (payload.get("five_years") or [])]
    if not barangays or forecast_year < 2000:
        return _error("barangays and a valid forecast_year are required")

    try:
        frame = _prepare_history(history, barangays)
    except (TypeError, ValueError) as exc:
        return _error(str(exc))

    if len(frame) < 24 or frame["quantity"].gt(0).sum() < 12:
        return _error("Not enough historical observations to train Random Forest reliably.", 422)

    # Time-aware holdout: validate on the newest 20% of chronological rows.
    split = max(1, int(len(frame) * 0.8))
    train = frame.iloc[:split]
    test = frame.iloc[split:]
    model = _make_model()
    model.fit(train[FEATURES], train["quantity"])

    metrics: dict[str, float | None] = {"mae": None, "r2": None, "training_rows": int(len(frame))}
    if not test.empty:
        validation = np.maximum(0, model.predict(test[FEATURES]))
        metrics["mae"] = round(float(mean_absolute_error(test["quantity"], validation)), 4)
        if len(test) > 1 and test["quantity"].nunique() > 1:
            metrics["r2"] = round(float(r2_score(test["quantity"], validation)), 4)

    # Refit using all available history before future inference.
    model.fit(frame[FEATURES], frame["quantity"])
    history_by_name: dict[str, list[float]] = defaultdict(list)
    for row in frame.itertuples():
        history_by_name[row.barangay].append(float(row.quantity))

    target_years = sorted(set(five_years + [forecast_year]))
    first_year = int(frame["year"].max())
    last_target = max(target_years)
    annual: dict[str, float] = defaultdict(float)
    monthly_target = [0.0] * 12
    barangay_target: dict[str, float] = defaultdict(float)

    for year in range(first_year + 1, last_target + 1):
        for month in range(1, 13):
            month_predictions = _predict_month(model, year, month, barangays, history_by_name)
            month_total = sum(month_predictions.values())
            annual[str(year)] += month_total
            if year == forecast_year:
                monthly_target[month - 1] = month_total
                for name, value in month_predictions.items():
                    barangay_target[name] += value

    return jsonify({
        "model": "RandomForestRegressor",
        "monthly_forecast": [round(v, 2) for v in monthly_target],
        "annual_forecast": {year: round(annual.get(str(year), 0.0), 2) for year in map(str, target_years)},
        "barangay_forecast": {name: round(value, 2) for name, value in barangay_target.items()},
        "metrics": metrics,
    })


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)

from __future__ import annotations
from contextlib import asynccontextmanager
from typing import Any, Literal
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from pokemon_ds.api_service import (
    ALLOWED_TOP_METRICS,
    AppState,
    build_app_state,
    get_top_pokemon_by_metric,
    predict_base_experience_from_stats,
    summarize_types,
)

# FastAPI app instance
app = FastAPI()

# Main FastAPI app definition with endpoints for health check, top Pokémon by metric, type summary, and base experience prediction
class BaseExperiencePredictionRequest(BaseModel):
    hp: float = Field(..., ge=1)
    attack: float = Field(..., ge=1)
    defense: float = Field(..., ge=1)
    special_attack: float = Field(..., ge=1, alias="special-attack")
    special_defense: float = Field(..., ge=1, alias="special-defense")
    speed: float = Field(..., ge=1)

    class Config:
        populate_by_name = True

# Prediction response model for the base experience prediction endpoint, including the model used, features, and predicted value
class BaseExperiencePredictionResponse(BaseModel):
    model: str
    features_used: list[str]
    predicted_base_experience: float


# App State and Lifespan for loading dataset and training regression model once on startup
@asynccontextmanager
async def lifespan(app: FastAPI):
  
    try:
        state: AppState = build_app_state()
        app.state.app_state = state
    except Exception as e:
        # If startup fails, we still start the app but endpoints can return errors.
        app.state.startup_error = str(e)
        app.state.app_state = None
    yield

# FastAPI app instance with defined endpoints and lifespan for startup initialization
app = FastAPI(
    title="Pokemon Data Science API",
    version="0.1.0",
    description="Analytics + ML API for the pokemon-datascience-report project.",
    lifespan=lifespan,
)


# Helper function to get the app state or raise an HTTPException if startup failed or state is not initialized
def _get_state_or_raise() -> AppState:
    startup_error = getattr(app.state, "startup_error", None)
    if startup_error:
        raise HTTPException(
            status_code=500,
            detail=f"API startup failed: {startup_error}",
        )

    state = getattr(app.state, "app_state", None)
    if state is None:
        raise HTTPException(status_code=500, detail="API state not initialized.")
    return state


# API Endpoints

# Health check endpoint to verify API is running and state is loaded
@app.get("/health")
def health() -> dict[str, Any]:
    
    startup_error = getattr(app.state, "startup_error", None)
    if startup_error:
        return {"status": "degraded", "startup_error": startup_error}

    state = getattr(app.state, "app_state", None)
    if state is None:
        return {"status": "degraded", "detail": "App state not initialized"}

    return {
        "status": "ok",
        "n_rows_loaded": int(len(state.pokemon_df)),
        "regression_model_loaded": state.regression_model is not None,
    }

# Return top N Pokémon by a specified metric
@app.get("/top")
def top_pokemon(
    metric: str = Query("total_stats", description="Metric to sort by"),
    n: int = Query(10, ge=1, le=50, description="Number of rows to return"),
) -> dict[str, Any]:
   
    state = _get_state_or_raise()

    try:
        rows = get_top_pokemon_by_metric(state.pokemon_df, metric=metric, n=n)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {
        "metric": metric,
        "n": n,
        "allowed_metrics": sorted(ALLOWED_TOP_METRICS),
        "results": rows,
    }


@app.get("/summary/types")
def type_summary() -> dict[str, Any]:
    """
    Return primary type counts and type occurrence counts.
    """
    state = _get_state_or_raise()

    try:
        return summarize_types(state.pokemon_df)
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/base-experience", response_model=BaseExperiencePredictionResponse)
def predict_base_experience(payload: BaseExperiencePredictionRequest):
    """
    Predict base_experience from base stats using the in-memory trained regression pipeline.

    The payload accepts:
    - hp
    - attack
    - defense
    - special-attack
    - special-defense
    - speed
    """
    state = _get_state_or_raise()

    # Convert Pydantic model to dict with original aliases expected by the modeling code
    raw = payload.model_dump(by_alias=True)

    try:
        result = predict_base_experience_from_stats(state.regression_model, raw)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    return result
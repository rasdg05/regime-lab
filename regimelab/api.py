"""FastAPI service: score a price window's regime, live.

Endpoints
---------
GET  /health              liveness probe (for Railway/Render/k8s).
GET  /metrics             tiny Prometheus-style counters (requests, latency).
POST /regime              body {"closes": [...], "window": 64}
                          -> {"irreversibility", "permutation_entropy",
                              "regime", "label"} for the most recent window.

This is intentionally stateless and dependency-light: the model is
parameter-free, so there's nothing to load or warm up. ``regime`` is a
human-readable string ("reversible" / "irreversible"); ``label`` is 0/1.

Run locally:  uvicorn regimelab.api:app --reload
"""
from __future__ import annotations

import time

import numpy as np

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel, Field
except Exception as e:      # pragma: no cover - only when extras missing
    raise ImportError(
        "the API needs the `serve` extra: pip install 'regime-lab[serve]'"
    ) from e

from .detectors.entropy import permutation_entropy
from .detectors.irreversibility import irreversibility

app = FastAPI(title="regime-lab", version="0.1.0",
              description="Parameter-free market-regime scoring (HVG time-irreversibility).")

# minimal in-process monitoring — enough to prove liveness + latency on a PaaS
_METRICS = {"requests_total": 0, "errors_total": 0, "latency_ms_sum": 0.0}

# default split learned offline on BTCUSDT 5m; callers can override per request.
DEFAULT_THRESHOLD = 0.34


class RegimeRequest(BaseModel):
    closes: list[float] = Field(..., min_length=8,
                                description="Ordered close prices, oldest first.")
    window: int = Field(64, ge=8, le=4096,
                        description="How many of the most recent closes to score.")
    threshold: float | None = Field(
        None, description="Irreversibility cut for the regime label; "
                          "defaults to a value learned offline.")


class RegimeResponse(BaseModel):
    irreversibility: float
    permutation_entropy: float
    regime: str
    label: int
    n_used: int


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "version": "0.1.0"}


@app.get("/metrics")
def metrics() -> dict:
    m = dict(_METRICS)
    n = max(m["requests_total"], 1)
    m["latency_ms_avg"] = round(m["latency_ms_sum"] / n, 3)
    return m


@app.post("/regime", response_model=RegimeResponse)
def regime(req: RegimeRequest) -> RegimeResponse:
    t0 = time.perf_counter()
    _METRICS["requests_total"] += 1
    try:
        y = np.asarray(req.closes[-req.window:], dtype=float)
        if len(y) < 8:
            raise HTTPException(status_code=422, detail="need >= 8 closes")
        irr = irreversibility(y)
        pe = permutation_entropy(y)
        thr = req.threshold if req.threshold is not None else DEFAULT_THRESHOLD
        label = int(irr > thr)
        return RegimeResponse(
            irreversibility=round(irr, 6),
            permutation_entropy=round(pe, 6),
            regime="irreversible" if label else "reversible",
            label=label,
            n_used=len(y),
        )
    except HTTPException:
        _METRICS["errors_total"] += 1
        raise
    except Exception as e:      # pragma: no cover
        _METRICS["errors_total"] += 1
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        _METRICS["latency_ms_sum"] += (time.perf_counter() - t0) * 1000.0

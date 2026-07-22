"""The live endpoint must answer health, score a window, and reject junk. Skips
cleanly if the optional serving deps aren't installed."""
import numpy as np
import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from regimelab.api import app  # noqa: E402

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_regime_scores_a_window():
    v = 0.0
    saw = []
    for i in range(200):
        v += 1.0 if i % 20 < 15 else -3.0
        saw.append(v)
    r = client.post("/regime", json={"closes": saw, "window": 128})
    assert r.status_code == 200
    body = r.json()
    assert body["label"] in (0, 1)
    assert body["regime"] in ("reversible", "irreversible")
    assert body["irreversibility"] >= 0.0
    assert body["n_used"] == 128


def test_regime_rejects_too_few_points():
    r = client.post("/regime", json={"closes": [1.0, 2.0, 3.0]})
    assert r.status_code == 422        # pydantic min_length


def test_metrics_counts_requests():
    before = client.get("/metrics").json()["requests_total"]
    client.post("/regime", json={"closes": list(np.arange(64.0))})
    after = client.get("/metrics").json()["requests_total"]
    assert after >= before + 1

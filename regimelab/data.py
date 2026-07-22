"""Data loading: public Binance Vision klines + a synthetic regime generator.

Two sources, one shape (a 1-D float array of closes, plus timestamps):

* ``load_binance_vision`` — downloads daily 5-minute kline zips from the public
  Binance Vision bucket (no API key, no auth). This is the "real data, anyone
  can reproduce" path the README leans on.
* ``synthetic_regimes`` — generates a series that alternates between a known
  trending regime and a known mean-reverting regime, with the ground-truth
  labels returned alongside. Used by the tests (offline, deterministic) and to
  sanity-check that a detector recovers regimes it *should* be able to see.
"""
from __future__ import annotations

import csv
import io
import urllib.request
import zipfile
from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np

__all__ = ["Series", "synthetic_regimes", "load_binance_vision"]

_BINANCE_VISION = "https://data.binance.vision/data/futures/um/daily/klines"


@dataclass
class Series:
    ts: np.ndarray          # int64 ms epoch
    close: np.ndarray       # float
    labels: np.ndarray | None = None    # ground truth (synthetic only)

    def __len__(self) -> int:
        return len(self.close)


def synthetic_regimes(
    n_blocks: int = 20,
    block_len: int = 300,
    seed: int = 0,
    rev_sigma: float = 0.006,
    irr_drift: float = 0.0009,
    irr_kappa: float = 0.15,
    irr_sigma: float = 0.0011,
    irr_jump_p: float = 0.10,
    irr_jump: float = -0.011,
) -> Series:
    """Alternating *reversible* / *irreversible* blocks with known labels.

    The two regimes are chosen to match what the HVG statistic actually
    measures — time-asymmetry — not trend strength:

    * label 0, "reversible": a driftless Gaussian random walk. Statistically
      time-symmetric (up-moves and down-moves interchangeable), so low
      irreversibility, and efficient — no rule beats costs.
    * label 1, "irreversible": mean reversion pulled by a slow up-drift but
      punctuated by rare *sharp down jumps* — the classic "stairs up, elevator
      down" asymmetry. High irreversibility, and a reversion (fade-the-move)
      rule has a genuine edge because the elevator-down over-shoots and snaps
      back.

    Returns a :class:`Series` whose ``labels`` is the ground truth per bar. This
    is only for unit tests and a sanity demo; the real results come from real
    data (:func:`load_binance_vision`).
    """
    rng = np.random.default_rng(seed)
    closes = [100.0]
    labels = []
    log_p = np.log(closes[0])
    for b in range(n_blocks):
        irreversible = (b % 2 == 1)
        for _ in range(block_len):
            if irreversible:
                shock = irr_drift + rng.normal(0, irr_sigma)
                if rng.random() < irr_jump_p:
                    shock += irr_jump                    # sharp asymmetric drop
                log_p += irr_kappa * (np.log(100.0) - log_p) * 0.02 + shock
                labels.append(1)
            else:
                log_p += rng.normal(0, rev_sigma)        # symmetric random walk
                labels.append(0)
            closes.append(float(np.exp(log_p)))
    close = np.asarray(closes[1:], dtype=float)
    labels = np.asarray(labels, dtype=int)
    ts = (np.arange(len(close), dtype=np.int64) * 300_000)      # 5m spacing
    return Series(ts=ts, close=close, labels=labels)


def _read_kline_zip(raw: bytes) -> list[tuple[int, float]]:
    rows = []
    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        name = zf.namelist()[0]
        with zf.open(name) as fh:
            text = io.TextIOWrapper(fh, encoding="utf-8")
            for r in csv.reader(text):
                if not r or not r[0].lstrip("-").isdigit():
                    continue                        # skip header row if present
                rows.append((int(r[0]), float(r[4])))   # open_time, close
    return rows


def load_binance_vision(
    symbol: str = "BTCUSDT",
    start: date | None = None,
    end: date | None = None,
    interval: str = "5m",
    timeout: int = 60,
) -> Series:
    """Download public USDⓈ-M futures klines from Binance Vision.

    ``start``/``end`` are inclusive dates (default: the 14 days ending
    yesterday). Missing days (weekends for some products, gaps) are skipped with
    a warning rather than failing the whole pull.
    """
    if end is None:
        end = date.today() - timedelta(days=1)
    if start is None:
        start = end - timedelta(days=13)
    ts_all: list[int] = []
    close_all: list[float] = []
    d = start
    while d <= end:
        url = f"{_BINANCE_VISION}/{symbol}/{interval}/{symbol}-{interval}-{d.isoformat()}.zip"
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                rows = _read_kline_zip(r.read())
            for t, c in rows:
                ts_all.append(t)
                close_all.append(c)
        except Exception as e:      # noqa: BLE001 - a missing day shouldn't abort
            print(f"[data] skip {symbol} {d}: {e}")
        d += timedelta(days=1)
    if not close_all:
        raise RuntimeError(f"no klines fetched for {symbol} {start}..{end}")
    order = np.argsort(ts_all, kind="mergesort")
    ts = np.asarray(ts_all, dtype=np.int64)[order]
    close = np.asarray(close_all, dtype=float)[order]
    return Series(ts=ts, close=close)

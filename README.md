# regime-lab

**A parameter-free market-regime detector, benchmarked honestly.**

Most "regime detection" write-ups stop at a pretty chart of coloured segments.
This one asks the only questions that survive contact with a skeptical reviewer:

1. Does the detector measure something *real and distinct* from the obvious
   baselines (realized volatility, a moving-average slope, a 2-state HMM)?
2. When you evaluate it the way you're supposed to — **purged walk-forward** so
   there's no leakage, **Deflated Sharpe Ratio** so multiple testing can't fool
   you — does any edge survive?

The star detector maps a price window to a directed **Horizontal Visibility
Graph** and measures its **time-irreversibility** (the KL divergence between the
in-degree and out-degree distributions). It has no tunable parameters in the
statistic itself, and it is grounded in published work rather than folklore.

> **Headline, stated honestly.** On 5-minute BTC, no simple rule — gated or not —
> earns a Deflated-Sharpe-certified edge after costs; the harness correctly
> refuses to certify any of them. But gating a rule on *any* regime detector
> beats trading it always-on, and cuts max drawdown by ~40%. The regime signal's
> value here is **defensive** (knowing when *not* to trade), and the deliverable
> is a reproducible, leak-free evaluation harness you can point at your own idea.

---

## What time-irreversibility actually measures (and what it doesn't)

It is **not** a trend detector. A clean linear ramp is perfectly time-*reversible*
(irreversibility ≈ 0). What lights the statistic up is temporal **asymmetry** —
the classic *"markets take the stairs up and the elevator down."* Measured on
canonical signals:

| signal                            | irreversibility |
|-----------------------------------|----------------:|
| linear ramp                       |          0.000  |
| Gaussian random walk              |          ~0.10  |
| AR(1) mean-reversion              |          ~0.10  |
| **sawtooth (slow up, fast down)** |        **17.9** |

That's exactly why it's worth benchmarking *separately* from volatility or slope
— it sees a different thing.

References: Lacasa et al., *From time series to complex networks: the visibility
graph*, PNAS 2008; Lacasa et al., *Time series irreversibility: a visibility
graph approach*, EPJ B 2012; Bailey & López de Prado, *The Deflated Sharpe
Ratio*, JPM 2014.

---

## Results

### Sanity — the detector recovers a regime it *should* see

On synthetic data that alternates a symmetric random walk with an asymmetric
"stairs up / elevator down" block, rolling irreversibility separates the two by
**~2.2×** (`test_synthetic_regimes_are_separable_by_irreversibility` pins this).
The statistic does what the theory promises.

### Real data — BTCUSDT, 5-minute, ~30 days (Binance Vision, public)

Base rule = mean-reversion (fade the last bar). Each detector *gates* that rule
(trade only in its "active" regime). Purged walk-forward, 6 folds, embargo 48
bars, 2 bps per turn. Reproduce with `scripts/run_benchmark.py`.

| strategy                 | Sharpe (per-bar) | Deflated Sharpe | max drawdown |
|--------------------------|-----------------:|----------------:|-------------:|
| always_flat (do nothing) |            0.000 |           0.000 |        0.000 |
| realized_vol (gated)     |           −0.068 |           0.000 |       −0.403 |
| trend_slope (gated)      |           −0.106 |           0.000 |       −0.570 |
| perm_entropy (gated)     |           −0.110 |           0.000 |       −0.542 |
| irreversibility (gated)  |           −0.135 |           0.000 |       −0.725 |
| always_on (no gating)    |           −0.159 |           0.000 |       −1.181 |

PBO (probability the in-sample winner is overfit, competing configs) = **0.000** —
the ranking is stable out-of-sample, it just isn't a *winning* ranking.

**Honest reading of this table:**

* No strategy clears a positive Deflated Sharpe. Simple fade/momentum rules on
  5-minute BTC do not have an edge once you charge realistic costs and correct
  for multiple testing. That is the *expected* result, and reporting it is the
  point — the harness is working.
* **Every** gated strategy loses less, and draws down less, than trading the rule
  always-on (−1.18 vs −0.40…−0.73). Turning the rule off in the wrong regime is
  worth something even when the rule itself is a loser.
* Realized volatility is a *strong* baseline (least-bad here). Irreversibility
  does not beat it on this particular rule/window — and this README says so
  rather than cherry-picking the cut where it does. Its distinct value is shown
  on the synthetic asymmetry test above; whether that converts to PnL on a given
  asset is an empirical question the harness is built to answer honestly.

Results vary by asset, base rule, and window — try your own with the script.

### Bonus — does order flow *anticipate* regime transitions?

A sharper question than "does the regime help you trade": can **taker order-flow
imbalance** (the aggressor side, computed from Binance's `taker_buy_base` column
at bar frequency) tell you a regime change is coming before price does? A
"transition" is a flip of the irreversibility label; the target is whether one
happens in the next 12 bars. Purged walk-forward, pooled OOS AUC — reproduce
with `python scripts/run_flow_experiment.py`.

| features                    | OOS AUC |
|-----------------------------|--------:|
| price only (returns + vol)  |  0.522  |
| price + order flow          |  0.513  |
| order flow only             |  0.505  |

**Order flow does not anticipate regime transitions here.** Flow-only is a coin
flip (0.505), and adding it to price features slightly *hurts*. Price/vol alone
carries a whisper of signal (0.522) but nothing you'd trade. A tidy negative for
the popular "order flow leads regime change" intuition, at 5-minute resolution
with this transition definition.

---

## Install & run

```bash
pip install -e ".[dev]"           # core + serving + hmm + pytest
pytest -q                          # 22 tests: statistic properties, no-leakage, DSR, API

# reproduce the benchmark on public data (no API key needed)
python scripts/run_benchmark.py --symbol BTCUSDT --days 30 --rule reversion
python scripts/run_benchmark.py --synthetic          # the sanity demo
```

Core detectors + evaluation depend only on **numpy**. `hmmlearn` (the HMM
baseline) and FastAPI (the endpoint) are optional extras.

---

## Live endpoint

A stateless FastAPI service scores the most recent window on demand. Because the
model is parameter-free there's nothing to load or warm up.

```bash
uvicorn regimelab.api:app --port 8000
curl -s localhost:8000/health
curl -s -X POST localhost:8000/regime \
  -H 'content-type: application/json' \
  -d '{"closes": [/* recent closes, oldest first */], "window": 64}'
# -> {"irreversibility": 0.41, "permutation_entropy": 0.62,
#     "regime": "irreversible", "label": 1, "n_used": 64}
```

`/health` and `/metrics` (request count, error count, average latency) are there
so a platform can liveness-probe it and you can watch it in production.

### Deploy (Railway / Render / Fly)

Local:

```bash
docker build -t regime-lab .
docker run -p 8000:8000 regime-lab
```

One-click on a PaaS — the repo ships the config so there's nothing to wire:

* **Render** — New → Blueprint → point at this repo (`render.yaml`, health check `/health`).
* **Railway** — New → Deploy from repo; `railway.json` pins the Dockerfile build and health check.

All of them inject `$PORT`; the image / `Procfile` serves `regimelab.api:app` on it.

---

## Layout

```
regimelab/
  detectors/
    irreversibility.py   # * HVG time-irreversibility (parameter-free)
    entropy.py           # permutation-entropy detector
    baselines.py         # realized-vol, MA-slope, 2-state Gaussian HMM
    base.py              # the one small interface they all share
  evaluation/
    deflated_sharpe.py   # DSR, expected-max-Sharpe, PBO (CSCV)
    walkforward.py       # purged walk-forward splits + leakage assert
    metrics.py           # Sharpe, drawdown, hit-rate, turnover
  data.py                # Binance Vision loader + synthetic regime generator
  benchmark.py           # the regime-gating experiment
  api.py                 # FastAPI live scoring
scripts/run_benchmark.py # CLI to reproduce the tables
tests/                   # statistic properties, no-leakage, DSR behaviour, API
```

## License

MIT.

"""regime-lab: an honest, reproducible benchmark for market-regime detection.

The star detector maps a price window to a directed Horizontal Visibility Graph
and measures its time-irreversibility (KL divergence of in/out degree
distributions). It is benchmarked — with purged walk-forward splits and the
Deflated Sharpe Ratio — against textbook baselines (HMM, realized volatility, a
moving-average slope rule) on public Binance Vision data.
"""
from .detectors.irreversibility import IrreversibilityDetector, irreversibility
from .detectors.entropy import PermutationEntropyDetector, permutation_entropy
from .detectors.baselines import (
    RealizedVolDetector, TrendSlopeDetector, HMMDetector,
)
from .benchmark import run_benchmark
from .flow_experiment import run_flow_experiment

__version__ = "0.1.0"

__all__ = [
    "IrreversibilityDetector", "irreversibility",
    "PermutationEntropyDetector", "permutation_entropy",
    "RealizedVolDetector", "TrendSlopeDetector", "HMMDetector",
    "run_benchmark", "run_flow_experiment",
]

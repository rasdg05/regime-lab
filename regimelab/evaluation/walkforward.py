"""Purged walk-forward splits with an embargo.

Standard k-fold cross-validation leaks in time series: a training bar right next
to a test bar shares almost the same information, so the test score is
optimistic. Walk-forward keeps train strictly before test; *purging* drops train
bars whose label horizon overlaps the test window, and an *embargo* adds a gap
after each test block so nothing immediately adjacent leaks back. This is the
López de Prado recipe (Advances in Financial Machine Learning, ch. 7).

Here labels are per-bar regime states with a 1-bar horizon, so purging reduces
to "train ends at least ``embargo`` bars before the test starts".
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["Fold", "purged_walk_forward_splits", "assert_no_leakage"]


@dataclass
class Fold:
    train: np.ndarray       # int index array
    test: np.ndarray


def purged_walk_forward_splits(n: int, n_splits: int = 5, embargo: int = 0,
                               min_train: int = 64) -> list[Fold]:
    """Expanding-window walk-forward over ``n`` ordered bars.

    Produces up to ``n_splits`` folds. Fold k trains on ``[0, test_start-embargo)``
    and tests on the k-th contiguous test block. Folds whose training window is
    shorter than ``min_train`` are skipped (not enough history to fit).
    """
    if n_splits < 1 or n < min_train + 2:
        return []
    test_blocks = np.array_split(np.arange(n), n_splits + 1)[1:]   # leave block 0 for train
    folds: list[Fold] = []
    for block in test_blocks:
        if len(block) == 0:
            continue
        test_start = int(block[0])
        train_end = test_start - embargo
        if train_end < min_train:
            continue
        train = np.arange(0, train_end)
        folds.append(Fold(train=train, test=np.asarray(block)))
    return folds


def assert_no_leakage(folds: list[Fold], embargo: int = 0) -> None:
    """Raise if any fold's train overlaps (within the embargo of) its test."""
    for k, f in enumerate(folds):
        if len(f.train) == 0 or len(f.test) == 0:
            continue
        if f.train.max() >= f.test.min() - embargo:
            raise AssertionError(
                f"fold {k}: train max {f.train.max()} leaks into test "
                f"start {f.test.min()} (embargo={embargo})")

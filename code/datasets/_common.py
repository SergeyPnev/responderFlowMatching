"""
Shared dataset utilities: :class:`LabelEncoder` (fit-on-train label mapping),
:class:`InfiniteDataLoader` and :class:`BaseDisentangleDataset`, the contract
every dataset module satisfies.
"""

from __future__ import annotations

import ast
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Label encoding
# ---------------------------------------------------------------------------

class LabelEncoder:
    """Fit on train, reuse on val for consistent label indices."""

    def __init__(self):
        self.label2idx: dict = {}
        self.idx2label: dict = {}

    def fit(self, labels: pd.Series) -> "LabelEncoder":
        unique = sorted(labels.dropna().unique(), key=str)
        self.label2idx = {lbl: i for i, lbl in enumerate(unique)}
        self.idx2label = {i: lbl for lbl, i in self.label2idx.items()}
        return self

    def transform(self, labels: pd.Series) -> np.ndarray:
        """Map labels to int.  Unknown labels get -1."""
        return labels.map(lambda x: self.label2idx.get(x, -1)).values.astype(np.int64)

    @property
    def num_classes(self) -> int:
        return len(self.label2idx)


# ---------------------------------------------------------------------------
# Infinite data loader
# ---------------------------------------------------------------------------

class _InfiniteSampler(torch.utils.data.Sampler):
    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            for batch in self.sampler:
                yield batch


class InfiniteDataLoader:
    def __init__(self, dataset, weights, batch_size, num_workers):
        if weights is not None:
            sampler = torch.utils.data.WeightedRandomSampler(
                weights, replacement=True, num_samples=batch_size
            )
        else:
            sampler = torch.utils.data.RandomSampler(dataset, replacement=True)

        batch_sampler = torch.utils.data.BatchSampler(
            sampler, batch_size=batch_size, drop_last=True
        )

        self._infinite_iterator = iter(
            torch.utils.data.DataLoader(
                dataset,
                batch_sampler=_InfiniteSampler(batch_sampler),
                num_workers=num_workers,
            )
        )

    def __iter__(self):
        while True:
            yield next(self._infinite_iterator)


# ---------------------------------------------------------------------------
# Abstract base dataset
# ---------------------------------------------------------------------------

class BaseDisentangleDataset(Dataset, ABC):
    """
    __getitem__ returns ``(image_tensor, labels_dict)`` where ``labels_dict``
    has at least ``"treatment"`` and ``"plate"`` (domain label), and ``"moa"``
    when ``return_moa=True``.
    """

    @abstractmethod
    def __getitem__(self, idx) -> Tuple[torch.Tensor, Dict[str, Any]]:
        ...

    @abstractmethod
    def __len__(self) -> int:
        ...

    @property
    @abstractmethod
    def num_treatments(self) -> int:
        ...

    @property
    @abstractmethod
    def num_plates(self) -> int:
        ...

    @property
    def num_moa(self) -> int:
        return 0


def transform_to_list(val):
    return ast.literal_eval(val)
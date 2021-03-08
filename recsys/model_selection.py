from collections import defaultdict
from typing import Generator, List

import numpy as np

from recsys.types import BatchType


def round_robin_kfold(batches: List[BatchType],
                      n_splits: int = 10) -> Generator:
    """
    Round robin k-fold cross validation.
    Useful when batches are sorted by sequence length, to keep the training
    set and validation set as balanced as possible in sequence length.
    Train indices are shuffled to try to reduce the bias in the gradient updates.
    """
    np.random.seed(42)
    n = len(batches)
    groups = defaultdict(list)

    group_id = 0
    for i in range(n):
        groups[group_id % n_splits].append(i)
        group_id += 1

    for i in range(n_splits):
        train_index = np.concatenate([group for group_id, group in groups.items()
                                      if group_id != i])
        valid_index = np.array(groups[i])
        np.random.shuffle(train_index)
        yield train_index, valid_index

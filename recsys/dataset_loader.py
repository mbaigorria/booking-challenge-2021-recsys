import sys
from collections import defaultdict
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader

if 'ipykernel' in sys.modules:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

from recsys.config import DEVICE
from recsys.types import BatchType


class BookingDataset(Dataset):

    def __init__(self,
                 df: pd.DataFrame,
                 features: List[str],
                 group_var='utrip_id'):
        sorted_groups = sorted(df.groupby(group_var), key=lambda g: len(g[1]), reverse=True)
        self.trips = [BookingDataset.pre_process(group, features) for _, group in tqdm(sorted_groups)]
        self.utrip_ids = [utrip_id for utrip_id, _ in sorted_groups]
        self.group_lengths = [len(g[1]) for g in sorted_groups]

    def __len__(self):
        return len(self.trips)

    def __getitem__(self, idx):
        return self.trips[idx]

    def get_ids(self):
        return pd.DataFrame({'utrip_id': self.utrip_ids,
                             'group_length': self.group_lengths})

    @staticmethod
    def pre_process(group: pd.DataFrame, features: List[str]):
        g = group[features].to_dict(orient='list')
        return {k: torch.LongTensor(np.array(v)) for k, v in g.items()}


def pad_collate(batch: List[BatchType]):
    """
    Unify observations in a padded batch dictionary.
    """
    batch_dict = defaultdict(list)
    lengths = []
    for d in batch:
        for k, v in d.items():
            batch_dict[k].append(v)
        # add the next city id if we are training
        if 'next_city_id' in d:
            batch_dict['last_city'].append(d['next_city_id'][-1])
        lengths.append(v.size())

    res = {k: pad_sequence(v, batch_first=True, padding_value=0)
           for k, v in batch_dict.items() if k != 'last_city'}

    # add last city id if we are training
    if 'next_city_id' in d:
        res['last_city'] = torch.tensor(batch_dict['last_city'])

    lengths = torch.tensor(lengths, dtype=torch.int64).squeeze()
    return res, lengths


def get_dataset_and_dataloader(df: pd.DataFrame,
                               features: List[str],
                               batch_size: int = 256) -> Tuple[BookingDataset, DataLoader]:
    """
    Get dataset and dataloader.
    """
    dataset = BookingDataset(df, features)
    data_loader = DataLoader(dataset,
                             batch_size=batch_size,
                             shuffle=False,
                             collate_fn=pad_collate)
    return dataset, data_loader


def batches_to_device(data_loader: DataLoader) -> np.array:
    """
    Batches to device.
    By pre-loading all batches in GPU for training, we avoid transferring data
    from memory to GPU on every fold. The risk of doing this is biasing the gradients,
    reason why we are then careful with the distribution of batches on each fold,
    also shuffling the batches every time we train a model.
    """
    if DEVICE == 'cpu':
        batches = np.array([({k: v for k, v in d.items()}, seq_len)
                            for (d, seq_len) in data_loader])
    else:
        batches = np.array([({k: v.cuda(non_blocking=True)
                              for k, v in d.items()}, seq_len) for (d, seq_len) in data_loader])

    return batches


def filter_batches_by_length(batches: List[BatchType], min_length: int = 3):
    """
    Filter batches to have a minimum length of `min_length`.
    """
    return list(filter(lambda b: b[1].min().item() > min_length, batches))

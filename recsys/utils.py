import functools
import gc
import json
import logging
import os
import subprocess
import sys
from datetime import datetime
from os import listdir
from os.path import isfile
from typing import List, Dict

import GPUtil as GPU
import humanize
import pandas as pd
import psutil
import torch
from torch.utils.data import DataLoader

if 'ipykernel' in sys.modules:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

from recsys.dataset_loader import BookingDataset, get_dataset_and_dataloader
from recsys.encoders import DatasetEncoder
from recsys.model import get_model_predictions, BookingNet
from recsys.paths import get_resources_path, get_path, get_model_ckpt_paths, get_model_arch_path
from recsys import config


def print_gpu_usage(gpu_id: int = 0):
    """
    Display GPU usage.
    """
    gpu_list = GPU.getGPUs()
    gpu = gpu_list[gpu_id]
    process = psutil.Process(os.getpid())
    logging.info(f"Gen RAM Free: {humanize.naturalsize(psutil.virtual_memory().available)}"
                 f" | Proc size: {humanize.naturalsize(process.memory_info().rss)}")
    logging.info("GPU RAM Free: {0:.0f}MB | Used: {1:.0f}MB | Util {2:3.0f}% | Total {3:.0f}MB".format(gpu.memoryFree,
                                                                                                       gpu.memoryUsed,
                                                                                                       gpu.memoryUtil * 100,
                                                                                                       gpu.memoryTotal))


def accuracy_at_k(submission: pd.DataFrame,
                  ground_truth: pd.DataFrame) -> Dict:
    """
    Calculates accuracy@k for k in {1, 4, 10} by group length and overall.
    """
    data_to_eval = submission.join(ground_truth, on='utrip_id')

    for k in [1, 4, 10]:
        data_to_eval[f'hits_at_{k}'] = data_to_eval.apply(
            lambda row: row['city_id'] in row[[f'city_id_{i}' for i in range(1, k + 1)]].values, axis=1)
    return {
        'accuracy@1': data_to_eval['hits_at_1'].mean(),
        'accuracy@4': data_to_eval['hits_at_4'].mean(),
        'accuracy@10': data_to_eval['hits_at_10'].mean(),
        'accuracy@4_by_pos': data_to_eval.groupby('group_length')['hits_at_4'].mean().to_dict()
    }


def get_submission(dataset: BookingDataset,
                   data_loader: DataLoader,
                   model: BookingNet,
                   checkpoint_path_list: List[str],
                   dataset_encoder: DatasetEncoder) -> pd.DataFrame:
    """
    Get submission from dataset.
    """
    assert len(checkpoint_path_list) > 0

    ensemble_batch_probs = None
    for checkpoint_path in tqdm(checkpoint_path_list):
        batch_probs_generator = get_model_predictions(model,
                                                      data_loader,
                                                      checkpoint_path)
        if ensemble_batch_probs is None:
            ensemble_batch_probs = list(batch_probs_generator)
        else:
            for i, batch_probs in enumerate(batch_probs_generator):
                ensemble_batch_probs[i] += batch_probs

    top_cities = torch.cat(
        [torch.topk(batch_submission, 10, dim=1).indices + 1
         for batch_submission in ensemble_batch_probs],
        axis=0
    )
    del ensemble_batch_probs
    cities_prediction = pd.DataFrame(top_cities.numpy(),
                                     columns=[f'city_id_{i}' for i in range(1, 11)])
    del top_cities
    gc.collect()

    for city_id in range(1, 11):
        cities_prediction[f'city_id_{city_id}'] = dataset_encoder.label_encoders['city_id'].inverse_transform(
            cities_prediction[f'city_id_{city_id}'] - 1).astype(int)

    submission = pd.concat([dataset.get_ids(), cities_prediction], axis=1)
    return submission


def get_ground_truth_from_dataset(df: pd.DataFrame,
                                  booking_dataset: BookingDataset,
                                  dataset_encoder: DatasetEncoder) -> pd.DataFrame:
    """
    Get ground truth from dataset. Assumes the df is sorted by checkin ASC.
    """
    ground_truth = df.groupby('utrip_id').tail(1)[['utrip_id', 'next_city_id']].set_index('utrip_id')
    ground_truth['city_id'] = (dataset_encoder
                               .label_encoders['city_id']
                               .inverse_transform(ground_truth['next_city_id'] - 1))
    if not ground_truth['city_id'].isnull().values.any():
        ground_truth['city_id'] = ground_truth['city_id'].astype(int)
    else:
        logging.warning("Warning: next_city_id has nulls")

    ground_truth = ground_truth.loc[booking_dataset.utrip_ids]  # reorder obs like batches
    ground_truth.drop(columns="next_city_id", inplace=True)
    return ground_truth


def get_count_distribution(df: pd.DataFrame,
                           by: str = 'utrip_id') -> pd.DataFrame:
    """
    Get count distribution from dataset.
    """
    df_dist = df.groupby(by)[by].count().value_counts(sort=True)
    df_dist /= df_dist.sum()
    return df_dist


def get_distribution_by_pos(**kwargs) -> pd.DataFrame:
    """
    Get distribution by pos from a list of key: dataframe pairs.
    """
    return functools.reduce(lambda a, b: a.join(b),
                            [get_count_distribution(df).to_frame(name)
                             for name, df in kwargs.items()]).sort_index()


def check_device() -> None:
    """
    Check if we are using GPU acceleration and warn the user.
    """
    if config.DEVICE != 'cuda':
        logging.warning('You are not using a GPU. If you are using colab, go to Runtime -> Change runtime type')
    else:
        current_gpu = subprocess.check_output(['nvidia-smi', '-L']).strip().decode('ascii')
        logging.info(f"Using {current_gpu}")


def get_trained_models() -> Dict:
    """
    Get dictionary of all models trained
    """
    base_path = get_path("architectures")
    model_paths = [f"{base_path}/{f}" for f in listdir(base_path) if isfile(f"{base_path}/{f}")]

    d = {}

    for path in model_paths:
        with open(path) as f:
            model_hash = path[-13:-5]
            d[model_hash] = json.load(f)
    return d


def get_final_submission(submission_set: pd.DataFrame,
                         model_hash: str,
                         dataset_encoder: DatasetEncoder) -> None:
    """
    Get final submission from model hash.
    """
    # create final submission
    dataset_submission, data_loader_submission = get_dataset_and_dataloader(
        df=submission_set,
        features=config.FEATURES_EMBEDDING
    )

    # get model parameters from hash
    with open(get_model_arch_path(model_hash)) as fhandle:
        model_parameters = json.load(fhandle)
    ckpt_list = get_model_ckpt_paths(model_hash=model_hash,
                                     checkpoint_type='accuracy_at_k')

    # load model and get predictions
    model = BookingNet(**model_parameters).to(config.DEVICE)
    predictions = get_submission(dataset_submission,
                                 data_loader_submission,
                                 model,
                                 ckpt_list,
                                 dataset_encoder)

    # build final csv and run sanity checks
    timestamp = datetime.now().strftime("%d_%m_%Y_%Hh_%Mm_%Ss")
    cols = ["utrip_id", "city_id_1", "city_id_2", "city_id_3", "city_id_4"]
    filename = f'submission_{model_hash}_{timestamp}'
    final_submission = predictions[cols]
    final_submission.to_csv(get_path(dirs="submissions",
                                     filename=filename,
                                     format='csv'),
                            index=False)
    submission_sanity_checks(final_submission)


def submission_sanity_checks(submission: pd.DataFrame) -> None:
    """
    Run submission sanity checks to make sure our dataframe is healthy.
    """
    _TOTAL_SUBMISSION_ROWS = 70662
    df = pd.read_csv(get_resources_path('booking_test_set.csv'),
                     dtype={'user_id': 'int32', 'city_id': 'int32'},
                     parse_dates=['checkin', 'checkout'])

    utrip_ids = set(df.utrip_id.unique())
    assert len(set(submission.utrip_id.unique()).intersection(utrip_ids)) == _TOTAL_SUBMISSION_ROWS
    assert submission.shape == (_TOTAL_SUBMISSION_ROWS, 5)
    assert submission.notna().values.all()

    df = pd.read_csv(get_resources_path('booking_train_set.csv'),
                     dtype={'user_id': 'int32', 'city_id': 'int32'},
                     parse_dates=['checkin', 'checkout'])

    # verify city ids
    city_ids = set(df.city_id.unique().astype(int))
    assert len(set(submission.city_id_1.unique()).difference(city_ids)) == 0
    assert len(set(submission.city_id_2.unique()).difference(city_ids)) == 0
    assert len(set(submission.city_id_3.unique()).difference(city_ids)) == 0
    assert len(set(submission.city_id_4.unique()).difference(city_ids)) == 0
    logging.info("Passed all sanity checks!")

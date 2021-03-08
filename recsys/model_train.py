import logging
import os
import sys
import time
from datetime import timedelta
from typing import Tuple, Dict, List

import pandas as pd
import torch
from recsys.model_selection import round_robin_kfold

from recsys import config

if 'ipykernel' in sys.modules:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

from recsys.config import DEVICE
from recsys.model import BookingNet
from recsys.paths import get_path, get_model_arch_path
from recsys.types import BatchType, OptimizerType


def get_model_metrics(batch: BatchType,
                      seq_len: torch.Tensor,
                      city_scores: torch.Tensor) -> Tuple:
    """
    Get model metrics, e.g. accuracy@1, accuracy@4.
    """
    bs, ts = batch['city_id'].shape
    predicted_cities = (city_scores
                        .argmax(1)
                        .view(-1, ts)
                        .gather(1, (seq_len.unsqueeze(1) - 1).to(DEVICE))
                        .squeeze(1) + 1)
    predicted_cities_top_k = (torch.topk(city_scores, 4, dim=1).indices.view(bs, -1, 4)
                              .gather(1, (torch.cat([seq_len.unsqueeze(1)] * 4, axis=1).view(-1, 1, 4) - 1)
                                      .to(DEVICE))).squeeze(1) + 1
    hits_at_1 = (predicted_cities == batch['last_city']).float().sum()
    hits_at_k = torch.sum(predicted_cities_top_k.eq(batch['last_city'].unsqueeze(1)), dim=1).float().sum()
    return hits_at_1, hits_at_k


def train_step(model: BookingNet,
               batch: List[BatchType]) -> Dict:
    """
    Training step, including loss evaluation and backprop.
    """
    batch, seq_len = batch
    model.optimizer.zero_grad(set_to_none=True)
    city_scores = model(batch, seq_len)
    city_scores = city_scores.view(-1, 39901)
    loss = model.get_loss(city_scores,
                          batch,
                          seq_len,
                          device=DEVICE)
    loss.backward()
    model.optimizer.step()
    return {
        'train_loss': loss.item()
    }


def validation_step(model: BookingNet,
                    batch: BatchType) -> Dict:
    """
    Validation step, including metric computation for batch.
    """
    batch, seq_len = batch
    city_scores = model(batch, seq_len)
    city_scores = city_scores.view(-1, 39901)
    loss = model.get_loss(city_scores,
                          batch,
                          seq_len)
    hits_at_1, hits_at_k = get_model_metrics(batch, seq_len, city_scores)
    obs = len(batch['city_id'])
    return {
        'valid_loss': loss.item(),
        'hits_at_1': hits_at_1.item(),
        'hits_at_k': hits_at_k.item(),
        'obs': obs
    }


def train_for_all_batches(model: BookingNet,
                          train_batches: List[BatchType]) -> Dict:
    """
    Train model on all given batches.
    """
    current_time = time.time()
    train_loss = 0
    model.train()
    for batch in train_batches:
        train_step_result = train_step(model, batch)
        train_loss += train_step_result['train_loss']
    train_loss /= len(train_batches)  # loss per batch
    ellapsed_time = timedelta(seconds=int(time.time() - current_time))
    return {
        'train_loss': train_loss,
        'ellapsed_time': ellapsed_time,
    }


def valid_for_all_batches(model: BookingNet,
                          valid_batches: List[BatchType]) -> Dict:
    """
    Run validation set metrics for all batches.
    """
    current_time = time.time()
    valid_result = {
        'valid_loss': 0,
        'hits_at_1': 0,
        'hits_at_k': 0,
        'obs': 0
    }
    model.eval()
    with torch.no_grad():
        for batch in valid_batches:
            batch_result = validation_step(model, batch)
            for key in valid_result.keys():
                valid_result[key] += batch_result[key]
    ellapsed_time = timedelta(seconds=int(time.time() - current_time))
    return {
        'valid_loss': valid_result['valid_loss'] / len(valid_batches),  # loss per batch
        'accuracy@1': valid_result['hits_at_1'] / valid_result['obs'],
        'accuracy@4': valid_result['hits_at_k'] / valid_result['obs'],
        'ellapsed_time_valid': ellapsed_time
    }


def model_checkpoint_exists(model_hash: str,
                            fold: int) -> bool:
    """
    Returns `true` if the model checkpoint given by the path exists, `false` otherwise.
    """
    ckpt_path = get_path(dirs=["models", model_hash],
                         filename=f"fold_{fold}_best_accuracy_at_k",
                         format="pt")
    return os.path.exists(ckpt_path)


def train_model(model: BookingNet,
                train_batches: List[BatchType],
                valid_batches: List[BatchType],
                epochs: int = 50,
                fold: int = 0,
                min_epochs_to_save: int = 20,
                verbose: bool = True) -> pd.DataFrame:
    """
    Train model from batches and save checkpoints of best models by accuracy.
    """
    epoch_report = {}
    best_accuracy_at_k = 0
    for epoch in tqdm(range(epochs)):
        train_report = train_for_all_batches(model, train_batches)
        valid_report = valid_for_all_batches(model, valid_batches)

        if epoch >= min_epochs_to_save and valid_report['accuracy@4'] > best_accuracy_at_k:
            best_accuracy_at_k = valid_report['accuracy@4']
            torch.save(model.state_dict(), get_path(dirs=["models", model.hash],
                                                    filename=f"fold_{fold}_best_accuracy_at_k",
                                                    format="pt"))

        r = dict(train_report)
        r.update(valid_report)
        epoch_report[epoch] = r

        if verbose:
            epoch_str = [f"Epoch: {epoch}",
                         f"train loss: {r['train_loss']:.4f}",
                         f"valid loss: {r['valid_loss']:.4f}",
                         f"accuracy@1: {r['accuracy@1']:.4f}",
                         f"accuracy@4: {r['accuracy@4']:.4f}",
                         f"time: {r['ellapsed_time']}"]
            epoch_str = ', '.join(epoch_str)
            logging.info(epoch_str)

    # save report
    pd.DataFrame(epoch_report).T.to_csv(get_path(dirs=["reports", model.hash],
                                                 hash=model.hash,
                                                 fold=fold,
                                                 format='csv'))

    with open(get_model_arch_path(model.hash), "w") as fhandle:
        fhandle.write(str(model))

    return pd.DataFrame(epoch_report).T


def train_model_for_folds(dataset_batches: List[BatchType],
                          train_set: pd.DataFrame,
                          model_configuration: Dict,
                          n_models: int = config.N_SPLITS,
                          min_epochs_to_save: int = 25,
                          skip_checkpoint=False) -> str:
    """
    Train `n_models` given a model configuration, returning the model hash.
    """
    for fold, (train_index, valid_index) in enumerate(round_robin_kfold(dataset_batches,
                                                                        n_splits=config.N_SPLITS)):
        if fold >= n_models:
            break

        model = BookingNet(**model_configuration).to(config.DEVICE)
        model.set_optimizer(optimizer_type=OptimizerType.ADAMW)
        model.set_entropy_weights(train_set)

        model_hash = model.hash

        if not skip_checkpoint and model_checkpoint_exists(model.hash, fold):
            continue

        train_batches = dataset_batches[train_index]
        valid_batches = dataset_batches[valid_index]
        # valid_batches = filter_batches_by_length(valid_batches)

        logging.info(f"Training model {model.hash} for fold {fold}")
        train_model(model,
                    train_batches,
                    valid_batches,
                    epochs=config.EPOCHS,
                    min_epochs_to_save=min_epochs_to_save,
                    fold=fold)

        # Empty CUDA memory
        del model
        torch.cuda.empty_cache()

    return model_hash

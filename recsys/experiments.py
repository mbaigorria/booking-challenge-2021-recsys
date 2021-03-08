import itertools
import logging
from typing import List, Tuple

import pandas as pd
from tqdm.notebook import tqdm

from recsys import config
from recsys.dataset import build_dataset, get_training_set_from_dataset, \
    get_test_set_from_dataset, set_future_features, get_submission_set_from_dataset
from recsys.dataset_loader import get_dataset_and_dataloader, batches_to_device
from recsys.encoders import DatasetEncoder
from recsys.model import BookingNet
from recsys.model_train import train_model_for_folds
from recsys.paths import get_model_ckpt_paths, get_path
from recsys.types import *
from recsys.utils import get_distribution_by_pos, get_submission, accuracy_at_k, print_gpu_usage
from recsys.utils import get_ground_truth_from_dataset, get_trained_models


def run_experiments(base_configuration: Dict,
                    experiments: List[Dict],
                    n_models: int,
                    dataset_batches: List[BatchType],
                    train_set: pd.DataFrame,
                    skip_checkpoint=False) -> None:
    """
    Given a base configuration in a dictionary, run experiments
    by overriding parameters of this base configuration with
    a list of overrides in `experiments`.
    """
    for model_overrides in tqdm(experiments):
        logging.info(model_overrides)
        model_configuration = dict(base_configuration, **model_overrides)
        train_model_for_folds(dataset_batches,
                              train_set,
                              model_configuration,
                              n_models=n_models,
                              skip_checkpoint=skip_checkpoint)


def get_base_configuration():
    """
    The base configuration describes our best model. Experiments
    change elements of this configuration to try to find an even
    better one.
    """
    return {
        'features_embedding': config.FEATURES_EMBEDDING,
        'hidden_size': int(config.EMBEDDING_SIZES['city_id'][1]),
        'output_size': int(config.EMBEDDING_SIZES['city_id'][0]),
        'embedding_sizes': config.EMBEDDING_SIZES,
        'n_layers': 2,
        'dropout': 0.3,
        'rnn_dropout': 0.1,
        'tie_embedding_and_projection': True,
        'model_type': ModelType.MANY_TO_MANY,
        'recurrent_type': RecurrentType.GRU,
        'weight_type': WeightType.UNWEIGHTED,
        'feature_projection_type': FeatureProjectionType.CONCATENATION,
        'num_folds': config.N_SPLITS,
        'batch_size': config.BATCH_SIZE
    }


def get_experiments() -> List:
    """
    An experiment is a dict that describes the parameters
    that will be overridden in the base configuration
    during an experiment.
    """
    params = ['model_type', 'weight_type', 'recurrent_type', 'tie_embedding_and_projection']
    return [
        dict(zip(params, p))
        for p in itertools.product(
            *map(list, [ModelType, WeightType, RecurrentType, [True, False]])
        )
    ]


def get_model_performance_data(test_set: pd.DataFrame,
                               dataset_encoder: DatasetEncoder,
                               model_hashes: List[str] = None) -> Tuple[pd.DataFrame, Dict]:
    """
    Get model performance data from all trained models.
    """
    booking_dataset_test, dataset_loader_test = get_dataset_and_dataloader(
        df=test_set,
        features=config.FEATURES_EMBEDDING
    )
    ground_truth_test = get_ground_truth_from_dataset(
        df=test_set,
        booking_dataset=booking_dataset_test,
        dataset_encoder=dataset_encoder
    )

    trained_models = get_trained_models()

    if model_hashes:
        trained_models = {h: trained_models[h] for h in model_hashes}

    df_rows = []
    accuracy_at_4_by_length = {}
    for model_hash, model_parameters in trained_models.items():
        try:
            ckpt_list = get_model_ckpt_paths(model_hash=model_hash,
                                             checkpoint_type='accuracy_at_k')
        except FileNotFoundError:
            continue

        d = {
            'single': ckpt_list[:1],
            'ensemble': ckpt_list
        }

        for model_type, ckpt_list in d.items():
            if model_type == 'ensemble' and len(ckpt_list) == 1:
                continue

            model = BookingNet(**model_parameters).to(config.DEVICE)
            predictions = get_submission(booking_dataset_test,
                                         dataset_loader_test,
                                         model,
                                         ckpt_list,
                                         dataset_encoder)
            accuracy = accuracy_at_k(predictions, ground_truth_test)
            model_parameters['num_models'] = len(ckpt_list)
            model_parameters['accuracy@1'] = accuracy['accuracy@1']
            model_parameters['accuracy@4'] = accuracy['accuracy@4']
            model_parameters['accuracy@10'] = accuracy['accuracy@10']
            model_parameters['hash'] = model_hash
            df = pd.DataFrame.from_dict(model_parameters, orient='index').T
            df_rows.append(pd.concat(
                [df, pd.DataFrame.from_dict(accuracy['accuracy@4_by_pos'], orient='index').T]
                , axis=1))
            accuracy_at_4_by_length[(model_hash, model_type)] = accuracy['accuracy@4_by_pos']
    return pd.concat(df_rows), accuracy_at_4_by_length


def filter_results_table(results: pd.DataFrame) -> pd.DataFrame:
    """
    Filter results table to get only attributes that change between models.
    """
    columns = ['model_type', 'recurrent_type', 'tie_embedding_and_projection',
               'weight_type', 'accuracy@1', 'accuracy@4', 'accuracy@10', 'hash']
    selected_columns = [col for col in results.columns.values
                        if results[col].apply(str).nunique() > 1
                        or col in columns]
    filtered_results = (results[selected_columns]
                        .sort_values("accuracy@4", ascending=False))

    decode = {
        'model_type': ModelType,
        'weight_type': WeightType,
        'recurrent_type': RecurrentType
    }

    for key, enum_type in decode.items():
        filtered_results[key] = (filtered_results[key]
                                 .apply(enum_type)
                                 .apply(lambda s: str(s).split('.')[1]))

    df_table = filtered_results[columns].sort_values(
        ["model_type", "recurrent_type", "tie_embedding_and_projection", "accuracy@4"],
        ascending=[True, True, False, False])
    return df_table


if __name__ == "__main__":
    # build and encode dataset
    dataset = build_dataset(reserved_obs=30000)
    de = DatasetEncoder(config.FEATURES_TO_ENCODE)
    de.fit_transform(dataset)
    set_future_features(dataset)

    submission_set = get_submission_set_from_dataset(dataset)

    # keep only observations before the last visit
    dataset = dataset[~dataset.next_city_id.isna()]

    # split training and test set from dataset
    train_set = get_training_set_from_dataset(dataset)
    test_set = get_test_set_from_dataset(dataset)

    logging.info(f"Training set: {train_set.shape}")
    logging.info(f"Test set: {test_set.shape}")
    logging.info(f"Dataset: {dataset.shape}")

    logging.info(get_distribution_by_pos(dataset=dataset,
                                         train_set=train_set[train_set.train == 1],
                                         test_set=test_set,
                                         submission=submission_set).head(10))

    _, dataset_loader = get_dataset_and_dataloader(
        train_set,
        features=config.FEATURES_EMBEDDING + ['next_city_id'],
        batch_size=config.BATCH_SIZE
    )
    dataset_batches_cuda = batches_to_device(dataset_loader)

    print_gpu_usage(0)

    # run experiments from base configuration
    base_configuration = get_base_configuration()
    experiments = get_experiments()
    run_experiments(base_configuration=base_configuration,
                    experiments=experiments,
                    n_models=1,
                    dataset_batches=dataset_batches_cuda,
                    train_set=train_set)

    # get and save results table
    results, _ = get_model_performance_data(test_set, de)

    filter_results_table(results).to_csv(
        get_path(
            filename='experiments',
            format='csv'),
        index=False
    )
